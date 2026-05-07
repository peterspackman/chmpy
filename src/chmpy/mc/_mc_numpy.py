"""Vectorised classic Marching Cubes (Lorensen 1987) in numpy.

Pure-numpy implementation of the marching-cubes isosurface extraction. The
classic algorithm has well-known topological ambiguities at saddle points;
the Lewiner variant in ``_mc_lewiner.pyx`` resolved those at the cost of a
~1500-LOC state machine. For smooth scalar fields like promolecule densities
and Hirshfeld stockholder weights, the differences are negligible — so we
ship the simpler classic algorithm here as the default.

The implementation is fully vectorised across the cube grid: there are no
Python-level loops over voxels. The hot path is per-direction edge
interpolation plus a single fancy-indexing step to assemble triangles.
"""

from __future__ import annotations

import base64

import numpy as np

from . import lookup_tables as _lt

__all__ = ["marching_cubes_classic"]


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

def _decode_table(args):
    shape, text = args
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype="int8").reshape(shape)


# (256, 16) int8: tri_table[case_id] = [e0, e1, e2, e3, e4, e5, ..., -1] giving
# up to 5 triangles (3 edge ids each); -1 is sentinel for "no more".
_TRI_TABLE = _decode_table(_lt.CASESCLASSIC).astype(np.int32)


# 12 edges of the unit cube, each described by (corner_a, corner_b) where
# corner i has relative position (xrel, yrel, zrel) = bit decomposition of i
# under the convention bit0 = x, bit1 = y, bit2 = z. Cube-corner numbering
# follows the Lewiner / scikit-image LUT (corner ids 0-7 cycle the bottom
# square then the top square):
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
#   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
_EDGE_CORNERS = np.array(
    [
        (0, 1),  # edge 0: x at (0,0,0)
        (1, 2),  # edge 1: y at (1,0,0)
        (2, 3),  # edge 2: x at (0,1,0)
        (3, 0),  # edge 3: y at (0,0,0)
        (4, 5),  # edge 4: x at (0,0,1)
        (5, 6),  # edge 5: y at (1,0,1)
        (6, 7),  # edge 6: x at (0,1,1)
        (7, 4),  # edge 7: y at (0,0,1)
        (0, 4),  # edge 8: z at (0,0,0)
        (1, 5),  # edge 9: z at (1,0,0)
        (2, 6),  # edge 10: z at (1,1,0)
        (3, 7),  # edge 11: z at (0,1,0)
    ],
    dtype=np.int32,
)


# Per-cube-edge offset that, given a cube position (cz, cy, cx), maps to the
# global edge index. Each entry is (direction, dz, dy, dx) where direction is
# 0=x, 1=y, 2=z and the (dz, dy, dx) offsets are added to the cube position to
# locate the edge in the global edge grid.
_EDGE_OFFSET = np.array(
    [
        (0, 0, 0, 0),  # edge 0: x-edge at (cz,   cy,   cx)
        (1, 0, 0, 1),  # edge 1: y-edge at (cz,   cy,   cx+1)
        (0, 0, 1, 0),  # edge 2: x-edge at (cz,   cy+1, cx)
        (1, 0, 0, 0),  # edge 3: y-edge at (cz,   cy,   cx)
        (0, 1, 0, 0),  # edge 4: x-edge at (cz+1, cy,   cx)
        (1, 1, 0, 1),  # edge 5: y-edge at (cz+1, cy,   cx+1)
        (0, 1, 1, 0),  # edge 6: x-edge at (cz+1, cy+1, cx)
        (1, 1, 0, 0),  # edge 7: y-edge at (cz+1, cy,   cx)
        (2, 0, 0, 0),  # edge 8: z-edge at (cz,   cy,   cx)
        (2, 0, 0, 1),  # edge 9: z-edge at (cz,   cy,   cx+1)
        (2, 0, 1, 1),  # edge 10: z-edge at (cz,   cy+1, cx+1)
        (2, 0, 1, 0),  # edge 11: z-edge at (cz,   cy+1, cx)
    ],
    dtype=np.int32,
)


def _interp_edges(values_a, values_b, level):
    """Linear interpolation parameter t in [0, 1] for each edge.

    Returns t = (level - va) / (vb - va), clipped to [0, 1] for numerical
    stability. Edges where va == vb are returned with t = 0.5 (midpoint), but
    those edges aren't crossed and won't contribute vertices anyway.
    """
    denom = values_b - values_a
    # Avoid divide-by-zero; the result is masked off later for non-crossing
    # edges, so any finite value is fine here.
    denom = np.where(denom == 0, np.float32(1.0), denom)
    t = (level - values_a) / denom
    return np.clip(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def marching_cubes_classic(volume, level, spacing=(1.0, 1.0, 1.0), step_size=1):
    """Extract an isosurface from a 3D scalar field using classic Marching Cubes.

    Args:
        volume: ``(M, N, P)`` array of scalar values. Indexing convention is
            ``volume[z, y, x]`` (matches scikit-image).
        level: contour value at which to extract the surface.
        spacing: per-axis voxel spacing ``(dz, dy, dx)``.
        step_size: take every Nth voxel along each axis. ``1`` means full
            resolution; larger values are coarser but cheaper.

    Returns:
        ``(vertices, faces, normals, values)`` where ``vertices`` is ``(V, 3)``
        in ``(z, y, x)`` order, ``faces`` is ``(F, 3)`` of vertex indices,
        ``normals`` is ``(V, 3)`` (unit-length gradient at each vertex), and
        ``values`` is ``(V,)`` (interpolated scalar value at each vertex,
        equal to ``level`` up to fp).
    """
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    if step_size > 1:
        volume = volume[::step_size, ::step_size, ::step_size]
        spacing = tuple(s * step_size for s in spacing)

    nz, ny, nx = volume.shape
    if nz < 2 or ny < 2 or nx < 2:
        raise ValueError("Input array must be at least 2x2x2.")

    level_f = np.float32(level)
    sz, sy, sx = (np.float32(s) for s in spacing)

    # Cube grid shape: (nz-1, ny-1, nx-1)
    above = volume > level_f

    # 8-bit case index per cube (corner i contributes bit i).
    # Corner offsets in (z, y, x) per the table at top of file.
    corner_offsets = np.array(
        [
            (0, 0, 0),  # 0
            (0, 0, 1),  # 1
            (0, 1, 1),  # 2
            (0, 1, 0),  # 3
            (1, 0, 0),  # 4
            (1, 0, 1),  # 5
            (1, 1, 1),  # 6
            (1, 1, 0),  # 7
        ],
        dtype=np.int32,
    )

    cube_case = np.zeros((nz - 1, ny - 1, nx - 1), dtype=np.uint16)
    for i, (dz, dy, dx) in enumerate(corner_offsets):
        cube_case |= (
            above[dz : dz + nz - 1, dy : dy + ny - 1, dx : dx + nx - 1].astype(
                np.uint16
            )
            << np.uint16(i)
        )

    # ----- per-direction active edge masks -----
    # Active edges are those whose two endpoints straddle the isovalue.
    x_active = above[:, :, :-1] != above[:, :, 1:]
    y_active = above[:, :-1, :] != above[:, 1:, :]
    z_active = above[:-1, :, :] != above[1:, :, :]

    n_x_verts = int(x_active.sum())
    n_y_verts = int(y_active.sum())
    n_z_verts = int(z_active.sum())
    n_total = n_x_verts + n_y_verts + n_z_verts

    if n_total == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    # Build per-direction id grids: -1 for inactive edges, otherwise the global
    # vertex index. y-edges follow x-edges in the global numbering; z-edges
    # follow y-edges.
    x_ids_global = np.full(x_active.shape, -1, dtype=np.int64)
    x_ids_global[x_active] = np.arange(n_x_verts, dtype=np.int64)
    y_ids_global = np.full(y_active.shape, -1, dtype=np.int64)
    y_ids_global[y_active] = np.arange(
        n_x_verts, n_x_verts + n_y_verts, dtype=np.int64
    )
    z_ids_global = np.full(z_active.shape, -1, dtype=np.int64)
    z_ids_global[z_active] = np.arange(
        n_x_verts + n_y_verts, n_total, dtype=np.int64
    )

    # ----- compute vertex positions and gradients only at active edges -----
    # Gradient is via central differences on the full volume. We need it for
    # vertex normals, interpolated linearly along each edge.
    gz, gy, gx = np.gradient(volume)
    gz = gz / sz
    gy = gy / sy
    gx = gx / sx

    vertices = np.empty((n_total, 3), dtype=np.float32)
    normals = np.empty((n_total, 3), dtype=np.float32)

    def _fill_dir_block(active, vol_a, vol_b, ga, gb, slot, axis_offset):
        """Compute interpolated vertex positions + gradients for one edge dir.

        ``vol_a, vol_b`` are the value arrays at the two endpoints (full grid
        view); ``ga, gb`` are stacked gradient arrays at the endpoints with
        shape ``(3, *vol_a.shape)`` (gz, gy, gx). ``slot`` is the destination
        ``slice`` in ``vertices`` / ``normals``. ``axis_offset`` is the
        ``(z, y, x)`` integer offset applied to the active index, with the
        edge axis handled separately via ``t``.
        """
        az, ay, ax = np.nonzero(active)
        va = vol_a[az, ay, ax]
        vb = vol_b[az, ay, ax]
        denom = vb - va
        denom = np.where(denom == 0, np.float32(1.0), denom)
        t = (level_f - va) / denom
        np.clip(t, 0.0, 1.0, out=t)

        zf = az.astype(np.float32) + np.float32(axis_offset[0]) * t
        yf = ay.astype(np.float32) + np.float32(axis_offset[1]) * t
        xf = ax.astype(np.float32) + np.float32(axis_offset[2]) * t
        vertices[slot, 0] = zf * sz
        vertices[slot, 1] = yf * sy
        vertices[slot, 2] = xf * sx

        # Gradient at vertex = lerp(g_a, g_b, t)
        normals[slot, 0] = ga[0][az, ay, ax] + t * (gb[0][az, ay, ax] - ga[0][az, ay, ax])
        normals[slot, 1] = ga[1][az, ay, ax] + t * (gb[1][az, ay, ax] - ga[1][az, ay, ax])
        normals[slot, 2] = ga[2][az, ay, ax] + t * (gb[2][az, ay, ax] - ga[2][az, ay, ax])

    # x-edges: a=(z,y,x), b=(z,y,x+1). Edge axis is x → (0, 0, 1) offset.
    _fill_dir_block(
        x_active,
        volume[:, :, :-1], volume[:, :, 1:],
        (gz[:, :, :-1], gy[:, :, :-1], gx[:, :, :-1]),
        (gz[:, :, 1:],  gy[:, :, 1:],  gx[:, :, 1:]),
        slice(0, n_x_verts),
        (0, 0, 1),
    )
    # y-edges
    _fill_dir_block(
        y_active,
        volume[:, :-1, :], volume[:, 1:, :],
        (gz[:, :-1, :], gy[:, :-1, :], gx[:, :-1, :]),
        (gz[:, 1:, :],  gy[:, 1:, :],  gx[:, 1:, :]),
        slice(n_x_verts, n_x_verts + n_y_verts),
        (0, 1, 0),
    )
    # z-edges
    _fill_dir_block(
        z_active,
        volume[:-1, :, :], volume[1:, :, :],
        (gz[:-1, :, :], gy[:-1, :, :], gx[:-1, :, :]),
        (gz[1:, :, :],  gy[1:, :, :],  gx[1:, :, :]),
        slice(n_x_verts + n_y_verts, n_total),
        (1, 0, 0),
    )

    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lengths[norm_lengths == 0] = 1.0
    normals /= norm_lengths

    # ----- per-cube triangle assembly -----
    # Restrict to cubes that actually contain triangles. For typical
    # surface workloads only ~1-3% of cubes are active so this skips an
    # enormous amount of work compared to processing the full grid.
    active_mask = (cube_case != 0) & (cube_case != 255)
    if not active_mask.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    cz_a, cy_a, cx_a = np.nonzero(active_mask)
    cube_case_a = cube_case[cz_a, cy_a, cx_a]
    tris_a = _TRI_TABLE[cube_case_a]  # (n_active, 16)

    # For each active cube, look up the global vertex id for each of its 12
    # edges (only as many gathers as we have edges).
    n_active = cz_a.shape[0]
    cube_edge_to_vid = np.empty((n_active, 12), dtype=np.int64)
    for e, (direction, dz, dy, dx) in enumerate(_EDGE_OFFSET):
        zz_e = cz_a + dz
        yy_e = cy_a + dy
        xx_e = cx_a + dx
        if direction == 0:
            cube_edge_to_vid[:, e] = x_ids_global[zz_e, yy_e, xx_e]
        elif direction == 1:
            cube_edge_to_vid[:, e] = y_ids_global[zz_e, yy_e, xx_e]
        else:
            cube_edge_to_vid[:, e] = z_ids_global[zz_e, yy_e, xx_e]

    # Map cube-edge ids in each triangle to global vertex ids; -1 entries
    # are sentinel "no triangle" markers.
    mask = tris_a >= 0
    tris_safe = np.where(mask, tris_a, 0).astype(np.int64)
    rows = np.arange(n_active)[:, None]
    face_vids_flat = cube_edge_to_vid[rows, tris_safe]
    faces = face_vids_flat[mask].reshape(-1, 3).astype(np.int32)

    # Vertex values are exactly the isolevel for true linear interpolation.
    values = np.full(vertices.shape[0], np.float32(level), dtype=np.float32)

    return vertices, faces, normals, values

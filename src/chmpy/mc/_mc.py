"""Marching cubes wrapper.

Thin wrapper over the pure-numpy classic MC implementation in ``_mc_numpy``.
The classic algorithm has well-known topological ambiguities at saddle points
that the Lewiner variant resolves at the cost of a ~1500-LOC state machine —
for the smooth scalar fields chmpy works with (promolecule densities,
Hirshfeld stockholder weights, void surfaces, etc.) the difference is
negligible in practice.

The previous implementation was a port of scikit-image's Lewiner cython
module; the public API kept here matches what those callers expected.

Acknowledgement:
    The ``CASESCLASSIC`` lookup table used by the underlying ``_mc_numpy``
    module is derived from the scikit-image project (BSD-3 licensed).
"""

from __future__ import annotations

import numpy as np

from ._mc_numpy import marching_cubes_classic as _marching_cubes_classic

__all__ = ["marching_cubes"]


def marching_cubes(
    volume,
    level=None,
    spacing=(1.0, 1.0, 1.0),
    gradient_direction="descent",
    step_size=1,
    allow_degenerate=True,
    use_classic=True,
):
    """Extract a triangular isosurface from a 3D scalar volume.

    Args:
        volume: ``(M, N, P)`` numpy array of scalar values. Indexing
            convention is ``volume[z, y, x]`` (matches scikit-image).
        level: contour value to extract; defaults to the midpoint of the
            volume's value range.
        spacing: voxel spacing along each axis ``(dz, dy, dx)``.
        gradient_direction: ``"descent"`` (default) flips the triangle winding
            so face normals point toward higher values (i.e. into the object
            for an enclosing isosurface). ``"ascent"`` keeps the right-handed
            winding the algorithm produces internally.
        step_size: take every Nth voxel along each axis. ``1`` means full
            resolution; larger values are coarser but cheaper.
        allow_degenerate: kept for backwards compatibility; currently has no
            effect — the numpy implementation does not generate zero-area
            triangles in the first place under typical inputs.
        use_classic: kept for backwards compatibility; the only available
            algorithm now is classic Marching Cubes.

    Returns:
        A 4-tuple ``(vertices, faces, normals, values)``:

        - ``vertices``: ``(V, 3)`` float32 array in ``(x, y, z)`` order.
        - ``faces``: ``(F, 3)`` int32 array of vertex indices.
        - ``normals``: ``(V, 3)`` float32 array of unit-length vertex normals.
        - ``values``: ``(V,)`` float32 array, equal to ``level`` up to fp.

    Raises:
        ValueError: input volume isn't 3-D, has any axis < 2, or ``level`` is
            outside the volume's value range.
        RuntimeError: no isosurface found at the given level.
    """
    if not isinstance(volume, np.ndarray) or volume.ndim != 3:
        raise ValueError("Input volume should be a 3D numpy array.")
    if any(s < 2 for s in volume.shape):
        raise ValueError("Input array must be at least 2x2x2.")

    volume = np.ascontiguousarray(volume, np.float32)

    if level is None:
        level = 0.5 * (volume.min() + volume.max())
    else:
        level = float(level)
        if level < volume.min() or level > volume.max():
            raise ValueError("Surface level must be within volume data range.")

    if len(spacing) != 3:
        raise ValueError("`spacing` must consist of three floats.")

    step_size = int(step_size)
    if step_size < 1:
        raise ValueError("step_size must be at least one.")

    if gradient_direction not in ("descent", "ascent"):
        raise ValueError(
            f"Incorrect input {gradient_direction!r} in `gradient_direction`."
        )

    vertices, faces, normals, values = _marching_cubes_classic(
        volume, level, spacing=spacing, step_size=step_size
    )

    if not len(vertices):
        raise RuntimeError("No surface found at the given iso value.")

    # Numpy MC returns vertices in (z, y, x); flip to (x, y, z) for parity with
    # the legacy output.
    vertices = np.ascontiguousarray(vertices[:, ::-1])
    normals = np.ascontiguousarray(normals[:, ::-1])

    if gradient_direction == "descent":
        faces = np.ascontiguousarray(faces[:, ::-1])

    return vertices, faces, normals, values

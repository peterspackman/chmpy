"""A rotatable collection of spheres and edges, decoupled from chemistry.

Everything below the Scene boundary is geometry: centres, radii, colours and a
camera. Nothing here knows what a crystal is, which is what makes it portable
to another language later if the frame rate ever demands it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.sparse import csgraph

from chmpy.core.element import Element
from chmpy.core.molecule import Molecule

from .canvas import Pixels, draw_cylinders, draw_spheres


@dataclass
class Scene:
    centers: np.ndarray  # (N, 3)
    radii: np.ndarray  # (N,)
    colors: np.ndarray  # (N, 3)
    bonds: list = field(default_factory=list)  # [(p0, p1, rgb0, rgb1, radius)]
    edges: list = field(default_factory=list)  # [(p0, p1)] in world coords
    axes: list = field(default_factory=list)  # [(start, end, rgb)] cell vectors
    origin: np.ndarray = None  # rotation pivot

    def __post_init__(self):
        if self.origin is None:
            self.origin = (
                self.centers.mean(axis=0) if len(self.centers) else np.zeros(3)
            )

    def __len__(self):
        return len(self.centers)

    def arrow_size(self):
        """(shaft, head) radii for the cell vectors, in world units."""
        span = float(np.linalg.norm(self.centers - self.origin, axis=1).max()) if len(
            self.centers
        ) else 1.0
        shaft = max(0.09, span * 0.014)
        return shaft, shaft * 2.4

    def geometry(self):
        """Every point the frame has to contain, with its padding radius.

        A bond or an arrow is a rod of finite thickness, so its endpoints need
        the same padding an atom gets; a cell edge is a hairline and needs
        none.
        """
        points = [self.centers]
        radii = [self.radii]

        def add(endpoints, radius):
            points.append(np.asarray(endpoints, dtype=float))
            radii.append(np.full(2, radius))

        for start, end, _rgb0, _rgb1, radius in self.bonds:
            add((start, end), radius)
        for start, end in self.edges:
            add((start, end), 0.0)
        head = self.arrow_size()[1]
        for start, end, _rgb in self.axes:
            add((start, end), head)
        return np.vstack(points), np.concatenate(radii)

    def view_extent(self, rotation, margin=1.04):
        """Half-width and half-height needed about the pivot, for this view.

        Measured about the rotation pivot rather than the projected centroid,
        so the structure does not drift across the frame as it turns. The
        margin keeps the outermost atom off the frame edge, where an exact
        fit would leave it sitting in the border row.
        """
        points, radii = self.geometry()
        p = (points - self.origin) @ rotation.T
        reach = np.abs(p[:, :2]) + radii[:, None]
        ex, ey = reach.max(axis=0) * margin
        return max(float(ex), 1e-3), max(float(ey), 1e-3)


def rotation_matrix(yaw, pitch, roll=0.0):
    """Rotation from yaw (about screen y), pitch (about screen x), roll (z)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def _rgb(atomic_numbers):
    return np.array([Element[int(z)].color[:3] for z in atomic_numbers], dtype=float)


BOND_RADIUS = 0.17


def _bond_segments(positions, elements, bonds, radius=BOND_RADIUS):
    """Bonds as coloured segments; sampling is deferred to render time."""
    colors = _rgb(elements)
    return [
        (positions[i], positions[j], colors[i] * 0.85, colors[j] * 0.85, radius)
        for i, j, _ in bonds
    ]


def molecule_scene(molecule, style="ball-and-stick"):
    """Spheres, and bonds when the style calls for them."""
    # the longest dimension of the molecule ends up across the screen
    pos = molecule.positions_in_molecular_axis_frame()
    z = molecule.atomic_numbers

    if style == "space-filling":
        radii = np.array([Element[int(n)].vdw_radius for n in z])
        return Scene(pos, radii, _rgb(z), origin=np.zeros(3))

    if molecule.bonds is None:
        molecule.guess_bonds()
    radii = np.array([Element[int(n)].ball_stick_radius for n in z]) * 0.9
    return Scene(
        pos,
        radii,
        _rgb(z),
        bonds=_bond_segments(pos, z, molecule.unique_bonds),
        origin=np.zeros(3),
    )


def crystal_scene(crystal, cells=(1, 1, 1), style="ball-and-stick"):
    """Spheres and bonds for a block of unit cells, plus the cell edges.

    A molecular crystal is built from whole molecules: chopping atoms at the
    cell boundary would draw half a molecule here and half a cell away, which
    is not what the structure looks like. A framework or ionic solid has no
    finite molecule to complete, so it falls back to the atoms of the block
    with bonds found by distance.
    """
    hi = tuple(int(c) for c in cells)
    molecular = not bonding_percolates(crystal)

    if molecular:
        pos, z, bonds = _molecular_block(crystal, hi, style)
    else:
        pos, z, bonds = _network_block(crystal, hi, style)

    if style == "ball-and-stick":
        radii = np.array([Element[int(n)].ball_stick_radius for n in z]) * 1.1
    else:
        radii = np.array([Element[int(n)].vdw_radius for n in z])

    centre = crystal.to_cartesian(np.array([[0.5, 0.5, 0.5]]) * np.array(hi))[0]
    scene = Scene(
        pos,
        radii,
        _rgb(z),
        edges=_cell_edges(crystal, hi),
        axes=_cell_axes(crystal),
        origin=centre,
    )
    if style == "ball-and-stick":
        scene.bonds = bonds
    return scene


#: a, b, c - the usual crystallographic red / green / blue, but muted enough
#: to stay readable against both a light and a dark terminal
AXIS_COLORS = ((228, 86, 86), (96, 196, 112), (104, 150, 244))


def _cell_edges(crystal, cells):
    """The twelve edges of every cell in the block, not one box around it all.

    Drawing a single box around a 2x2x2 block hides the lattice: what you want
    to see is the repeat, so each cell gets its own outline.
    """
    corners = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float
    )
    cart = crystal.to_cartesian(corners)
    pairs = [
        (cart[a], cart[b])
        for a in range(8)
        for b in range(a + 1, 8)
        if np.count_nonzero(corners[a] != corners[b]) == 1
    ]
    shifts = crystal.to_cartesian(
        np.array(list(product(*(range(n) for n in cells))), dtype=float)
    )
    return [(p0 + s, p1 + s) for s in shifts for p0, p1 in pairs]


def _cell_axes(crystal):
    """The three cell vectors as coloured arrows from the cell origin."""
    origin = crystal.to_cartesian(np.zeros((1, 3)))[0]
    direct = crystal.unit_cell.direct
    return [(origin, origin + direct[i], AXIS_COLORS[i]) for i in range(3)]


def _molecular_block(crystal, cells, style):
    """Whole molecules of the unit cell, tiled by lattice translations."""
    molecules = crystal.unit_cell_molecules()
    shifts = crystal.to_cartesian(
        np.array(list(product(*(range(n) for n in cells))), dtype=float)
    )
    positions, elements, bonds = [], [], []
    for shift in shifts:
        for mol in molecules:
            pos = mol.positions + shift
            positions.append(pos)
            elements.append(mol.atomic_numbers)
            if style == "ball-and-stick":
                bonds.extend(
                    _bond_segments(pos, mol.atomic_numbers, mol.unique_bonds)
                )
    return np.vstack(positions), np.concatenate(elements), bonds


def _network_block(crystal, cells, style):
    """Atoms of the block with bonds by distance, for frameworks and salts."""
    # slab bounds are inclusive at both ends, so n cells spans 0 .. n-1
    last = tuple(n - 1 for n in cells)
    slab = crystal.slab(bounds=((0, 0, 0), last))
    pos, z = slab["cart_pos"], slab["element"]
    bonds = []
    if style == "ball-and-stick":
        mol = Molecule.from_arrays(z, pos)
        mol.guess_bonds()
        bonds = _bond_segments(pos, z, mol.unique_bonds)
    return pos, z, bonds


def bonding_percolates(crystal, tolerance=0.4):
    """True when covalent bonding runs through the lattice.

    Each component is unwrapped by walking a spanning tree and accumulating the
    cell translation on every edge. If a remaining edge then disagrees with the
    shifts that walk assigned, the component closes a loop through a lattice
    translation - it is a framework or a chain, not a finite molecule.
    """
    graph, edge_cells = crystal.unit_cell_connectivity(tolerance=tolerance)
    n_comp, labels = csgraph.connected_components(csgraph=graph, directed=False)
    shifts = np.zeros((graph.shape[0], 3))
    for comp in range(n_comp):
        root = int(np.flatnonzero(labels == comp)[0])
        order, pred = csgraph.breadth_first_order(
            csgraph=graph, i_start=root, directed=False
        )
        for j in order[1:]:
            i = pred[j]
            shifts[j] = (
                shifts[i] - edge_cells[(j, i)]
                if j < i
                else shifts[i] + edge_cells[(i, j)]
            )
    return any(
        not np.allclose(shifts[j] - shifts[i], cell)
        for (i, j), cell in edge_cells.items()
    )


def render_scene(
    scene,
    rotation=None,
    width=88,
    zoom=1.0,
    edge_color=None,
    depth_cue=True,
    height=None,
    theme=None,
    shading="lit",
    extent=None,
):
    """Rasterise a Scene into a Pixels image under the given rotation."""
    from .canvas import theme_for

    theme = theme or theme_for(None)
    edge_color = edge_color or theme.edge
    r = np.eye(3) if rotation is None else rotation
    pts = (scene.centers - scene.origin) @ r.T

    height = height or width
    # default to a tight fit for this view; callers that rotate interactively
    # pass their own extent so the framing does not breathe frame to frame
    ex, ey = extent if extent is not None else scene.view_extent(r)
    z = max(zoom, 1e-3)
    ex, ey = ex / z, ey / z
    # both dimensions have to fit, or a frame wider than it is tall - which
    # every terminal is - crops the structure top and bottom
    scale = min((width - 1) / (2 * ex), (height - 1) / (2 * ey))
    origin = (-0.5 * width / scale, 0.5 * height / scale)

    px = Pixels(height, width)
    cue = None
    if depth_cue and len(pts):
        cue = (pts[:, 2].max(), pts[:, 2].min() - 1.0)

    shared = {"depth_cue": cue, "fade_to": theme.background,
              "shading": shading}
    zbuffer = draw_spheres(
        px, pts, scene.radii, scene.colors, scale, origin, **shared
    )
    if scene.bonds:
        starts, ends, c0, c1, radii = _bond_arrays(scene, r)
        draw_cylinders(
            px, starts, ends, radii, c0, c1, scale, origin,
            zbuffer=zbuffer, **shared,
        )

    for p0, p1 in scene.edges:
        a = (np.asarray(p0) - scene.origin) @ r.T
        b = (np.asarray(p1) - scene.origin) @ r.T
        _line3(px, a, b, scale, origin, edge_color, zbuffer=zbuffer)

    if scene.axes:
        _draw_axes(px, scene, r, scale, origin, zbuffer, shared)
    return px


def _draw_axes(px, scene, rotation, scale, origin, zbuffer, shared):
    """The a/b/c cell vectors, as solid arrows rather than hairlines.

    These are the one part of the frame that should read even when it crosses
    a dense structure, so they are real geometry through the same z-buffer -
    a shaft cylinder with a sphere for the head.
    """
    shaft, head_r = scene.arrow_size()
    starts = (np.array([a[0] for a in scene.axes]) - scene.origin) @ rotation.T
    ends = (np.array([a[1] for a in scene.axes]) - scene.origin) @ rotation.T
    colors = np.array([a[2] for a in scene.axes], dtype=float)

    # shorten the shaft so it stops where the head begins
    direction = ends - starts
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = direction / np.maximum(length, 1e-9)
    shaft_ends = ends - unit * head_r

    # reference geometry keeps its full colour: fading a/b/c into the distance
    # would make the frame hardest to read exactly where the cell is deepest
    unfaded = {**shared, "depth_cue": None, "fade_to": None}
    draw_cylinders(
        px, starts, shaft_ends, np.full(len(starts), shaft),
        colors, colors, scale, origin, zbuffer=zbuffer, **unfaded,
    )
    draw_spheres(
        px, ends, np.full(len(ends), head_r), colors, scale, origin,
        zbuffer=zbuffer, **unfaded,
    )


def _line3(px, a, b, scale, origin, color, alpha=0.45, zbuffer=None):
    """A cell edge, drawn only where it is actually in front of the structure.

    The edge is a hairline rather than geometry, so it is composited rather
    than rasterised - but it still has to respect the depth buffer, or the
    box floats on top of the atoms instead of running through them.
    """
    h, w = px.shape
    p0 = np.array([(a[0] - origin[0]) * scale, (origin[1] - a[1]) * scale])
    p1 = np.array([(b[0] - origin[0]) * scale, (origin[1] - b[1]) * scale])
    n = int(np.max(np.abs(p1 - p0))) + 1
    t = np.linspace(0, 1, 2 * n)

    cols = np.rint(p0[0] + t * (p1[0] - p0[0])).astype(int)
    rows = np.rint(p0[1] + t * (p1[1] - p0[1])).astype(int)
    zs = a[2] + t * (b[2] - a[2])

    keep = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if zbuffer is not None:
        keep &= zs >= zbuffer[np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)]
    rows, cols = rows[keep], cols[keep]
    if not len(rows):
        return

    col = np.asarray(color, dtype=float)
    covered = px.alpha[rows, cols]
    px.rgb[rows[covered], cols[covered]] = (
        (1 - alpha) * px.rgb[rows[covered], cols[covered]] + alpha * col
    )
    px.rgb[rows[~covered], cols[~covered]] = col * 0.55
    px.alpha[rows, cols] = True


def _bond_arrays(scene, rotation):
    """Bond endpoints in view space, as arrays for the cylinder rasteriser."""
    starts = np.array([b[0] for b in scene.bonds], dtype=float)
    ends = np.array([b[1] for b in scene.bonds], dtype=float)
    starts = (starts - scene.origin) @ rotation.T
    ends = (ends - scene.origin) @ rotation.T
    c0 = np.array([b[2] for b in scene.bonds], dtype=float)
    c1 = np.array([b[3] for b in scene.bonds], dtype=float)
    radii = np.array([b[4] for b in scene.bonds], dtype=float)
    return starts, ends, c0, c1, radii


# --------------------------------------------------------------- aiming ----


def look_along(direction, up_hint=None):
    """Rotation that puts `direction` out of the screen, toward the viewer.

    The returned matrix has the screen basis as its rows, matching what
    `render_scene` expects. `up_hint` fixes the roll; the component of it
    perpendicular to the view direction becomes screen-up.
    """
    w = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        raise ValueError("view direction must be non-zero")
    w = w / norm

    if up_hint is None:
        # whichever cardinal axis is least parallel to the view direction
        up_hint = np.eye(3)[int(np.argmin(np.abs(w)))]
    up_hint = np.asarray(up_hint, dtype=float)

    right = np.cross(up_hint, w)
    if np.linalg.norm(right) < 1e-9:  # hint was parallel; pick another
        up_hint = np.eye(3)[(int(np.argmin(np.abs(w))) + 1) % 3]
        right = np.cross(up_hint, w)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(w, right), w])


def lattice_direction(crystal, indices, kind="uvw"):
    """Cartesian vector for a [uvw] zone axis or an (hkl) plane normal.

    [uvw] is a direction in the direct lattice, u*a + v*b + w*c. (hkl) names a
    plane, and the direction of interest is its normal, which is the
    reciprocal-lattice vector h*a* + k*b* + l*c*. For a non-orthogonal cell
    these are genuinely different directions, so the distinction matters.
    """
    indices = np.asarray(indices, dtype=float)
    cell = crystal.unit_cell
    if kind == "hkl":
        return indices @ cell.reciprocal_lattice
    return indices @ cell.direct


#: named directions, resolved against the lattice unless marked cartesian
NAMED = {
    "a": ((1, 0, 0), "uvw"),
    "b": ((0, 1, 0), "uvw"),
    "c": ((0, 0, 1), "uvw"),
    "a*": ((1, 0, 0), "hkl"),
    "b*": ((0, 1, 0), "hkl"),
    "c*": ((0, 0, 1), "hkl"),
    "x": ((1, 0, 0), "cartesian"),
    "y": ((0, 1, 0), "cartesian"),
    "z": ((0, 0, 1), "cartesian"),
}


def parse_direction(text):
    """Parse a view direction into (indices, kind).

    Accepts crystallographic notation - '[uvw]' for a zone axis, '(hkl)' for a
    plane normal, bare digits defaulting to a zone axis - as well as the named
    axes a/b/c, their reciprocals a*/b*/c*, and the cartesian x/y/z.

    Indices may be run together in the usual way, with a minus sign binding to
    the digit that follows ('1-10' is [1,-1,0]), or separated by spaces or
    commas when they need more than one digit ('10 -2 1').
    """
    raw = text.strip().lower()
    if not raw:
        raise ValueError("no direction given")
    if raw in NAMED:
        return NAMED[raw]

    kind = "uvw"
    if raw.startswith("(") and raw.endswith(")"):
        kind, raw = "hkl", raw[1:-1]
    elif raw.startswith("[") and raw.endswith("]"):
        kind, raw = "uvw", raw[1:-1]
    elif raw.startswith("{") and raw.endswith("}"):
        kind, raw = "hkl", raw[1:-1]
    raw = raw.strip()

    if any(sep in raw for sep in " ,;"):
        parts = [p for p in re.split(r"[\s,;]+", raw) if p]
    else:
        # run-together form: a minus sign belongs to the digit after it
        parts = re.findall(r"-?\d", raw)
        if "".join(parts) != raw.replace(" ", ""):
            raise ValueError(f"cannot read direction {text!r}")

    if len(parts) != 3:
        raise ValueError(f"need three indices, got {len(parts)} from {text!r}")
    try:
        indices = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"cannot read direction {text!r}") from exc
    if not any(indices):
        raise ValueError("direction must be non-zero")
    return indices, kind


def format_direction(indices, kind):
    """The crystallographic spelling of a parsed direction.

    Multi-digit indices are spaced out, since running them together would
    produce something this module could not read back.
    """
    joiner = " " if max(abs(i) for i in indices) > 9 else ""
    body = joiner.join(str(i) for i in indices)
    if kind == "hkl":
        return f"({body})"
    if kind == "cartesian":
        return "xyz"[int(np.argmax(np.abs(indices)))]
    return f"[{body}]"


def view_rotation(indices, kind="uvw", crystal=None):
    """Rotation looking along a direction, resolved against a lattice if given.

    For a crystal the roll is fixed by whichever cell vector is closest to
    perpendicular to the view, so an axis-aligned view comes out the way it
    would be drawn on paper rather than at an arbitrary twist.
    """
    if crystal is None or kind == "cartesian":
        return look_along(indices)
    w = lattice_direction(crystal, indices, kind)
    direct = crystal.unit_cell.direct
    cosine = np.abs(direct @ w) / (
        np.linalg.norm(direct, axis=1) * np.linalg.norm(w)
    )
    return look_along(w, direct[int(np.argmin(cosine))])

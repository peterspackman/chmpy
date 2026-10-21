"""The k nearest neighbours of every atom, periodic images included.

Structure identification -- common neighbour analysis and its relatives --
asks for a fixed number of nearest neighbours rather than every pair inside a
cutoff, so this complements `NeighborList` rather than replacing it.

Periodicity is handled by padding the cell with the images of atoms close to
its faces and searching one KD-tree over atoms and images. The padding is
chosen from the density and checked afterwards: the search is exact for an
atom whose k-th neighbour lies inside the padding, and the padding grows for
any that does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

#: the initial padding as a multiple of the radius of a sphere expected to hold
#: k atoms at the mean density
PADDING_FACTOR = 1.5


@dataclass(frozen=True)
class NearestNeighbors:
    """The `k` nearest neighbours of each atom, nearest first.

    Atoms with fewer than `k` neighbours (only possible without full
    periodicity) have their missing entries filled with index -1, a NaN
    vector and an infinite distance.

    Attributes:
        indices: (N, k) index of each neighbour
        vectors: (N, k, 3) displacement from the atom to the image of each
            neighbour, Angstroms
        distances: (N, k) lengths of `vectors`, ascending along each row
    """

    indices: np.ndarray
    vectors: np.ndarray
    distances: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def k(self) -> int:
        return self.indices.shape[1]

    def __repr__(self) -> str:
        return f"<NearestNeighbors {len(self)} atoms, k={self.k}>"


def nearest_neighbors(
    positions, k, cell=None, pbc=None, workers=-1
) -> NearestNeighbors:
    """The `k` nearest neighbours of every atom.

    An atom's own periodic images count as neighbours, so a one-atom cell has
    a full set.

    Args:
        positions: (N, 3) Cartesian positions, Angstroms
        k: neighbours per atom
        cell: (3, 3) lattice vectors as rows, or None for an isolated system.
            Rows along non-periodic directions are ignored and may be zero.
        pbc: (3,) which directions are periodic, or a single bool. Defaults to
            fully periodic when a cell is given.
        workers: threads for the search, -1 for one per CPU

    Returns:
        NearestNeighbors
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    pbc = _resolve_pbc(cell, pbc)
    n = len(positions)
    if n == 0:
        return NearestNeighbors(
            np.zeros((0, k), dtype=np.int64),
            np.zeros((0, k, 3)),
            np.zeros((0, k)),
        )

    if not pbc.any():
        return _query(positions, np.arange(n), n, k, workers)

    lattice = _complete_lattice(np.asarray(cell, dtype=np.float64), pbc, positions)
    reciprocal = np.linalg.inv(lattice)
    fractional = positions @ reciprocal
    fractional[:, pbc] -= np.floor(fractional[:, pbc])

    radius = _initial_radius(n, lattice, pbc, k)
    while True:
        points, source = _padded(fractional, lattice, reciprocal, pbc, radius)
        result = _query(points, source, n, k, workers)
        furthest = result.distances[:, -1].max()
        if furthest <= radius:
            return result
        # every atom within `furthest` of another is present next time round
        radius = furthest if np.isfinite(furthest) else 2.0 * radius


def _resolve_pbc(cell, pbc):
    if pbc is None:
        pbc = cell is not None
    pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)).copy()
    if pbc.any() and cell is None:
        raise ValueError("periodic boundaries need a cell")
    return pbc


def _complete_lattice(cell, pbc, positions):
    """The cell with non-periodic rows replaced, so it can be inverted.

    The replacements are orthogonal to the periodic rows and span the atoms,
    which keeps fractional coordinates along them meaningful; they are never
    wrapped or padded.
    """
    lattice = cell.copy()
    periodic = lattice[pbc]
    if len(periodic) == 3:
        return lattice
    _, _, vt = np.linalg.svd(np.vstack([periodic, np.zeros((3 - len(periodic), 3))]))
    normals = vt[len(periodic) :]
    for row, normal in zip(np.flatnonzero(~pbc), normals, strict=True):
        extent = np.ptp(positions @ normal)
        lattice[row] = normal * max(extent, 1.0)
    return lattice


def _initial_radius(n, lattice, pbc, k):
    """A sphere expected to hold `k + 1` atoms at the cell's mean density."""
    volume = abs(np.linalg.det(lattice))
    return PADDING_FACTOR * (3.0 * (k + 1) * volume / (4.0 * np.pi * n)) ** (1 / 3)


def _padded(fractional, lattice, reciprocal, pbc, radius):
    """Atoms followed by every image within `radius` of the cell.

    A point within `radius` of the cell lies within `radius * |b_i|` of it in
    fractional coordinate `i`, where `b_i` is a reciprocal vector. Padding one
    axis at a time, starting from the already-padded set, reaches the edge and
    corner images as well as the face ones.

    Returns:
        (points, source): (M, 3) Cartesian coordinates, the first N of which
        are the atoms themselves, and the atom each point is an image of
    """
    reach = radius * np.linalg.norm(reciprocal, axis=0)
    points = fractional
    source = np.arange(len(fractional))
    for axis in np.flatnonzero(pbc):
        span = int(np.ceil(reach[axis]))
        coordinate = points[:, axis]
        new_points, new_source = [points], [source]
        for shift in range(-span, span + 1):
            if shift == 0:
                continue
            moved = coordinate + shift
            keep = (moved > -reach[axis]) & (moved < 1.0 + reach[axis])
            image = points[keep]
            image[:, axis] += shift
            new_points.append(image)
            new_source.append(source[keep])
        points = np.concatenate(new_points)
        source = np.concatenate(new_source)
    return points @ lattice, source


def _query(points, source, n, k, workers):
    """k nearest neighbours of the first `n` points among all of them.

    Queries run in the tree's own order, which keeps consecutive queries
    spatially close whatever order the atoms came in.
    """
    tree = cKDTree(points, leafsize=8, balanced_tree=False, compact_nodes=False)
    order = tree.indices[tree.indices < n]
    distances, found = tree.query(points[order], k + 1, workers=workers)

    # drop each atom from its own list; if more than k other points coincide
    # with it, it may not be there, and the furthest entry goes instead
    if np.array_equal(found[:, 0], order):
        found, distances = found[:, 1:], distances[:, 1:]
    else:
        is_self = found == order[:, None]
        is_self[~is_self.any(axis=1), -1] = True
        keep = ~is_self
        found = found[keep].reshape(-1, k)
        distances = distances[keep].reshape(-1, k)

    found = np.ascontiguousarray(found)
    missing = found == len(points)
    any_missing = bool(missing.any())
    if any_missing:
        found[missing] = 0

    # gather each coordinate from its own contiguous array, not rows of (M, 3)
    vectors = np.empty((n, k, 3))
    components = np.ascontiguousarray(points.T)
    for c in range(3):
        vectors[order, :, c] = components[c][found] - components[c][order, None]
    indices = np.empty((n, k), dtype=np.int64)
    indices[order] = source[found]
    out_distances = np.empty((n, k))
    out_distances[order] = distances
    if any_missing:
        missing_rows = np.empty((n, k), dtype=bool)
        missing_rows[order] = missing
        indices[missing_rows] = -1
        vectors[missing_rows] = np.nan
    return NearestNeighbors(indices, vectors, out_distances)

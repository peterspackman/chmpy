"""Neighbour lists, with a Verlet skin.

A `NeighborList` is bound to a cutoff rather than to a structure, so a
calculator can hold one and reuse it across a whole relaxation. Two things make
that worth doing rather than enumerating pairs afresh each step:

* **A skin.** Pairs are enumerated out to `cutoff + skin` and the list is
  reused while the geometry has not moved far enough for a new pair to have
  entered the cutoff. The criterion accounts for a moving cell as well as
  moving atoms, since a pair vector is `r_j - r_i + S h`; see `_can_reuse`.
  Only the distances are recomputed on a reuse.
* **Caching by version.** Asking twice about one geometry costs nothing.

Pairs come from `vesin` when it is installed and from the numpy cell list in
`_celllist` otherwise, except for mixed periodicity -- slabs and wires -- which
neither handles and which falls back to a KD-tree over periodic images.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from ._celllist import cell_list_pairs


@dataclass(frozen=True)
class Neighbors:
    """Pairs within a cutoff, and the vectors between them.

    `vectors[p]` runs from atom `i[p]` to the image of atom `j[p]` displaced by
    `shifts[p]` lattice vectors, i.e.
    ``r[j] + shifts @ cell - r[i]``. That is the quantity a pair potential
    differentiates, and the one a virial needs, so it is computed once here
    rather than by every caller.

    Attributes:
        i: (P,) first atom of each pair
        j: (P,) second atom of each pair
        shifts: (P, 3) integer lattice translations applied to `j`
        vectors: (P, 3) displacement from `i` to the image of `j`, Angstroms
        distances: (P,) lengths of `vectors`, Angstroms
        full: whether each pair appears twice (once from each end)
    """

    i: np.ndarray
    j: np.ndarray
    shifts: np.ndarray
    vectors: np.ndarray
    distances: np.ndarray
    full: bool

    def __len__(self) -> int:
        return len(self.i)

    def __repr__(self) -> str:
        kind = "full" if self.full else "half"
        return f"<Neighbors {len(self)} {kind} pairs>"


class NeighborList:
    """Pairs within `cutoff`, rebuilt only when the geometry demands it.

    Args:
        cutoff: maximum pair distance in Angstroms
        full: return each pair from both ends. A half list is the natural input
            to a pair potential; a full list is what most message-passing
            models want.
        skin: extra range to enumerate, so small moves reuse the pair list.
            Zero disables reuse. A tenth of the cutoff is usually a good trade.
        self_pairs: include an atom with its own periodic images
    """

    def __init__(
        self,
        cutoff: float,
        full: bool = False,
        skin: float = 0.0,
        self_pairs: bool = True,
    ):
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)
        self.full = bool(full)
        self.skin = float(skin)
        self.self_pairs = bool(self_pairs)
        self._built_for = None  # (system, positions, cell) the list was built at
        self._pairs = None  # (i, j, shifts) out to cutoff + skin
        self._max_shift = 0.0  # largest |S| in the list, for the reuse bound
        self._cached = None  # (system, version, Neighbors) of the last call
        self.rebuilds = 0
        self.reuses = 0

    def __repr__(self) -> str:
        kind = "full" if self.full else "half"
        return f"<NeighborList cutoff={self.cutoff} {kind} skin={self.skin}>"

    def compute(self, system) -> Neighbors:
        """The pairs of `system` within the cutoff.

        Args:
            system: a `System`

        Returns:
            Neighbors
        """
        if (
            self._cached is not None
            and self._cached[0] is system
            and self._cached[1] == system.version
        ):
            return self._cached[2]

        positions = system.positions
        cell = system.cell
        if not self._can_reuse(system, positions, cell):
            self._build(system)
        else:
            self.reuses += 1

        i, j, shifts = self._pairs
        vectors = positions[j] - positions[i]
        if np.any(system.pbc):
            vectors = vectors + shifts @ cell
        distances = np.linalg.norm(vectors, axis=1)

        if self.skin > 0:
            inside = distances <= self.cutoff
            i, j, shifts = i[inside], j[inside], shifts[inside]
            vectors, distances = vectors[inside], distances[inside]

        neighbors = Neighbors(i, j, shifts, vectors, distances, self.full)
        self._cached = (system, system.version, neighbors)
        return neighbors

    # -- internals -----------------------------------------------------------

    def _can_reuse(self, system, positions, cell) -> bool:
        """Verlet criterion, extended to a cell that is also moving.

        A pair's vector is `r_j - r_i + S h`, so between two geometries it
        changes by `(dr_j - dr_i) + S dh`, whose length is at most
        `2 max|dr| + ||S|| ||dh||`. While that stays inside the skin, no pair
        can have entered the cutoff without being in the list already.

        Demanding the cell be *unchanged*, which is the obvious version of this
        test, throws the list away on every step of a variable-cell relaxation
        -- exactly the case the skin is meant to help.
        """
        if self._pairs is None or self.skin <= 0:
            return False
        built_system, built_positions, built_cell = self._built_for
        if built_system is not system or built_positions.shape != positions.shape:
            return False

        moved = np.linalg.norm(positions - built_positions, axis=1).max(initial=0.0)
        change = 2.0 * moved
        if self._max_shift > 0:
            deformation = np.linalg.norm(np.asarray(cell) - built_cell, ord=2)
            change += self._max_shift * deformation
        return change <= self.skin

    def _build(self, system) -> None:
        self.rebuilds += 1
        cutoff = self.cutoff + self.skin
        i, j, shifts = _enumerate_pairs(
            system, cutoff, full=self.full, self_pairs=self.self_pairs
        )
        self._pairs = (i, j, shifts)
        self._max_shift = (
            float(np.linalg.norm(shifts, axis=1).max()) if len(shifts) else 0.0
        )
        self._built_for = (system, np.array(system.positions), np.array(system.cell))


def _enumerate_pairs(system, cutoff, full, self_pairs):
    """Pairs within `cutoff`.

    `vesin` when it is installed, the numpy cell list otherwise, and a KD-tree
    over periodic images for the mixed-periodicity case that neither handles.
    """
    periodic = bool(np.all(system.pbc))
    isolated = not np.any(system.pbc)
    if periodic or isolated:
        if _vesin_available():
            return _vesin_pairs(system, cutoff, full, periodic, self_pairs)
        return cell_list_pairs(
            system.positions,
            system.cell,
            system.pbc,
            cutoff,
            full=full,
            self_pairs=self_pairs,
        )
    return _kdtree_pairs(system, cutoff, full, self_pairs)


@lru_cache(maxsize=1)
def _vesin_available() -> bool:
    """Whether vesin can be imported, asked once rather than per rebuild."""
    try:
        import vesin  # noqa: F401
    except ImportError:
        return False
    return True


def _vesin_pairs(system, cutoff, full, periodic, self_pairs):
    from vesin import NeighborList as VesinNeighborList

    box = np.asarray(system.cell) if periodic else np.zeros((3, 3))
    i, j, shifts = VesinNeighborList(cutoff=cutoff, full_list=full).compute(
        points=np.asarray(system.positions, dtype=np.float64),
        box=np.asarray(box, dtype=np.float64),
        periodic=periodic,
        quantities="ijS",
    )
    i = i.astype(np.int64)
    j = j.astype(np.int64)
    shifts = shifts.astype(np.float64)
    if not self_pairs:
        keep = i != j
        i, j, shifts = i[keep], j[keep], shifts[keep]
    return i, j, shifts


def _kdtree_pairs(system, cutoff, full, self_pairs):
    """Mixed periodicity: a KD-tree over the images within reach.

    Only slabs and wires come here; the fully periodic and fully isolated cases
    have faster paths above.
    """
    positions = np.asarray(system.positions)
    n = len(positions)
    pbc = np.asarray(system.pbc)
    cell = np.asarray(system.cell)

    reciprocal = np.linalg.inv(cell)
    repeats = np.where(
        pbc, np.ceil(cutoff * np.linalg.norm(reciprocal, axis=0)).astype(int), 0
    )
    ranges = [np.arange(-r, r + 1) for r in repeats]
    offsets = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T

    images = (positions[None, :, :] + (offsets @ cell)[:, None, :]).reshape(-1, 3)
    image_atom = np.tile(np.arange(n), len(offsets))
    image_shift = np.repeat(offsets, n, axis=0)

    pairs = cKDTree(positions).sparse_distance_matrix(
        cKDTree(images), cutoff, output_type="ndarray"
    )
    i = pairs["i"].astype(np.int64)
    k = pairs["j"].astype(np.int64)
    j = image_atom[k]
    shifts = image_shift[k].astype(np.float64)

    keep = pairs["v"] > 1e-12
    if not self_pairs:
        keep &= j != i
    i, j, shifts = i[keep], j[keep], shifts[keep]

    if not full:
        keep = _half_list_mask(i, j, shifts)
        i, j, shifts = i[keep], j[keep], shifts[keep]
    return i, j, shifts


def _half_list_mask(i, j, shifts):
    """Keep one of each pair: `j > i`, or a canonical half of the self-images."""
    total = shifts.sum(axis=1)
    positive_half = (total > 0) | (
        (total == 0) & ((shifts[:, 2] > 0) | ((shifts[:, 2] == 0) & (shifts[:, 1] > 0)))
    )
    return (j > i) | ((j == i) & positive_half)

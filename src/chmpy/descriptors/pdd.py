"""Pointwise distance distributions (PDD) and average minimum distances (AMD).

Widdowson and Kurlin (NeurIPS 2022; Widdowson et al., MATCH 87, 529 (2022)):
for each atom, the distances to its `k` nearest neighbours among every atom
of the periodic structure, its own images included, nearest first. Atoms whose
rows are the same by symmetry are kept as one row, so a PDD has one row per
asymmetric unit atom, averaged over that atom's copies in the unit cell, with
a weight: the fraction of the unit cell's atoms the row stands for. The AMD is
the weighted mean of the rows.

Two PDDs whose rows correspond, atom by atom, compare entry by entry
(`pdd_rms`); with `match_rows`, rows are first paired within each element by
least difference, which a copy relabelled by its molecule's own symmetry
needs. Otherwise -- different Z, a supercell, a different atom order --
the earth mover's distance (`pdd_emd`) matches rows by transporting weight
between them, at a cost of the largest difference between two rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chmpy.calc.nearest import nearest_neighbors


@dataclass(frozen=True, eq=False)
class PDD:
    """A pointwise distance distribution.

    Attributes:
        rows: (m, k) nearest-neighbour distances, Angstroms, ascending along
            each row
        weights: (m,) fraction of the unit cell's atoms each row stands for
        numbers: (m,) atomic number of each row's atoms
    """

    rows: np.ndarray
    weights: np.ndarray
    numbers: np.ndarray

    @property
    def k(self) -> int:
        return self.rows.shape[1]

    @property
    def amd(self) -> np.ndarray:
        "(k,) average minimum distances: the weighted mean of the rows"
        return self.weights @ self.rows

    def __repr__(self) -> str:
        return f"<PDD {len(self.rows)} rows, k={self.k}>"


def pdd(crystal, k: int = 100, groups=None) -> PDD:
    """The PDD of a crystal.

    Args:
        crystal: a `Crystal`
        k: neighbours per atom
        groups: (N,) the row each unit cell atom (in `unit_cell_atoms()` order)
            belongs to, rows numbered from 0. By default its asymmetric unit
            atom, so a row averages the atom's symmetry copies. For a P1 cell
            holding several copies of a molecule, each with its atoms in the
            same order, `np.arange(N) % atoms_per_molecule` gives one row per
            atom of the molecule.

    Returns:
        PDD
    """
    atoms = crystal.unit_cell_atoms()
    groups = atoms["asym_atom"] if groups is None else np.asarray(groups)
    if len(groups) != len(atoms["frac_pos"]):
        raise ValueError(f"{len(groups)} groups for {len(atoms['frac_pos'])} atoms")
    distances = nearest_neighbors(
        atoms["cart_pos"], k, cell=crystal.unit_cell.direct
    ).distances
    _, first, groups, counts = np.unique(
        groups, return_index=True, return_inverse=True, return_counts=True
    )
    numbers = np.asarray(atoms["element"])
    if np.any(numbers != numbers[first][groups]):
        raise ValueError("a row's atoms must all be of one element")
    rows = np.zeros((len(counts), k))
    np.add.at(rows, groups, distances)
    return PDD(rows / counts[:, None], counts / counts.sum(), numbers[first])


def amd(crystal, k: int = 100) -> np.ndarray:
    "(k,) average minimum distances of a crystal, Angstroms"
    return pdd(crystal, k).amd


def pdd_rms(a: PDD, b: PDD, match_rows: bool = False) -> float:
    """The root mean square difference of two PDDs, entry by entry.

    Their rows must correspond: the same atoms, in the same order. With
    `match_rows`, the rows of each element are first paired by least total
    squared difference, so a structure whose atoms were relabelled -- by its
    molecule's own symmetry, say -- still compares row for row.
    """
    if a.rows.shape != b.rows.shape:
        raise ValueError(
            f"{a.rows.shape} and {b.rows.shape} rows do not correspond; use pdd_emd"
        )
    if not match_rows:
        return float(np.sqrt(np.mean((a.rows - b.rows) ** 2)))
    from scipy.optimize import linear_sum_assignment

    if sorted(a.numbers) != sorted(b.numbers):
        raise ValueError("the rows are of different elements; use pdd_emd")
    total = 0.0
    for z in np.unique(a.numbers):
        ra, rb = a.rows[a.numbers == z], b.rows[b.numbers == z]
        cost = np.sum((ra[:, None] - rb[None]) ** 2, axis=-1)
        i, j = linear_sum_assignment(cost)
        total += cost[i, j].sum()
    return float(np.sqrt(total / a.rows.size))


def pdd_emd(a: PDD, b: PDD) -> float:
    """The earth mover's distance between two PDDs.

    The least total cost of moving each PDD's row weights onto the other's,
    where moving weight w from row i to row j costs w times the largest
    difference between the two rows (the L-infinity distance). Rows need not
    correspond and their numbers may differ.
    """
    from scipy.optimize import linprog

    if a.k != b.k:
        raise ValueError(f"k differs: {a.k} and {b.k}")
    cost = np.abs(a.rows[:, None, :] - b.rows[None, :, :]).max(axis=-1)
    m, n = cost.shape
    # flow[i, j] >= 0 with row sums a.weights and column sums b.weights
    supply = np.kron(np.eye(m), np.ones(n))
    demand = np.kron(np.ones(m), np.eye(n))
    result = linprog(
        cost.ravel(),
        A_eq=np.vstack([supply, demand]),
        b_eq=np.concatenate([a.weights, b.weights]),
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"the transport problem failed: {result.message}")
    return float(result.fun)

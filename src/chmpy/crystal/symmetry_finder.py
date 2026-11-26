"""
Native symmetry detection for crystal structures.

Finds the full symmetry of a crystal from its atomic positions and lattice
vectors using a hierarchical, numpy-vectorized algorithm:

1. Metric tensor filter - keeps rotations R where R^T G R = G
2. Translation search - finds candidate translations for each rotation
3. Full verification - checks all atoms map to same-element partners
4. Space group identification - feeds results into identify_standard_setting()
"""

import logging
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np

from .integer_lattice import lattice_basis
from .lattice_reduction import niggli_reduce
from .symmetry_operation import SymmetryOperation

LOG = logging.getLogger(__name__)


def _build_rotation_table() -> np.ndarray:
    """Every integer matrix with entries in {-1, 0, 1} and determinant +-1.

    In a Niggli reduced basis every symmetry operation's rotation is one of
    these, whatever the setting.

    Returns:
        (K, 3, 3) integer array of candidate rotation matrices
    """
    entries = np.array(list(product((-1, 0, 1), repeat=9))).reshape(-1, 3, 3)
    det = np.round(np.linalg.det(entries))
    return entries[np.abs(det) == 1]


_CRYSTALLOGRAPHIC_ROTATIONS = _build_rotation_table()


def _metric_compatible_rotations(G: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Return rotations compatible with metric tensor G.

    A rotation R is compatible if R^T G R = G (preserves the lattice metric).

    Args:
        G: (3, 3) metric tensor
        atol: absolute tolerance on the metric, in squared Angstroms

    Returns:
        (K, 3, 3) float array of compatible rotations
    """
    Rs = _CRYSTALLOGRAPHIC_ROTATIONS.astype(float)
    # R^T @ G @ R for all rotations: einsum 'kji,jl,klm->kim'
    RtGR = np.einsum('kji,jl,klm->kim', Rs, G, Rs)
    mask = np.all(np.abs(RtGR - G) < atol, axis=(1, 2))
    return Rs[mask]


def _find_translations(R, positions, elements, atol, A):
    """Find all translations t such that (R, t) is a valid symmetry operation.

    Uses rarest-element reference atom to minimize candidate translations.

    Args:
        R: (3, 3) rotation matrix
        positions: (N, 3) fractional coordinates
        elements: (N,) integer atomic numbers
        atol: Cartesian tolerance in Angstroms
        A: (3, 3) direct lattice matrix (row vectors)

    Returns:
        list of (3,) translation vectors
    """
    # Pick reference: atom of rarest element
    unique_els, counts = np.unique(elements, return_counts=True)
    rare_el = unique_els[np.argmin(counts)]
    ref_mask = elements == rare_el
    ref_idx = np.where(ref_mask)[0][0]

    # Candidate translations: t = x_target - R @ x_ref (mod 1)
    Rx_ref = R @ positions[ref_idx]
    target_positions = positions[ref_mask]
    candidates = (target_positions - Rx_ref) % 1.0

    valid = []
    for t in candidates:
        if _verify_operation(R, t, positions, elements, atol, A):
            valid.append(t)
    return valid


def _verify_operation(R, t, positions, elements, atol, A):
    """Check if (R, t) maps every atom to a same-element atom within tolerance.

    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        positions: (N, 3) fractional coordinates
        elements: (N,) integer atomic numbers
        atol: Cartesian tolerance in Angstroms
        A: (3, 3) direct lattice matrix (row vectors)

    Returns:
        True if (R, t) is a valid symmetry operation
    """
    transformed = (positions @ R.T + t) % 1.0
    # Pairwise fractional differences with minimum image convention
    diff = transformed[:, None, :] - positions[None, :, :]
    diff = diff - np.round(diff)
    # Convert to Cartesian distances
    cart_diff = diff @ A
    dist_sq = np.sum(cart_diff ** 2, axis=-1)
    # Element match mask
    el_match = elements[:, None] == elements[None, :]
    dist_sq[~el_match] = np.inf
    min_dist_sq = np.min(dist_sq, axis=1)
    return np.all(min_dist_sq < atol ** 2)


def find_symmetry_operations(unit_cell, positions, elements, atol=0.01):
    """Find all symmetry operations of a crystal structure.

    Args:
        unit_cell: UnitCell object
        positions: (N, 3) fractional coordinates of all atoms in unit cell
        elements: (N,) integer atomic numbers
        atol: Cartesian tolerance in Angstroms

    Returns:
        list[SymmetryOperation]: all detected symmetry operations
    """
    G = unit_cell.metric_tensor
    A = unit_cell.direct

    # Stage 1: metric tensor filter, as loose as the position tolerance
    longest = np.sqrt(np.max(np.diag(G)))
    compatible_rots = _metric_compatible_rotations(G, atol=2 * atol * longest + 1e-6)
    LOG.debug(
        "Metric tensor filter: %d / %d rotations compatible",
        len(compatible_rots), len(_CRYSTALLOGRAPHIC_ROTATIONS),
    )

    # Stage 2+3: find and verify translations for each rotation
    symops = []
    for R in compatible_rots:
        for t in _find_translations(R, positions, elements, atol, A):
            symops.append(SymmetryOperation(R, t))

    LOG.debug("Found %d symmetry operations", len(symops))
    return symops


def find_asymmetric_unit_indices(positions, elements, symops, A, atol=0.01):
    """Group atoms into orbits under symmetry operations and pick representatives.

    Args:
        positions: (N, 3) fractional coordinates
        elements: (N,) integer atomic numbers
        symops: list of SymmetryOperation objects
        A: (3, 3) direct lattice matrix
        atol: Cartesian tolerance in Angstroms

    Returns:
        list of indices into positions array, one per orbit
    """
    N = len(positions)
    orbit_id = np.full(N, -1, dtype=int)
    current_orbit = 0

    for i in range(N):
        if orbit_id[i] >= 0:
            continue
        orbit_id[i] = current_orbit
        # Apply all symops to atom i, find which atoms they map to
        for symop in symops:
            transformed = (symop.rotation @ positions[i] + symop.translation) % 1.0
            # Find matching atom
            diff = positions - transformed
            diff = diff - np.round(diff)
            cart_diff = diff @ A
            dist_sq = np.sum(cart_diff ** 2, axis=1)
            # Must match element
            same_el = elements == elements[i]
            dist_sq[~same_el] = np.inf
            j = np.argmin(dist_sq)
            if dist_sq[j] < atol ** 2:
                orbit_id[j] = current_orbit
        current_orbit += 1

    # Pick one representative per orbit (the lowest index)
    representatives = []
    for oid in range(current_orbit):
        members = np.where(orbit_id == oid)[0]
        representatives.append(members[0])

    return representatives


@dataclass
class FoundSymmetry:
    """A structure's symmetry, in a reduced basis of its primitive lattice.

    Attributes:
        transform: (3, 3) P, the basis being P @ direct (rows)
        origin_shift: (3,) added to fractional coordinates in that basis,
            which puts every translation on the 1/24 grid
        operations: the symmetry operations in that basis
        positions: (M, 3) fractional coordinates of the primitive cell's atoms
        indices: (M,) which input atom each of those is
    """

    transform: np.ndarray
    origin_shift: np.ndarray
    operations: list
    positions: np.ndarray
    indices: np.ndarray


def _primitive_basis(translations, atol_frac):
    """Q with Q @ direct a basis of the lattice the pure translations add to."""
    generators = np.eye(3, dtype=int).tolist()
    for t in translations:
        t = (np.asarray(t) + 0.5) % 1 - 0.5
        if np.abs(t).max() < atol_frac:
            continue
        fractions = [Fraction(float(x)).limit_denominator(96) for x in t]
        if np.abs(np.array(fractions, dtype=float) - t).max() > atol_frac:
            raise ValueError(f"Pure translation {t} is not rational")
        generators.append(fractions)
    return np.array(lattice_basis(generators), dtype=float)


def _nullspace(M, tol=1e-8):
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > tol))
    return vt[rank:].T


def _rational_origin(operations):
    """An origin shift s putting every translation on the 1/24 grid.

    Each translation t splits into an intrinsic part w (the screw or glide
    component, the same for any origin) and a location part t - w, which
    the shift changes by (I - R) s. The origin is placed on each
    operation's fixed axis or plane in turn, most constrained first, within
    the freedom the earlier ones leave.
    """
    identity = np.eye(3)
    shift = np.zeros(3)
    free = identity
    ordered = sorted(
        operations, key=lambda op: -np.linalg.matrix_rank(identity - op.rotation)
    )
    for op in ordered:
        R, t = op.rotation, op.translation
        order = next(
            n
            for n in range(1, 7)
            if np.allclose(np.linalg.matrix_power(R, n), identity)
        )
        w = sum(np.linalg.matrix_power(R, k) @ t for k in range(order)) / order
        w = np.round(24 * w) / 24  # a lattice vector over the order
        A = (identity - R) @ free
        if np.abs(A).max() < 1e-12:
            continue
        rhs = -(t - w) - (identity - R) @ shift
        # the location is only defined modulo the lattice: take the nearest
        # solution to the current origin
        y, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        shift = shift + free @ y
        free = free @ _nullspace(A)
        if free.shape[1] == 0:
            break
    return shift


def find_symmetry(unit_cell, positions, elements, atol=0.01):
    """The symmetry of a structure given as all the atoms in a cell.

    The cell may be any cell of the lattice (a supercell, a centred cell,
    an arbitrary basis): the pure translations give the primitive lattice,
    whose Niggli reduced basis is searched for operations.

    Args:
        unit_cell: UnitCell holding the atoms
        positions: (N, 3) fractional coordinates of all its atoms
        elements: (N,) integer atomic numbers
        atol: Cartesian tolerance in Angstroms

    Returns:
        FoundSymmetry, or None if the operations' translations cannot be
        made rational
    """
    A = unit_cell.direct
    positions = np.asarray(positions) % 1.0
    elements = np.asarray(elements)
    atol_frac = atol / np.min(np.linalg.norm(A, axis=1))

    pure = _find_translations(np.eye(3), positions, elements, atol, A)
    Q = _primitive_basis(pure, atol_frac)
    reduced = niggli_reduce(Q @ A)
    P = reduced.transform @ Q
    cell = type(unit_cell)(P @ A)

    # one copy of each atom in the primitive cell
    frac = (positions @ np.linalg.inv(P)) % 1.0
    keep = []
    for i in range(len(frac)):
        d = frac[keep] - frac[i]
        d -= np.round(d)
        if not keep or np.min(np.linalg.norm(d @ cell.direct, axis=1)) > atol:
            keep.append(i)
    keep = np.array(keep)
    frac, kept_elements = frac[keep], elements[keep]

    operations = find_symmetry_operations(cell, frac, kept_elements, atol=atol)
    shift = _rational_origin(operations)
    frac = (frac + shift) % 1.0
    exact = []
    for op in operations:
        t = op.translation + (np.eye(3) - op.rotation) @ shift
        t24 = np.round(24 * t)
        if np.linalg.norm((t - t24 / 24) @ cell.direct) > atol:
            return None
        exact.append(SymmetryOperation(np.round(op.rotation), (t24 % 24) / 24))
    if not all(
        _verify_operation(
            op.rotation, op.translation, frac, kept_elements, atol, cell.direct
        )
        for op in exact
    ):
        return None
    return FoundSymmetry(P, shift, exact, frac, keep)

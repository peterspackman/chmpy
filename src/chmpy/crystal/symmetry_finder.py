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

import numpy as np

from .space_group import SG_FROM_NUMBER
from .symmetry_operation import SymmetryOperation, decode_symm_int

LOG = logging.getLogger(__name__)


def _build_rotation_table() -> np.ndarray:
    """Extract all unique crystallographic rotation matrices from space group data.

    Returns:
        (K, 3, 3) integer array of unique rotation matrices
    """
    unique = {}
    for settings in SG_FROM_NUMBER.values():
        for sg in settings:
            for code in sg.symops:
                rot_code = code % 19683
                if rot_code not in unique:
                    R, _ = decode_symm_int(rot_code)
                    unique[rot_code] = np.round(R).astype(int)
    return np.array(list(unique.values()))


_CRYSTALLOGRAPHIC_ROTATIONS = _build_rotation_table()


def _metric_compatible_rotations(G: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Return rotations compatible with metric tensor G.

    A rotation R is compatible if R^T G R = G (preserves the lattice metric).

    Args:
        G: (3, 3) metric tensor
        atol: absolute tolerance for comparison

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

    # Stage 1: metric tensor filter
    compatible_rots = _metric_compatible_rotations(G, atol=1e-4)
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

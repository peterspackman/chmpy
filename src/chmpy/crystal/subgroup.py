"""
Subgroup enumeration for crystallographic space groups.

This module provides tools to enumerate translationengleiche (t-) subgroups
of a space group, identify their space group type, and expand the asymmetric
unit when reducing symmetry.

Primary use case: expanding asymmetric units to contain complete molecules
(e.g., urea with Z' = 0.5 -> Z' = 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from itertools import permutations, product as iterproduct
from math import gcd
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .space_group import SG_FROM_SYMOPS, SG_FROM_NUMBER, SpaceGroup
from .space_group_table import SpaceGroupTable
from .symmetry_operation import (
    SymmetryOperation,
    encode_symm_int,
    decode_symm_int,
)

if TYPE_CHECKING:
    from .asymmetric_unit import AsymmetricUnit

LOG = logging.getLogger(__name__)

# Rotation trace → rotation order (for det=+1 proper rotations)
_TRACE_TO_ORDER = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}

# Map from sorted (trace, det) multiset → point group symbol
# Generated programmatically from symmorphic representative space groups
# for each of the 32 crystallographic point groups.
_POINT_GROUP_TABLE: dict[tuple[tuple[int, int], ...], str] = {
    # Triclinic
    ((3, 1),): "1",
    ((-3, -1), (3, 1)): "-1",
    # Monoclinic
    ((-1, 1), (3, 1)): "2",
    ((1, -1), (3, 1)): "m",
    ((-3, -1), (-1, 1), (1, -1), (3, 1)): "2/m",
    # Orthorhombic
    ((-1, 1), (-1, 1), (-1, 1), (3, 1)): "222",
    ((-1, 1), (1, -1), (1, -1), (3, 1)): "mm2",
    ((-3, -1), (-1, 1), (-1, 1), (-1, 1), (1, -1), (1, -1), (1, -1), (3, 1)): "mmm",
    # Tetragonal
    ((-1, 1), (1, 1), (1, 1), (3, 1)): "4",
    ((-1, -1), (-1, -1), (-1, 1), (3, 1)): "-4",
    ((-3, -1), (-1, -1), (-1, -1), (-1, 1), (1, -1), (1, 1), (1, 1), (3, 1)): "4/m",
    ((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (1, 1), (1, 1), (3, 1)): "422",
    ((-1, 1), (1, -1), (1, -1), (1, -1), (1, -1), (1, 1), (1, 1), (3, 1)): "4mm",
    ((-1, -1), (-1, -1), (-1, 1), (-1, 1), (-1, 1), (1, -1), (1, -1), (3, 1)): "-42m",
    ((-3, -1), (-1, -1), (-1, -1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, 1), (1, 1), (3, 1)): "4/mmm",
    # Trigonal
    ((0, 1), (0, 1), (3, 1)): "3",
    ((-3, -1), (0, -1), (0, -1), (0, 1), (0, 1), (3, 1)): "-3",
    ((-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (3, 1)): "32",
    ((0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (3, 1)): "3m",
    ((-3, -1), (-1, 1), (-1, 1), (-1, 1), (0, -1), (0, -1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (3, 1)): "-3m",
    # Hexagonal
    ((-1, 1), (0, 1), (0, 1), (2, 1), (2, 1), (3, 1)): "6",
    ((-2, -1), (-2, -1), (0, 1), (0, 1), (1, -1), (3, 1)): "-6",
    ((-3, -1), (-2, -1), (-2, -1), (-1, 1), (0, -1), (0, -1), (0, 1), (0, 1), (1, -1), (2, 1), (2, 1), (3, 1)): "6/m",
    ((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (2, 1), (2, 1), (3, 1)): "622",
    ((-1, 1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (2, 1), (2, 1), (3, 1)): "6mm",
    ((-2, -1), (-2, -1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (1, -1), (3, 1)): "-6m2",
    ((-3, -1), (-2, -1), (-2, -1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, -1), (0, -1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (2, 1), (2, 1), (3, 1)): "6/mmm",
    # Cubic
    ((-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (3, 1)): "23",
    ((-3, -1), (-1, 1), (-1, 1), (-1, 1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (3, 1)): "m-3",
    ((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (3, 1)): "432",
    ((-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (3, 1)): "-43m",
    ((-3, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, -1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, -1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (3, 1)): "m-3m",
}

# Basis transformation matrices for common crystal system descents.
# Each is tried when direct lookup fails.
# P transforms coordinates: x_new = P @ x_old
# Supported centering vectors (duodecimal encoding).
_KNOWN_CENTERINGS = {
    (6, 6, 6),  # I-centering: (1/2, 1/2, 1/2)
    (4, 8, 8),  # R-centering: (1/3, 2/3, 2/3)
    (8, 4, 4),  # R-centering: (2/3, 1/3, 1/3)
}


def _integer_null_space(A_int: np.ndarray) -> list[np.ndarray]:
    """Find integer basis vectors for the null space of an integer matrix.

    Uses SVD to find null-space directions, then rounds to the nearest
    integer vector and verifies A @ v = 0 exactly.

    Args:
        A_int: Integer matrix (m, n)

    Returns:
        List of integer column vectors spanning null(A)
    """
    A = A_int.astype(float)
    _, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-8)
    null_vecs = []

    for i in range(rank, A.shape[1]):
        v = Vt[i].real
        # Round directly — for integer matrices the null vector direction
        # always rounds to a valid integer vector
        v_cand = np.round(v)
        if not np.allclose(v_cand, 0) and np.allclose(A @ v_cand, 0, atol=1e-8):
            entries = [abs(int(x)) for x in v_cand if abs(x) > 0.5]
            if entries:
                g = reduce(gcd, entries)
                null_vecs.append((v_cand / g).astype(int))
                continue

        # Fallback: try small integer scales
        for scale in range(1, 25):
            vi = np.round(v * scale)
            if np.allclose(vi, 0):
                continue
            if np.allclose(A @ vi, 0, atol=1e-8):
                entries = [abs(int(x)) for x in vi if abs(x) > 0.5]
                if entries:
                    g = reduce(gcd, entries)
                    null_vecs.append((vi / g).astype(int))
                break

    return null_vecs


def _compute_symmetry_adapted_transforms(
    rotations: list[np.ndarray],
) -> list[np.ndarray]:
    """Compute candidate basis transforms from rotation eigenvectors.

    For each non-trivial rotation matrix R, extracts:
    - null(R - I): eigenvectors with eigenvalue +1 (rotation axes)
    - null(R + I): eigenvectors with eigenvalue -1 (mirror normals)

    Assembles sets of 3 independent vectors into candidate Q matrices,
    returns P = Q^{-1} for those that produce valid crystallographic
    rotations (integer entries in {-1, 0, 1}).

    This replaces hardcoded centering transform tables with a systematic
    approach that works for any point group and orientation.

    Args:
        rotations: List of (3,3) integer rotation matrices

    Returns:
        List of P matrices (may have fractional entries, e.g. det=1/2)
    """
    I3 = np.eye(3, dtype=int)

    # Collect all symmetry-adapted vectors (deduplicated up to sign)
    all_vecs = []
    for R in rotations:
        R_int = np.round(R).astype(int)
        if np.allclose(R_int, I3) or np.allclose(R_int, -I3):
            continue

        for A in [R_int - I3, R_int + I3]:
            for v in _integer_null_space(A):
                is_dup = any(np.allclose(np.cross(v, e), 0) for e in all_vecs)
                if not is_dup:
                    all_vecs.append(v)

    # Supplement with cross products if needed
    if len(all_vecs) == 2:
        cross = np.cross(all_vecs[0], all_vecs[1])
        if not np.allclose(cross, 0):
            all_vecs.append(cross.astype(int))

    # If still fewer than 3 independent vectors, supplement with standard
    # basis vectors. This handles pure rotation groups (e.g. C3) where
    # only the rotation axis is extracted from eigenvectors.
    if len(all_vecs) < 3:
        for e in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
            is_dup = any(np.allclose(np.cross(e, v), 0) for v in all_vecs)
            if not is_dup:
                all_vecs.append(e)

    if len(all_vecs) < 3:
        return []

    # Try all ordered triples (with sign flips) as columns of Q
    valid_Ps = []
    seen = set()

    for perm in permutations(range(len(all_vecs)), 3):
        for signs in iterproduct([-1, 1], repeat=3):
            Q = np.column_stack([
                signs[0] * all_vecs[perm[0]],
                signs[1] * all_vecs[perm[1]],
                signs[2] * all_vecs[perm[2]],
            ]).astype(float)

            det_Q = np.linalg.det(Q)
            if abs(det_Q) < 1e-8:
                continue

            P = np.linalg.inv(Q)

            # Deduplicate
            key = tuple(np.round(P * 24).astype(int).flatten())
            if key in seen:
                continue
            seen.add(key)

            # Verify: all conjugated rotations must be integer with entries
            # in {-1, 0, 1} (valid crystallographic rotations)
            valid = True
            for R in rotations:
                R_new = P @ R @ Q
                R_int = np.round(R_new)
                if not np.allclose(R_new, R_int, atol=1e-6):
                    valid = False
                    break
                if np.any(np.abs(R_int) > 1.5):
                    valid = False
                    break

            if valid:
                valid_Ps.append(P)

    return valid_Ps


def _detect_centering(
    ops: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray | None:
    """Detect centering translation in a set of symmetry operations.

    Returns a centering vector if ALL rotations appear with the same
    multiplicity and the translation differences form a consistent
    centering lattice. Handles I-centering (multiplicity 2) and
    R-centering (multiplicity 3).

    All arithmetic is done in duodecimal integers (translation * 12)
    to avoid floating-point comparison issues.

    Args:
        ops: List of (rotation, translation) pairs

    Returns:
        Centering vector (3,) if detected, None otherwise
    """
    # Group operations by rotation matrix, storing translations as
    # integer duodecimal triples (t * 12, mod 12)
    rot_groups: dict[tuple, list[tuple[int, int, int]]] = {}
    for R, t in ops:
        rot_key = tuple(np.round(R).astype(int).flatten())
        t12 = tuple(round(x * 12) % 12 for x in t)
        if rot_key not in rot_groups:
            rot_groups[rot_key] = []
        rot_groups[rot_key].append(t12)

    # All rotations must have the same multiplicity > 1
    multiplicities = [len(ts) for ts in rot_groups.values()]
    if len(set(multiplicities)) != 1:
        return None
    mult = multiplicities[0]
    if mult < 2:
        return None

    # Extract candidate centering vectors from the first rotation group
    first_ts = list(rot_groups.values())[0]
    t0 = first_ts[0]
    centering_key = None
    for t12 in first_ts[1:]:
        diff = tuple((t12[i] - t0[i]) % 12 for i in range(3))
        if diff == (0, 0, 0):
            return None  # Zero difference means not a centering
        if diff in _KNOWN_CENTERINGS:
            centering_key = diff
            break

    if centering_key is None:
        return None

    # Verify: for every rotation group, adding the centering vector
    # maps each translation to another translation in the group
    for rot_key, ts in rot_groups.items():
        ts_set = set(ts)
        for t12 in ts:
            shifted = tuple((t12[i] + centering_key[i]) % 12 for i in range(3))
            if shifted not in ts_set:
                return None

    return np.array([c / 12.0 for c in centering_key])


def _reduce_centered_ops(
    ops: list[tuple[np.ndarray, np.ndarray]],
    centering: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reduce centered ops to primitive by keeping one from each group.

    For each unique rotation, keeps the translation closest to the origin
    (smallest norm after wrapping to [0, 1)). Works for any centering
    multiplicity (2 for I, 3 for R, etc.).

    Args:
        ops: List of (rotation, translation) pairs (must be centered)
        centering: The centering vector

    Returns:
        List of n/mult (rotation, translation) pairs
    """
    rot_groups: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
    for R, t in ops:
        key = tuple(np.round(R).astype(int).flatten())
        if key not in rot_groups:
            rot_groups[key] = []
        rot_groups[key].append((R, t))

    reduced = []
    for key, pairs in rot_groups.items():
        # Keep the one with smallest translation norm (wrapped to [0, 1))
        best = min(pairs, key=lambda rt: np.linalg.norm(rt[1] % 1.0))
        reduced.append(best)

    return reduced


_BASIS_TRANSFORMS = [
    # Tetragonal → orthorhombic (45° rotation in ab-plane)
    np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=float),
    # Tetragonal → orthorhombic (other variant)
    np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float),
    # Tetragonal → monoclinic (unique axis b, a'=a-b, c'=c)
    np.array([[1, -1, 0], [0, 0, 1], [1, 1, 0]], dtype=float),
    # Tetragonal → monoclinic (unique axis b, a'=a+b, c'=c)
    np.array([[1, 1, 0], [0, 0, 1], [-1, 1, 0]], dtype=float),
    # Hexagonal → orthorhombic
    np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float),
    # Hexagonal → monoclinic
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
    # Cubic → rhombohedral: sign-flips to map 3-fold axes to standard [1,1,1]
    # Single sign flips (map body diagonals [-1,1,1], [1,-1,1], [1,1,-1] to [1,1,1])
    np.diag([-1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, 1.0]),
    np.diag([1.0, 1.0, -1.0]),
    # Double sign flips (needed for subgroups where 3-fold rotations involve
    # coordinate permutations with two sign changes)
    np.diag([-1.0, -1.0, 1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([1.0, -1.0, -1.0]),
    # Cubic → tetragonal: axis permutations for non-standard unique axis
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),  # [100] → [001]
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),  # [010] → [001]
]


@dataclass(frozen=True)
class StandardSettingResult:
    """Result of identifying the standard ITA setting for a set of symops.

    Attributes:
        sg_number: ITA space group number
        sg_symbol: Short Hermann-Mauguin symbol
        choice: Space group choice (e.g. "b", "2", "")
        basis_transform: P matrix (3x3), None if identity
        origin_shift: p vector (3,), None if zero shift
        target_symops: Integer codes of the matched standard setting
    """

    sg_number: int
    sg_symbol: str
    choice: str
    basis_transform: np.ndarray | None
    origin_shift: np.ndarray | None
    target_symops: tuple[int, ...]


@dataclass(frozen=True)
class SubgroupResult:
    """Result of computing a t-subgroup.

    Attributes:
        symop_indices: Indices into parent group's symop list
        index: [G:H] = |G|/|H|, the index of the subgroup
        space_group_number: ITA number if matches known group, else None
        space_group_symbol: Symbol if identified, else None
        point_group_symbol: Point group symbol (always identified)
        z_prime_factor: Factor by which Z' increases (equals the index)
    """

    symop_indices: tuple[int, ...]
    index: int
    space_group_number: int | None
    space_group_symbol: str | None
    point_group_symbol: str
    z_prime_factor: float


def compute_closure(generators: frozenset[int], sg_table: SpaceGroupTable) -> frozenset[int]:
    """Compute group closure of generator indices using the Cayley table.

    Uses the multiplication and inverse tables for O(1) composition,
    iteratively expanding the set until no new elements are produced.

    Args:
        generators: Frozenset of symop indices (into sg_table)
        sg_table: Precomputed space group table with Cayley table

    Returns:
        Frozenset of all symop indices in the closure
    """
    closure = set(generators)
    changed = True
    while changed:
        changed = False
        new_elements = set()
        for a in closure:
            # Add inverse
            inv_a = int(sg_table.inverse_table[a])
            if inv_a not in closure:
                new_elements.add(inv_a)
            # Add all products with existing elements
            for b in closure:
                product = int(sg_table.mult_table[a, b])
                if product not in closure and product not in new_elements:
                    new_elements.add(product)
        if new_elements:
            closure.update(new_elements)
            changed = True
    return frozenset(closure)


def _identify_point_group(rotations: list[np.ndarray]) -> str:
    """Identify the point group from a set of rotation matrices.

    Uses the (trace, determinant) fingerprint of each rotation matrix
    to identify the abstract point group.

    Args:
        rotations: List of (3,3) rotation matrices

    Returns:
        Point group symbol (e.g., "mm2", "4", "-1")
    """
    fingerprint = []
    for R in rotations:
        tr = int(round(np.trace(R)))
        det = int(round(np.linalg.det(R)))
        fingerprint.append((tr, det))
    fingerprint_key = tuple(sorted(fingerprint))

    result = _POINT_GROUP_TABLE.get(fingerprint_key)
    if result is not None:
        return result

    # Fallback: describe by order and type
    n_ops = len(rotations)
    has_inversion = any(
        np.allclose(R, -np.eye(3)) for R in rotations
    )
    if has_inversion:
        return f"order-{n_ops} (centrosymmetric)"
    return f"order-{n_ops}"


def _solve_origin_shift(
    sub_ops: list[tuple[np.ndarray, np.ndarray]],
    candidate_symop_codes: tuple[int, ...],
) -> np.ndarray | None:
    """Solve for origin shift p that maps subgroup ops to candidate ops.

    For operations (R_i, t_i) and candidate (R_i, t_i'), solves for
    origin shift p such that:
        t_i - t_i' = (R_i - I) @ p  (mod 1) for all i

    Strategy: generate candidate p vectors from individual constraints
    (trying integer shifts on b), then verify each candidate against
    all constraints mod 1.

    Args:
        sub_ops: List of (rotation, translation) for the subgroup
        candidate_symop_codes: Integer codes of candidate space group

    Returns:
        (3,) origin shift vector, or None if no valid shift exists
    """
    # Decode candidate operations
    cand_ops = []
    for code in candidate_symop_codes:
        R, t = decode_symm_int(code)
        cand_ops.append((R, t))

    # Build rotation → translation maps
    sub_rot_map: dict[tuple, np.ndarray] = {}
    for R, t in sub_ops:
        key = tuple(np.round(R).astype(int).flatten())
        sub_rot_map[key] = t

    cand_rot_map: dict[tuple, np.ndarray] = {}
    for R, t in cand_ops:
        key = tuple(np.round(R).astype(int).flatten())
        cand_rot_map[key] = t

    # Rotations must match as sets
    if set(sub_rot_map.keys()) != set(cand_rot_map.keys()):
        return None

    # Collect linear constraints: (R_i - I) @ p = delta_t_i (mod 1)
    I3 = np.eye(3)
    constraints_A = []
    constraints_b = []

    for rot_key in sub_rot_map:
        R = np.array(rot_key).reshape(3, 3).astype(float)
        if np.allclose(R, I3):
            continue
        delta_t = sub_rot_map[rot_key] - cand_rot_map[rot_key]
        A = R - I3
        constraints_A.append(A)
        constraints_b.append(delta_t)

    if not constraints_A:
        return np.zeros(3)  # Only identity, trivially matches with zero shift

    # Stack all constraints for verification
    A_full = np.vstack(constraints_A)
    b_full = np.concatenate(constraints_b)

    # First try: direct lstsq
    p, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=None)
    if _verify_origin_shift(p, constraints_A, constraints_b):
        return p

    # Generate candidate p vectors from individual constraints with
    # integer corrections. For each constraint A_i @ p = b_i + n_i,
    # solve for p with different integer vectors n_i, then verify
    # against ALL constraints mod 1.
    candidates_seen = set()
    shifts = [-1, 0, 1]

    for i, (A, b) in enumerate(zip(constraints_A, constraints_b)):
        for n in iterproduct(shifts, repeat=3):
            b_shifted = b + np.array(n, dtype=float)
            # Solve this single constraint (3 eqs, 3 unknowns)
            try:
                rank = np.linalg.matrix_rank(A, tol=1e-6)
                if rank == 3:
                    p_cand = np.linalg.solve(A, b_shifted)
                elif rank >= 1:
                    p_cand, _, _, _ = np.linalg.lstsq(A, b_shifted, rcond=None)
                else:
                    continue
            except np.linalg.LinAlgError:
                continue

            # Round to avoid floating point noise in deduplication
            key = tuple(np.round(p_cand * 24).astype(int))
            if key in candidates_seen:
                continue
            candidates_seen.add(key)

            if _verify_origin_shift(p_cand, constraints_A, constraints_b):
                return p_cand

    # Try pairs of constraints for under-determined single constraints
    for i in range(len(constraints_A)):
        for j in range(i + 1, len(constraints_A)):
            A_pair = np.vstack([constraints_A[i], constraints_A[j]])
            for ni in iterproduct(shifts, repeat=3):
                for nj in iterproduct(shifts, repeat=3):
                    b_pair = np.concatenate([
                        constraints_b[i] + np.array(ni, dtype=float),
                        constraints_b[j] + np.array(nj, dtype=float),
                    ])
                    try:
                        p_cand, _, _, _ = np.linalg.lstsq(A_pair, b_pair, rcond=None)
                    except np.linalg.LinAlgError:
                        continue

                    key = tuple(np.round(p_cand * 24).astype(int))
                    if key in candidates_seen:
                        continue
                    candidates_seen.add(key)

                    if _verify_origin_shift(p_cand, constraints_A, constraints_b):
                        return p_cand

    return None


def _verify_origin_shift(
    p: np.ndarray,
    constraints_A: list[np.ndarray],
    constraints_b: list[np.ndarray],
    atol: float = 1e-3,
) -> bool:
    """Verify that p satisfies all constraints A_i @ p = b_i (mod 1)."""
    for A, b in zip(constraints_A, constraints_b):
        residual = A @ p - b
        residual = residual - np.round(residual)
        if not np.allclose(residual, 0, atol=atol):
            return False
    return True


def _apply_basis_transform(
    P: np.ndarray,
    ops: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Apply basis transform P to operations. Returns None if invalid.

    Transforms each (R, t) pair via R' = P R P^{-1}, t' = (P t) mod 1.
    Returns None if any transformed rotation has non-integer entries
    or entries outside {-1, 0, 1}.

    Args:
        P: (3,3) basis transformation matrix
        ops: List of (rotation, translation) pairs

    Returns:
        Transformed operations, or None if any rotation is invalid
    """
    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return None

    transformed = []
    for R, t in ops:
        R_new = P @ R @ P_inv
        R_int = np.round(R_new)
        if not np.allclose(R_new, R_int, atol=1e-6):
            return None
        if np.any(np.abs(R_int) > 1.5):
            return None
        t_new = (P @ t) % 1.0
        transformed.append((R_int, t_new))

    return transformed


def _match_ops_to_known_settings(
    ops: list[tuple[np.ndarray, np.ndarray]],
) -> tuple | None:
    """Try exact match then origin shift against all known space group settings.

    Returns (sgdata, origin_shift) on success, None on failure.
    sgdata is the matched entry from SG_FROM_SYMOPS or SG_FROM_NUMBER.
    origin_shift is None for exact matches, or a (3,) vector for origin shifts.

    Args:
        ops: List of (rotation, translation) pairs

    Returns:
        (sgdata, origin_shift) or None
    """
    n_ops = len(ops)

    # Try exact integer code match
    try:
        codes = tuple(sorted(encode_symm_int(R, t) for R, t in ops))
        sgdata = SG_FROM_SYMOPS.get(codes)
        if sgdata is not None:
            return (sgdata, None)
    except (ValueError, IndexError):
        pass

    # Try origin shift: match rotations, solve for shift vector
    rot_codes = sorted(encode_symm_int(R, np.zeros(3)) for R, _ in ops)

    for _sgnum_str, settings in SG_FROM_NUMBER.items():
        for candidate in settings:
            if len(candidate.symops) != n_ops:
                continue
            cand_rot_codes = sorted(code % 19683 for code in candidate.symops)
            if cand_rot_codes != [c % 19683 for c in rot_codes]:
                continue
            p = _solve_origin_shift(ops, tuple(candidate.symops))
            if p is not None:
                return (candidate, p)

    return None


def _identify_match(
    sub_ops: list[tuple[np.ndarray, np.ndarray]],
) -> tuple | None:
    """Run all matching tiers. Returns (sgdata, origin_shift, basis_transform) or None.

    Tier 1+2: Direct match (exact or origin shift, no basis transform)
    Tier 3+4: Hardcoded basis transforms + match
    Tier 5:   Centering reduction + eigenvector transforms + match

    Args:
        sub_ops: List of (rotation, translation) pairs

    Returns:
        (sgdata, origin_shift, basis_transform) or None
    """
    # Tier 1+2: Direct match (exact or origin shift, no basis transform)
    match = _match_ops_to_known_settings(sub_ops)
    if match is not None:
        return (*match, None)

    # Tier 3+4: Hardcoded basis transforms
    for P in _BASIS_TRANSFORMS:
        transformed = _apply_basis_transform(P, sub_ops)
        if transformed is None:
            continue
        match = _match_ops_to_known_settings(transformed)
        if match is not None:
            return (*match, P)

    # Tier 5: Centering reduction + eigenvector transforms
    centering = _detect_centering(sub_ops)
    if centering is not None:
        reduced = _reduce_centered_ops(sub_ops, centering)
        for P in _compute_symmetry_adapted_transforms([R for R, _ in reduced]):
            transformed = _apply_basis_transform(P, reduced)
            if transformed is None:
                continue
            match = _match_ops_to_known_settings(transformed)
            if match is not None:
                return (*match, P)

    return None


def _identify_space_group(
    symop_indices: Sequence[int],
    sg_table: SpaceGroupTable,
) -> tuple[int | None, str | None, str]:
    """Identify the space group of a subgroup from its symop indices.

    Uses a multi-tier strategy via _identify_match():
    1. Exact match via sorted integer codes
    2. Origin shift only
    3. Basis transformation + exact match
    4. Basis transformation + origin shift
    5. Centering reduction + eigenvector transforms
    6. Point group identification (always succeeds)

    Args:
        symop_indices: Indices into the parent group's symop list
        sg_table: The parent group's SpaceGroupTable

    Returns:
        (space_group_number, space_group_symbol, point_group_symbol)
    """
    # Decode operations
    sub_ops = []
    for idx in symop_indices:
        code = int(sg_table.idx_to_symop[idx])
        R, t = decode_symm_int(code)
        sub_ops.append((R, t))

    # Identify point group
    rotations = [R for R, _ in sub_ops]
    pg_symbol = _identify_point_group(rotations)

    # If point group not recognized (e.g. "order-N"), try with centering-reduced
    # rotations — centered ops double each rotation, confusing the fingerprint.
    if pg_symbol.startswith("order-"):
        centering_for_pg = _detect_centering(sub_ops)
        if centering_for_pg is not None:
            reduced_for_pg = _reduce_centered_ops(sub_ops, centering_for_pg)
            pg_symbol = _identify_point_group([R for R, _ in reduced_for_pg])

    # Run all matching tiers
    result = _identify_match(sub_ops)
    if result is not None:
        sgdata, _origin_shift, _basis_transform = result
        return sgdata.number, sgdata.short, pg_symbol

    return None, None, pg_symbol


def identify_standard_setting(
    symops: list[SymmetryOperation],
) -> StandardSettingResult | None:
    """Identify which standard ITA setting a set of symops corresponds to.

    Uses a multi-tier strategy via _identify_match():
    1. Exact code match — already standard
    2. Origin shift only (P=I)
    3. Basis transform P + exact match
    4. Basis transform P + origin shift p
    5. Centering reduction + eigenvector transforms

    Args:
        symops: List of SymmetryOperation objects

    Returns:
        StandardSettingResult if a match is found, None otherwise
    """
    sub_ops = [(s.rotation, s.translation) for s in symops]

    result = _identify_match(sub_ops)
    if result is None:
        return None

    sgdata, origin_shift, basis_transform = result
    has_shift = origin_shift is not None and not np.allclose(origin_shift, 0, atol=1e-6)
    return StandardSettingResult(
        sg_number=sgdata.number,
        sg_symbol=sgdata.short,
        choice=sgdata.choice,
        basis_transform=basis_transform,
        origin_shift=origin_shift if has_shift else None,
        target_symops=tuple(sgdata.symops),
    )


class SubgroupEnumerator:
    """Enumerate and verify t-subgroups of a space group.

    Uses BFS with closure checking to enumerate translationengleiche
    subgroups (same lattice, fewer point operations).

    Attributes:
        sg_table: Precomputed SpaceGroupTable for the parent group
        parent_sg: The parent SpaceGroup
    """

    def __init__(self, sg_table: SpaceGroupTable, parent_sg: SpaceGroup):
        self.sg_table = sg_table
        self.parent_sg = parent_sg

    @classmethod
    def from_space_group(cls, sg: SpaceGroup) -> SubgroupEnumerator:
        """Create a SubgroupEnumerator from a SpaceGroup.

        Args:
            sg: The space group to enumerate subgroups of

        Returns:
            SubgroupEnumerator ready to enumerate subgroups
        """
        sg_table = SpaceGroupTable.from_space_group(sg)
        return cls(sg_table, sg)

    def enumerate_all(self, max_index: int = 8) -> list[SubgroupResult]:
        """Enumerate all t-subgroups up to a given index.

        Uses BFS: starts from the identity, iteratively adds generators,
        computes closure, and collects unique subgroups.

        Args:
            max_index: Maximum index [G:H] to consider. Subgroups with
                |G|/|H| > max_index are excluded.

        Returns:
            List of SubgroupResult for each unique subgroup found,
            sorted by index (ascending) then by number of symops.
        """
        n_ops = self.sg_table.n_ops
        identity_idx = self.sg_table.identity_idx()
        min_size = max(1, n_ops // max_index)

        # BFS: each state is (closure, last_generator_tried)
        # We only try generators with index > last_gen to avoid duplicates
        # visited tracks all closures we've seen (to avoid re-exploring)
        # found_subgroups collects closures that meet the min_size filter
        visited: set[frozenset[int]] = set()
        found_subgroups: set[frozenset[int]] = set()
        queue: list[tuple[frozenset[int], int]] = [
            (frozenset([identity_idx]), 0)
        ]

        while queue:
            current, last_gen = queue.pop(0)
            closure = compute_closure(current, self.sg_table)

            if closure in visited:
                continue
            visited.add(closure)

            if len(closure) >= min_size:
                found_subgroups.add(closure)

            # Try adding each element as new generator
            for g in range(last_gen, n_ops):
                if g not in closure:
                    queue.append((closure | {g}, g + 1))

        # Build results
        results = []
        full_group = frozenset(range(n_ops))
        for subgroup in found_subgroups:
            if subgroup == full_group:
                continue  # Skip the parent group itself

            index = n_ops // len(subgroup)
            if index > max_index:
                continue

            sg_number, sg_symbol, pg_symbol = _identify_space_group(
                tuple(subgroup), self.sg_table
            )
            results.append(
                SubgroupResult(
                    symop_indices=tuple(sorted(subgroup)),
                    index=index,
                    space_group_number=sg_number,
                    space_group_symbol=sg_symbol,
                    point_group_symbol=pg_symbol,
                    z_prime_factor=float(index),
                )
            )

        results.sort(key=lambda r: (r.index, len(r.symop_indices)))
        return results

    def find_by_index(self, index: int) -> list[SubgroupResult]:
        """Find all subgroups with a specific index.

        Args:
            index: The desired index [G:H]

        Returns:
            List of SubgroupResult with the specified index
        """
        all_subgroups = self.enumerate_all(max_index=index)
        return [sg for sg in all_subgroups if sg.index == index]

    def find_for_target_z_prime(
        self, current_z_prime: float, target: float
    ) -> list[SubgroupResult]:
        """Find subgroups that achieve a target Z'.

        Args:
            current_z_prime: Current Z' value
            target: Desired Z' value

        Returns:
            List of SubgroupResult that would produce the target Z'
        """
        if target <= current_z_prime:
            return []
        ratio = target / current_z_prime
        # ratio must be close to an integer (the index)
        index = round(ratio)
        if abs(ratio - index) > 0.01:
            return []
        return self.find_by_index(index)

    def verify_is_subgroup(self, symop_indices: Sequence[int]) -> bool:
        """Verify that a set of symop indices forms a valid subgroup.

        Checks closure, identity, and inverses using the Cayley table.

        Args:
            symop_indices: Indices to verify

        Returns:
            True if the indices form a valid subgroup
        """
        idx_set = set(symop_indices)
        n_ops = self.sg_table.n_ops

        # Check all indices are valid
        if not all(0 <= i < n_ops for i in idx_set):
            return False

        # Check identity
        if self.sg_table.identity_idx() not in idx_set:
            return False

        # Check closure and inverses
        for a in idx_set:
            # Inverse must be in set
            if int(self.sg_table.inverse_table[a]) not in idx_set:
                return False
            # All products must be in set
            for b in idx_set:
                if int(self.sg_table.mult_table[a, b]) not in idx_set:
                    return False

        # Check Lagrange's theorem: |H| divides |G|
        if n_ops % len(idx_set) != 0:
            return False

        return True


def _nearest_image(pos: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Find the periodic image of pos nearest to reference.

    Shifts pos by integer lattice translations so that each fractional
    coordinate is within [-0.5, 0.5) of the reference coordinate.

    Args:
        pos: Fractional coordinates to shift
        reference: Reference fractional coordinates

    Returns:
        Shifted fractional coordinates nearest to reference
    """
    diff = pos - reference
    return pos - np.round(diff)


def _deduplicate_asymmetric_unit(
    asymmetric_unit: "AsymmetricUnit",
    symops: list[SymmetryOperation],
    tolerance: float = 1e-4,
) -> tuple[list[int], list]:
    """Remove redundant atoms from an asymmetric unit.

    Real CIF files often contain atoms that are already related by parent
    group symmetry operations. This function identifies a minimal set of
    truly independent atoms.

    For each atom, we check if any previous atom of the same element can
    be mapped onto it by a symmetry operation. If so, the atom is redundant.

    Args:
        asymmetric_unit: The (potentially redundant) asymmetric unit
        symops: Parent group's symmetry operations
        tolerance: Tolerance for position comparison

    Returns:
        Tuple of (unique_indices, redundancy_map) where unique_indices
        lists the indices of truly independent atoms and redundancy_map[i]
        is the index of the unique atom that atom i is equivalent to.
    """
    n_atoms = len(asymmetric_unit.elements)
    # redundancy_map[i] = index of the unique representative for atom i
    redundancy_map = list(range(n_atoms))
    unique_indices = []

    for i in range(n_atoms):
        elem_i = asymmetric_unit.elements[i]
        pos_i = asymmetric_unit.positions[i]

        is_redundant = False
        for j in unique_indices:
            # Only atoms of the same element can be symmetry-equivalent
            if asymmetric_unit.elements[j] != elem_i:
                continue

            pos_j = asymmetric_unit.positions[j]

            # Check if any symmetry operation maps pos_j to pos_i
            for symop in symops:
                mapped = symop.apply(pos_j.reshape(1, 3)).flatten()
                diff = (mapped - pos_i) % 1.0
                diff = np.where(diff > 0.5, diff - 1.0, diff)
                if np.allclose(diff, 0.0, atol=tolerance):
                    is_redundant = True
                    redundancy_map[i] = j
                    break
            if is_redundant:
                break

        if not is_redundant:
            unique_indices.append(i)

    return unique_indices, redundancy_map


def expand_asymmetric_unit(
    asymmetric_unit: "AsymmetricUnit",
    symops: list[SymmetryOperation],
    subgroup_indices: Sequence[int],
    sg_table: SpaceGroupTable,
    tolerance: float = 1e-4,
) -> "AsymmetricUnit":
    """Expand asymmetric unit when reducing to a subgroup.

    First deduplicates the input asymmetric unit (removing atoms that are
    already related by parent group symmetry), then for each unique atom
    computes its stabilizer, finds coset representatives (symops in G but
    not accounted for by the subgroup), and generates new independent atoms.

    Args:
        asymmetric_unit: The original asymmetric unit (may contain redundancy)
        symops: Parent group's symmetry operations
        subgroup_indices: Indices of symops forming the subgroup
        sg_table: Parent group's SpaceGroupTable
        tolerance: Tolerance for position comparison

    Returns:
        New AsymmetricUnit with expanded atom list
    """
    from .asymmetric_unit import AsymmetricUnit
    from .site_symmetry import SiteSymmetry

    # Step 1: Deduplicate the input asymmetric unit
    unique_indices, _ = _deduplicate_asymmetric_unit(
        asymmetric_unit, symops, tolerance
    )
    if len(unique_indices) < len(asymmetric_unit.elements):
        LOG.info(
            "Deduplicated asymmetric unit: %d -> %d atoms",
            len(asymmetric_unit.elements),
            len(unique_indices),
        )

    subgroup_set = set(subgroup_indices)
    n_parent = len(symops)
    n_subgroup = len(subgroup_indices)

    new_elements = []
    new_positions = []
    new_labels = []

    for i in unique_indices:
        elem = asymmetric_unit.elements[i]
        pos = asymmetric_unit.positions[i]

        # Compute stabilizer in parent group
        site_sym = SiteSymmetry.from_position(pos, symops, tolerance)
        stabilizer_codes = set(site_sym.stabilizer_symop_codes)

        # Map stabilizer codes to indices
        stabilizer_indices = set()
        for code in stabilizer_codes:
            if code in sg_table.symop_to_idx:
                stabilizer_indices.add(sg_table.symop_to_idx[code])

        # Compute stabilizer in subgroup = Stab_G(x) ∩ H
        stab_in_subgroup = stabilizer_indices & subgroup_set

        # Multiplicity in parent = |G| / |Stab_G|
        mult_parent = n_parent // len(stabilizer_indices)
        # Multiplicity in subgroup = |H| / |Stab_H|
        mult_subgroup = n_subgroup // len(stab_in_subgroup) if stab_in_subgroup else n_subgroup

        # Number of new independent copies = mult_parent / mult_subgroup
        # This is how many orbits the single parent orbit splits into
        # under the subgroup. Equivalently, [Stab_G(x) : Stab_G(x) ∩ H].
        n_copies = mult_parent // mult_subgroup if mult_parent > mult_subgroup else 1

        label_base = (
            asymmetric_unit.labels[i]
            if i < len(asymmetric_unit.labels)
            else f"{elem}{i+1}"
        )

        if n_copies <= 1:
            # No expansion needed for this atom
            new_elements.append(elem)
            new_positions.append(pos.copy())
            new_labels.append(str(label_base))
            continue

        # Find coset representatives: we need n_copies distinct images.
        # Always start with the original position, then find additional
        # copies nearby (closest periodic image to original).
        seen_positions = [pos.copy()]
        new_elements.append(elem)
        new_positions.append(pos.copy())
        new_labels.append(str(label_base))

        for g_idx in range(n_parent):
            raw_pos = symops[g_idx].apply(pos.reshape(1, 3)).flatten()

            # Find nearest periodic image to original position
            new_pos = _nearest_image(raw_pos, pos)

            # Check if this position is equivalent to any seen position
            # under the subgroup operations
            is_new = True
            for seen_pos in seen_positions:
                for h_idx in subgroup_indices:
                    h_pos = symops[h_idx].apply(seen_pos.reshape(1, 3)).flatten()
                    diff = new_pos - h_pos
                    diff = diff - np.round(diff)
                    if np.allclose(diff, 0.0, atol=tolerance):
                        is_new = False
                        break
                if not is_new:
                    break

            if is_new:
                seen_positions.append(new_pos)
                new_elements.append(elem)
                new_positions.append(new_pos)
                suffix = chr(ord('a') + len(seen_positions) - 1)
                new_labels.append(f"{label_base}{suffix}")

                if len(seen_positions) >= n_copies:
                    break

    positions_array = np.array(new_positions)
    return AsymmetricUnit(
        new_elements, positions_array, labels=new_labels
    )

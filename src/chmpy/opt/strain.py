"""Cell deformation, and which deformations a structure's symmetry allows.

A cell degree of freedom is an amplitude along a symmetry-adapted strain
direction, not a lattice parameter and not a raw Voigt component. That choice
is what keeps a cell optimisation inside the structure's space group.

The strains a crystal may undergo without losing its space group are those
invariant under every rotation `R` in its point group, `eps = R eps R^T`. Those
form a linear subspace of the symmetric 3x3 tensors, and this module builds an
orthonormal basis for it by projection:

    P = (1 / |G|) sum_R  T(R),        T(R): eps -> R eps R^T

`P` is the projector onto the invariant subspace, and its eigenvectors of
eigenvalue 1 are the basis. Nothing here consults a lattice type or assumes the
cell is in a standard setting: a rhombohedral cell with its three-fold along
`a + b + c` and one in the hexagonal setting both come out right, because both
are described by their own rotation matrices.

Mapping one degree of freedom to one Voigt component -- the obvious scheme --
cannot express these subspaces at all. Tetragonal requires `eps_xx = eps_yy`,
cubic requires all three equal, and rhombohedral requires an isotropic part
plus a uniaxial one along an axis that need not be a Cartesian direction. Left
uncoupled, a cell optimisation walks the metric out of its lattice type within
a few steps.

Strain is the symmetric tensor `eps` in `r -> (I + eps) r`, and because every
inner product here is the Frobenius one, no engineering factor of two on the
shear components ever appears.
"""

from __future__ import annotations

import numpy as np

#: strains beyond these stop being a small deformation: at -0.4 on all three
#: diagonals, det(I + eps) is 0.216, and the shear bound keeps the cell angles
#: away from degeneracy
MAX_NORMAL_STRAIN = 0.4
MAX_SHEAR_STRAIN = 0.3


def symmetric_basis() -> np.ndarray:
    """An orthonormal basis of the symmetric 3x3 tensors, under Frobenius.

    Returns:
        (6, 3, 3) with the three normal directions first, then the shears
    """
    basis = np.zeros((6, 3, 3))
    for k in range(3):
        basis[k, k, k] = 1.0
    root_half = 1.0 / np.sqrt(2.0)
    for k, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
        basis[k, a, b] = basis[k, b, a] = root_half
    return basis


#: the basis `to_vector` and `from_vector` work in
SYMMETRIC_BASIS = symmetric_basis()


def to_vector(tensor) -> np.ndarray:
    """A symmetric (3, 3) tensor as its (6,) coordinates in `SYMMETRIC_BASIS`.

    Unlike engineering Voigt notation this is an isometry: lengths and inner
    products are the same in both representations.
    """
    return np.einsum("kab,...ab->...k", SYMMETRIC_BASIS, np.asarray(tensor))


def from_vector(vector) -> np.ndarray:
    """The symmetric (3, 3) tensor with these coordinates in `SYMMETRIC_BASIS`."""
    return np.einsum("...k,kab->...ab", np.asarray(vector), SYMMETRIC_BASIS)


def rotation_action(rotation) -> np.ndarray:
    """The (6, 6) matrix of `eps -> R eps R^T` in `SYMMETRIC_BASIS`."""
    rotation = np.asarray(rotation, dtype=float)
    rotated = np.einsum("ac,kcd,bd->kab", rotation, SYMMETRIC_BASIS, rotation)
    return to_vector(rotated).T


def invariant_strain_basis(rotations, tolerance: float = 1e-8) -> np.ndarray:
    """An orthonormal basis for the strains a set of rotations leaves invariant.

    Args:
        rotations: (M, 3, 3) Cartesian point operations of the structure. Pass
            an empty list for no symmetry, which gives all six directions.
        tolerance: how far an eigenvalue of the projector may sit from 1

    Returns:
        (k, 3, 3) orthonormal symmetric tensors spanning the allowed strains,
        with `k` between 1 (cubic) and 6 (triclinic)
    """
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    if len(rotations) == 0:
        return SYMMETRIC_BASIS.copy()

    projector = np.mean([rotation_action(r) for r in rotations], axis=0)
    projector = 0.5 * (projector + projector.T)
    values, vectors = np.linalg.eigh(projector)
    kept = vectors[:, values > 1.0 - tolerance]
    if kept.shape[1] == 0:
        raise ValueError(
            "no strain is invariant under these operations, which cannot happen "
            "for a genuine point group -- check that the rotations are Cartesian"
        )
    return from_vector(_tidy(kept).T)


def _tidy(vectors: np.ndarray) -> np.ndarray:
    """Make the basis reproducible: a fixed sign and a stable order.

    `eigh` is free to return any orthonormal basis of a degenerate subspace,
    and the strain basis ends up in optimiser state and in test expectations,
    so it should not depend on the LAPACK build. Sorting by which component
    each vector leads with, and fixing the sign of that component, is enough.
    """
    vectors = np.asarray(vectors, dtype=float)
    cleaned = np.where(np.abs(vectors) < 1e-12, 0.0, vectors)
    leading = np.argmax(np.abs(cleaned) > 1e-12, axis=0)
    signs = np.sign(cleaned[leading, np.arange(cleaned.shape[1])])
    cleaned = cleaned * np.where(signs == 0, 1.0, signs)
    return cleaned[:, np.lexsort((np.arange(cleaned.shape[1]), leading))]


def cartesian_rotations(crystal) -> np.ndarray:
    """The Cartesian point operations of a crystal's space group.

    A symmetry operation acts on fractional column coordinates as `R x + t`.
    With the lattice vectors stored as the rows of `A`, a Cartesian column
    vector is `r = A^T x`, so the same operation in Cartesian space is
    `A^T R A^-T`.

    Args:
        crystal: a `chmpy.crystal.Crystal`

    Returns:
        (M, 3, 3) Cartesian rotation matrices, duplicates removed
    """
    direct = np.asarray(crystal.unit_cell.direct).T
    inverse = np.linalg.inv(direct)
    seen = {}
    for operation in crystal.space_group.symmetry_operations:
        cartesian = direct @ np.asarray(operation.rotation, dtype=float) @ inverse
        seen[np.round(cartesian, 9).tobytes()] = cartesian
    rotations = np.array(list(seen.values()))
    _check_orthogonal(rotations, crystal)
    return rotations


def _check_orthogonal(rotations, crystal, tolerance: float = 1e-6) -> None:
    """A point operation must come out orthogonal in Cartesian space.

    It does not when the cell's metric disagrees with the space group -- a
    tetragonal group on a cell whose a and b differ, say. Every symmetry
    argument downstream then quietly fails: the symmetry copies of a molecule
    come out sheared, and forces transform by the wrong matrix. Better to say
    so here than to return a strain basis that is not one.
    """
    error = np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3)).max(
        initial=0.0
    )
    if error > tolerance:
        a, b, c = crystal.unit_cell.lengths
        alpha, beta, gamma = np.degrees(crystal.unit_cell.angles)
        raise ValueError(
            f"the cell of {crystal} is not consistent with its space group: its "
            f"operations are not orthogonal in Cartesian space (off by {error:.2e}). "
            f"a={a:.4f} b={b:.4f} c={c:.4f} "
            f"alpha={alpha:.3f} beta={beta:.3f} gamma={gamma:.3f} for "
            f"{crystal.space_group.symbol} ({crystal.space_group.lattice_type})"
        )


def deformation(amplitudes, basis) -> np.ndarray:
    """The deformation matrix `I + eps` for amplitudes along a strain basis.

    Args:
        amplitudes: (k,) amplitudes
        basis: (k, 3, 3) strain basis

    Returns:
        (3, 3) deformation to apply as `cell @ F` and `positions @ F`
    """
    return np.eye(3) + np.einsum("k,kab->ab", np.asarray(amplitudes), basis)


def strain_of(amplitudes, basis) -> np.ndarray:
    """The strain tensor `eps` for these amplitudes."""
    return np.einsum("k,kab->ab", np.asarray(amplitudes), basis)


def step_fraction(amplitudes, step, basis) -> float:
    """The largest `t` in [0, 1] keeping `eps(q + t * dq)` inside the bounds.

    The whole step is scaled by one factor rather than each component being
    clamped separately. Clamping per component breaks a trust region twice
    over: the geometry evaluated is not the step the predicted reduction was
    computed for, and the unclamped step is still handed to the curvature
    update as the secant. It would also push the strain straight out of the
    symmetry subspace the basis exists to define.
    """
    current = strain_of(amplitudes, basis)
    change = strain_of(step, basis)
    fraction = 1.0
    for a in range(3):
        for b in range(a, 3):
            limit = MAX_NORMAL_STRAIN if a == b else MAX_SHEAR_STRAIN
            here, delta = current[a, b], change[a, b]
            if delta == 0.0:
                continue
            target = limit if delta > 0 else -limit
            if abs(here + delta) > limit:
                fraction = min(fraction, (target - here) / delta)
    return max(fraction, 0.0)


# ---------------------------------------------------------------------------
# Elastic tensors
#
# An elastic tensor acts on strains, so in the orthonormal basis above it is a
# symmetric 6x6 matrix, and the same rotation action that defines the allowed
# strains defines the allowed elastic tensors: C = T(R) C T(R)^T for every R in
# the point group. The invariant subspace is found the same way, by projection,
# and its dimension is the number of independent elastic constants -- 21 for
# triclinic down to 3 for cubic, arrived at rather than looked up.
# ---------------------------------------------------------------------------

#: engineering Voigt strain is (exx, eyy, ezz, 2eyz, 2exz, 2exy) while this
#: module's coordinates are (exx, eyy, ezz, sqrt2 eyz, sqrt2 exz, sqrt2 exy)
VOIGT_SCALE = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])


def elastic_from_voigt(voigt) -> np.ndarray:
    """A Voigt 6x6 elastic matrix in this module's orthonormal basis."""
    return np.asarray(voigt, dtype=float) * np.outer(VOIGT_SCALE, VOIGT_SCALE)


def elastic_to_voigt(tensor) -> np.ndarray:
    """An orthonormal-basis elastic matrix as an engineering Voigt 6x6."""
    return np.asarray(tensor, dtype=float) / np.outer(VOIGT_SCALE, VOIGT_SCALE)


def _symmetric_matrix_basis(n: int = 6) -> np.ndarray:
    """An orthonormal basis of the symmetric n x n matrices, under Frobenius."""
    basis = []
    for i in range(n):
        for j in range(i, n):
            matrix = np.zeros((n, n))
            if i == j:
                matrix[i, i] = 1.0
            else:
                matrix[i, j] = matrix[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(matrix)
    return np.array(basis)


def invariant_elastic_basis(rotations, tolerance: float = 1e-8) -> np.ndarray:
    """An orthonormal basis for the elastic tensors a point group allows.

    Args:
        rotations: (M, 3, 3) Cartesian point operations. An empty list gives
            the full 21-dimensional triclinic space.
        tolerance: how far an eigenvalue of the projector may sit from 1

    Returns:
        (k, 6, 6) orthonormal symmetric matrices, in this module's basis, where
        `k` is the number of independent elastic constants
    """
    space = _symmetric_matrix_basis(6)
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    if len(rotations) == 0:
        return space

    actions = np.array([rotation_action(rotation) for rotation in rotations])
    # <B_a, T B_b T^T> averaged over the group
    transformed = np.einsum("rij,bjk,rlk->rbil", actions, space, actions)
    projector = np.einsum("aij,rbij->ab", space, transformed) / len(rotations)
    projector = 0.5 * (projector + projector.T)

    values, vectors = np.linalg.eigh(projector)
    kept = _tidy(vectors[:, values > 1.0 - tolerance])
    return np.einsum("ak,aij->kij", kept, space)


def project_elastic(tensor, basis) -> np.ndarray:
    """The part of an elastic tensor its symmetry allows.

    Finite differences and a finite ionic relaxation leave a little of the
    tensor outside the invariant subspace. Projecting it back is not cosmetic:
    the components that symmetry forbids are pure noise, and leaving them in
    puts a spurious anisotropy into every modulus derived from the tensor.

    Args:
        tensor: (6, 6) in this module's orthonormal basis
        basis: (k, 6, 6) from `invariant_elastic_basis`

    Returns:
        (6, 6) projected tensor
    """
    tensor = np.asarray(tensor, dtype=float)
    tensor = 0.5 * (tensor + tensor.T)
    return np.einsum("k,kij->ij", np.einsum("kij,ij->k", basis, tensor), basis)

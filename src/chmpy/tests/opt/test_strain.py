import numpy as np
import pytest

from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell
from chmpy.opt.strain import (
    SYMMETRIC_BASIS,
    cartesian_rotations,
    deformation,
    from_vector,
    invariant_strain_basis,
    rotation_action,
    step_fraction,
    to_vector,
)

# space group, cell lengths, cell angles, expected number of strain freedoms
LATTICES = [
    (1, (5, 6, 7), (80, 85, 95), 6),
    (14, (5, 6, 7), (90, 102, 90), 4),
    (19, (5, 6, 7), (90, 90, 90), 3),
    (92, (5, 5, 7), (90, 90, 90), 2),
    (152, (5, 5, 7), (90, 90, 120), 2),
    (194, (5, 5, 7), (90, 90, 120), 2),
    (225, (5, 5, 5), (90, 90, 90), 1),
]


def crystal(number, lengths, angles):
    cell = UnitCell.from_lengths_and_angles(lengths, np.radians(angles))
    return Crystal(
        cell,
        SpaceGroup(number),
        AsymmetricUnit([Element[6]], np.array([[0.11, 0.21, 0.31]])),
    )


def test_symmetric_basis_is_orthonormal():
    gram = np.einsum("kab,lab->kl", SYMMETRIC_BASIS, SYMMETRIC_BASIS)
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-14)


def test_vector_round_trip_is_an_isometry():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(3, 3))
    tensor = 0.5 * (tensor + tensor.T)
    vector = to_vector(tensor)
    np.testing.assert_allclose(from_vector(vector), tensor, atol=1e-14)
    assert np.linalg.norm(vector) == pytest.approx(np.linalg.norm(tensor))


def test_rotation_action_agrees_with_the_tensor_transformation():
    rng = np.random.default_rng(1)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    tensor = rng.normal(size=(3, 3))
    tensor = 0.5 * (tensor + tensor.T)
    np.testing.assert_allclose(
        from_vector(rotation_action(rotation) @ to_vector(tensor)),
        rotation @ tensor @ rotation.T,
        atol=1e-13,
    )


@pytest.mark.parametrize("number,lengths,angles,expected", LATTICES)
def test_number_of_strain_freedoms(number, lengths, angles, expected):
    basis = invariant_strain_basis(
        cartesian_rotations(crystal(number, lengths, angles))
    )
    assert len(basis) == expected


@pytest.mark.parametrize("number,lengths,angles,expected", LATTICES)
def test_basis_is_orthonormal_and_invariant(number, lengths, angles, expected):
    structure = crystal(number, lengths, angles)
    rotations = cartesian_rotations(structure)
    basis = invariant_strain_basis(rotations)

    gram = np.einsum("kab,lab->kl", basis, basis)
    np.testing.assert_allclose(gram, np.eye(len(basis)), atol=1e-10)

    for rotation in rotations:
        transformed = np.einsum("ac,kcd,bd->kab", rotation, basis, rotation)
        # every rotated basis tensor must still lie in the span
        residual = transformed - np.einsum(
            "kl,lab->kab", np.einsum("kab,lab->kl", transformed, basis), basis
        )
        assert np.abs(residual).max() < 1e-10


@pytest.mark.parametrize("number,lengths,angles,expected", LATTICES)
def test_a_basis_strain_preserves_the_lattice_type(number, lengths, angles, expected):
    """Deforming along the basis must leave the cell in the same lattice type."""
    structure = crystal(number, lengths, angles)
    basis = invariant_strain_basis(cartesian_rotations(structure))
    direct = np.asarray(structure.unit_cell.direct)

    rng = np.random.default_rng(number)
    amplitudes = rng.normal(scale=0.05, size=len(basis))
    deformed = direct @ deformation(amplitudes, basis)

    cell = UnitCell(deformed)
    strained = Crystal(cell, structure.space_group, structure.asymmetric_unit)
    # the cell still supports its space group's operations
    cartesian_rotations(strained)

    original = UnitCell(direct)
    _assert_metric_relations_hold(original, cell)


def _assert_metric_relations_hold(before, after):
    """Equal lengths stay equal, right angles stay right."""
    a0, b0, c0 = before.lengths
    a1, b1, c1 = after.lengths
    for x0, y0, x1, y1 in ((a0, b0, a1, b1), (a0, c0, a1, c1), (b0, c0, b1, c1)):
        if abs(x0 - y0) < 1e-9:
            assert abs(x1 - y1) < 1e-9
    for before_angle, after_angle in zip(before.angles, after.angles, strict=True):
        if abs(before_angle - np.pi / 2) < 1e-9:
            assert abs(after_angle - np.pi / 2) < 1e-9
        if abs(before_angle - 2 * np.pi / 3) < 1e-9:
            assert abs(after_angle - 2 * np.pi / 3) < 1e-9


def test_no_symmetry_gives_all_six_directions():
    assert len(invariant_strain_basis([])) == 6


def test_a_cell_inconsistent_with_its_space_group_is_refused():
    """A tetragonal group on a cell whose a and b differ is not a structure."""
    broken = crystal(92, (5.0, 5.4, 7.0), (90, 90, 90))
    with pytest.raises(ValueError, match="not consistent with its space group"):
        cartesian_rotations(broken)


def test_step_fraction_respects_the_normal_and_shear_bounds():
    basis = invariant_strain_basis([])
    zero = np.zeros(6)
    assert step_fraction(zero, np.array([0.2, 0, 0, 0, 0, 0]), basis) == 1.0
    assert step_fraction(zero, np.array([0.8, 0, 0, 0, 0, 0]), basis) == pytest.approx(
        0.5
    )
    # a shear basis tensor has two equal off-diagonal entries of 1/sqrt(2)
    shear = np.array([0, 0, 0, 0, 0, 1.0])
    fraction = step_fraction(zero, shear, basis)
    assert fraction == pytest.approx(0.3 * np.sqrt(2.0))


# the number of independent elastic constants each Laue class allows, from the
# dimension of the invariant subspace rather than a table
ELASTIC_CONSTANTS = [
    (2, (5, 6, 7), (80, 85, 95), 21),  # triclinic
    (14, (5, 6, 7), (90, 102, 90), 13),  # monoclinic
    (19, (5, 6, 7), (90, 90, 90), 9),  # orthorhombic
    (76, (5, 5, 7), (90, 90, 90), 7),  # tetragonal, Laue 4/m
    (92, (5, 5, 7), (90, 90, 90), 6),  # tetragonal, Laue 4/mmm
    (147, (5, 5, 7), (90, 90, 120), 7),  # trigonal, Laue -3
    (152, (5, 5, 7), (90, 90, 120), 6),  # trigonal, Laue -3m
    (194, (5, 5, 7), (90, 90, 120), 5),  # hexagonal
    (225, (5, 5, 5), (90, 90, 90), 3),  # cubic
]


@pytest.mark.parametrize("number,lengths,angles,expected", ELASTIC_CONSTANTS)
def test_number_of_independent_elastic_constants(number, lengths, angles, expected):
    from chmpy.opt.strain import invariant_elastic_basis

    structure = crystal(number, lengths, angles)
    basis = invariant_elastic_basis(cartesian_rotations(structure))
    assert len(basis) == expected

    gram = np.einsum("kij,lij->kl", basis, basis)
    np.testing.assert_allclose(gram, np.eye(len(basis)), atol=1e-10)


def test_no_symmetry_allows_all_twenty_one_elastic_constants():
    from chmpy.opt.strain import invariant_elastic_basis

    assert len(invariant_elastic_basis([])) == 21


def test_the_voigt_conversion_round_trips():
    from chmpy.opt.strain import elastic_from_voigt, elastic_to_voigt

    rng = np.random.default_rng(0)
    voigt = rng.normal(size=(6, 6))
    voigt = voigt + voigt.T
    np.testing.assert_allclose(
        elastic_to_voigt(elastic_from_voigt(voigt)), voigt, atol=1e-12
    )


def test_the_voigt_conversion_preserves_the_strain_energy():
    """Both conventions must give the same energy for the same deformation."""
    from chmpy.opt.strain import elastic_from_voigt, to_vector

    rng = np.random.default_rng(1)
    voigt_matrix = rng.normal(size=(6, 6))
    voigt_matrix = voigt_matrix + voigt_matrix.T

    strain = rng.normal(size=(3, 3))
    strain = 0.5 * (strain + strain.T)
    engineering = np.array(
        [
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            2 * strain[1, 2],
            2 * strain[0, 2],
            2 * strain[0, 1],
        ]
    )
    orthonormal = to_vector(strain)

    assert engineering @ voigt_matrix @ engineering == pytest.approx(
        orthonormal @ elastic_from_voigt(voigt_matrix) @ orthonormal
    )

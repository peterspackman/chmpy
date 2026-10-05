"""Symmetry on a P1 system: atom maps, and relaxing within any subgroup."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import System
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt.elastic import voigt_strain
from chmpy.opt.symmetry import (
    AtomMap,
    InvariantAtomic,
    SymmetryAdapted,
    crystal_atom_map,
    map_atoms,
)
from chmpy.opt.trust_region import TrustRegion

from .. import TEST_FILES
from .test_symmetry_coordinates import GaussianPairs, numeric_gradient

ACETIC_ACID = TEST_FILES["acetic_acid.cif"]
R3C = TEST_FILES["r3c_example.cif"]


def crystal(lengths, angles, number, elements, positions):
    return Crystal(
        UnitCell.from_lengths_and_angles(lengths, np.radians(angles)),
        SpaceGroup(number),
        AsymmetricUnit([Element[e] for e in elements], np.array(positions, float)),
    )


def rutile():
    "Ti on 2a (fixed), O on 4f (x, x, 0): one freedom between them"
    return crystal(
        (4.6, 4.6, 2.96), (90, 90, 90), 136, [18, 36], [[0, 0, 0], [0.3, 0.3, 0]]
    )


def hexagonal():
    "2c (fixed) and 6l (x, 2x, 1/2) of P6/mmm, in a 120-degree cell"
    return crystal(
        (5.0, 5.0, 7.0),
        (90, 90, 120),
        191,
        [18, 36],
        [[1 / 3, 2 / 3, 0], [0.2, 0, 0.5]],
    )


def test_the_map_reproduces_every_image_exactly():
    structure = rutile()
    system = System.from_crystal(structure)
    atom_map = crystal_atom_map(structure, system)
    fractional = system.scaled_positions
    for g in range(len(atom_map)):
        image = fractional @ atom_map.rotations[g].T + atom_map.translations[g]
        expected = fractional[atom_map.permutation[g]] + atom_map.shifts[g]
        np.testing.assert_allclose(image, expected, atol=1e-10)


def test_an_operation_that_is_not_a_symmetry_is_refused():
    system = System.from_crystal(rutile())
    quarter_turn_about_x = ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], [0, 0, 0])
    with pytest.raises(ValueError, match="not symmetric"):
        map_atoms(system, [quarter_turn_about_x])


def test_a_strain_keeps_the_operations_that_commute_with_it():
    structure = crystal((5.3,) * 3, (90, 90, 90), 225, [18], [[0, 0, 0]])
    system = System.from_crystal(structure)
    atom_map = crystal_atom_map(structure, system)
    assert len(atom_map) == 192
    # xx leaves 4/mmm about x: 16 of 48 point operations, times 4 centrings
    assert len(atom_map.preserving_strain(voigt_strain(0, 1.0), system.cell)) == 64
    # a shear leaves mmm about the diagonals: 8 of 48
    assert len(atom_map.preserving_strain(voigt_strain(5, 1.0), system.cell)) == 32


@pytest.mark.parametrize(
    "structure, expected", [(rutile(), 1), (hexagonal(), 1)], ids=["rutile", "hex"]
)
def test_special_positions_lose_their_forbidden_freedoms(structure, expected):
    system = System.from_crystal(structure)
    coordinates = InvariantAtomic(system, crystal_atom_map(structure, system))
    assert coordinates.n_dof == expected


@pytest.mark.parametrize("path", [ACETIC_ACID, R3C])
def test_the_freedoms_match_the_asymmetric_unit(path):
    """Same count from the space group as from its action on the P1 cell."""
    structure = Crystal.load(path)
    system = System.from_crystal(structure)
    invariant = InvariantAtomic(system, crystal_atom_map(structure, system))
    assert invariant.n_dof == SymmetryAdapted(structure, cell=False).n_dof


@pytest.mark.parametrize("path", [ACETIC_ACID, R3C])
def test_gradient_matches_finite_differences(path):
    structure = Crystal.load(path)
    system = System.from_crystal(structure)
    coordinates = InvariantAtomic(system, crystal_atom_map(structure, system))
    calc = GaussianPairs()
    analytic = coordinates.gradient(calc(coordinates.system, coordinates.wanted))
    numeric = numeric_gradient(coordinates, calc)
    scale = max(float(np.abs(numeric).max()), 1e-12)
    assert np.abs(analytic - numeric).max() < 1e-6 * scale


def test_any_step_stays_exactly_symmetric():
    structure = hexagonal()
    system = System.from_crystal(structure)
    atom_map = crystal_atom_map(structure, system)
    coordinates = InvariantAtomic(system, atom_map)
    coordinates.set(np.random.default_rng(0).normal(scale=0.05, size=coordinates.n_dof))
    assert len(crystal_atom_map(structure, coordinates.system, tolerance=1e-9)) == 24


def test_relaxing_in_a_subgroup_keeps_it_and_converges():
    structure = Crystal.load(R3C)
    system = System.from_crystal(structure)
    sheared = system.copy()
    sheared.set_cell(
        np.asarray(system.cell) @ (np.eye(3) + voigt_strain(3, 0.02)), True
    )
    kept = crystal_atom_map(structure, system).preserving_strain(
        voigt_strain(3, 1.0), system.cell
    )
    assert 1 < len(kept) < len(structure.space_group.symmetry_operations)

    coordinates = InvariantAtomic(sheared, kept)
    outcome = TrustRegion(coordinates, GaussianPairs()).run(fmax=1e-4, steps=300)
    assert outcome.converged
    assert len(map_atoms(coordinates.system, _operations(kept), 1e-8)) == len(kept)


def test_the_identity_map_leaves_every_atom_free():
    system = System([18, 18], [[0, 0, 0], [2.0, 2.0, 2.0]], np.eye(3) * 4.0, True)
    assert InvariantAtomic(system, AtomMap.identity(2)).n_dof == 6


def _operations(atom_map):
    return list(zip(atom_map.rotations, atom_map.translations, strict=True))


class _NotQuiteEquivariant(GaussianPairs):
    """Adds a uniform, symmetry-forbidden force, mimicking a model that is
    not exactly equivariant."""

    push = np.array([0.03, 0.0, 0.0])

    def compute(self, system, want):
        result = super().compute(system, want)
        if result.forces is None:
            return result
        return type(result)(
            energy=result.energy,
            forces=result.forces + self.push,
            stress=result.stress,
            energies=result.energies,
            volume=result.volume,
        )


def test_symmetrising_removes_exactly_what_the_group_forbids():
    structure = Crystal.load(R3C)
    system = System.from_crystal(structure)
    atom_map = crystal_atom_map(structure, system)
    forces = GaussianPairs()(system, ("energy", "forces")).forces
    symmetric = atom_map.symmetrise(forces, system.cell)
    np.testing.assert_allclose(atom_map.symmetrise(symmetric, system.cell), symmetric)
    # R-3c is centrosymmetric, so a uniform force is entirely forbidden
    uniform = np.tile([0.03, 0.0, 0.0], (len(system), 1))
    np.testing.assert_allclose(
        atom_map.symmetrise(forces + uniform, system.cell), symmetric, atol=1e-12
    )


@pytest.mark.parametrize("cell", [False, True])
def test_a_forbidden_force_does_not_block_convergence(cell):
    from chmpy.opt import relax

    result = relax(
        Crystal.load(ACETIC_ACID),
        _NotQuiteEquivariant(),
        cell=cell,
        fmax=0.01,
        smax=0.05,
        steps=300,
    )
    assert result.converged
    assert result.measures["fmax"] <= 0.01
    assert result.measures["forbidden"] == pytest.approx(0.03, abs=0.005)


def test_a_large_forbidden_force_is_warned_about(caplog):
    from chmpy.opt import relax

    class Pushy(_NotQuiteEquivariant):
        push = np.array([0.3, 0.0, 0.0])

    with caplog.at_level("WARNING"):
        relax(Crystal.load(ACETIC_ACID), Pushy(), cell=False, fmax=0.01, steps=300)
    assert "symmetry=False" in caplog.text


def test_a_small_forbidden_force_is_not(caplog):
    from chmpy.opt import relax

    with caplog.at_level("WARNING"):
        relax(
            Crystal.load(ACETIC_ACID),
            _NotQuiteEquivariant(),
            cell=False,
            fmax=0.01,
            steps=300,
        )
    assert "symmetry=False" not in caplog.text

"""The one-call entry point."""

import numpy as np
import pytest

from chmpy import Crystal, Molecule
from chmpy.calc import LennardJones, System
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt import Atomic, AtomicStrain, SymmetryAdapted, coordinates_for, relax


def fcc_argon(a=5.0):
    cell = UnitCell.from_lengths_and_angles((a, a, a), np.radians((90, 90, 90)))
    return Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )


def argon_dimer(r=3.0):
    return Molecule.from_arrays(np.array([18, 18]), np.array([[0.0, 0, 0], [r, 0, 0]]))


def test_a_crystal_relaxes_its_asymmetric_unit_by_default():
    assert isinstance(coordinates_for(fcc_argon()), SymmetryAdapted)


def test_symmetry_off_relaxes_the_whole_cell():
    assert isinstance(coordinates_for(fcc_argon(), symmetry=False), AtomicStrain)


def test_a_fixed_cell_gives_atomic_coordinates():
    assert isinstance(coordinates_for(fcc_argon(), symmetry=False, cell=False), Atomic)


def test_a_molecule_gives_atomic_coordinates():
    assert isinstance(coordinates_for(argon_dimer()), Atomic)


def test_relaxing_a_crystal_gives_back_a_crystal_in_its_space_group():
    crystal = fcc_argon(5.0)
    result = relax(crystal, LennardJones(), fmax=1e-4, smax=1e-3, steps=50)
    assert result.converged
    assert isinstance(result.structure, Crystal)
    assert result.structure.space_group == crystal.space_group
    assert result.structure.unit_cell.lengths[0] != pytest.approx(5.0)


def test_relaxing_without_symmetry_gives_back_a_p1_crystal():
    result = relax(
        fcc_argon(5.0), LennardJones(), symmetry=False, fmax=1e-4, smax=1e-3, steps=50
    )
    assert result.structure.space_group.international_tables_number == 1


def test_relaxing_a_molecule_gives_back_a_molecule():
    result = relax(argon_dimer(3.0), LennardJones(), fmax=1e-5, steps=60)
    assert result.converged
    assert isinstance(result.structure, Molecule)
    separation = np.linalg.norm(
        result.structure.positions[1] - result.structure.positions[0]
    )
    # the LJ pair minimum is at 2^(1/6) sigma
    assert separation == pytest.approx(2 ** (1 / 6) * 3.4, rel=1e-4)


def test_relaxing_a_system_gives_back_the_same_system():
    system = System.from_crystal(fcc_argon(5.0))
    result = relax(system, LennardJones(), fmax=1e-4, smax=1e-3, steps=50)
    assert result.structure is system


def test_pressure_is_accepted_end_to_end():
    at_zero = relax(fcc_argon(5.0), LennardJones(), fmax=1e-4, smax=1e-3, steps=80)
    compressed = relax(
        fcc_argon(5.0),
        LennardJones(),
        pressure=3.0,
        fmax=1e-4,
        smax=1e9,
        steps=80,
    )
    assert (
        compressed.structure.unit_cell.volume() < at_zero.structure.unit_cell.volume()
    )


def test_fixed_atoms_are_passed_through():
    crystal = Crystal.load(_acetic())
    result = relax(
        crystal,
        LennardJones(sigma=1.0, epsilon=1e-4),
        fixed=np.array([True] + [False] * 7),
        steps=2,
    )
    np.testing.assert_allclose(
        result.structure.asymmetric_unit.positions[0],
        crystal.asymmetric_unit.positions[0],
        atol=1e-12,
    )


def _acetic():
    from .. import TEST_FILES

    return TEST_FILES["acetic_acid.cif"]

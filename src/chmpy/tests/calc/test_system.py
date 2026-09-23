import numpy as np
import pytest

from chmpy.calc import System


def water():
    return System([8, 1, 1], [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


def cubic(a=4.0):
    return System([18, 18], [[0, 0, 0], [a / 2, a / 2, a / 2]], np.eye(3) * a, True)


def test_arrays_are_read_only():
    system = water()
    with pytest.raises(ValueError):
        system.positions[0, 0] = 1.0
    with pytest.raises(ValueError):
        system.cell[0, 0] = 1.0


def test_version_tracks_geometry_changes():
    system = cubic()
    assert system.version == 0
    system.set_positions(system.positions + 0.1)
    assert system.version == 1
    system.set_cell(system.cell * 1.01)
    assert system.version == 2


def test_isolated_system_has_no_volume_or_fractional_coordinates():
    system = water()
    assert system.volume == 0.0
    assert not system.periodic
    with pytest.raises(ValueError):
        _ = system.scaled_positions


def test_scaled_positions_round_trip():
    system = cubic(5.0)
    scaled = system.scaled_positions
    np.testing.assert_allclose(scaled, [[0, 0, 0], [0.5, 0.5, 0.5]], atol=1e-12)
    system.set_scaled_positions(scaled)
    np.testing.assert_allclose(system.positions, [[0, 0, 0], [2.5, 2.5, 2.5]])


def test_set_cell_can_carry_the_atoms():
    system = cubic(4.0)
    system.set_cell(np.eye(3) * 8.0, scale_atoms=True)
    np.testing.assert_allclose(system.positions[1], [4.0, 4.0, 4.0])

    system = cubic(4.0)
    system.set_cell(np.eye(3) * 8.0, scale_atoms=False)
    np.testing.assert_allclose(system.positions[1], [2.0, 2.0, 2.0])


def test_periodic_system_needs_a_cell():
    with pytest.raises(ValueError, match="needs a cell"):
        System([1], [[0, 0, 0]], np.zeros((3, 3)), True)


def test_wrapped_brings_atoms_into_the_cell():
    system = System([1, 1], [[0, 0, 0], [9.0, -3.0, 0.5]], np.eye(3) * 4.0, True)
    wrapped = system.wrapped()
    np.testing.assert_allclose(wrapped.positions[1], [1.0, 1.0, 0.5], atol=1e-12)


def test_crystal_round_trip():
    from chmpy import Crystal

    crystal = Crystal.from_molecule(_molecule())
    system = System.from_crystal(crystal)
    assert len(system) == len(crystal.unit_cell_atoms()["element"])
    np.testing.assert_allclose(system.cell, crystal.unit_cell.direct)

    back = system.to_crystal()
    np.testing.assert_allclose(back.unit_cell.direct, system.cell)
    np.testing.assert_allclose(
        np.sort(back.asymmetric_unit.atomic_numbers), np.sort(system.numbers)
    )


def test_molecule_round_trip():
    molecule = _molecule()
    system = System.from_molecule(molecule)
    np.testing.assert_allclose(system.positions, molecule.positions)
    np.testing.assert_allclose(system.to_molecule().positions, molecule.positions)


def test_ase_round_trip():
    ase = pytest.importorskip("ase")  # noqa: F841
    system = cubic(4.2)
    atoms = system.to_ase()
    back = System.from_ase(atoms)
    np.testing.assert_allclose(back.positions, system.positions)
    np.testing.assert_allclose(back.cell, system.cell)
    assert list(back.pbc) == [True, True, True]


def _molecule():
    from chmpy import Molecule

    return Molecule.from_arrays(
        np.array([8, 1, 1]),
        np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]),
    )

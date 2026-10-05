"""The ASE bridge, checked against ASE's own Lennard-Jones."""

import numpy as np
import pytest

from chmpy.calc import Calculator, LennardJones, System

pytest.importorskip("ase")

EPSILON, SIGMA, CUTOFF = 0.0103, 3.4, 8.0


def argon():
    a = 5.3
    positions = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    ) * a + np.array(
        [[0.05, -0.02, 0.01], [0, 0.03, 0], [-0.04, 0, 0.02], [0.01, 0.01, -0.03]]
    )
    return System([18] * 4, positions, np.eye(3) * a, True)


def ase_lennard_jones():
    from ase.calculators.lj import LennardJones as AseLJ

    return AseLJ(epsilon=EPSILON, sigma=SIGMA, rc=CUTOFF, smooth=False)


def test_wrapped_ase_calculator_matches_ase(request):
    from ase.filters import FrechetCellFilter  # noqa: F401  (ase import sanity)

    system = argon()
    atoms = system.to_ase()
    atoms.calc = ase_lennard_jones()

    calc = Calculator.from_ase(ase_lennard_jones())
    result = calc(system, ("energy", "forces", "stress"))

    assert result.energy == pytest.approx(atoms.get_potential_energy())
    np.testing.assert_allclose(result.forces, atoms.get_forces(), atol=1e-12)
    np.testing.assert_allclose(
        result.stress_voigt, atoms.get_stress(voigt=True), atol=1e-12
    )


def test_our_lennard_jones_matches_ase():
    """Same potential, same cutoff convention: the numbers must agree."""
    system = argon()
    atoms = system.to_ase()
    atoms.calc = ase_lennard_jones()

    ours = LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF)
    result = ours(system, ("energy", "forces", "stress"))

    assert result.energy == pytest.approx(atoms.get_potential_energy(), rel=1e-10)
    np.testing.assert_allclose(result.forces, atoms.get_forces(), atol=1e-10)
    np.testing.assert_allclose(
        result.stress_voigt, atoms.get_stress(voigt=True), atol=1e-12
    )


def test_adapter_keeps_one_atoms_object_across_calls():
    system = argon()
    calc = Calculator.from_ase(ase_lennard_jones())
    calc.energy(system)
    first = calc._atoms
    system.set_positions(system.positions + 0.01)
    calc.energy(system)
    assert calc._atoms is first


def test_exposing_a_calculator_to_ase_round_trips():
    system = argon()
    ours = LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF)
    reference = ours(system, ("energy", "forces", "stress"))

    atoms = system.to_ase()
    atoms.calc = ours.as_ase()
    assert atoms.get_potential_energy() == pytest.approx(reference.energy)
    np.testing.assert_allclose(atoms.get_forces(), reference.forces, atol=1e-12)
    np.testing.assert_allclose(
        atoms.get_stress(voigt=True), reference.stress_voigt, atol=1e-12
    )


def test_an_ase_optimiser_can_drive_our_calculator():
    from ase.optimize import LBFGS

    system = argon()
    atoms = system.to_ase()
    atoms.calc = LennardJones(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF).as_ase()
    LBFGS(atoms, logfile=None).run(fmax=1e-3, steps=100)
    assert np.abs(atoms.get_forces()).max() < 1e-3


def test_info_reaches_the_ase_calculator():
    """Some models want a charge, a spin or a field in `atoms.info`.

    MACE-Polar is the case in point: it reads `charge`, `spin` and
    `external_field` off the Atoms object and fails without them. A `System`
    carries them in `info`, and the adapter has to put them where the model
    looks -- including on later calls, since the Atoms object is reused rather
    than rebuilt.
    """
    seen = []

    class WantsInfo(ase_lennard_jones().__class__):
        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            missing = {"charge", "spin", "external_field"} - set(atoms.info)
            if missing:
                raise KeyError(f"missing from atoms.info: {sorted(missing)}")
            seen.append(dict(atoms.info))
            super().calculate(atoms, list(properties), system_changes or [])

    system = argon()
    system.info.update(charge=0, spin=1, external_field=[0.0, 0.0, 0.0])
    calc = Calculator.from_ase(
        WantsInfo(epsilon=EPSILON, sigma=SIGMA, rc=CUTOFF, smooth=False)
    )

    calc.energy(system)
    system.set_positions(system.positions + 0.01)
    calc.energy(system)

    assert len(seen) == 2
    assert all(entry["charge"] == 0 and entry["spin"] == 1 for entry in seen)


def test_system_from_crystal_takes_model_inputs():
    from chmpy import Crystal

    from .. import TEST_FILES

    crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
    system = System.from_crystal(crystal, charge=0, spin=1)
    assert system.info == {"charge": 0, "spin": 1}


def test_a_calculator_that_initialises_on_new_numbers_works():
    """EMT only initialises when `system_changes` includes "numbers"."""
    from ase.calculators.emt import EMT

    system = System([29] * 4, argon().positions * 0.7, np.eye(3) * 3.7, True)
    atoms = system.to_ase()
    atoms.calc = EMT()

    calc = Calculator.from_ase(EMT())
    result = calc(system, ("energy", "forces"))
    assert result.energy == pytest.approx(atoms.get_potential_energy())
    np.testing.assert_allclose(result.forces, atoms.get_forces(), atol=1e-10)

    changed = System([29, 29, 79, 79], system.positions, system.cell, True)
    atoms = changed.to_ase()
    atoms.calc = EMT()
    assert calc(changed, ("energy",)).energy == pytest.approx(
        atoms.get_potential_energy()
    )


def test_a_result_is_not_changed_by_the_next_evaluation():
    """EMT reuses its force array between calls; results must hold copies."""
    from ase.calculators.emt import EMT

    system = System([29] * 4, argon().positions * 0.7, np.eye(3) * 3.7, True)
    calc = Calculator.from_ase(EMT())
    first = calc(system, ("energy", "forces", "energies"))
    kept = first.forces.copy(), first.energies.copy()

    moved = system.copy()
    moved.set_positions(system.positions + 0.05)
    calc(moved, ("energy", "forces", "energies"))

    np.testing.assert_array_equal(first.forces, kept[0])
    np.testing.assert_array_equal(first.energies, kept[1])


class _Broken:
    """Fails identically whichever way it is called."""

    implemented_properties = ["energy", "forces"]
    calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=None):
        type(self).calls += 1
        raise RuntimeError("Traceback...\n  lots of TorchScript\nscatter(): bad index")

    def get_potential_energy(self, atoms=None):
        self.calculate(atoms)

    def get_property(self, name, atoms=None, allow_calculation=True):
        self.calculate(atoms)

    def reset(self):
        pass


def test_a_model_that_fails_either_way_raises_its_own_error_once():
    calc = Calculator.from_ase(_Broken())
    with pytest.raises(RuntimeError) as raised:
        calc(argon(), ("energy",))
    assert str(raised.value).endswith("scatter(): bad index")
    assert "failed both" not in str(raised.value)
    # the direct route stays enabled
    assert calc._direct


def test_a_long_error_is_quoted_by_its_last_line():
    from chmpy.calc.adapters.ase import _first_line

    error = RuntimeError("Traceback...\n  frame\n\nscatter(): bad index\n")
    assert _first_line(error) == "RuntimeError: scatter(): bad index"

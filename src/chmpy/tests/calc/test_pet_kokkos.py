"""The pet-kokkos backend, against finite differences and its own ASE calculator.

Skipped unless pet-kokkos is installed and the model can be found or fetched.
"""

import numpy as np
import pytest

from chmpy.calc import Calculator, PropertyNotAvailable, System

pytest.importorskip("pet_kokkos")

MODEL = "pet-mad-xs"


@pytest.fixture(scope="module")
def calc():
    from chmpy.calc.adapters.pet_kokkos import PetKokkosCalculator

    try:
        return PetKokkosCalculator(MODEL)
    except Exception as exc:  # no network, no downloaded model
        pytest.skip(f"{MODEL} is not available: {exc}")


def silicon_carbide(seed=0):
    """A distorted zinc-blende cell, so no symmetry hides a sign error."""
    rng = np.random.default_rng(seed)
    a = 4.36
    cell = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    cell = cell @ (np.eye(3) + rng.normal(scale=0.03, size=(3, 3)))
    scaled = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    scaled = scaled + rng.normal(scale=0.02, size=scaled.shape)
    return System([6, 14], scaled @ cell, cell, True)


def water():
    return System([8, 1, 1], [[0, 0, 0.119], [0, 0.763, -0.477], [0, -0.763, -0.477]])


def test_gradients_match_finite_differences(calc):
    # fp32 energies: the step has to be large enough for the difference of two
    # energies to rise above their rounding, and the tolerance to match
    check = calc.check_gradients(silicon_carbide(), step=2e-3, tolerance=2e-3)
    assert check.ok, str(check)


def test_matches_its_own_ase_calculator(calc):
    pytest.importorskip("ase")
    from pet_kokkos.ase import PETCalculator

    system = silicon_carbide(seed=1)
    want = ("energy", "forces", "stress", "energies")
    ours = calc(system, want)
    theirs = Calculator.from_ase(PETCalculator(MODEL))(system, want)
    assert ours.energy == pytest.approx(theirs.energy, abs=1e-6)
    np.testing.assert_allclose(ours.forces, theirs.forces, atol=1e-6)
    np.testing.assert_allclose(ours.stress, theirs.stress, atol=1e-8)
    np.testing.assert_allclose(ours.energies, theirs.energies, atol=1e-6)


def test_per_atom_energies_sum_to_the_total(calc):
    result = calc(silicon_carbide(seed=2), "energies")
    assert result.energies.sum() == pytest.approx(result.energy, abs=1e-6)


def test_batch_matches_one_at_a_time(calc):
    systems = [silicon_carbide(seed=s) for s in range(3)] + [water()]
    want = ("energy", "forces")
    batched = calc.batch(systems, want)
    calc.clear_cache()
    for system, result in zip(systems, batched, strict=True):
        single = calc(system, want)
        # the batch reduces in a different order, so fp32 rounding differs
        assert result.energy == pytest.approx(single.energy, abs=1e-4)
        np.testing.assert_allclose(result.forces, single.forces, atol=1e-4)


def test_isolated_molecule_has_forces_but_no_stress(calc):
    system = water()
    assert calc(system, "forces").forces.shape == (3, 3)
    with pytest.raises(PropertyNotAvailable, match="periodic"):
        calc(system, "stress")


def test_unknown_element_is_named(calc):
    system = System([118, 1], [[0, 0, 0], [0, 0, 1.0]])
    with pytest.raises(PropertyNotAvailable, match="118"):
        calc(system)


def test_energy_noise_is_float32(calc):
    assert calc.energy_noise(-100.0) == pytest.approx(100 * np.finfo(np.float32).eps)

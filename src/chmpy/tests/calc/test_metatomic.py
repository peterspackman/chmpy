"""The native metatomic backend, against metatomic's own ASE calculator.

Skipped unless metatomic, torch and a PET-MAD checkpoint are all present.
"""

import numpy as np
import pytest

from chmpy.calc import Calculator, System

pytest.importorskip("torch")
pytest.importorskip("metatomic.torch")
pytest.importorskip("pet_mad")


@pytest.fixture(scope="module")
def pet():
    """PET-MAD, loaded once. Skips if the checkpoint is not already cached."""
    import warnings

    from pet_mad.calculator import PETMADCalculator

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return PETMADCalculator(version="latest")
    except Exception as exc:  # no network, no cached checkpoint
        pytest.skip(f"PET-MAD is not available: {exc}")


@pytest.fixture(scope="module")
def native(pet):
    from chmpy.calc.adapters.metatomic import MetatomicCalculator

    return MetatomicCalculator(pet._model)


def argon(a=5.3, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    ) * a + rng.normal(scale=0.05, size=(4, 3))
    return System([18] * 4, positions, np.eye(3) * a, True)


def test_matches_metatomics_own_ase_calculator(native, pet):
    system = argon(seed=1)
    want = ("energy", "forces", "stress")
    ours = native(system, want)
    theirs = Calculator.from_ase(pet)(system, want)

    # The model runs in float32 and the two paths enumerate pairs in different
    # orders, so the sums round differently. This is agreement to the model's
    # own precision, not a looser standard.
    assert ours.energy == pytest.approx(theirs.energy, abs=5e-5)
    np.testing.assert_allclose(ours.forces, theirs.forces, atol=1e-4)
    np.testing.assert_allclose(ours.stress, theirs.stress, atol=1e-6)


def test_a_batch_gives_the_same_answers_as_one_at_a_time(native):
    systems = [argon(5.3, seed=s) for s in range(4)]
    want = ("energy", "forces", "stress")

    individually = [native(system.copy(), want) for system in systems]
    batched = native.batch([system.copy() for system in systems], want)

    for one, many in zip(individually, batched, strict=True):
        assert one.energy == pytest.approx(many.energy, abs=5e-5)
        np.testing.assert_allclose(one.forces, many.forces, atol=1e-4)
        np.testing.assert_allclose(one.stress, many.stress, atol=1e-6)


def test_a_batch_is_one_call_into_the_model(native):
    systems = [argon(5.3, seed=s).copy() for s in range(5)]
    native.stats.reset()
    native.batch(systems, ("energy", "forces"))
    assert native.stats.batched_calls == 1
    assert native.stats.calls == 5


def test_forces_agree_with_finite_differences(native):
    """The model's own gradients, checked against its own energies.

    A float32 model puts a floor on how well this can do: an energy carries
    about 1e-5 eV of rounding, so a central difference over a step `h` carries
    `1e-5 / h` of noise. The step is large for a finite difference, and the
    tolerance is relative, for that reason and not to paper over a discrepancy.
    """
    system = argon(5.3, seed=2)
    check = native.check_gradients(system, step=1e-2, tolerance=0.05)
    assert check.ok, str(check)


def test_the_neighbour_list_is_reused_across_small_steps(native):
    # the calculator is shared across this module, so count the change
    before = [(nl.rebuilds, nl.reuses) for _, nl in native._neighbor_lists]
    system = argon(5.3, seed=3)
    for _ in range(5):
        native(system, ("energy", "forces"))
        system.set_positions(system.positions + 1e-3)
    after = [(nl.rebuilds, nl.reuses) for _, nl in native._neighbor_lists]

    rebuilds = sum(a[0] - b[0] for a, b in zip(after, before, strict=True))
    reuses = sum(a[1] - b[1] for a, b in zip(after, before, strict=True))
    assert reuses >= 4  # only the first geometry should need enumerating
    assert rebuilds == 1


def test_per_atom_energies_sum_to_the_total(native):
    system = argon(5.3, seed=4)
    result = native(system, ("energy", "energies"))
    assert result.energies.shape == (len(system),)
    assert float(result.energies.sum()) == pytest.approx(result.energy, abs=1e-5)


def test_the_backends_agree_on_a_geometry_neither_of_them_chose(native, pet):
    """Rule out the neighbour lists as a source of energy differences.

    Two optimisers that stop at different points inside the same tolerance box
    report different energies, which invites the conclusion that one of them
    computes the energy differently. This pins it down: on a geometry neither
    produced, the two backends agree to the model's own precision.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    system = argon(5.1, seed=11)
    atoms = system.to_ase()
    atoms.calc = pet
    LBFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=5)

    elsewhere = System.from_ase(atoms)
    ours = native(elsewhere.copy(), ("energy",)).energy
    theirs = Calculator.from_ase(pet)(elsewhere.copy(), ("energy",)).energy
    # one unit in the last place of a float32 energy of this magnitude
    assert abs(ours - theirs) <= 4 * abs(ours) * np.finfo(np.float32).eps

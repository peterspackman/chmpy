import numpy as np
import pytest

from chmpy.calc import (
    Calculator,
    LennardJones,
    PropertyNotAvailable,
    Result,
    System,
)


class Harmonic(Calculator):
    """Springs to fixed reference sites: analytic everything, trivially checkable."""

    provides = {"energy", "forces", "stress"}

    def __init__(self, reference, k=1.5, **kwargs):
        super().__init__(**kwargs)
        self.reference = np.asarray(reference, dtype=float)
        self.k = k

    def compute(self, system, want):
        d = system.positions - self.reference
        stress = None
        if "stress" in want:
            # E depends on positions only, so dE/deps = sum_i r_i (x) dE/dr_i
            stress = (system.positions.T @ (self.k * d)) / system.volume
            stress = 0.5 * (stress + stress.T)
        return Result(
            energy=0.5 * self.k * float((d * d).sum()),
            forces=-self.k * d if "forces" in want else None,
            stress=stress,
            volume=system.volume,
        )


class EnergyOnly(Calculator):
    """Declares only an energy, so everything else has to be differenced."""

    provides = {"energy"}

    def __init__(self, reference, k=1.5):
        self.reference = np.asarray(reference, dtype=float)
        self.k = k
        self.n_calls = 0

    def compute(self, system, want):
        self.n_calls += 1
        d = system.positions - self.reference
        return Result(energy=0.5 * self.k * float((d * d).sum()))


def argon(a=5.3, displaced=True):
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    if displaced:
        positions = positions + np.array(
            [[0.05, -0.02, 0.01], [0, 0.03, 0], [-0.04, 0, 0.02], [0.01, 0.01, -0.03]]
        )
    return System([18] * 4, positions, np.eye(3) * a, True)


def test_a_calculator_that_skips_super_init_still_works():
    """Forgetting super().__init__() must not cost caching or statistics."""
    system = argon()
    calc = EnergyOnly(system.positions * 0.99)
    assert calc.energy(system) > 0
    assert calc.stats.calls == 1
    calc.energy(system)
    assert calc.stats.cache_hits == 1


def test_finite_difference_forces_match_analytic():
    system = argon()
    reference = system.positions * 0.97
    analytic = Harmonic(reference).forces(system)
    numeric = EnergyOnly(reference).forces(system)
    np.testing.assert_allclose(numeric, analytic, atol=1e-7)


def test_finite_difference_stress_matches_analytic():
    system = argon()
    reference = system.positions * 0.97
    analytic = Harmonic(reference).stress(system)
    numeric = EnergyOnly(reference).stress(system)
    np.testing.assert_allclose(numeric, analytic, atol=1e-8)


def test_finite_difference_forces_cost_six_evaluations_per_atom():
    system = argon()
    calc = EnergyOnly(system.positions * 0.97)
    calc.forces(system)
    assert calc.stats.finite_difference_calls == 6 * len(system)


def test_results_are_cached_until_the_geometry_changes():
    system = argon()
    calc = Harmonic(system.positions * 0.97)

    calc.energy(system)
    calc.forces(system)  # a different property: must recompute
    calc.forces(system)  # same geometry and property: must not
    assert calc.stats.calls == 2
    assert calc.stats.cache_hits == 1

    system.set_positions(system.positions + 0.01)
    calc.forces(system)
    assert calc.stats.calls == 3


def test_check_gradients_passes_for_a_correct_calculator():
    system = argon()
    check = LennardJones().check_gradients(system)
    assert check.ok, str(check)


def test_check_gradients_catches_a_wrong_sign():
    class Broken(LennardJones):
        def compute(self, system, want):
            result = super().compute(system, want)
            return Result(
                energy=result.energy,
                forces=-result.forces if result.forces is not None else None,
                stress=result.stress,
                energies=result.energies,
                volume=result.volume,
            )

    check = Broken().check_gradients(argon())
    assert not check.ok
    assert "forces" in str(check)


def test_stress_of_an_isolated_system_is_refused_clearly():
    system = System([18, 18], [[0, 0, 0], [3.8, 0, 0]])
    with pytest.raises(PropertyNotAvailable, match="only defined for a system"):
        LennardJones().stress(system)


def test_shapes_are_validated_with_the_calculator_named():
    class WrongShape(Calculator):
        provides = {"energy", "forces"}

        def compute(self, system, want):
            return Result(energy=0.0, forces=np.zeros((len(system) + 1, 3)))

    with pytest.raises(ValueError, match="WrongShape returned forces with shape"):
        WrongShape().forces(argon())


def test_non_finite_results_are_refused():
    class Diverges(Calculator):
        def compute(self, system, want):
            return Result(energy=np.inf)

    with pytest.raises(PropertyNotAvailable, match="non-finite energy"):
        Diverges().energy(argon())


def test_sum_of_two_calculators_adds_every_property():
    system = argon()
    a = Harmonic(system.positions * 0.97, k=1.0)
    b = Harmonic(system.positions * 1.02, k=2.0)
    total = a + b

    result = total(system, ("energy", "forces", "stress"))
    np.testing.assert_allclose(result.energy, a.energy(system) + b.energy(system))
    np.testing.assert_allclose(result.forces, a.forces(system) + b.forces(system))
    np.testing.assert_allclose(result.stress, a.stress(system) + b.stress(system))


def test_sums_flatten_rather_than_nest():
    system = argon()
    a, b, c = (Harmonic(system.positions) for _ in range(3))
    assert len((a + b + c).calculators) == 3


def test_shifted_removes_a_per_element_reference():
    system = argon()
    calc = LennardJones()
    shifted = calc.shifted({18: -1.5})
    assert shifted.energy(system) == pytest.approx(calc.energy(system) + 6.0)
    np.testing.assert_allclose(shifted.forces(system), calc.forces(system))


def test_batch_evaluates_every_system_and_uses_the_cache():
    systems = [argon(a) for a in (5.2, 5.3, 5.4)]
    calc = LennardJones()
    energies = [r.energy for r in calc.batch(systems)]
    assert len(energies) == 3
    assert calc.stats.calls == 3

    again = [r.energy for r in calc.batch(systems)]
    np.testing.assert_allclose(again, energies)
    assert calc.stats.cache_hits == 3


def test_unknown_property_names_are_rejected():
    with pytest.raises(ValueError, match="unknown property"):
        LennardJones()(argon(), ("energy", "dipole"))


class FromASequence(Calculator):
    """Returns a fixed cycle of energies, for testing precision inference."""

    provides = {"energy"}

    def __init__(self, values):
        self.values = list(values)
        self.index = -1

    def compute(self, system, want):
        self.index = (self.index + 1) % len(self.values)
        return Result(energy=self.values[self.index])


def _feed(calc, count=4):
    system = System([18], [[0.0, 0.0, 0.0]])
    for _ in range(count):
        calc.energy(system)
        system.set_positions(system.positions)


def test_float32_energies_are_recognised():
    values = [float(np.float32(v)) for v in (-192.737534, -192.7376, -192.73742)]
    calc = FromASequence(values)
    _feed(calc)
    assert calc.energy_noise(-192.7) == pytest.approx(192.7 * np.finfo(np.float32).eps)


def test_float64_energies_are_recognised():
    calc = FromASequence([-192.737534123, -192.73760001, -192.7374200003])
    _feed(calc)
    assert calc.energy_noise(-192.7) < 1e-12


def test_one_round_energy_does_not_convict_a_float64_calculator():
    """The single-sample rule called -1.0 a float32 model. Three do not."""
    calc = FromASequence([-1.0, -192.737534123, -0.5])
    _feed(calc)
    assert calc.energy_noise(-192.7) < 1e-12


def test_a_declared_precision_is_never_second_guessed():
    calc = FromASequence([-1.0, -0.5, -100.0])
    calc.energy_precision = float(np.finfo(np.float64).eps)
    _feed(calc)
    assert calc.energy_noise(-100.0) < 1e-12


def test_an_undecided_calculator_assumes_full_precision():
    """One repeated energy is not evidence of anything."""
    calc = FromASequence([-192.5])
    _feed(calc)
    assert calc.energy_noise(-192.5) < 1e-12

import numpy as np
import pytest

from chmpy.calc import Calculator, LennardJones, Result, System
from chmpy.opt.coordinates import Atomic, AtomicStrain
from chmpy.opt.curvature import LimitedMemoryBFGS
from chmpy.opt.trust_region import TrustRegion, TrustRegionOptions, _dogleg


class HarmonicField(Calculator):
    """A quadratic energy surface in Cartesian coordinates, exactly solvable."""

    provides = {"energy", "forces"}

    def __init__(self, hessian, centre):
        self.hessian = np.asarray(hessian, dtype=float)
        self.centre = np.asarray(centre, dtype=float)

    def compute(self, system, want):
        displacement = (system.positions - self.centre).ravel()
        gradient = self.hessian @ displacement
        return Result(
            energy=0.5 * float(displacement @ gradient),
            forces=-gradient.reshape(-1, 3),
        )


def harmonic(n_atoms=2, seed=0, displacement=0.05):
    """A random positive-definite quadratic and a start displaced from its minimum."""
    rng = np.random.default_rng(seed)
    n = 3 * n_atoms
    a = rng.normal(size=(n, n))
    hessian = a @ a.T + 0.5 * np.eye(n)
    centre = rng.normal(size=(n_atoms, 3)) * 2.0
    system = System(
        [18] * n_atoms, centre + rng.normal(size=(n_atoms, 3)) * displacement
    )
    return system, HarmonicField(hessian, centre)


def argon(a=5.3, rattle=0.0, seed=0):
    scaled = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cell = np.eye(3) * a
    positions = scaled @ cell
    if rattle:
        positions = positions + np.random.default_rng(seed).normal(
            scale=rattle, size=positions.shape
        )
    return System([18] * 4, positions, cell, True)


def test_dogleg_takes_the_newton_step_when_it_fits():
    hessian = np.diag([2.0, 4.0])
    gradient = np.array([1.0, 1.0])
    step, predicted = _dogleg(
        LimitedMemoryBFGS(2, initial=hessian), gradient, radius=10.0
    )
    np.testing.assert_allclose(step, [-0.5, -0.25])
    assert predicted == pytest.approx(0.5 * (1.0 / 2 + 1.0 / 4))


def test_dogleg_stays_inside_the_radius():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=(5, 5))
        hessian = a @ a.T + np.eye(5)
        gradient = rng.normal(size=5)
        step, predicted = _dogleg(
            LimitedMemoryBFGS(5, initial=hessian), gradient, radius=0.3
        )
        assert np.linalg.norm(step) <= 0.3 + 1e-12
        assert predicted > 0


def test_dogleg_handles_an_indefinite_model():
    """Where the Newton step is nonsense, go downhill to the boundary."""
    hessian = np.diag([1.0, -2.0])
    gradient = np.array([0.5, 0.5])
    step, _ = _dogleg(LimitedMemoryBFGS(2, initial=hessian), gradient, radius=0.4)
    assert np.linalg.norm(step) == pytest.approx(0.4)
    assert gradient @ step < 0


def test_powell_damping_accepts_pairs_plain_bfgs_would_have_to_skip():
    """On an indefinite surface, damping learns from every step."""
    rng = np.random.default_rng(3)
    probes = np.random.default_rng(31)
    true_hessian = np.diag([2.0, 1.0, -0.5, -1.5])  # a saddle, as at a bad start
    model = LimitedMemoryBFGS(4)
    negative_curvature = applied = 0
    for _ in range(20):
        step = rng.normal(size=4) * 0.05
        change = true_hessian @ step
        if change @ step <= 0:
            negative_curvature += 1  # plain BFGS would have to skip this one
        applied += model.update(step, change)
        # a positive-definite model has a positive quadratic form; the probe
        # gets its own stream so it cannot shift the steps being counted
        probe = probes.normal(size=4)
        assert float(probe @ model.matvec(probe)) > 0
    assert negative_curvature > 3
    assert applied == 20


def test_a_collapsed_model_falls_back_rather_than_failing():
    """Damping caps the curvature it learns along directions that curve down.

    Fed enough of them, the model's eigenvalues in those directions go to zero
    and the Cholesky factorisation fails. That is not an error: the dogleg
    takes the steepest-descent leg, which is the right step where the model
    has nothing to say.
    """
    model = LimitedMemoryBFGS(2, initial=np.diag([1.0, 1e-18]))
    step, predicted = _dogleg(model, np.array([0.3, 0.1]), radius=0.2)
    assert np.all(np.isfinite(step))
    assert np.linalg.norm(step) <= 0.2 + 1e-12
    assert predicted > 0


def test_exact_hessian_reaches_a_quadratic_minimum_in_one_step():
    system, calc = harmonic(n_atoms=2, seed=4, displacement=0.02)
    relaxation = TrustRegion(Atomic(system), calc, hessian=calc.hessian).run(
        fmax=1e-10, steps=5
    )
    assert relaxation.converged
    assert relaxation.steps == 1
    np.testing.assert_allclose(system.positions, calc.centre, atol=1e-12)


def test_a_quadratic_converges_from_the_identity_model():
    system, calc = harmonic(n_atoms=3, seed=5, displacement=0.05)
    relaxation = TrustRegion(Atomic(system), calc).run(fmax=1e-9, steps=100)
    assert relaxation.converged
    np.testing.assert_allclose(system.positions, calc.centre, atol=1e-8)


def test_fixed_cell_relaxation_of_a_rattled_crystal():
    system = argon(5.3, rattle=0.15, seed=1)
    calc = LennardJones()
    relaxation = TrustRegion(Atomic(system), calc).run(fmax=1e-4, steps=200)
    assert relaxation.converged
    assert relaxation.measures["fmax"] < 1e-4


def test_variable_cell_relaxation_meets_both_criteria():
    system = argon(5.0, rattle=0.1, seed=2)
    calc = LennardJones()
    relaxation = TrustRegion(AtomicStrain(system), calc).run(
        fmax=1e-4, smax=1e-3, steps=300
    )
    assert relaxation.converged
    assert relaxation.measures["fmax"] < 1e-4
    assert relaxation.measures["smax"] < 1e-3
    # a relaxed fcc argon crystal sits near the LJ nearest-neighbour distance
    assert 5.0 < np.linalg.norm(system.cell[0]) < 6.5


def test_convergence_criteria_are_physical_quantities():
    """fmax is eV/A and smax is GPa: neither is diluted by the other."""
    system = argon(5.0, rattle=0.1, seed=3)
    calc = LennardJones()
    relaxation = TrustRegion(AtomicStrain(system), calc).run(
        fmax=1e-4, smax=1e-3, steps=300
    )
    result = calc(system, ("energy", "forces", "stress"))
    assert np.abs(result.forces).max() == pytest.approx(
        relaxation.measures["fmax"], rel=1e-9
    )
    assert np.abs(result.stress_gpa).max() == pytest.approx(
        relaxation.measures["smax"], rel=1e-9
    )


def test_rejected_steps_do_not_cost_a_second_evaluation():
    """Returning to the accepted point must come out of the cache."""
    system = argon(5.0, rattle=0.2, seed=4)
    calc = LennardJones()
    # a deliberately over-large starting radius, so some steps are rejected
    options = TrustRegionOptions(delta0=3.0)
    relaxation = TrustRegion(Atomic(system), calc, options=options).run(
        fmax=1e-4, steps=200
    )
    rejected = sum(1 for step in relaxation.history if not step.accepted)
    assert rejected > 0
    # one evaluation for the starting point, one per trial step, and nothing
    # for the rollbacks
    assert relaxation.evaluations == relaxation.steps + 1


def test_the_radius_recovers_from_a_collapse():
    """A badly scaled start should restart rather than die at the floor."""
    system = argon(4.2, rattle=0.4, seed=7)
    calc = LennardJones()
    options = TrustRegionOptions(delta0=1.0)
    relaxation = TrustRegion(AtomicStrain(system), calc, options=options).run(
        fmax=1e-3, smax=1e-2, steps=400
    )
    assert relaxation.converged


def test_external_pressure_compresses_the_cell():
    calc = LennardJones()
    relaxed = argon(5.3, rattle=0.05, seed=8)
    TrustRegion(AtomicStrain(relaxed), calc).run(fmax=1e-4, smax=1e-3, steps=300)

    compressed = argon(5.3, rattle=0.05, seed=8)
    relaxation = TrustRegion(AtomicStrain(compressed), calc, pressure=2.0).run(
        fmax=1e-4, smax=1e9, steps=300
    )
    assert relaxation.converged
    assert compressed.volume < relaxed.volume
    # at equilibrium the internal stress balances the applied pressure:
    # minimising E + PV gives sigma = -P I, i.e. a hydrostatic pressure of +P
    result = calc(compressed, ("energy", "forces", "stress"))
    assert result.pressure == pytest.approx(2.0, abs=0.05)


def test_a_logger_sees_every_step():
    system = argon(5.2, rattle=0.1, seed=9)
    lines = []
    TrustRegion(Atomic(system), LennardJones()).run(
        fmax=1e-3, steps=50, logger=lines.append
    )
    assert lines and all("E=" in line for line in lines)

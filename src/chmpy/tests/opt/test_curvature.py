"""The quadratic model: dense and limited-memory must be the same model."""

import numpy as np
import pytest

from chmpy.calc import LennardJones, System
from chmpy.opt.coordinates import AtomicStrain
from chmpy.opt.curvature import (
    DEFAULT_MEMORY,
    FULL_MODEL_LIMIT,
    LimitedMemoryBFGS,
    for_size,
)
from chmpy.opt.trust_region import TrustRegion


def quadratic(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + 0.5 * np.eye(n)


def feed(model, hessian, count=8, seed=1):
    """Curvature pairs from an exact quadratic."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        step = rng.normal(size=model.n) * 0.1
        model.update(step, hessian @ step)
    return model


@pytest.mark.parametrize("diagonal", [None, "given"])
def test_the_forward_and_inverse_are_the_same_model(diagonal):
    """solve(matvec(v)) == v.

    A dogleg takes its Newton step from `solve` and scores it with `matvec`.
    When the two were unwound through different `B0` -- a scalar scaling
    applied to one of them only -- it was taking a step from one model and
    judging it against another.
    """
    n = 30
    rng = np.random.default_rng(2)
    d = None if diagonal is None else np.abs(rng.normal(size=n)) + 0.5
    model = feed(LimitedMemoryBFGS(n, memory=8, diagonal=d), quadratic(n))
    vector = rng.normal(size=n)
    np.testing.assert_allclose(model.solve(model.matvec(vector)), vector, atol=1e-10)


def test_the_model_stays_positive_definite_under_damping():
    n = 12
    saddle = np.diag(np.concatenate([np.ones(8), -np.ones(4)]))
    for model in (LimitedMemoryBFGS(n, memory=6), LimitedMemoryBFGS(n, memory=2)):
        rng = np.random.default_rng(6)
        for _ in range(15):
            step = rng.normal(size=n) * 0.05
            assert model.update(step, saddle @ step)
            # a positive-definite model has a positive quadratic form
            probe = rng.normal(size=n)
            assert float(probe @ model.matvec(probe)) > 0


def test_only_the_last_pairs_are_kept():
    model = LimitedMemoryBFGS(20, memory=5)
    feed(model, quadratic(20), count=12)
    assert len(model) == 5


def test_rescaling_is_a_change_of_coordinates():
    """A rescale and its inverse must leave the model where it started."""
    n = 10
    hessian = quadratic(n, seed=8)
    rng = np.random.default_rng(9)
    old = np.abs(rng.normal(size=n)) + 0.5
    new = old * (1.0 + 0.1 * rng.normal(size=n))

    limited = feed(LimitedMemoryBFGS(n, memory=8), hessian)
    vector = rng.normal(size=n)
    reference = limited.solve(vector)
    limited.rescale(old, new)
    limited.rescale(new, old)
    np.testing.assert_allclose(limited.solve(vector), reference, atol=1e-10)


def test_for_size_builds_a_model_at_every_size():
    assert isinstance(for_size(50), LimitedMemoryBFGS)
    assert isinstance(for_size(FULL_MODEL_LIMIT + 1), LimitedMemoryBFGS)


def test_a_small_starting_model_is_kept_whole():
    """Its couplings are the reason limited memory beats dense at small n."""
    initial = quadratic(30, seed=10)
    model = for_size(30, initial=initial)
    assert model._matrix is not None
    off_diagonal = model._matrix - np.diag(np.diag(model._matrix))
    assert np.abs(off_diagonal).max() > 1e-6


def test_a_large_starting_model_is_reduced_to_its_diagonal():
    n = FULL_MODEL_LIMIT + 1
    initial = np.diag(np.linspace(1.0, 3.0, n))
    model = for_size(n, initial=initial)
    assert model._matrix is None
    expected = np.diag(initial) / np.mean(np.diag(initial))
    np.testing.assert_allclose(model._shape, expected, atol=1e-12)


def test_a_full_starting_model_round_trips():
    n = 25
    rng = np.random.default_rng(11)
    b0 = quadratic(n, seed=12)
    model = feed(LimitedMemoryBFGS(n, memory=6, initial=b0), quadratic(n, seed=13))
    vector = rng.normal(size=n)
    np.testing.assert_allclose(model.solve(model.matvec(vector)), vector, atol=1e-9)


def test_the_memory_length_does_not_change_where_it_lands():
    """Keeping three curvature pairs or twenty finds the same minimum."""

    def run(memory):
        positions = np.array(
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        ) * 5.0 + np.array(
            [[0.08, 0, 0], [0, 0.05, 0], [0, 0, -0.06], [0.02, 0.02, 0.02]]
        )
        system = System([18] * 4, positions, np.eye(3) * 5.0, True)
        coordinates = AtomicStrain(system)
        model = LimitedMemoryBFGS(coordinates.n_dof, memory=memory)
        return TrustRegion(coordinates, LennardJones(), model=model).run(
            fmax=1e-5, smax=1e-4, steps=300
        )

    short, long = run(3), run(20)
    assert short.converged and long.converged
    assert short.energy == pytest.approx(long.energy, abs=1e-7)


def test_an_indefinite_starting_model_falls_back_to_its_diagonal():
    """An analytic Hessian at a saddle is a legitimate thing to be handed."""
    indefinite = np.diag([1.0, -2.0, 3.0])
    model = LimitedMemoryBFGS(3, initial=indefinite)
    assert model._matrix is None  # the couplings could not be used
    probe = np.array([0.3, 0.5, -0.2])
    assert float(probe @ model.matvec(probe)) > 0


def test_the_default_memory_is_what_it_says():
    assert LimitedMemoryBFGS(10).memory == DEFAULT_MEMORY

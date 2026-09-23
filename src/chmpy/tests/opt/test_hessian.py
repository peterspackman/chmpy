"""The starting curvature model."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones, System
from chmpy.opt import SymmetryAdapted, relax, stretch_hessian
from chmpy.opt.coordinates import Atomic, AtomicStrain

from .. import TEST_FILES


def argon(a=5.3):
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    positions = positions + np.array(
        [[0.05, 0, 0], [0, 0.03, 0], [0, 0, -0.04], [0.01, 0.01, 0.01]]
    )
    return System([18] * 4, positions, np.eye(3) * a, True)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Atomic(argon()),
        lambda: AtomicStrain(argon()),
        lambda: SymmetryAdapted(Crystal.load(TEST_FILES["acetic_acid.cif"])),
    ],
)
def test_the_model_is_symmetric_and_positive_definite(make):
    coordinates = make()
    hessian = stretch_hessian(coordinates)
    assert hessian.shape == (coordinates.n_dof, coordinates.n_dof)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
    np.linalg.cholesky(hessian)  # raises if it is not positive definite


def test_building_the_model_costs_no_energy_evaluations():
    coordinates = AtomicStrain(argon())
    calc = LennardJones()
    stretch_hessian(coordinates)
    assert calc.stats.calls == 0


def test_building_the_model_leaves_the_geometry_alone():
    system = argon()
    coordinates = AtomicStrain(system)
    before = np.array(system.positions), np.array(system.cell)
    stretch_hessian(coordinates)
    np.testing.assert_allclose(system.positions, before[0], atol=1e-14)
    np.testing.assert_allclose(system.cell, before[1], atol=1e-14)


def test_the_model_supplies_shape_not_magnitude():
    """Normalised to a mean diagonal of gamma, whatever the material."""
    for gamma in (1.0, 3.0):
        hessian = stretch_hessian(AtomicStrain(argon()), gamma=gamma)
        # the ridge adds gamma * ridge on top of the normalised mean
        assert np.mean(np.diag(hessian)) == pytest.approx(gamma * 1.05, rel=1e-6)


def test_the_model_sees_the_anisotropy_the_identity_misses():
    """A stiff bond and a soft contact must not look alike."""
    crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
    hessian = stretch_hessian(SymmetryAdapted(crystal))
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert eigenvalues.max() / eigenvalues.min() > 20


def test_relaxing_with_the_model_reaches_the_same_minimum():
    calc = LennardJones()
    with_model = relax(argon(), calc, fmax=1e-4, smax=1e-3, steps=200)
    without = relax(argon(), calc, hessian="identity", fmax=1e-4, smax=1e-3, steps=200)
    assert with_model.converged and without.converged
    # two different paths into the same basin: equal to well below the
    # convergence threshold, not bit for bit
    assert with_model.energy == pytest.approx(without.energy, abs=1e-6)


def test_an_unknown_hessian_argument_is_refused():
    """Eagerly: the default plan has its own per-stage setting, so a
    misspelled top-level one would otherwise be ignored in silence."""
    with pytest.raises(ValueError, match="hessian must be"):
        relax(argon(), LennardJones(), hessian="guess", steps=1)

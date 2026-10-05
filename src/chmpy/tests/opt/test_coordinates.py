"""Every parameterisation must report the gradient of the energy it produces."""

import numpy as np
import pytest

from chmpy.calc import LennardJones, System
from chmpy.opt.coordinates import Atomic, AtomicStrain, Strain, pressure_term
from chmpy.opt.strain import invariant_strain_basis


def argon(a=5.3, skew=False):
    cell = np.eye(3) * a
    if skew:
        cell = np.array([[a, 0, 0], [0.4 * a, 0.9 * a, 0], [0.1 * a, 0.2 * a, 1.1 * a]])
    scaled = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    scaled = scaled + np.array(
        [[0.01, -0.02, 0.01], [0, 0.03, 0], [-0.02, 0, 0.02], [0.01, 0.01, -0.01]]
    )
    return System([18] * 4, scaled @ cell, cell, True)


def numeric_gradient(coordinates, calc, step=1e-6):
    """dE/dx by central differences in degree-of-freedom space."""
    x = coordinates.get()
    gradient = np.zeros_like(x)
    for k in range(len(x)):
        shifted = x.copy()
        shifted[k] = x[k] + step
        coordinates.set(shifted)
        plus = calc.energy(coordinates.system)
        shifted[k] = x[k] - step
        coordinates.set(shifted)
        minus = calc.energy(coordinates.system)
        gradient[k] = (plus - minus) / (2 * step)
    coordinates.set(x)
    return gradient


def analytic_gradient(coordinates, calc):
    return coordinates.gradient(calc(coordinates.system, coordinates.wanted))


@pytest.mark.parametrize("skew", [False, True])
def test_atomic_gradient(skew):
    coordinates = Atomic(argon(skew=skew))
    calc = LennardJones()
    np.testing.assert_allclose(
        analytic_gradient(coordinates, calc),
        numeric_gradient(coordinates, calc),
        atol=1e-7,
    )


@pytest.mark.parametrize("skew", [False, True])
def test_strain_gradient_at_the_reference(skew):
    coordinates = Strain(argon(skew=skew))
    calc = LennardJones()
    np.testing.assert_allclose(
        analytic_gradient(coordinates, calc),
        numeric_gradient(coordinates, calc),
        atol=1e-7,
    )


def test_strain_gradient_away_from_the_reference():
    """The F^-T factor only shows up once the cell has actually deformed."""
    coordinates = Strain(argon(skew=True))
    calc = LennardJones()
    coordinates.set(np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.03]))
    np.testing.assert_allclose(
        analytic_gradient(coordinates, calc),
        numeric_gradient(coordinates, calc),
        atol=1e-7,
    )


@pytest.mark.parametrize("skew", [False, True])
def test_atomic_strain_gradient(skew):
    coordinates = AtomicStrain(argon(skew=skew))
    calc = LennardJones()
    np.testing.assert_allclose(
        analytic_gradient(coordinates, calc),
        numeric_gradient(coordinates, calc),
        atol=1e-7,
    )


def test_atomic_strain_gradient_away_from_the_reference():
    coordinates = AtomicStrain(argon(skew=True))
    calc = LennardJones()
    x = coordinates.get()
    x[: coordinates.n_atomic] += 0.03
    x[coordinates.n_atomic :] = [0.04, -0.02, 0.03, 0.01, 0.02, -0.01]
    coordinates.set(x)
    np.testing.assert_allclose(
        analytic_gradient(coordinates, calc),
        numeric_gradient(coordinates, calc),
        atol=1e-7,
    )


def test_a_constrained_strain_basis_gives_fewer_dofs_and_a_correct_gradient():
    system = argon()
    basis = invariant_strain_basis(_cubic_rotations())
    coordinates = AtomicStrain(system, basis=basis)
    assert len(basis) == 1
    assert coordinates.n_dof == 3 * len(system) + 1
    calc = LennardJones()
    np.testing.assert_allclose(
        analytic_gradient(coordinates, calc),
        numeric_gradient(coordinates, calc),
        atol=1e-7,
    )


def test_fixed_atoms_do_not_appear_as_degrees_of_freedom():
    system = argon()
    fixed = np.array([True, False, False, False])
    coordinates = AtomicStrain(system, fixed=fixed)
    assert coordinates.n_atomic == 9
    before = np.array(system.positions[0])
    x = coordinates.get()
    x[:9] += 0.2
    coordinates.set(x)
    np.testing.assert_allclose(system.positions[0], before)


def test_scale_is_angstroms_of_displacement_per_unit_dof():
    system = argon(5.3)
    coordinates = AtomicStrain(system)
    scale = coordinates.scale()
    np.testing.assert_allclose(scale[: coordinates.n_atomic], 1.0)

    # one unit of a strain amplitude should move an atom by about `scale`
    x = coordinates.get()
    before = np.array(system.positions)
    x[coordinates.n_atomic] = 1.0 / scale[-1]
    coordinates.set(x)
    moved = np.linalg.norm(system.positions - before, axis=1).max()
    assert 0.2 < moved < 2.0


def test_reanchor_preserves_the_geometry():
    system = argon(skew=True)
    coordinates = AtomicStrain(system)
    x = coordinates.get()
    x[coordinates.n_atomic :] = [0.6, -0.1, 0.05, 0, 0, 0]
    coordinates.set(x)
    cell, positions = np.array(system.cell), np.array(system.positions)

    new_x = coordinates.reanchor(x)
    assert new_x is not None
    np.testing.assert_allclose(new_x[coordinates.n_atomic :], 0.0, atol=1e-14)
    coordinates.set(new_x)
    np.testing.assert_allclose(system.cell, cell, atol=1e-12)
    np.testing.assert_allclose(system.positions, positions, atol=1e-12)


def test_step_fraction_scales_the_whole_step():
    coordinates = Strain(argon())
    x = np.zeros(6)
    step = np.array([10.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    fraction = coordinates.step_fraction(x, step)
    assert 0 < fraction < 1
    # the limit binds on the first component, and the second is scaled by the
    # same factor rather than left alone
    assert (fraction * step)[0] == pytest.approx(0.4)


def test_external_pressure_gradient_matches_d_pv_dq():
    system = argon()
    coordinates = Strain(system)
    pressure_gpa = 5.0
    analytic = pressure_term(coordinates, pressure_gpa)

    from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

    pressure = pressure_gpa / EV_PER_ANGSTROM3_TO_GPA
    x = coordinates.get()
    numeric = np.zeros_like(x)
    for k in range(len(x)):
        for sign in (1, -1):
            shifted = x.copy()
            shifted[k] += sign * 1e-6
            coordinates.set(shifted)
            numeric[k] += sign * pressure * system.volume
    numeric /= 2e-6
    coordinates.set(x)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)


def _cubic_rotations():
    """The 48 operations of m-3m, as Cartesian matrices."""
    import itertools

    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            rotations.append(matrix)
    return np.array(rotations)

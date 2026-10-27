"""The Einstein-Debye vibrational free energy."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt import elastic_tensor, relax
from chmpy.util.unit import EV_TO_KJ_PER_MOL
from chmpy.vib import (
    DebyeModel,
    debye_model,
    einstein_debye,
    einstein_debye_properties,
    force_constants,
)
from chmpy.vib.einstein_debye import debye_function
from chmpy.vib.thermo import BOLTZMANN_EV, HBAR_EV_S

#: (T, F) in kJ/mol per molecule from ThermPropDebye's own example, test.clmz
#: (C11 = C22 = C33 = 10 GPa and nothing else, a 10 A cube, 100 amu, Z = 4)
THERMPROPDEBYE = [(2.0, 0.1919), (50.0, 0.0713), (100.0, -0.3206), (300.0, -3.0449)]


def test_the_acoustic_part_reproduces_thermpropdebye():
    model = debye_model(np.diag([10.0, 10, 10, 0, 0, 0]), 10.0 * np.eye(3), 100.0)
    temperatures, expected = np.array(THERMPROPDEBYE).T
    result = einstein_debye_properties([], model, temperatures, z=4)
    np.testing.assert_allclose(result.per_molecule("free_energy"), expected, atol=1e-4)


def test_the_debye_frequency_of_an_isotropic_cubic_crystal():
    """Each direction's boundary is |(a, b, c)| pi / a away, and the velocities
    are the same along all of them."""
    bulk, shear, a, mass = 20.0, 8.0, 7.0, 400.0
    lam = bulk - 2 * shear / 3
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] = lam + 2 * shear
    c[np.arange(3, 6), np.arange(3, 6)] = shear
    model = debye_model(c, a * np.eye(3), mass)

    rho = model.density
    v_l = np.sqrt((bulk + 4 * shear / 3) * 1e9 / rho)
    v_t = np.sqrt(shear * 1e9 / rho)
    lengths = np.array([1.0] * 3 + [np.sqrt(2)] * 6 + [np.sqrt(3)] * 4)
    k_cb = lengths.mean() * np.pi / (a * 1e-10)
    expected = 2 / np.pi * k_cb * (v_l + 2 * v_t) / 3
    assert model.frequency == pytest.approx(expected, rel=1e-12)
    assert model.mean_velocity == pytest.approx((v_l + 2 * v_t) / 3, rel=1e-12)


def test_the_debye_function_limits():
    assert debye_function(0.0) == 1.0
    assert debye_function(1e-4) == pytest.approx(1 - 3 / 8 * 1e-4, rel=1e-8)
    assert debye_function(60.0) == pytest.approx(np.pi**4 / (5 * 60.0**3), rel=1e-10)
    # continuous across the switch to the asymptote
    assert debye_function(150.0 - 1e-9) == pytest.approx(
        debye_function(150.0 + 1e-9), rel=1e-8
    )


def test_debye_thermodynamics_are_consistent():
    model = DebyeModel(
        frequency=2 * np.pi * 1.5e12, velocities=np.ones((13, 3)), density=1.0
    )
    T, h = 120.0, 1e-3
    F = lambda t: model.properties(t)[0]  # noqa: E731
    U = lambda t: model.properties(t)[3]  # noqa: E731
    free, entropy, heat_capacity, internal = model.properties(T)
    assert entropy == pytest.approx(-(F(T + h) - F(T - h)) / (2 * h), rel=1e-6)
    assert heat_capacity == pytest.approx((U(T + h) - U(T - h)) / (2 * h), rel=1e-6)
    assert internal == pytest.approx(free + T * entropy, rel=1e-12)
    # classical limit: three modes, 3k
    assert model.properties(1e5)[2] == pytest.approx(3 * BOLTZMANN_EV, rel=1e-3)
    # zero point is 9/8 hbar omega_D
    assert model.properties(0.0)[0] == pytest.approx(
        9 / 8 * HBAR_EV_S * model.frequency
    )


def test_imaginary_optical_modes_are_left_out_and_counted(caplog):
    model = DebyeModel(frequency=1e12, velocities=np.ones((13, 3)), density=1.0)
    omega = 2 * np.pi * np.array([-0.5e12, 1e12, 2e12])
    result = einstein_debye_properties(omega, model, [100.0])
    assert result.n_imaginary == 1
    assert "imaginary" in caplog.text
    clean = einstein_debye_properties(omega[1:], model, [100.0])
    assert result.free_energy[0] == pytest.approx(clean.free_energy[0])


def test_per_molecule_needs_z():
    model = DebyeModel(frequency=1e12, velocities=np.ones((13, 3)), density=1.0)
    with pytest.raises(ValueError, match="molecules per cell"):
        einstein_debye_properties([], model, [100.0]).per_molecule()


@pytest.fixture(scope="module")
def argon():
    cell = UnitCell.from_lengths_and_angles((5.3, 5.3, 5.3), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    relaxed = relax(crystal, calc, fmax=1e-6, smax=1e-5, steps=200).structure
    return relaxed, calc


def test_unit_cell_gamma_modes_match_a_supercell(argon):
    """A 1x1x1 displacement gives the exact Gamma-point modes: every image of a
    displaced atom moves with it, which is what a q = 0 mode is."""
    relaxed, calc = argon
    tensor = elastic_tensor(relaxed, calc, strain=0.002)
    result = einstein_debye(relaxed, calc, [0.0, 50.0], elastic=tensor)
    supercell = force_constants(relaxed, calc, cutoff=8.0)
    np.testing.assert_allclose(
        result.optical_frequencies,
        supercell.optical_frequencies(units="rad/s"),
        rtol=1e-4,
    )
    # four argon atoms, each its own molecule
    assert result.z == 4
    assert len(result.optical_frequencies) == 9
    assert result.n_imaginary == 0
    assert result.free_energy[0] == pytest.approx(result.zero_point)
    np.testing.assert_allclose(
        result.per_molecule("free_energy"),
        result.free_energy * EV_TO_KJ_PER_MOL / 4,
    )
    np.testing.assert_allclose(
        result.free_energy, result.optical[:, 0] + result.acoustic[:, 0]
    )

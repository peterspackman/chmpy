"""Harmonic thermodynamics from force constants."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt import relax
from chmpy.vib import force_constants
from chmpy.vib.thermo import (
    BOLTZMANN_EV,
    HBAR_EV_S,
    harmonic_free_energy,
    harmonic_properties,
    q_mesh,
    thermal_properties,
)

#: one mode, ~200 cm-1, in rad/s
OMEGA = 2 * np.pi * 6.0e12


def test_zero_temperature_free_energy_is_the_zero_point_energy():
    assert harmonic_free_energy(np.array([OMEGA]), 0.0) == pytest.approx(
        0.5 * HBAR_EV_S * OMEGA
    )


def test_high_temperature_limit_is_classical():
    T = 1.0e5
    kT = BOLTZMANN_EV * T
    classical = kT * np.log(HBAR_EV_S * OMEGA / kT)
    assert harmonic_free_energy(np.array([OMEGA]), T) == pytest.approx(
        classical, rel=1e-4
    )


def test_entropy_and_heat_capacity_are_derivatives_of_the_free_energy():
    frequencies = np.array([OMEGA, 0.5 * OMEGA, 3 * OMEGA])
    T, h = 150.0, 1e-3
    F = lambda t: harmonic_free_energy(frequencies, t)  # noqa: E731
    _, S, C = harmonic_properties(frequencies, T)
    assert S == pytest.approx(-(F(T + h) - F(T - h)) / (2 * h), rel=1e-6)
    S_of = lambda t: harmonic_properties(frequencies, t)[1]  # noqa: E731
    assert C == pytest.approx(T * (S_of(T + h) - S_of(T - h)) / (2 * h), rel=1e-6)


def test_heat_capacity_reaches_the_equipartition_limit():
    _, _, C = harmonic_properties(np.array([OMEGA]), 1.0e5)
    assert C == pytest.approx(BOLTZMANN_EV, rel=1e-4)


def test_q_mesh_is_gamma_centred_and_complete():
    q = q_mesh((3, 4, 2))
    assert len(q) == 24
    assert np.any(np.all(q == 0, axis=1))
    assert np.all(q >= -0.5) and np.all(q < 0.5)
    assert len(np.unique(np.round(q, 12), axis=0)) == 24


@pytest.fixture(scope="module")
def argon_constants():
    cell = UnitCell.from_lengths_and_angles((5.3, 5.3, 5.3), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    relaxed = relax(crystal, calc, fmax=1e-6, smax=1e-5).structure
    return force_constants(relaxed, calc, cutoff=8.0)


def test_free_energy_converges_with_the_mesh(argon_constants):
    values = [
        thermal_properties(argon_constants, [80.0], mesh=(n, n, n)).free_energy[0]
        for n in (4, 6, 8, 10)
    ]
    steps = np.abs(np.diff(values))
    assert np.all(steps[1:] < 0.5 * steps[:-1])


def test_argon_has_no_imaginary_modes(argon_constants):
    result = thermal_properties(argon_constants, [0.0, 50.0], mesh=(6, 6, 6))
    assert result.n_imaginary == 0
    assert result.zero_point > 0
    # four atoms in the conventional cell: 12 modes, 3 acoustic at Gamma skipped
    assert result.n_modes == 12 * 6**3 - 3
    assert result.free_energy[0] == pytest.approx(result.zero_point)
    assert result.free_energy[1] < result.free_energy[0]

"""Quasi-harmonic geometry, against a direct free-energy scan."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt import relax
from chmpy.vib import force_constants, thermal_properties
from chmpy.vib.qha import qha_structure, thermal_stress

MESH = (4, 4, 4)
CUTOFF = 8.0


def fcc_argon(a):
    cell = UnitCell.from_lengths_and_angles((a, a, a), np.radians((90, 90, 90)))
    return Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )


@pytest.fixture(scope="module")
def calc():
    return LennardJones(epsilon=0.0103, sigma=3.4, cutoff=CUTOFF)


@pytest.fixture(scope="module")
def static(calc):
    return relax(fcc_argon(5.3), calc, fmax=1e-7, smax=1e-6).structure


def free_energy(calc, a, temperature):
    crystal = fcc_argon(a)
    constants = force_constants(crystal, calc, cutoff=CUTOFF)
    vibrational = thermal_properties(constants, [temperature], mesh=MESH)
    from chmpy.calc import System

    return calc.energy(System.from_crystal(crystal)) + vibrational.free_energy[0]


def lattice(crystal):
    return float(crystal.unit_cell.lengths[0])


def test_thermal_stress_is_the_derivative_of_the_free_energy(calc, static):
    a, T, h = lattice(static), 40.0, 1e-4
    sigma = thermal_stress(static, calc, T, mesh=MESH, cutoff=CUTOFF)
    volume = static.unit_cell.volume()
    # a uniform strain e scales a by (1 + e): dF/de = 3 V sigma_xx for cubic
    numeric = (
        free_energy(calc, a * (1 + h), T) - free_energy(calc, a * (1 - h), T)
    ) / (2 * h)
    from chmpy.calc import System
    from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

    static_part = (
        3
        * volume
        * calc(System.from_crystal(static), ("energy", "stress")).stress[0, 0]
    )
    expected = (numeric - static_part) / (3 * volume) * EV_PER_ANGSTROM3_TO_GPA
    assert sigma.stress_gpa[0, 0] == pytest.approx(expected, rel=2e-2)
    np.testing.assert_allclose(np.diag(sigma.stress_gpa), sigma.stress_gpa[0, 0])


def test_qha_lattice_matches_a_direct_free_energy_scan(calc, static):
    T = 40.0
    result = qha_structure(static, calc, T, mesh=MESH, cutoff=CUTOFF)
    assert result.converged

    grid = lattice(static) * np.linspace(1.0, 1.04, 9)
    values = [free_energy(calc, a, T) for a in grid]
    coefficients = np.polyfit(grid, values, 4)
    fine = np.linspace(grid[0], grid[-1], 4001)
    scanned = fine[np.argmin(np.polyval(coefficients, fine))]

    assert lattice(result.structure) == pytest.approx(scanned, abs=2e-3)
    assert lattice(result.structure) > lattice(static)


def test_zero_temperature_leaves_only_zero_point_expansion(calc, static):
    cold = qha_structure(static, calc, 0.0, mesh=MESH, cutoff=CUTOFF)
    warm = qha_structure(static, calc, 40.0, mesh=MESH, cutoff=CUTOFF)
    assert lattice(static) < lattice(cold.structure) < lattice(warm.structure)

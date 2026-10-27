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


def hcp_argon():
    cell = UnitCell.from_lengths_and_angles(
        (3.75, 3.75, 6.12), np.radians((90, 90, 120))
    )
    return Crystal(
        cell,
        SpaceGroup(194),
        AsymmetricUnit([Element[18]], np.array([[1 / 3, 2 / 3, 0.25]])),
    )


def test_anisotropic_expansion_matches_a_lattice_parameter_grid(calc):
    """hcp has two free lattice parameters; c and a expand by different amounts."""
    from chmpy.calc import System

    T = 40.0
    static = relax(hcp_argon(), calc, fmax=1e-7, smax=1e-6, steps=800).structure
    a0, _, c0 = static.unit_cell.lengths
    result = qha_structure(static, calc, T, mesh=MESH, cutoff=CUTOFF, fmax=1e-6)
    assert result.converged
    a, _, c = result.structure.unit_cell.lengths

    strains = np.linspace(0.0, 0.04, 9)
    G = np.zeros((9, 9))
    for i, ea in enumerate(strains):
        for j, ec in enumerate(strains):
            crystal = Crystal(
                UnitCell.from_lengths_and_angles(
                    (a0 * (1 + ea), a0 * (1 + ea), c0 * (1 + ec)),
                    np.radians((90, 90, 120)),
                ),
                static.space_group,
                static.asymmetric_unit,
            )
            constants = force_constants(crystal, calc, cutoff=CUTOFF)
            G[i, j] = (
                calc.energy(System.from_crystal(crystal))
                + thermal_properties(constants, [T], mesh=MESH).free_energy[0]
            )
    x, y = (v.ravel() for v in np.meshgrid(strains, strains, indexing="ij"))

    def design(x, y):
        return np.stack(
            [
                np.ones_like(x),
                x,
                y,
                x * x,
                x * y,
                y * y,
                x**3,
                x * x * y,
                x * y * y,
                y**3,
            ],
            axis=1,
        )

    coefficients = np.linalg.lstsq(design(x, y), G.ravel(), rcond=None)[0]
    fx, fy = (v.ravel() for v in np.meshgrid(*[np.linspace(0, 0.04, 801)] * 2))
    best = np.argmin(design(fx, fy) @ coefficients)

    assert a / a0 - 1 == pytest.approx(fx[best], abs=2e-4)
    assert c / c0 - 1 == pytest.approx(fy[best], abs=2e-4)
    assert c / a > c0 / a0  # c expands more than a


def test_bfgs_converges_faster_than_the_static_hessian(calc):
    static = relax(hcp_argon(), calc, fmax=1e-7, smax=1e-6, steps=800).structure
    result = qha_structure(static, calc, 40.0, mesh=MESH, cutoff=CUTOFF, fmax=1e-6)
    # the static Hessian alone took 7 iterations to 1e-3 GPa
    assert result.converged and len(result.history) <= 5


def test_imaginary_modes_stop_the_calculation():
    """A saddle point held only by symmetry has no harmonic free energy."""
    from chmpy.tests.opt.test_elastic import TwoSpecies, _crystal
    from chmpy.vib.qha import ImaginaryModes

    potential = TwoSpecies()
    start = _crystal(
        (4.0, 4.0, 8.0),
        (90, 90, 90),
        123,
        [18, 36, 18],
        [[0, 0, 0.27], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
    )
    static = relax(start, potential, fmax=1e-7, smax=1e-6, steps=800).structure
    with pytest.raises(ImaginaryModes, match="imaginary modes"):
        qha_structure(static, potential, 30.0, mesh=(3, 3, 3), cutoff=9.0)


def test_an_unstable_trial_cell_is_a_rejected_step(calc, monkeypatch, caplog):
    """A step that lands where the crystal is dynamically unstable is shortened,
    not fatal; only an unstable current cell stops the calculation."""
    from chmpy.vib import qha

    static = relax(hcp_argon(), calc, fmax=1e-7, smax=1e-6, steps=800).structure
    original = qha._FreeEnergySurface.vibrational
    limit = 0.012

    def unstable_beyond_limit(self, q):
        if np.abs(q).max() > limit + self.strain:
            raise qha.ImaginaryModes("unstable")
        return original(self, q)

    monkeypatch.setattr(qha._FreeEnergySurface, "vibrational", unstable_beyond_limit)
    with caplog.at_level("WARNING"):
        result = qha_structure(static, calc, 40.0, mesh=MESH, cutoff=CUTOFF, fmax=1e-6)
    assert not result.converged
    assert np.abs(result.strain).max() <= limit + 0.005 + 1e-9
    assert result.history[-1]["gradient_gpa"] < result.history[0]["gradient_gpa"]


def test_the_volume_scan_matches_the_strained_cell_for_a_cubic_crystal(calc, static):
    """For a cubic crystal isotropic scaling loses nothing, so both routes must
    find the same cell and free energy."""
    from chmpy.vib.qha import qha_volume

    scan = qha_volume(
        static,
        calc,
        [20.0, 40.0],
        scales=np.linspace(0.98, 1.12, 9),
        mesh=MESH,
        cutoff=CUTOFF,
        fmax=1e-6,
    )
    assert scan.in_range.all()
    for k, temperature in enumerate(scan.temperatures):
        direct = qha_structure(
            static, calc, temperature, mesh=MESH, cutoff=CUTOFF, fmax=1e-6
        )
        assert scan.volume_change[k] == pytest.approx(direct.volume_change, abs=2e-4)
        assert scan.free_energy[k] == pytest.approx(direct.free_energy, abs=5e-5)
    assert np.all(scan.thermal_expansion > 0)

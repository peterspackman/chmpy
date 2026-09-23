"""Elastic constants: symmetry, ionic relaxation, and an independent check."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones, PairPotential, System
from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.ext.elastic_tensor import ElasticTensor
from chmpy.opt import elastic_tensor, relax
from chmpy.opt.elastic import voigt_strain
from chmpy.opt.strain import cartesian_rotations


class TwoSpecies(PairPotential):
    """Different radii per element, so internal coordinates respond to strain."""

    cutoff = 9.0
    radii = {18: 3.4, 36: 3.9}

    def pair(self, r, zi, zj):
        sigma = 0.5 * (
            np.vectorize(self.radii.get)(zi) + np.vectorize(self.radii.get)(zj)
        )
        x = (sigma / r) ** 6
        return 0.01 * 4 * (x * x - x), 0.01 * 4 * (-12 * x * x + 6 * x) / r


def fcc_argon(a=5.3):
    cell = UnitCell.from_lengths_and_angles((a, a, a), np.radians((90, 90, 90)))
    return Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )


@pytest.fixture(scope="module")
def relaxed_argon():
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=10.0)
    result = relax(fcc_argon(), calc, fmax=1e-6, smax=1e-5, steps=200)
    assert result.converged
    return result.structure, calc


@pytest.fixture(scope="module")
def binary():
    a = 6.0
    fractional = np.array([[0, 0, 0], [0.5, 0.52, 0.48], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cell = np.diag([a, a * 1.05, a * 0.95])
    system = System([18, 36, 18, 36], fractional @ cell, cell, True)
    calc = TwoSpecies()
    result = relax(system, calc, fmax=1e-6, smax=1e-5, steps=400)
    assert result.converged
    return result.structure, calc


def test_voigt_strain_carries_the_engineering_factor():
    np.testing.assert_allclose(np.diag(voigt_strain(0, 0.01)), [0.01, 0, 0])
    shear = voigt_strain(5, 0.01)  # xy
    assert shear[0, 1] == shear[1, 0] == pytest.approx(0.005)
    assert np.trace(shear) == 0.0


def test_a_cubic_crystal_gives_a_cubic_tensor(relaxed_argon):
    crystal, calc = relaxed_argon
    result = elastic_tensor(crystal, calc, strain=0.002)
    c = result.c_voigt

    assert result.n_independent == 3
    assert c[0, 0] == pytest.approx(c[1, 1]) == pytest.approx(c[2, 2])
    assert c[3, 3] == pytest.approx(c[4, 4]) == pytest.approx(c[5, 5])
    assert c[0, 1] == pytest.approx(c[0, 2]) == pytest.approx(c[1, 2])
    np.testing.assert_allclose(c[0:3, 3:6], 0.0, atol=1e-12)
    # the raw finite differences were already cubic, so the projection is a
    # formality rather than a repair
    assert result.symmetry_residual < 1e-8


def test_the_bulk_modulus_matches_an_energy_volume_fit(relaxed_argon):
    """An independent route to the same number, through no shared code."""
    crystal, calc = relaxed_argon
    c = elastic_tensor(crystal, calc, strain=0.002).c_voigt
    from_elastic = (c[0, 0] + 2 * c[0, 1]) / 3

    system = System.from_crystal(crystal)
    volume0 = system.volume
    energies, volumes = [], []
    for scale in np.linspace(0.985, 1.015, 13):
        trial = system.copy()
        trial.set_cell(np.asarray(system.cell) * scale ** (1 / 3), scale_atoms=True)
        energies.append(calc.energy(trial))
        volumes.append(trial.volume)
    curvature = np.polyval(np.polyder(np.polyfit(volumes, energies, 3), 2), volume0)
    from_energy = volume0 * curvature * EV_PER_ANGSTROM3_TO_GPA

    assert from_elastic == pytest.approx(from_energy, rel=5e-3)


def test_a_central_pair_potential_satisfies_the_cauchy_relation(relaxed_argon):
    """C12 = C44 for a cubic crystal held together by central pair forces.

    Exact for an untruncated central potential at zero pressure. Here it holds
    to 3e-5, limited by the O(strain^2) error of the finite difference and by
    the force discontinuity at the cutoff -- both of which are the reasons to
    check a relation like this rather than a tabulated number.
    """
    crystal, calc = relaxed_argon
    c = elastic_tensor(crystal, calc, strain=0.002).c_voigt
    assert c[0, 1] == pytest.approx(c[3, 3], rel=1e-4)


def test_relaxing_the_ions_can_only_soften(binary):
    """C_clamped - C_relaxed is positive semidefinite. It is a theorem."""
    structure, calc = binary
    relaxed = elastic_tensor(structure, calc, strain=0.002, relax_ions=True, fmax=1e-6)
    clamped = elastic_tensor(structure, calc, strain=0.002, relax_ions=False)

    difference = clamped.c_voigt - relaxed.c_voigt
    assert np.linalg.eigvalsh(difference).min() > -1e-6
    # and it matters: this structure's shear constants halve
    assert relaxed.c_voigt[5, 5] < 0.5 * clamped.c_voigt[5, 5]


def test_clamped_ions_are_much_cheaper(binary):
    structure, calc = binary
    relaxed = elastic_tensor(structure, calc, strain=0.002, relax_ions=True, fmax=1e-6)
    clamped = elastic_tensor(structure, calc, strain=0.002, relax_ions=False)
    assert clamped.evaluations == 13  # one reference plus twelve strains
    assert relaxed.evaluations > clamped.evaluations


def test_the_tensor_is_stable_for_a_relaxed_structure(relaxed_argon):
    crystal, calc = relaxed_argon
    result = elastic_tensor(crystal, calc, strain=0.002)
    assert result.tensor.is_stable()
    assert result.tensor.eigenvalues.min() > 0


def test_symmetrising_a_tensor_by_hand_matches_the_calculation(relaxed_argon):
    crystal, calc = relaxed_argon
    raw = elastic_tensor(crystal, calc, strain=0.002, symmetry=False)
    projected = raw.tensor.symmetrised(cartesian_rotations(crystal))
    done_inline = elastic_tensor(crystal, calc, strain=0.002, symmetry=True)
    np.testing.assert_allclose(projected.c_voigt, done_inline.c_voigt, atol=1e-10)


def test_symmetrising_removes_what_the_group_forbids():
    """A deliberately wrong tensor, projected onto cubic symmetry."""
    broken = np.diag([10.0, 11.0, 12.0, 5.0, 5.5, 6.0])
    broken[0, 1] = broken[1, 0] = 4.0
    crystal = fcc_argon()
    projected = ElasticTensor(broken).symmetrised(cartesian_rotations(crystal)).c_voigt

    assert projected[0, 0] == pytest.approx(11.0)  # the average of 10, 11, 12
    assert projected[3, 3] == pytest.approx(5.5)
    assert projected[0, 1] == pytest.approx(4.0 / 3)  # only one of three was set
    np.testing.assert_allclose(projected[0:3, 3:6], 0.0, atol=1e-12)


def test_a_non_periodic_structure_is_refused():
    system = System([18, 18], [[0, 0, 0], [3.8, 0, 0]])
    with pytest.raises(ValueError, match="need a periodic structure"):
        elastic_tensor(system, LennardJones())


def test_an_unrelaxed_reference_is_reported_and_warned_about(caplog):
    crystal = fcc_argon(4.6)  # heavily compressed
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=10.0)
    with caplog.at_level("WARNING"):
        result = elastic_tensor(crystal, calc, strain=0.002)
    assert result.residual_stress > 0.5
    assert "relax it first" in caplog.text


def test_a_singular_tensor_says_so_rather_than_raising_attributeerror():
    with pytest.raises(ValueError, match="singular"):
        ElasticTensor(np.zeros((6, 6)))


def test_a_noisy_tensor_is_flagged_rather_than_returned_quietly(caplog):
    """The symmetry residual is the only free error estimate, so it has to shout.

    Measured on benzene with PET-MAD: at a strain of 0.004 and an ionic
    tolerance sitting on the model's force-noise floor, the tensor came back
    with a *negative* bulk modulus and a fifth of it outside the
    symmetry-allowed subspace. Nothing else in the result said so.
    """

    class Noisy(TwoSpecies):
        """Adds a reproducible error to the stress, as an unconverged run would."""

        def compute(self, system, want):
            result = super().compute(system, want)
            if result.stress is None:
                return result
            rng = np.random.default_rng(
                abs(hash(np.asarray(system.positions).tobytes())) % 2**32
            )
            noise = rng.normal(scale=1e-3, size=(3, 3))
            return type(result)(
                energy=result.energy,
                forces=result.forces,
                stress=result.stress + 0.5 * (noise + noise.T),
                energies=result.energies,
                volume=result.volume,
            )

    cell = UnitCell.from_lengths_and_angles((6.0, 6.0, 6.0), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    with caplog.at_level("WARNING"):
        result = elastic_tensor(crystal, Noisy(), strain=0.002, relax_ions=False)

    assert result.noise_fraction > 0.15
    assert "of its largest constant" in caplog.text


def test_noise_fraction_is_small_for_a_converged_tensor(relaxed_argon):
    crystal, calc = relaxed_argon
    assert elastic_tensor(crystal, calc, strain=0.002).noise_fraction < 1e-6


def test_model_inputs_reach_the_calculator(relaxed_argon):
    """A polarisable model needs a charge and a spin at every strained geometry."""
    crystal, calc = relaxed_argon
    seen = []

    class NeedsInfo(LennardJones):
        def compute(self, system, want):
            seen.append(dict(system.info))
            return super().compute(system, want)

    elastic_tensor(crystal, NeedsInfo(), strain=0.002, info={"charge": 0, "spin": 1})
    assert seen and all(entry == {"charge": 0, "spin": 1} for entry in seen)


def test_asymmetry_catches_noise_that_symmetry_cannot():
    """A triclinic crystal forbids nothing, so it needs the other estimate.

    `symmetry_residual` is identically zero for P1 and P-1 -- all 21 constants
    are allowed -- which is exactly where a silently wrong tensor is least
    likely to be noticed. The asymmetry of the raw tensor works everywhere.
    """
    from chmpy.opt.strain import cartesian_rotations, invariant_elastic_basis

    cell = UnitCell.from_lengths_and_angles((5.0, 5.4, 6.1), np.radians((85, 95, 100)))
    crystal = Crystal(
        cell,
        SpaceGroup(2),
        AsymmetricUnit(
            [Element[18], Element[36]], np.array([[0.1, 0.15, 0.2], [0.6, 0.4, 0.7]])
        ),
    )
    assert len(invariant_elastic_basis(cartesian_rotations(crystal))) == 21

    class Noisy(TwoSpecies):
        def compute(self, system, want):
            result = super().compute(system, want)
            if result.stress is None:
                return result
            rng = np.random.default_rng(
                abs(hash(np.asarray(system.positions).tobytes())) % 2**32
            )
            return type(result)(
                energy=result.energy,
                forces=result.forces,
                stress=result.stress + rng.normal(scale=3e-4, size=(3, 3)),
                energies=result.energies,
                volume=result.volume,
            )

    result = elastic_tensor(crystal, Noisy(), strain=0.002, relax_ions=False)
    # nothing is forbidden, so the projection is the identity up to round-off
    assert result.symmetry_residual < 1e-9
    assert result.asymmetry > 0.0
    assert result.noise == result.asymmetry
    assert result.noise_fraction > 0.01


def test_auto_strain_escalates_until_the_noise_is_acceptable(caplog):
    """A strain too small for the calculator's noise should be grown, not kept."""

    class NoisyBelowAStrain(TwoSpecies):
        """Stress noise fixed in size, so a bigger strain gives a better ratio."""

        def compute(self, system, want):
            result = super().compute(system, want)
            if result.stress is None:
                return result
            rng = np.random.default_rng(
                abs(hash(np.asarray(system.cell).tobytes())) % 2**32
            )
            noise = rng.normal(scale=1e-5, size=(3, 3))
            return type(result)(
                energy=result.energy,
                forces=result.forces,
                stress=result.stress + 0.5 * (noise + noise.T),
                energies=result.energies,
                volume=result.volume,
            )

    cell = UnitCell.from_lengths_and_angles((6.0, 6.0, 6.0), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    calc = NoisyBelowAStrain()
    tight = elastic_tensor(crystal, calc, strain=0.001, relax_ions=False)
    auto = elastic_tensor(crystal, calc, strain="auto", relax_ions=False)

    assert auto.strain > 0.001
    assert auto.noise_fraction < tight.noise_fraction


def test_an_explicit_strain_is_not_second_guessed(relaxed_argon):
    crystal, calc = relaxed_argon
    assert elastic_tensor(crystal, calc, strain=0.002).strain == 0.002

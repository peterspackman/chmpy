"""Force constants, and the checks that say the bookkeeping is right."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones, System
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt import elastic_tensor, relax
from chmpy.vib import (
    ForceConstants,
    christoffel_velocities,
    compare_with_elastic,
    density,
    force_constants,
)
from chmpy.vib.christoffel import full_tensor


@pytest.fixture(scope="module")
def argon():
    """Relaxed fcc argon, its force constants, and its elastic tensor."""
    cell = UnitCell.from_lengths_and_angles((5.3, 5.3, 5.3), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    relaxed = relax(crystal, calc, fmax=1e-6, smax=1e-5, steps=200).structure
    constants = force_constants(relaxed, calc, cutoff=8.0)
    tensor = elastic_tensor(relaxed, calc, strain=0.002)
    return relaxed, calc, constants, tensor


def test_the_acoustic_modes_vanish_at_gamma(argon):
    """Translating the crystal costs nothing; the sum rule says so exactly."""
    _, _, constants, _ = argon
    assert constants.acoustic_error("cm-1") < 1e-3


def test_optical_frequencies_are_real_and_ordered(argon):
    _, _, constants, _ = argon
    frequencies = constants.frequencies([0, 0, 0], units="cm-1")
    assert len(frequencies) == 3 * constants.n_atoms
    assert np.all(np.diff(frequencies) >= -1e-9)
    assert np.all(frequencies[3:] > 1.0)  # nothing imaginary at a real minimum


def test_sound_velocities_match_the_elastic_tensor(argon):
    """The sharp one.

    Force constants and elastic constants come from completely different
    calculations -- forces under displacement in a supercell, stresses under
    strain -- and have to agree on the slope of the acoustic branches. An error
    in the supercell bookkeeping or in the multiplicity weights of the phase
    sum shows up here and essentially nowhere else.
    """
    _, _, constants, tensor = argon
    comparison = compare_with_elastic(constants, tensor.c_voigt)
    assert comparison["max_relative_error"] < 1e-3


def test_the_comparison_sees_the_anisotropy(argon):
    """Not just the magnitudes: the right three velocities in each direction."""
    _, _, constants, tensor = argon
    comparison = compare_with_elastic(constants, tensor.c_voigt, directions=[[1, 1, 0]])
    velocities = comparison["phonon"][0]
    # a cubic crystal along [110] has three distinct acoustic branches
    assert velocities[0] < velocities[1] < velocities[2]
    np.testing.assert_allclose(velocities, comparison["elastic"][0], rtol=1e-3)


def test_the_supercell_size_does_not_change_the_velocities():
    """The multiplicity weights are what make this true.

    A small supercell reaches a pair through several equally short lattice
    translations and measures their sum once; the weights share it back out.
    Were they wrong, the answer would depend on the supercell -- so this is the
    test that pins them.

    The cutoff is short enough that 2x2x2 already contains the interaction; a
    supercell narrower than twice the cutoff would be wrong for a reason that
    has nothing to do with the weights.
    """
    cell = UnitCell.from_lengths_and_angles((5.3, 5.3, 5.3), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=4.5)
    relaxed = relax(crystal, calc, fmax=1e-6, smax=1e-5, steps=200).structure
    tensor = elastic_tensor(relaxed, calc, strain=0.002)

    errors = []
    for repeats in ((2, 2, 2), (3, 3, 3)):
        constants = force_constants(relaxed, calc, supercell=repeats)
        assert constants.acoustic_error("cm-1") < 1e-3
        errors.append(
            compare_with_elastic(constants, tensor.c_voigt)["max_relative_error"]
        )
    assert max(errors) < 5e-3, errors


def test_force_constants_are_symmetric_under_exchange(argon):
    """Phi_ab(i,j,S) = Phi_ba(j,i,-S): the same derivative from either end."""
    _, _, constants, _ = argon
    index = {
        (int(i), int(j), *np.rint(shift).astype(int)): position
        for position, ((i, j), shift) in enumerate(
            zip(constants.pairs, constants.shifts, strict=True)
        )
    }
    checked = 0
    for position, ((i, j), shift) in enumerate(
        zip(constants.pairs, constants.shifts, strict=True)
    ):
        partner = index.get((int(j), int(i), *(-np.rint(shift).astype(int))))
        if partner is None:
            continue
        np.testing.assert_allclose(
            constants.blocks[position], constants.blocks[partner].T, atol=1e-10
        )
        checked += 1
    assert checked > 10


def test_the_dynamical_matrix_is_hermitian(argon):
    _, _, constants, _ = argon
    matrix = constants.dynamical_matrix([0.13, -0.27, 0.41])
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


def test_frequencies_are_the_same_at_q_and_minus_q(argon):
    """Time-reversal symmetry, and a check on the phase convention."""
    _, _, constants, _ = argon
    q = np.array([0.2, 0.1, -0.35])
    np.testing.assert_allclose(
        constants.frequencies(q, "cm-1"), constants.frequencies(-q, "cm-1"), atol=1e-8
    )


def test_the_unit_conversion_is_right():
    """One oscillator of known force constant and mass.

    A single conversion in one place is the whole defence against the unit
    confusion that lattice dynamics is famous for, so it gets a closed-form
    test: omega = sqrt(k / m).
    """
    k, mass = 4.0, 12.0  # eV/A^2, amu
    constants = ForceConstants(
        pairs=np.array([[0, 0]]),
        shifts=np.zeros((1, 3)),
        blocks=np.array([np.eye(3) * k]),
        weights=np.ones(1),
        masses=np.array([mass]),
        cell=np.eye(3) * 10.0,
        positions=np.zeros((1, 3)),
    )
    from chmpy.vib.force_constants import (
        EV_PER_ANGSTROM2_AMU_TO_RAD2_S2,
        RAD_S_TO_PER_CM,
    )

    expected = np.sqrt(k / mass * EV_PER_ANGSTROM2_AMU_TO_RAD2_S2)
    np.testing.assert_allclose(
        constants.frequencies([0, 0, 0], "rad/s"), expected, rtol=1e-12
    )

    # and independently, straight from SI constants rather than through the
    # module's own factor, so the two cannot be wrong together
    electron_volt, angstrom, amu = 1.602176634e-19, 1e-10, 1.66053906660e-27
    speed_of_light = 2.99792458e10  # cm/s
    omega = np.sqrt(k * electron_volt / angstrom**2 / (mass * amu))
    np.testing.assert_allclose(expected, omega, rtol=1e-12)
    assert expected * RAD_S_TO_PER_CM == pytest.approx(
        omega / (2 * np.pi * speed_of_light), rel=1e-12
    )
    # ~301 cm^-1 for a 4 eV/A^2 spring on carbon
    assert 300 < expected * RAD_S_TO_PER_CM < 302


def test_christoffel_reproduces_isotropic_velocities():
    """An isotropic solid has one longitudinal and two degenerate transverse."""
    bulk, shear = 100.0, 40.0  # GPa
    lam = bulk - 2 * shear / 3
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[0, 0] = c[1, 1] = c[2, 2] = lam + 2 * shear
    c[3, 3] = c[4, 4] = c[5, 5] = shear
    rho = 3000.0

    for direction in ([1, 0, 0], [1, 1, 0], [1, 1, 1], [0.3, -0.7, 0.2]):
        velocities = christoffel_velocities(c, rho, direction)
        np.testing.assert_allclose(
            velocities[:2], np.sqrt(shear * 1e9 / rho), rtol=1e-10
        )
        np.testing.assert_allclose(
            velocities[2], np.sqrt((bulk + 4 * shear / 3) * 1e9 / rho), rtol=1e-10
        )


def test_full_tensor_round_trips_the_voigt_indices():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(6, 6))
    c = c + c.T
    tensor = full_tensor(c)
    assert tensor[0, 0, 0, 0] == c[0, 0]
    assert tensor[1, 2, 0, 1] == c[3, 5]
    assert tensor[2, 1, 1, 0] == c[3, 5]  # minor symmetries
    np.testing.assert_allclose(tensor, tensor.transpose(1, 0, 2, 3))
    np.testing.assert_allclose(tensor, tensor.transpose(2, 3, 0, 1))


def test_density_of_a_known_cell():
    # one carbon-12 in a 10 A cube
    assert density(np.array([12.0]), np.eye(3) * 10.0) == pytest.approx(
        12.0 * 1.66053906660e-27 / 1e-27, rel=1e-9
    )


def test_a_non_periodic_structure_is_refused():
    system = System([18, 18], [[0, 0, 0], [3.8, 0, 0]])
    with pytest.raises(ValueError, match="need a periodic structure"):
        force_constants(system, LennardJones())

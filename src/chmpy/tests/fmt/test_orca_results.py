import numpy as np
import pytest

from chmpy.fmt.orca_results import (
    absorption_spectrum,
    broaden,
    ecd_spectrum,
    parse_hess_file,
    parse_property_file,
    vibrational_spectrum,
)

from .. import TEST_FILES

TDDFT = TEST_FILES["h2co_tddft.orca.property.txt"]
HESS = TEST_FILES["h2co_freq.orca.hess"]


@pytest.fixture(scope="module")
def blocks():
    return parse_property_file(TDDFT)


def test_every_block_is_read_in_order(blocks):
    names = [b.name for b in blocks]
    assert names[:2] == ["Calculation_Status", "Geometry"]
    assert names.count("CIS_Absorption_Spectrum") == 2
    assert names[-1] == "Calculation_Timings"


def test_scalar_types(blocks):
    status = blocks[0]
    assert status["Status"] == "NORMAL TERMINATION"
    assert status["version"] == "6.1.1"
    assert status.geometry_index == 1
    cis = next(b for b in blocks if b.name == "CIS_Energies")
    assert cis["mode"] == "TDA"
    assert cis["multP1"] is False
    assert cis["e0"] == pytest.approx(-114.35604861026040)
    assert cis.comments["e0"] == "Ground state energy"


def test_coordinates_keep_their_units(blocks):
    geometry = next(b for b in blocks if b.name == "Geometry")
    elements, positions = geometry["CartesianCoordinates"]
    assert elements == ["C", "O", "H", "H"]
    assert positions.shape == (4, 3)
    assert positions[1, 2] == pytest.approx(1.285013771066)
    assert geometry.units["CartesianCoordinates"] == "Bohr"


def test_arrays_wider_than_one_chunk(blocks):
    table = next(b for b in blocks if b.name == "CIS_Absorption_Spectrum")[
        "ExcitationEnergies"
    ]
    assert table.shape == (6, 11)
    # the last chunk holds columns 8-10
    assert table[3, 9] == pytest.approx(1.2044583240852615e-01)


def test_integer_arrays(blocks):
    states = next(b for b in blocks if b.name == "CIS_Absorption_Spectrum")["States"]
    assert states.dtype.kind == "i"
    np.testing.assert_array_equal(states[:, 2], [1, 2, 3, 4, 5, 6])


def test_absorption_matches_the_printed_table(blocks):
    """Values from the ABSORPTION SPECTRUM table in the same run's output."""
    spectrum = absorption_spectrum(blocks)
    assert spectrum.method == "CIS"
    assert spectrum.representation == "Length"
    np.testing.assert_allclose(
        spectrum.energies,
        [3.998049, 8.228510, 9.112358, 9.657945, 10.189616, 11.075669],
        atol=1e-6,
    )
    np.testing.assert_allclose(spectrum.wavelengths[:2], [310.1, 150.7], atol=0.05)
    assert spectrum.intensities[1] == pytest.approx(0.154965008, abs=1e-9)
    np.testing.assert_allclose(spectrum.moments[1], [0.0, -0.87675, 0.0], atol=1e-5)
    np.testing.assert_allclose(spectrum.moments[3], [0.0, 0.0, 0.12045], atol=1e-5)


def test_the_velocity_representation(blocks):
    spectrum = absorption_spectrum(blocks, representation="Velocity")
    assert spectrum.intensities[1] == pytest.approx(0.082208437, abs=1e-9)
    np.testing.assert_allclose(spectrum.moments[1], [0.0, 0.19310, 0.0], atol=1e-5)


def test_ecd_matches_the_printed_table(blocks):
    spectrum = ecd_spectrum(TDDFT)
    assert spectrum.kind == "ecd"
    np.testing.assert_allclose(spectrum.moments[0], [0.0, 0.0, 0.58277], atol=1e-5)
    np.testing.assert_allclose(spectrum.moments[2], [0.0, -0.72675, 0.0], atol=1e-5)


def test_a_missing_spectrum_names_what_there_is(blocks):
    with pytest.raises(ValueError, match="found \\['CIS_Absorption_Spectrum'\\]"):
        absorption_spectrum(blocks, method="STEOM")


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_broadening_keeps_the_area(shape):
    grid = np.linspace(-200, 200, 400001)
    profile = broaden([-1.0, 2.0], [0.3, 0.7], grid, fwhm=0.5, shape=shape)
    assert np.trapezoid(profile, grid) == pytest.approx(1.0, rel=2e-3)


def test_broadening_has_the_requested_width():
    grid = np.linspace(-3, 3, 60001)
    for shape in ("gaussian", "lorentzian"):
        profile = broaden([0.0], [1.0], grid, fwhm=0.5, shape=shape)
        above = grid[profile >= 0.5 * profile.max()]
        assert above[-1] - above[0] == pytest.approx(0.5, abs=1e-3)


def test_an_unknown_line_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        broaden([0.0], [1.0], [0.0], 0.1, shape="voigt")


def test_the_absorption_profile_integrates_to_the_oscillator_strengths(blocks):
    spectrum = absorption_spectrum(blocks)
    grid = np.linspace(0, 16, 4001)
    area = np.trapezoid(spectrum.profile(grid, fwhm=0.3), grid)
    assert area == pytest.approx(spectrum.intensities.sum(), rel=1e-6)


# -- the Hessian file -------------------------------------------------------------


@pytest.fixture(scope="module")
def hess():
    return parse_hess_file(HESS)


def test_hess_blocks(hess):
    assert hess["multiplicity"] == 1
    assert hess["frequency_scale_factor"] == 1.0
    assert hess["hessian"].shape == (12, 12)
    np.testing.assert_allclose(hess["hessian"], hess["hessian"].T, atol=1e-6)
    # the last chunk of the Hessian holds columns 10 and 11
    assert hess["hessian"][1, 10] == pytest.approx(-2.3832578906e-01)
    elements, masses, positions = hess["atoms"]
    assert elements == ["C", "O", "H", "H"]
    np.testing.assert_allclose(masses, [12.011, 15.999, 1.008, 1.008])
    assert positions[1, 2] == pytest.approx(1.132260249936)
    assert hess["dipole_derivatives"].shape == (12, 3)
    assert hess["polarizability_derivatives"].shape == (12, 6)


def test_vibrations_match_the_printed_tables(hess):
    """Values from the IR and RAMAN SPECTRUM tables of the same run."""
    spectrum = vibrational_spectrum(hess)
    assert len(spectrum) == 6
    np.testing.assert_array_equal(spectrum.mode_indices, [6, 7, 8, 9, 10, 11])
    np.testing.assert_allclose(
        spectrum.frequencies,
        [1199.85, 1269.26, 1533.00, 1896.71, 2901.77, 2967.32],
        atol=0.005,
    )
    np.testing.assert_allclose(
        spectrum.ir_intensities, [1.67, 13.27, 6.09, 114.78, 57.97, 148.80], atol=0.005
    )
    np.testing.assert_allclose(
        spectrum.transition_dipoles[3], [0.0, 0.0, 0.061131], atol=1e-6
    )
    np.testing.assert_allclose(
        spectrum.raman_activities,
        [1.349803, 7.134662, 14.255087, 4.701745, 188.160451, 94.364583],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        spectrum.depolarization,
        [0.75, 0.75, 0.674533, 0.525675, 0.165816, 0.75],
        atol=1e-4,
    )


def test_normal_modes_are_unit_columns(hess):
    spectrum = vibrational_spectrum(hess)
    assert spectrum.modes.shape == (12, 6)
    np.testing.assert_allclose(np.linalg.norm(spectrum.modes, axis=0), 1.0, atol=1e-6)


def test_imaginary_modes_are_kept(hess):
    frequencies = hess["vibrational_frequencies"].copy()
    frequencies[6] = -350.0
    spectrum = vibrational_spectrum({**hess, "vibrational_frequencies": frequencies})
    assert len(spectrum) == 6
    np.testing.assert_allclose(spectrum.imaginary, [-350.0])
    assert "1 imaginary" in repr(spectrum)


def test_without_raman_data(hess):
    trimmed = {k: v for k, v in hess.items() if k != "raman_spectrum"}
    spectrum = vibrational_spectrum(trimmed)
    assert spectrum.raman_activities is None
    with pytest.raises(ValueError, match="Polar"):
        spectrum.raman_profile(np.linspace(0, 4000, 10))


def test_the_ir_profile_integrates_to_the_intensities(hess):
    spectrum = vibrational_spectrum(hess)
    grid = np.linspace(0, 4000, 40001)
    area = np.trapezoid(spectrum.ir_profile(grid, fwhm=10.0, shape="gaussian"), grid)
    assert area == pytest.approx(spectrum.ir_intensities.sum(), rel=1e-6)


def test_a_hess_file_without_frequencies_is_refused():
    with pytest.raises(ValueError, match="no vibrational frequencies"):
        vibrational_spectrum("$orca_hessian_file\n\n$multiplicity\n  1\n\n$end\n")

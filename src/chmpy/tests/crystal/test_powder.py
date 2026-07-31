"""Tests for powder X-ray diffraction pattern simulation.

Self-contained tests cover the atomic form factors, reflection generation,
structure factors and the assembled pattern. A final test cross-validates the
relative intensities against PLATON when it is available on the PATH.
"""

import logging
import shutil
import subprocess
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np

from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell
from chmpy.crystal.powder import (
    generate_hkl,
    plot_powder_patterns,
    powder_pattern,
    structure_factors,
    structure_factors_fft,
)
from chmpy.crystal.scattering_factors import AVAILABLE, atomic_form_factor
from chmpy.util.optional import have

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class ScatteringFactorTestCase(unittest.TestCase):
    def test_forward_scattering_equals_electron_count(self):
        # f(0) should equal the atomic number for neutral atoms
        for z in (1, 6, 7, 8, 26, 92):
            f0 = atomic_form_factor([z], [0.0])[0, 0]
            self.assertAlmostEqual(f0, z, delta=0.05)

    def test_known_carbon_values(self):
        # close to International Tables values for neutral C; the Waasmaier-Kirfel
        # 5-Gaussian fit differs from the Cromer-Mann 4-Gaussian by ~1%
        f = atomic_form_factor([6], [0.0, 0.2, 0.5])[0]
        np.testing.assert_allclose(f, [6.0, 3.58, 1.685], atol=0.03)

    def test_decreasing_with_angle(self):
        f = atomic_form_factor([8], [0.0, 0.1, 0.2, 0.4, 0.6])[0]
        self.assertTrue(np.all(np.diff(f) < 0))

    def test_missing_element_raises(self):
        missing = next(z for z in range(1, 99) if z not in AVAILABLE)
        with self.assertRaises(ValueError):
            atomic_form_factor([missing], [0.0])


class GenerateHklTestCase(unittest.TestCase):
    def setUp(self):
        self.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_resolution_limit_respected(self):
        d_min = 1.0
        hkl, d = generate_hkl(self.crystal.unit_cell, d_min)
        self.assertTrue(np.all(d >= d_min - 1e-9))
        self.assertFalse(np.any(np.all(hkl == 0, axis=1)))  # no (0,0,0)

    def test_d_spacing_matches_reciprocal_vector(self):
        hkl, d = generate_hkl(self.crystal.unit_cell, 1.5)
        q = hkl @ self.crystal.unit_cell.reciprocal_lattice
        np.testing.assert_allclose(d, 1.0 / np.linalg.norm(q, axis=1))

    def test_index_bound_is_complete(self):
        # brute-force enumerate a wide box; generate_hkl must find all with d>=d_min
        d_min = 1.5
        uc = self.crystal.unit_cell
        hkl, _ = generate_hkl(uc, d_min)
        found = {tuple(h) for h in hkl}
        brute = []
        for h in product(range(-15, 16), repeat=3):
            if any(h):
                q = np.array(h) @ uc.reciprocal_lattice
                if 1.0 / np.linalg.norm(q) >= d_min:
                    brute.append(h)
        self.assertEqual(found, set(brute))


class StructureFactorTestCase(unittest.TestCase):
    def setUp(self):
        self.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_friedel_symmetry(self):
        # |F(h)| == |F(-h)| in the absence of anomalous dispersion
        hkl = np.array([[1, 2, 3], [2, 0, 1], [0, 1, 4], [3, 1, 2]])
        d = 1.0 / np.linalg.norm(
            hkl @ self.crystal.unit_cell.reciprocal_lattice, axis=1
        )
        f = np.abs(structure_factors(self.crystal, hkl, d))
        fbar = np.abs(structure_factors(self.crystal, -hkl, d))
        np.testing.assert_allclose(f, fbar, rtol=1e-10)

    def test_systematic_absences_have_zero_intensity(self):
        # cross-check against the reflection-condition engine: any reflection it
        # flags absent must have a vanishing structure factor
        sg = self.crystal.space_group
        hkl, d = generate_hkl(self.crystal.unit_cell, 1.0)
        f2 = np.abs(structure_factors(self.crystal, hkl, d)) ** 2
        absent = sg.is_systematically_absent(hkl)
        self.assertTrue(absent.any())
        self.assertLess(f2[absent].max(), 1e-6 * f2.max())


class PowderPatternTestCase(unittest.TestCase):
    def setUp(self):
        self.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_two_theta_follows_bragg(self):
        pattern = powder_pattern(self.crystal, wavelength=1.5406)
        expected = np.degrees(2 * np.arcsin(1.5406 / (2 * pattern.d_spacing)))
        np.testing.assert_allclose(pattern.two_theta, expected)

    def test_intensities_nonnegative_and_in_range(self):
        pattern = powder_pattern(
            self.crystal, wavelength=1.5406, two_theta_range=(5, 50)
        )
        self.assertTrue(np.all(pattern.intensity >= 0))
        self.assertTrue(np.all(pattern.two_theta >= 5))
        self.assertTrue(np.all(pattern.two_theta <= 50))

    def test_absent_reflections_excluded_from_pattern(self):
        pattern = powder_pattern(self.crystal, two_theta_range=(5, 60))
        absent = self.crystal.space_group.is_systematically_absent(pattern.hkl)
        # absent reflections may appear but must carry no intensity
        self.assertLess(
            pattern.intensity[absent].max(initial=0.0), 1e-6 * pattern.intensity.max()
        )

    def test_profile_binning(self):
        pattern = powder_pattern(self.crystal, two_theta_range=(5, 50))
        centres, intensity = pattern.profile(
            two_theta_range=(5, 50), num_bins=900, fwhm=0.1
        )
        self.assertEqual(len(centres), 900)
        self.assertEqual(len(intensity), 900)
        self.assertGreater(intensity.sum(), 0)

    def test_crystal_method(self):
        pattern = self.crystal.powder_pattern(two_theta_range=(5, 40))
        self.assertGreater(len(pattern.hkl), 0)

    def test_empty_range_returns_empty_pattern(self):
        # a 2-theta window with no reflections must not crash
        pattern = powder_pattern(self.crystal, two_theta_range=(1.0, 1.5))
        self.assertEqual(len(pattern), 0)
        centres, intensity = pattern.profile()
        self.assertEqual(intensity.sum(), 0.0)

    def test_peaks_sorted_and_normalized(self):
        pattern = powder_pattern(self.crystal, two_theta_range=(5, 50))
        peaks = pattern.peaks(n=5)
        self.assertEqual(len(peaks), 5)
        self.assertTrue(np.all(np.diff(peaks["intensity"]) <= 0))  # descending
        self.assertAlmostEqual(peaks["intensity"][0], 100.0)
        for field in ("h", "k", "l", "d", "two_theta", "multiplicity"):
            self.assertIn(field, peaks.dtype.names)

    def test_profile_defaults_to_pattern_range(self):
        # profile() with no range must cover the full pattern, not a fixed window
        pattern = powder_pattern(self.crystal, two_theta_range=(5, 80))
        centres, _ = pattern.profile()
        self.assertAlmostEqual(centres[-1], 80.0, delta=0.1)

    def test_repr_and_len(self):
        pattern = powder_pattern(self.crystal, two_theta_range=(5, 50))
        self.assertEqual(len(pattern), len(pattern.hkl))
        self.assertIn("PowderPattern", repr(pattern))
        self.assertNotIn("\n", repr(pattern))  # concise, no array dump

    @unittest.skipUnless(have("matplotlib"), "needs chmpy[plots]")
    def test_plot_returns_axes(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pattern = powder_pattern(self.crystal, two_theta_range=(5, 50))
        ax = pattern.plot()
        self.assertGreater(len(ax.get_lines()[0].get_xdata()), 0)
        self.assertIn("theta", ax.get_xlabel())
        # draws onto a supplied Axes too
        _, ax2 = plt.subplots()
        self.assertIs(pattern.plot(ax=ax2), ax2)
        plt.close("all")

    @unittest.skipUnless(have("matplotlib"), "needs chmpy[plots]")
    def test_plot_overlay(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        patterns = [
            powder_pattern(self.crystal, wavelength=wl, two_theta_range=(5, 50))
            for wl in (1.5406, 0.7107)
        ]
        ax = plot_powder_patterns(patterns, labels=["Cu", "Mo"], offset=10.0)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        # second pattern is offset above the first
        self.assertGreater(lines[1].get_ydata().max(), 100)
        # baseline pattern is drawn in front (higher zorder) of the stacked one
        self.assertGreater(lines[0].get_zorder(), lines[1].get_zorder())
        self.assertIsNotNone(ax.get_legend())
        plt.close("all")

        # optional horizontal shear offsets each pattern sideways too
        ax = plot_powder_patterns(patterns, offset=10.0, x_offset=2.0)
        lines = ax.get_lines()
        shift = lines[1].get_xdata()[0] - lines[0].get_xdata()[0]
        self.assertAlmostEqual(shift, 2.0, places=3)
        self.assertGreater(ax.get_xlim()[1], 50.0)  # widened to fit the shift
        plt.close("all")


class FftStructureFactorTestCase(unittest.TestCase):
    def setUp(self):
        self.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.hkl, self.d = generate_hkl(self.crystal.unit_cell, 1.2)

    def test_fft_matches_direct(self):
        direct = structure_factors(self.crystal, self.hkl, self.d)
        fft = structure_factors_fft(self.crystal, self.hkl, self.d)
        scale = (np.abs(direct) ** 2).max()
        rel = np.abs(np.abs(fft) ** 2 - np.abs(direct) ** 2) / scale
        self.assertLess(rel.max(), 1e-3)

    def test_methods_agree_in_powder_pattern(self):
        direct = powder_pattern(self.crystal, method="direct", dtype=np.float64)
        fft = powder_pattern(self.crystal, method="fft")
        # same reflections in the same order; relative intensities agree
        np.testing.assert_array_equal(direct.hkl, fft.hkl)
        scale = direct.f2.max()
        np.testing.assert_allclose(fft.f2, direct.f2, atol=1e-3 * scale)

    def test_auto_selects_direct_for_small_cell(self):
        # acetic acid is tiny -> auto should match the direct result exactly
        auto = powder_pattern(self.crystal, method="auto", dtype=np.float64)
        direct = powder_pattern(self.crystal, method="direct", dtype=np.float64)
        np.testing.assert_array_equal(auto.f2, direct.f2)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            powder_pattern(self.crystal, method="nonsense")


class CubicGeometryTestCase(unittest.TestCase):
    def test_simple_cubic_peak_positions(self):
        # primitive cubic, a=4: d(hkl) = a/sqrt(h^2+k^2+l^2)
        uc = UnitCell.cubic(4.0)
        sg = SpaceGroup(1)
        asym = AsymmetricUnit([Element["C"]], np.array([[0.0, 0.0, 0.0]]))
        crystal = Crystal(uc, sg, asym)
        pattern = powder_pattern(crystal, wavelength=1.5406, two_theta_range=(5, 90))
        # the largest d-spacing should be (100): d = 4.0
        self.assertAlmostEqual(pattern.d_spacing.max(), 4.0, places=6)


@unittest.skipUnless(shutil.which("platon"), "platon not available")
class PlatonValidationTestCase(unittest.TestCase):
    """Compare relative intensities against PLATON's generated structure factors."""

    def test_structure_factors_match_platon(self):
        cif = Path(TEST_FILES["acetic_acid.cif"])
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp, "acetic_acid.cif")
            local.write_text(cif.read_text())
            try:
                subprocess.run(
                    ["platon", "-o", local.name],
                    input="ASYM GENERATE\n",
                    text=True,
                    capture_output=True,
                    cwd=tmp,
                    timeout=60,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:  # pragma: no cover
                self.skipTest(f"platon run failed: {exc}")

            gener = Path(tmp, "acetic_acid_gener.hkl")
            if not gener.exists():  # pragma: no cover
                self.skipTest("platon produced no _gener.hkl")

            rows = []
            for line in gener.read_text().splitlines():
                if line.startswith("_"):
                    break
                tokens = line.split()
                if len(tokens) >= 4:
                    h, k, ll = int(tokens[0]), int(tokens[1]), int(tokens[2])
                    if (h, k, ll) != (0, 0, 0):
                        rows.append((h, k, ll, float(tokens[3])))

        ref = np.array(rows)
        hkl = ref[:, :3].astype(int)
        f2_platon = ref[:, 3]

        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        d = 1.0 / np.linalg.norm(hkl @ crystal.unit_cell.reciprocal_lattice, axis=1)
        f2_mine = np.abs(structure_factors(crystal, hkl, d)) ** 2

        r = np.corrcoef(f2_mine, f2_platon)[0, 1]
        self.assertGreater(r, 0.99)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

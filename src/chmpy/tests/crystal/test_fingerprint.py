import unittest

import numpy as np

from chmpy.crystal import Crystal
from chmpy.crystal.fingerprint import (
    filtered_histogram,
    filtered_histogram_by_elements,
    fingerprint_histogram,
)

from .. import TEST_FILES


class FingerprintTestCase(unittest.TestCase):
    def setUp(self):
        self.acetic_acid = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.surfaces = self.acetic_acid.hirshfeld_surfaces(
            separation=1.0, radius=3.8, kind="mol"
        )
        self.mesh = self.surfaces[0]

    def test_fingerprint_histogram(self):
        """Test basic fingerprint histogram generation."""
        hist, xedges, yedges = fingerprint_histogram(self.mesh, bins=50)

        self.assertEqual(hist.shape, (50, 50))
        self.assertEqual(len(xedges), 51)
        self.assertEqual(len(yedges), 51)
        self.assertGreater(np.sum(hist), 0)

    def test_filtered_histogram_by_atomic_numbers(self):
        """Test filtered histogram using atomic numbers."""
        # Filter for C...H contacts (C=6, H=1)
        hist, xedges, yedges = filtered_histogram(self.mesh, 6, 1, bins=50)

        self.assertEqual(hist.shape, (50, 50))
        self.assertGreaterEqual(np.sum(hist), 0)

    def test_filtered_histogram_by_elements(self):
        """Test filtered histogram using element symbols."""
        # Filter for C...H contacts using strings
        hist_str, xedges_str, yedges_str = filtered_histogram_by_elements(
            self.mesh, "C", "H", bins=50
        )

        # Filter for C...H contacts using atomic numbers
        hist_int, xedges_int, yedges_int = filtered_histogram_by_elements(
            self.mesh, 6, 1, bins=50
        )

        # Results should be identical
        self.assertEqual(hist_str.shape, hist_int.shape)
        np.testing.assert_array_equal(hist_str, hist_int)
        np.testing.assert_array_equal(xedges_str, xedges_int)
        np.testing.assert_array_equal(yedges_str, yedges_int)

    def test_filtered_histogram_different_element_pairs(self):
        """Test filtered histograms for different element pairs."""
        ch_hist, _, _ = filtered_histogram_by_elements(self.mesh, "C", "H", bins=50)
        oh_hist, _, _ = filtered_histogram_by_elements(self.mesh, "O", "H", bins=50)
        hh_hist, _, _ = filtered_histogram_by_elements(self.mesh, "H", "H", bins=50)

        # All histograms should have the same shape
        self.assertEqual(ch_hist.shape, oh_hist.shape)
        self.assertEqual(ch_hist.shape, hh_hist.shape)

        # Histograms should be different (not all zeros)
        self.assertGreater(np.sum(ch_hist), 0)
        self.assertGreater(np.sum(oh_hist), 0)
        self.assertGreater(np.sum(hh_hist), 0)

    def test_filtered_histogram_custom_range(self):
        """Test filtered histogram with custom range."""
        xrange = (0.5, 2.5)
        yrange = (0.5, 2.5)
        hist, xedges, yedges = filtered_histogram_by_elements(
            self.mesh, "C", "H", bins=50, xrange=xrange, yrange=yrange
        )

        self.assertAlmostEqual(xedges[0], xrange[0])
        self.assertAlmostEqual(xedges[-1], xrange[1])
        self.assertAlmostEqual(yedges[0], yrange[0])
        self.assertAlmostEqual(yedges[-1], yrange[1])

    def test_filtered_histogram_samples_per_edge(self):
        """Test filtered histogram with different samples_per_edge."""
        hist1, _, _ = filtered_histogram_by_elements(
            self.mesh, "C", "H", bins=50, samples_per_edge=2
        )
        hist2, _, _ = filtered_histogram_by_elements(
            self.mesh, "C", "H", bins=50, samples_per_edge=4
        )

        # Histograms should have same shape but potentially different values
        self.assertEqual(hist1.shape, hist2.shape)

    def test_filtered_histogram_nonexistent_pair(self):
        """Test filtered histogram for element pair that doesn't exist."""
        # Try filtering for a pair that likely doesn't exist (e.g., N...N in acetic acid)
        hist, _, _ = filtered_histogram_by_elements(self.mesh, "N", "N", bins=50)

        # Should return empty histogram (all zeros)
        self.assertEqual(np.sum(hist), 0)

    def test_element_string_case_insensitive(self):
        """Test that element strings are case insensitive."""
        hist1, _, _ = filtered_histogram_by_elements(self.mesh, "C", "H", bins=50)
        hist2, _, _ = filtered_histogram_by_elements(self.mesh, "c", "h", bins=50)

        np.testing.assert_array_equal(hist1, hist2)

    def test_include_inverse(self):
        """Test include_inverse parameter for symmetric contacts."""
        # Get histogram for C...H only
        ch_hist, _, _ = filtered_histogram_by_elements(
            self.mesh, "C", "H", bins=50, include_inverse=False
        )

        # Get histogram for H...C only
        hc_hist, _, _ = filtered_histogram_by_elements(
            self.mesh, "H", "C", bins=50, include_inverse=False
        )

        # Get histogram for both C...H and H...C
        combined_hist, _, _ = filtered_histogram_by_elements(
            self.mesh, "C", "H", bins=50, include_inverse=True
        )

        # Combined histogram should have at least as many counts as either individual one
        self.assertGreaterEqual(np.sum(combined_hist), np.sum(ch_hist))
        self.assertGreaterEqual(np.sum(combined_hist), np.sum(hc_hist))

    def test_include_inverse_symmetric(self):
        """Test that include_inverse is symmetric for same element pairs."""
        # For H...H contacts, including inverse should not change the result
        hh_hist, _, _ = filtered_histogram_by_elements(
            self.mesh, "H", "H", bins=50, include_inverse=False
        )
        hh_hist_inv, _, _ = filtered_histogram_by_elements(
            self.mesh, "H", "H", bins=50, include_inverse=True
        )

        # Should be the same for symmetric pairs
        np.testing.assert_array_equal(hh_hist, hh_hist_inv)


if __name__ == "__main__":
    unittest.main()

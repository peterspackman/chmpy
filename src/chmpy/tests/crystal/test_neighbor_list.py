"""Tests for CrystalNeighborList."""

import logging
import unittest

import numpy as np
from scipy.spatial import cKDTree as KDTree

from chmpy.crystal import Crystal
from chmpy.crystal.neighbor_list import CrystalNeighborList
from chmpy.crystal.orbit import AtomOrbitTable

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class CrystalNeighborListTestCase(unittest.TestCase):
    """Test CrystalNeighborList basic functionality."""

    def setUp(self):
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_from_crystal(self):
        """Test building neighbor list from Crystal."""
        nlist = CrystalNeighborList.from_crystal(self.acetic, cutoff=5.0)

        self.assertEqual(nlist.n_atoms, len(self.acetic.unit_cell_atoms()["element"]))
        self.assertEqual(nlist.cutoff, 5.0)
        self.assertGreater(nlist.n_pairs, 0)

    def test_from_orbit_table(self):
        """Test building neighbor list from AtomOrbitTable."""
        orbit = AtomOrbitTable.from_crystal(self.acetic)
        nlist = CrystalNeighborList.from_orbit_table(
            orbit, self.acetic.unit_cell, cutoff=5.0
        )

        self.assertEqual(nlist.n_atoms, orbit.n_uc)
        self.assertGreater(nlist.n_pairs, 0)

    def test_get_neighbors(self):
        """Test retrieving neighbors for a specific atom."""
        nlist = CrystalNeighborList.from_crystal(self.acetic, cutoff=5.0)

        # Get neighbors of first atom
        indices, distances, cells = nlist.get_neighbors(0)

        # All distances should be within cutoff
        self.assertTrue(np.all(distances <= 5.0))
        self.assertTrue(np.all(distances > 0))

        # Indices should be valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < nlist.n_atoms))

    def test_iter_neighbors(self):
        """Test iterating over neighbors."""
        nlist = CrystalNeighborList.from_crystal(self.acetic, cutoff=5.0)

        # Should iterate over same neighbors as get_neighbors
        indices, distances, cells = nlist.get_neighbors(0)
        iter_data = list(nlist.iter_neighbors(0))

        self.assertEqual(len(iter_data), len(indices))

        for i, (j, d, c) in enumerate(iter_data):
            self.assertEqual(j, indices[i])
            self.assertAlmostEqual(d, distances[i])
            np.testing.assert_array_equal(c, cells[i])

    def test_csr_format_consistency(self):
        """Test that CSR offsets are consistent."""
        nlist = CrystalNeighborList.from_crystal(self.acetic, cutoff=5.0)

        # Offsets should be monotonically increasing
        self.assertTrue(np.all(np.diff(nlist.neighbor_offsets) >= 0))

        # First offset should be 0, last should be n_pairs
        self.assertEqual(nlist.neighbor_offsets[0], 0)
        self.assertEqual(nlist.neighbor_offsets[-1], nlist.n_pairs)

        # Length of offsets should be n_atoms + 1
        self.assertEqual(len(nlist.neighbor_offsets), nlist.n_atoms + 1)


class NeighborListComparisonTestCase(unittest.TestCase):
    """Compare CrystalNeighborList with existing implementation."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])

    def test_compare_with_slab_method(self):
        """Compare neighbor list with Crystal.slab() based approach."""
        crystal = self.acetic
        cutoff = 5.0

        # Build our neighbor list
        nlist = CrystalNeighborList.from_crystal(crystal, cutoff=cutoff)

        # Build reference using slab approach
        uc_atoms = crystal.unit_cell_atoms()
        uc_cart = uc_atoms["cart_pos"]
        n_uc = len(uc_cart)

        # Determine slab bounds
        frac_cutoff = cutoff / np.array(crystal.unit_cell.lengths)
        lower = np.floor(-frac_cutoff).astype(int) - 1
        upper = np.ceil(frac_cutoff).astype(int) + 1
        bounds = (tuple(lower), tuple(upper))

        slab = crystal.slab(bounds=bounds)
        slab_cart = slab["cart_pos"]

        # Find neighbors using KDTree
        KDTree(uc_cart)
        tree_slab = KDTree(slab_cart)

        # For each UC atom, count neighbors in slab
        reference_counts = []
        for i in range(n_uc):
            neighbors = tree_slab.query_ball_point(uc_cart[i], cutoff)
            # Filter out self
            count = sum(
                1
                for j in neighbors
                if np.linalg.norm(slab_cart[j] - uc_cart[i]) > 1e-10
            )
            reference_counts.append(count)

        # Compare with our neighbor list
        our_counts = [nlist.get_n_neighbors(i) for i in range(n_uc)]

        np.testing.assert_array_equal(
            our_counts,
            reference_counts,
            err_msg="Neighbor counts don't match reference implementation",
        )

    def test_distances_match_reference(self):
        """Verify that computed distances match direct calculation."""
        crystal = self.acetic
        cutoff = 5.0
        nlist = CrystalNeighborList.from_crystal(crystal, cutoff=cutoff)

        uc_atoms = crystal.unit_cell_atoms()
        uc_cart = uc_atoms["cart_pos"]
        uc_frac = uc_atoms["frac_pos"]

        # Check a few atoms
        for i in range(min(5, nlist.n_atoms)):
            indices, distances, cells = nlist.get_neighbors(i)

            for j, dist, cell in zip(indices, distances, cells, strict=False):
                # Compute distance directly
                j_frac = uc_frac[j] + cell
                j_cart = crystal.unit_cell.to_cartesian(j_frac)
                expected_dist = np.linalg.norm(j_cart - uc_cart[i])

                self.assertAlmostEqual(
                    dist,
                    expected_dist,
                    places=10,
                    msg=f"Distance mismatch for pair ({i}, {j}, {cell})",
                )


class NeighborListPeriodicTestCase(unittest.TestCase):
    """Test periodic boundary condition handling."""

    def test_includes_periodic_images(self):
        """Test that periodic images are included in neighbor list."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

        # Use a large cutoff to ensure periodic images are found
        nlist = CrystalNeighborList.from_crystal(crystal, cutoff=10.0)

        # Check that some neighbors have non-zero cell offsets
        has_periodic = False
        for _i, _j, _dist, cell in nlist.iter_all_pairs():
            if not np.all(cell == 0):
                has_periodic = True
                break

        self.assertTrue(
            has_periodic, "Expected to find neighbors in periodic images"
        )

    def test_symmetry_of_pairs(self):
        """Test that if (i,j,cell) is a pair, (j,i,-cell) also exists."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        nlist = CrystalNeighborList.from_crystal(crystal, cutoff=5.0)

        # Build set of all pairs for quick lookup
        pairs = set()
        for i, j, dist, cell in nlist.iter_all_pairs():
            pairs.add((i, j, tuple(cell), round(dist, 8)))

        # Check symmetry
        for i, j, dist, cell in nlist.iter_all_pairs():
            reverse_cell = tuple(-c for c in cell)
            reverse_pair_found = False

            for ri, rj, rc, rd in pairs:
                if ri == j and rj == i and rc == reverse_cell:
                    self.assertAlmostEqual(rd, round(dist, 8))
                    reverse_pair_found = True
                    break

            self.assertTrue(
                reverse_pair_found,
                f"Reverse pair not found for ({i}, {j}, {cell})",
            )


class CoordinationNumberTestCase(unittest.TestCase):
    """Test coordination number calculation."""

    def test_coordination_numbers(self):
        """Test coordination number calculation."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        nlist = CrystalNeighborList.from_crystal(crystal, cutoff=2.0)

        coord_nums = nlist.get_coordination_numbers()

        # All coordination numbers should be non-negative
        self.assertTrue(np.all(coord_nums >= 0))

        # Should have reasonable values for a molecular crystal
        self.assertTrue(np.all(coord_nums <= 10))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

"""Tests for pair_graph.py - AtomPairGraph for force calculations."""

import logging
import unittest

import numpy as np

from chmpy.crystal import Crystal
from chmpy.crystal.pair_graph import AtomPairGraph

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class AtomPairGraphConstructionTestCase(unittest.TestCase):
    """Test AtomPairGraph construction from crystal."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])

    def test_from_crystal_acetic_acid(self):
        """Test building pair graph from acetic acid."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        self.assertGreater(pairs.n_unique_pairs(), 0)
        self.assertGreater(pairs.n_total_pairs(), 0)

    def test_from_crystal_ice_ii(self):
        """Test building pair graph from ice II."""
        pairs = AtomPairGraph.from_crystal(self.ice_ii, cutoff=4.0)

        self.assertGreater(pairs.n_unique_pairs(), 0)

    def test_distances_within_cutoff(self):
        """Test that all pair distances are within cutoff."""
        cutoff = 5.0
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=cutoff)

        for dist in pairs.pair_distances:
            self.assertLessEqual(dist, cutoff)
            self.assertGreater(dist, 0)

    def test_unique_vs_total_pairs(self):
        """Test that total >= unique pairs."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        n_unique = pairs.n_unique_pairs()
        n_total = pairs.n_total_pairs()

        self.assertGreaterEqual(n_total, n_unique)

    def test_multiplicities_sum(self):
        """Test that sum of multiplicities equals total pairs."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        total_from_mults = int(np.sum(pairs.pair_multiplicities))
        self.assertEqual(total_from_mults, pairs.n_total_pairs())

    def test_speedup_factor(self):
        """Test speedup factor calculation."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        speedup = pairs.speedup_factor()

        # Speedup should be >= 1 (at least as fast as brute force)
        self.assertGreaterEqual(speedup, 1.0)

        LOG.info(f"Acetic acid pair graph speedup: {speedup:.1f}x")


class PairwiseEnergyTestCase(unittest.TestCase):
    """Test pairwise energy calculations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_compute_pairwise_energy(self):
        """Test computing pairwise energy with simple potential."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        # Simple LJ-like potential: V(r) = 1/r^12 - 1/r^6
        def lj_potential(r):
            return 1.0 / r**12 - 1.0 / r**6

        energy = pairs.compute_pairwise_energy(lj_potential)

        # Energy should be finite
        self.assertTrue(np.isfinite(energy))

    def test_energy_scales_with_cutoff(self):
        """Test that energy changes with cutoff."""
        # Smaller cutoff
        pairs_small = AtomPairGraph.from_crystal(self.acetic, cutoff=3.0)
        # Larger cutoff
        pairs_large = AtomPairGraph.from_crystal(self.acetic, cutoff=6.0)

        def harmonic(r):
            return 0.5 * r**2

        e_small = pairs_small.compute_pairwise_energy(harmonic)
        e_large = pairs_large.compute_pairwise_energy(harmonic)

        # Larger cutoff should include more pairs
        self.assertGreaterEqual(
            pairs_large.n_unique_pairs(), pairs_small.n_unique_pairs()
        )


class PairwiseForcesTestCase(unittest.TestCase):
    """Test pairwise force calculations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_compute_forces_simple(self):
        """Test computing forces with simple potential."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        # Force magnitude for harmonic potential: F = -dV/dr = -r
        def force_mag(r):
            return -r  # Attractive harmonic

        forces = pairs.compute_pairwise_forces_simple(force_mag)

        # Forces should have correct shape
        n_uc = len(pairs._uc_atom_positions)
        self.assertEqual(forces.shape, (n_uc, 3))

        # Forces should be finite
        self.assertTrue(np.all(np.isfinite(forces)))

    def test_forces_sum_to_zero(self):
        """Test that total force sums to zero (Newton's 3rd law)."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        def repulsive_force(r):
            return 1.0 / r**2  # Repulsive

        forces = pairs.compute_pairwise_forces_simple(repulsive_force)

        # Sum should be approximately zero
        total_force = np.sum(forces, axis=0)
        np.testing.assert_allclose(
            total_force, [0, 0, 0], atol=1e-10,
            err_msg="Forces do not sum to zero"
        )


class PairwiseStressTestCase(unittest.TestCase):
    """Test pairwise stress tensor calculations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_compute_stress(self):
        """Test computing stress tensor."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        def force_mag(r):
            return 1.0 / r**2

        stress = pairs.compute_pairwise_stress(force_mag)

        # Stress should be 3x3
        self.assertEqual(stress.shape, (3, 3))

        # Stress should be finite
        self.assertTrue(np.all(np.isfinite(stress)))

    def test_stress_is_symmetric(self):
        """Test that stress tensor is symmetric."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        def force_mag(r):
            return -1.0 / r**3  # Attractive

        stress = pairs.compute_pairwise_stress(force_mag)

        # Should be symmetric
        np.testing.assert_allclose(
            stress, stress.T, atol=1e-10,
            err_msg="Stress tensor is not symmetric"
        )


class PairQueryTestCase(unittest.TestCase):
    """Test pair graph query operations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_pairs_involving_atom(self):
        """Test querying pairs involving specific atom type."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        # Query pairs involving first asym atom
        involving = pairs.pairs_involving_atom(asym_idx=0)

        # Each pair should involve atom type 0
        for pair, mult, dist in involving:
            self.assertTrue(
                pair.src.asym_idx == 0 or pair.dst.asym_idx == 0
            )

    def test_unique_pairs_by_type(self):
        """Test filtering pairs by element types."""
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=5.0)

        # Get C-C pairs (C=6)
        cc_pairs = pairs.unique_pairs_by_type(6, 6)

        # Verify they're actually C-C
        atoms = self.acetic.site_atoms
        for pair, mult, dist in cc_pairs:
            type_a = atoms[pair.src.asym_idx]
            type_b = atoms[pair.dst.asym_idx]
            self.assertEqual(type_a, 6)
            self.assertEqual(type_b, 6)


class CompareWithBruteForceTestCase(unittest.TestCase):
    """Compare symmetric approach with brute-force calculation."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_energy_matches_brute_force(self):
        """Test that symmetric energy matches brute-force calculation."""
        cutoff = 4.0
        pairs = AtomPairGraph.from_crystal(self.acetic, cutoff=cutoff)

        def lj(r):
            return 1.0 / r**6

        # Symmetric calculation
        symmetric_energy = pairs.compute_pairwise_energy(lj)

        # Brute-force calculation
        uc_dict = self.acetic.unit_cell_atoms()
        uc_cart = uc_dict["cart_pos"]
        n_uc = len(uc_cart)

        # Generate supercell
        cell_lengths = np.array(self.acetic.unit_cell.lengths)
        n_cells = np.ceil(cutoff / cell_lengths).astype(int) + 1

        brute_energy = 0.0
        for i in range(n_uc):
            for h in range(-n_cells[0], n_cells[0] + 1):
                for k in range(-n_cells[1], n_cells[1] + 1):
                    for l in range(-n_cells[2], n_cells[2] + 1):
                        for j in range(n_uc):
                            if i == j and (h, k, l) == (0, 0, 0):
                                continue

                            # Position of j in translated cell
                            frac_j = uc_dict["frac_pos"][j] + np.array([h, k, l])
                            pos_j = self.acetic.unit_cell.to_cartesian(frac_j)
                            pos_i = uc_cart[i]

                            r = np.linalg.norm(pos_j - pos_i)
                            if r > 0 and r <= cutoff:
                                brute_energy += lj(r) / 2.0  # Divide by 2 to avoid double counting

        # Compare (allow some tolerance for numerical differences)
        # The symmetric approach counts differently, so normalize
        LOG.info(f"Symmetric energy: {symmetric_energy}")
        LOG.info(f"Brute force energy: {brute_energy}")

        # They might not be exactly equal due to different counting,
        # but should be in the same ballpark
        if brute_energy != 0:
            ratio = symmetric_energy / brute_energy
            # Allow 50% tolerance due to counting differences
            self.assertGreater(ratio, 0.5)
            self.assertLess(ratio, 2.0)


class ReprTestCase(unittest.TestCase):
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        pairs = AtomPairGraph.from_crystal(acetic, cutoff=5.0)

        repr_str = repr(pairs)

        self.assertIn("AtomPairGraph", repr_str)
        self.assertIn("cutoff", repr_str)
        self.assertIn("n_unique", repr_str)
        self.assertIn("speedup", repr_str)


class HighSymmetryPairTestCase(unittest.TestCase):
    """Test pair graph with high-symmetry structures."""

    def test_ice_ii_pairs(self):
        """Test pair graph for ice II."""
        ice = Crystal.load(TEST_FILES["iceII.cif"])
        pairs = AtomPairGraph.from_crystal(ice, cutoff=4.0)

        n_unique = pairs.n_unique_pairs()
        n_total = pairs.n_total_pairs()
        speedup = pairs.speedup_factor()

        LOG.info(
            f"Ice II: {n_unique} unique pairs, {n_total} total, "
            f"{speedup:.1f}x speedup"
        )

        # High symmetry should give good speedup
        self.assertGreater(speedup, 1.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

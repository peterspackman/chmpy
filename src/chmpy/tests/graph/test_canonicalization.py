"""Tests for graph canonicalization."""

import logging
import unittest

import numpy as np

from chmpy.graph import MolecularGraph
from chmpy.graph.canonicalization import (
    canonical_atom_map,
    canonical_ordering,
    equivalent_atoms,
    morgan_fingerprint,
    morgan_fingerprint_bits,
    reorder_graph,
    tanimoto_similarity,
)

LOG = logging.getLogger(__name__)


def make_benzene_graph() -> MolecularGraph:
    """Create benzene - all carbons should be equivalent."""
    atomic_numbers = np.array([6, 6, 6, 6, 6, 6])
    positions = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [-0.5, 0.866, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -0.866, 0.0],
        [0.5, -0.866, 0.0],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_propane_graph() -> MolecularGraph:
    """Create propane - terminal carbons equivalent, central different."""
    atomic_numbers = np.array([6, 6, 6])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    edges = [(0, 1), (1, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_methane_graph() -> MolecularGraph:
    """Create methane (carbon + 4 hydrogens)."""
    atomic_numbers = np.array([6, 1, 1, 1, 1])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_toluene_graph() -> MolecularGraph:
    """Create toluene (benzene with methyl) - has different atom types."""
    atomic_numbers = np.array([6, 6, 6, 6, 6, 6, 6])  # 6 ring C + 1 methyl C
    positions = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [-0.5, 0.866, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.5, -0.866, 0.0],
        [0.5, -0.866, 0.0],
        [2.5, 0.0, 0.0],  # Methyl carbon
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


class CanonicalOrderingTestCase(unittest.TestCase):
    """Test cases for canonical ordering."""

    def test_ordering_deterministic(self):
        """Test that canonical ordering is deterministic."""
        graph = make_benzene_graph()

        ordering1 = canonical_ordering(graph)
        ordering2 = canonical_ordering(graph)

        np.testing.assert_array_equal(ordering1, ordering2)

    def test_ordering_complete(self):
        """Test that ordering contains all indices exactly once."""
        graph = make_benzene_graph()
        ordering = canonical_ordering(graph)

        self.assertEqual(len(ordering), 6)
        self.assertEqual(set(ordering), {0, 1, 2, 3, 4, 5})

    def test_ordering_single_atom(self):
        """Test ordering with single atom."""
        atomic_numbers = np.array([6])
        positions = np.array([[0, 0, 0]])
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, [])

        ordering = canonical_ordering(graph)
        self.assertEqual(len(ordering), 1)
        self.assertEqual(ordering[0], 0)

    def test_ordering_empty_graph(self):
        """Test ordering with empty graph."""
        atomic_numbers = np.array([], dtype=np.int32)
        positions = np.zeros((0, 3))
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, [])

        ordering = canonical_ordering(graph)
        self.assertEqual(len(ordering), 0)

    def test_ordering_different_atoms(self):
        """Test that different atom types get different ranks."""
        # Water: O and 2 H
        atomic_numbers = np.array([8, 1, 1])
        positions = np.zeros((3, 3))
        edges = [(0, 1), (0, 2)]
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, edges)

        ordering = canonical_ordering(graph)

        # Oxygen should have different rank than hydrogens
        self.assertNotEqual(ordering[0], ordering[1])
        self.assertNotEqual(ordering[0], ordering[2])


class EquivalentAtomsTestCase(unittest.TestCase):
    """Test cases for finding equivalent atoms."""

    def test_benzene_all_equivalent(self):
        """All carbons in benzene should be equivalent."""
        graph = make_benzene_graph()
        equiv = equivalent_atoms(graph)

        # Should have one group with all 6 carbons
        self.assertEqual(len(equiv), 1)
        self.assertEqual(equiv[0], {0, 1, 2, 3, 4, 5})

    def test_propane_symmetry(self):
        """Terminal carbons in propane should be equivalent."""
        graph = make_propane_graph()
        equiv = equivalent_atoms(graph)

        # Should have two groups: terminals and central
        self.assertEqual(len(equiv), 2)

        # Find the group with 2 atoms (terminals)
        terminals = [g for g in equiv if len(g) == 2][0]
        central = [g for g in equiv if len(g) == 1][0]

        self.assertEqual(terminals, {0, 2})
        self.assertEqual(central, {1})

    def test_methane_hydrogens_equivalent(self):
        """All hydrogens in methane should be equivalent."""
        graph = make_methane_graph()
        equiv = equivalent_atoms(graph)

        # Should have 2 groups: 1 carbon, 4 hydrogens
        self.assertEqual(len(equiv), 2)

        # Find the hydrogen group
        h_group = [g for g in equiv if len(g) == 4][0]
        self.assertEqual(h_group, {1, 2, 3, 4})

    def test_toluene_asymmetry(self):
        """Toluene has multiple atom classes due to methyl substituent."""
        graph = make_toluene_graph()
        equiv = equivalent_atoms(graph)

        # Toluene should have multiple classes
        # - Ipso carbon (connected to methyl)
        # - Ortho carbons (2)
        # - Meta carbons (2)
        # - Para carbon (1)
        # - Methyl carbon (1)
        self.assertGreater(len(equiv), 1)


class MorganFingerprintTestCase(unittest.TestCase):
    """Test cases for Morgan fingerprints."""

    def test_fingerprint_shape(self):
        """Test fingerprint has correct shape."""
        graph = make_benzene_graph()
        fp = morgan_fingerprint(graph, radius=2, n_bits=2048)

        self.assertEqual(fp.shape, (2048,))
        self.assertEqual(fp.dtype, bool)

    def test_fingerprint_nonzero(self):
        """Test fingerprint has some bits set."""
        graph = make_benzene_graph()
        fp = morgan_fingerprint(graph, radius=2)

        self.assertGreater(np.sum(fp), 0)

    def test_fingerprint_deterministic(self):
        """Test fingerprint is deterministic."""
        graph = make_benzene_graph()

        fp1 = morgan_fingerprint(graph, radius=2)
        fp2 = morgan_fingerprint(graph, radius=2)

        np.testing.assert_array_equal(fp1, fp2)

    def test_fingerprint_bits(self):
        """Test fingerprint_bits returns a set."""
        graph = make_benzene_graph()
        bits = morgan_fingerprint_bits(graph, radius=2)

        self.assertIsInstance(bits, set)
        self.assertGreater(len(bits), 0)

    def test_similar_molecules_similar_fingerprint(self):
        """Similar molecules should have similar fingerprints."""
        # Benzene vs toluene
        benzene = make_benzene_graph()
        toluene = make_toluene_graph()

        fp_benzene = morgan_fingerprint(benzene, radius=2)
        fp_toluene = morgan_fingerprint(toluene, radius=2)

        similarity = tanimoto_similarity(fp_benzene, fp_toluene)

        # Should have some similarity (toluene contains benzene substructure)
        self.assertGreater(similarity, 0.2)

    def test_different_molecules_different_fingerprint(self):
        """Different molecules should have different fingerprints."""
        benzene = make_benzene_graph()
        methane = make_methane_graph()

        fp_benzene = morgan_fingerprint(benzene, radius=2)
        fp_methane = morgan_fingerprint(methane, radius=2)

        # Should not be identical
        self.assertFalse(np.array_equal(fp_benzene, fp_methane))


class TanimotoSimilarityTestCase(unittest.TestCase):
    """Test cases for Tanimoto similarity."""

    def test_identical_fingerprints(self):
        """Identical fingerprints should have similarity 1.0."""
        graph = make_benzene_graph()
        fp = morgan_fingerprint(graph)

        similarity = tanimoto_similarity(fp, fp)
        self.assertAlmostEqual(similarity, 1.0)

    def test_empty_fingerprints(self):
        """Empty fingerprints should have similarity 1.0."""
        fp1 = np.zeros(2048, dtype=bool)
        fp2 = np.zeros(2048, dtype=bool)

        similarity = tanimoto_similarity(fp1, fp2)
        self.assertAlmostEqual(similarity, 1.0)

    def test_disjoint_fingerprints(self):
        """Completely disjoint fingerprints should have similarity 0.0."""
        fp1 = np.zeros(100, dtype=bool)
        fp2 = np.zeros(100, dtype=bool)
        fp1[:50] = True
        fp2[50:] = True

        similarity = tanimoto_similarity(fp1, fp2)
        self.assertAlmostEqual(similarity, 0.0)


class CanonicalAtomMapTestCase(unittest.TestCase):
    """Test cases for canonical atom mapping."""

    def test_map_complete(self):
        """Map should contain all atoms."""
        graph = make_benzene_graph()
        mapping = canonical_atom_map(graph)

        self.assertEqual(len(mapping), 6)
        self.assertEqual(set(mapping.keys()), {0, 1, 2, 3, 4, 5})

    def test_map_values_unique(self):
        """Map values should be unique."""
        graph = make_benzene_graph()
        mapping = canonical_atom_map(graph)

        values = list(mapping.values())
        self.assertEqual(len(values), len(set(values)))


class ReorderGraphTestCase(unittest.TestCase):
    """Test cases for graph reordering."""

    def test_reorder_preserves_size(self):
        """Reordering should preserve number of atoms and bonds."""
        graph = make_benzene_graph()
        reordered = reorder_graph(graph)

        self.assertEqual(reordered.n_atoms, graph.n_atoms)
        self.assertEqual(reordered.n_bonds, graph.n_bonds)

    def test_reorder_preserves_elements(self):
        """Reordering should preserve element composition."""
        graph = make_toluene_graph()
        reordered = reorder_graph(graph)

        original_elements = sorted(graph.atomic_numbers)
        reordered_elements = sorted(reordered.atomic_numbers)

        np.testing.assert_array_equal(original_elements, reordered_elements)

    def test_reorder_deterministic(self):
        """Reordering should be deterministic."""
        graph = make_benzene_graph()

        reordered1 = reorder_graph(graph)
        reordered2 = reorder_graph(graph)

        np.testing.assert_array_equal(
            reordered1.atomic_numbers, reordered2.atomic_numbers
        )


if __name__ == "__main__":
    unittest.main()

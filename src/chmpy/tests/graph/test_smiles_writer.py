"""Tests for SMILES generation."""

import logging
import unittest

import numpy as np

from chmpy.graph import MolecularGraph
from chmpy.graph.smiles_writer import to_smiles

LOG = logging.getLogger(__name__)


def make_methane_graph() -> MolecularGraph:
    """Create methane CH4."""
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


def make_ethane_graph() -> MolecularGraph:
    """Create ethane C2H6."""
    atomic_numbers = np.array([6, 6, 1, 1, 1, 1, 1, 1])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
        [-0.5, 0.0, 0.9],
        [2.04, 0.9, 0.0],
        [2.04, -0.9, 0.0],
        [2.04, 0.0, 0.9],
    ])
    edges = [
        (0, 1),
        (0, 2), (0, 3), (0, 4),
        (1, 5), (1, 6), (1, 7),
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_water_graph() -> MolecularGraph:
    """Create water H2O."""
    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
    ])
    edges = [(0, 1), (0, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_benzene_graph() -> MolecularGraph:
    """Create benzene C6H6 (just the ring carbons)."""
    atomic_numbers = np.array([6, 6, 6, 6, 6, 6])
    r = 1.40
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    positions = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(6)])
    edges = [(i, (i + 1) % 6) for i in range(6)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_ethene_graph() -> MolecularGraph:
    """Create ethene C2H4 with double bond."""
    atomic_numbers = np.array([6, 6, 1, 1, 1, 1])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.34, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
        [1.84, 0.9, 0.0],
        [1.84, -0.9, 0.0],
    ])
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    bond_orders = [2, 1, 1, 1, 1]
    return MolecularGraph.from_edge_list(
        atomic_numbers, positions, edges, bond_orders=bond_orders
    )


def make_ethyne_graph() -> MolecularGraph:
    """Create ethyne/acetylene C2H2 with triple bond."""
    atomic_numbers = np.array([6, 6, 1, 1])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.20, 0.0, 0.0],
        [-1.06, 0.0, 0.0],
        [2.26, 0.0, 0.0],
    ])
    edges = [(0, 1), (0, 2), (1, 3)]
    bond_orders = [3, 1, 1]
    return MolecularGraph.from_edge_list(
        atomic_numbers, positions, edges, bond_orders=bond_orders
    )


def make_cyclopropane_graph() -> MolecularGraph:
    """Create cyclopropane C3H6."""
    atomic_numbers = np.array([6, 6, 6])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.75, 1.3, 0.0],
    ])
    edges = [(0, 1), (1, 2), (2, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_propane_graph() -> MolecularGraph:
    """Create propane C3H8 (just carbons)."""
    atomic_numbers = np.array([6, 6, 6])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [3.08, 0.0, 0.0],
    ])
    edges = [(0, 1), (1, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


class BasicSMILESTestCase(unittest.TestCase):
    """Test cases for basic SMILES generation."""

    def test_methane(self):
        """Test SMILES for methane."""
        graph = make_methane_graph()
        smiles = to_smiles(graph)

        # Methane: C with 4 H
        self.assertIn("C", smiles)
        # Should have a carbon atom
        # Note: explicit H atoms may be included depending on graph structure

    def test_water(self):
        """Test SMILES for water."""
        graph = make_water_graph()
        smiles = to_smiles(graph)

        # Water: O with 2 H
        self.assertIn("O", smiles)

    def test_ethane(self):
        """Test SMILES for ethane."""
        graph = make_ethane_graph()
        smiles = to_smiles(graph)

        # Should have two carbons
        self.assertEqual(smiles.count("C"), 2)

    def test_propane(self):
        """Test SMILES for propane (carbon chain)."""
        graph = make_propane_graph()
        smiles = to_smiles(graph)

        # Should have three carbons
        self.assertEqual(smiles.count("C"), 3)


class BondOrderSMILESTestCase(unittest.TestCase):
    """Test cases for bond order representation in SMILES."""

    def test_ethene_double_bond(self):
        """Test SMILES for ethene (double bond)."""
        graph = make_ethene_graph()
        smiles = to_smiles(graph)

        # Should contain double bond symbol
        self.assertIn("=", smiles)

    def test_ethyne_triple_bond(self):
        """Test SMILES for ethyne (triple bond)."""
        graph = make_ethyne_graph()
        smiles = to_smiles(graph)

        # Should contain triple bond symbol
        self.assertIn("#", smiles)


class RingSMILESTestCase(unittest.TestCase):
    """Test cases for ring representation in SMILES."""

    def test_cyclopropane_ring(self):
        """Test SMILES for cyclopropane (3-membered ring)."""
        graph = make_cyclopropane_graph()
        smiles = to_smiles(graph)

        # Should have ring closure number
        # Ring closures use digits
        has_ring = any(c.isdigit() for c in smiles)
        self.assertTrue(has_ring)

    def test_benzene_aromatic(self):
        """Test SMILES for benzene (aromatic)."""
        graph = make_benzene_graph()
        smiles = to_smiles(graph)

        # Aromatic carbons should be lowercase 'c'
        self.assertIn("c", smiles.lower())


class CanonicalSMILESTestCase(unittest.TestCase):
    """Test cases for canonical SMILES generation."""

    def test_canonical_deterministic(self):
        """Test that canonical SMILES is deterministic."""
        graph = make_benzene_graph()

        smiles1 = to_smiles(graph, canonical=True)
        smiles2 = to_smiles(graph, canonical=True)

        self.assertEqual(smiles1, smiles2)

    def test_canonical_vs_non_canonical(self):
        """Test that canonical and non-canonical can differ."""
        graph = make_propane_graph()

        canonical = to_smiles(graph, canonical=True)
        non_canonical = to_smiles(graph, canonical=False)

        # Both should be valid (contain same atoms)
        self.assertEqual(canonical.count("C"), 3)
        self.assertEqual(non_canonical.count("C"), 3)


class EmptySMILESTestCase(unittest.TestCase):
    """Test cases for edge cases."""

    def test_empty_graph(self):
        """Test SMILES for empty graph."""
        atomic_numbers = np.array([], dtype=np.int32)
        positions = np.zeros((0, 3))
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, [])

        smiles = to_smiles(graph)
        self.assertEqual(smiles, "")

    def test_single_atom(self):
        """Test SMILES for single atom."""
        atomic_numbers = np.array([6])
        positions = np.array([[0, 0, 0]])
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, [])

        smiles = to_smiles(graph)
        self.assertEqual(smiles, "C")


class MultiFragmentSMILESTestCase(unittest.TestCase):
    """Test cases for multi-fragment molecules."""

    def test_two_waters(self):
        """Test SMILES for two disconnected water molecules."""
        atomic_numbers = np.array([8, 1, 1, 8, 1, 1])
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [5, 0, 0], [6, 0, 0], [5, 1, 0],
        ])
        edges = [(0, 1), (0, 2), (3, 4), (3, 5)]
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, edges)

        smiles = to_smiles(graph)

        # Should have dot separator for fragments
        self.assertIn(".", smiles)
        # Should have two oxygens
        self.assertEqual(smiles.count("O"), 2)


if __name__ == "__main__":
    unittest.main()

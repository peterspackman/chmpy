"""Tests for MolecularGraph core functionality."""

import logging
import unittest
from copy import deepcopy

import numpy as np

from chmpy import Molecule
from chmpy.graph import MolecularGraph

from .. import TEST_FILES

LOG = logging.getLogger(__name__)

_WATER = None
_SDF_MOL = None


class MolecularGraphTestCase(unittest.TestCase):
    """Test cases for MolecularGraph class."""

    @staticmethod
    def load_water():
        global _WATER
        if _WATER is None:
            _WATER = Molecule.load(TEST_FILES["water.xyz"])
        return deepcopy(_WATER)

    @staticmethod
    def load_sdf_mol():
        global _SDF_MOL
        if _SDF_MOL is None:
            _SDF_MOL = Molecule.load(TEST_FILES["DB09563.sdf"])
        return deepcopy(_SDF_MOL)

    def test_from_molecule_water(self):
        """Test creating graph from water molecule."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        self.assertEqual(graph.n_atoms, 3)
        # Water: O-H, O-H = 2 bonds
        self.assertEqual(graph.n_bonds, 2)

        # Check atomic numbers (O=8, H=1, H=1)
        self.assertEqual(graph.atomic_numbers[0], 8)  # Oxygen
        self.assertEqual(graph.atomic_numbers[1], 1)  # Hydrogen
        self.assertEqual(graph.atomic_numbers[2], 1)  # Hydrogen

    def test_from_molecule_sdf(self):
        """Test creating graph from SDF molecule."""
        mol = self.load_sdf_mol()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        self.assertEqual(graph.n_atoms, 21)
        # Verify graph has reasonable number of bonds
        self.assertGreater(graph.n_bonds, 0)

    def test_neighbors(self):
        """Test neighbor retrieval."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        # Oxygen (index 0) should have 2 hydrogen neighbors
        o_neighbors = graph.neighbors(0)
        self.assertEqual(len(o_neighbors), 2)

        # Each hydrogen should have 1 neighbor (oxygen)
        h1_neighbors = graph.neighbors(1)
        h2_neighbors = graph.neighbors(2)
        self.assertEqual(len(h1_neighbors), 1)
        self.assertEqual(len(h2_neighbors), 1)

    def test_degree(self):
        """Test degree calculation."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        degrees = graph.degree
        self.assertEqual(degrees[0], 2)  # Oxygen
        self.assertEqual(degrees[1], 1)  # Hydrogen
        self.assertEqual(degrees[2], 1)  # Hydrogen

    def test_has_bond(self):
        """Test bond existence check."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        # O-H bonds exist
        self.assertTrue(graph.has_bond(0, 1))
        self.assertTrue(graph.has_bond(0, 2))
        self.assertTrue(graph.has_bond(1, 0))  # Symmetric

        # H-H bond does not exist
        self.assertFalse(graph.has_bond(1, 2))

    def test_connected_components_single(self):
        """Test connected components for single molecule."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        n_components, labels = graph.connected_components()
        self.assertEqual(n_components, 1)
        # All atoms in same component
        self.assertTrue(np.all(labels == labels[0]))

    def test_edges(self):
        """Test edge list retrieval."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        edges = graph.edges()
        self.assertEqual(len(edges), 2)
        # Edges should be sorted (i < j)
        for i, j in edges:
            self.assertLess(i, j)

    def test_subgraph(self):
        """Test subgraph extraction."""
        mol = self.load_sdf_mol()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        # Extract first 5 atoms
        subgraph = graph.subgraph(np.array([0, 1, 2, 3, 4]))
        self.assertEqual(subgraph.n_atoms, 5)
        self.assertEqual(len(subgraph.atomic_numbers), 5)
        self.assertEqual(subgraph.positions.shape, (5, 3))

    def test_from_edge_list(self):
        """Test creating graph from edge list."""
        # Simple methane-like structure
        atomic_numbers = np.array([6, 1, 1, 1, 1])  # C + 4H
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]  # C-H bonds

        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, edges)

        self.assertEqual(graph.n_atoms, 5)
        self.assertEqual(graph.n_bonds, 4)
        self.assertEqual(graph.degree[0], 4)  # Carbon
        self.assertEqual(graph.degree[1], 1)  # Hydrogen

    def test_from_edge_list_with_bond_orders(self):
        """Test creating graph with bond orders."""
        # Ethene-like: C=C with hydrogens
        atomic_numbers = np.array([6, 6, 1, 1, 1, 1])
        positions = np.zeros((6, 3))  # Positions not important for this test
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
        bond_orders = [2, 1, 1, 1, 1]  # C=C double bond

        graph = MolecularGraph.from_edge_list(
            atomic_numbers, positions, edges, bond_orders=bond_orders
        )

        self.assertEqual(graph.bond_order(0, 1), 2)  # Double bond
        self.assertEqual(graph.bond_order(0, 2), 1)  # Single bond

    def test_bond_distance(self):
        """Test bond distance retrieval."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        # O-H bond should have reasonable distance
        dist = graph.bond_distance(0, 1)
        self.assertIsNotNone(dist)
        self.assertGreater(dist, 0.5)  # > 0.5 Angstrom
        self.assertLess(dist, 1.5)  # < 1.5 Angstrom

    def test_all_neighbors(self):
        """Test getting all neighbor lists at once."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        all_neighbors = graph.all_neighbors()
        self.assertEqual(len(all_neighbors), 3)
        self.assertEqual(len(all_neighbors[0]), 2)  # Oxygen has 2 neighbors
        self.assertEqual(len(all_neighbors[1]), 1)  # H has 1 neighbor

    def test_repr(self):
        """Test string representation."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        repr_str = repr(graph)
        self.assertIn("MolecularGraph", repr_str)
        self.assertIn("n_atoms=3", repr_str)
        self.assertIn("n_bonds=2", repr_str)

    def test_len(self):
        """Test __len__ method."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = MolecularGraph.from_molecule(mol)

        self.assertEqual(len(graph), 3)


if __name__ == "__main__":
    unittest.main()

"""Integration tests for graph module with Molecule class."""

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


class MoleculeGraphIntegrationTestCase(unittest.TestCase):
    """Test integration of graph module with Molecule class."""

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

    def test_molecule_graph_property(self):
        """Test that Molecule.graph returns a MolecularGraph."""
        mol = self.load_water()
        mol.guess_bonds()
        graph = mol.graph

        self.assertIsInstance(graph, MolecularGraph)
        self.assertEqual(graph.n_atoms, 3)

    def test_molecule_graph_cached(self):
        """Test that the graph property is cached."""
        mol = self.load_water()
        mol.guess_bonds()

        graph1 = mol.graph
        graph2 = mol.graph

        self.assertIs(graph1, graph2)

    def test_molecule_to_smiles(self):
        """Test SMILES generation from Molecule."""
        mol = self.load_water()
        mol.guess_bonds()

        smiles = mol.to_smiles()

        self.assertIsInstance(smiles, str)
        self.assertIn("O", smiles)

    def test_molecule_find_rings_water(self):
        """Test ring finding for water (no rings)."""
        mol = self.load_water()
        mol.guess_bonds()

        rings = mol.find_rings()

        self.assertEqual(len(rings), 0)

    def test_molecule_is_aromatic_water(self):
        """Test aromaticity check for water (not aromatic)."""
        mol = self.load_water()
        mol.guess_bonds()

        self.assertFalse(mol.is_aromatic())

    def test_molecule_morgan_fingerprint(self):
        """Test Morgan fingerprint generation."""
        mol = self.load_water()
        mol.guess_bonds()

        fp = mol.morgan_fingerprint(radius=2, n_bits=1024)

        self.assertEqual(fp.shape, (1024,))
        self.assertTrue(any(fp))  # Should have some bits set

    def test_sdf_molecule_graph(self):
        """Test graph creation from SDF molecule."""
        mol = self.load_sdf_mol()
        mol.guess_bonds()
        graph = mol.graph

        self.assertEqual(graph.n_atoms, 21)

    def test_sdf_molecule_smiles(self):
        """Test SMILES generation from SDF molecule."""
        mol = self.load_sdf_mol()
        mol.guess_bonds()

        smiles = mol.to_smiles()

        self.assertIsInstance(smiles, str)
        self.assertGreater(len(smiles), 0)


class MolecularGraphFromMoleculeTestCase(unittest.TestCase):
    """Test MolecularGraph.from_molecule() method."""

    @staticmethod
    def load_water():
        global _WATER
        if _WATER is None:
            _WATER = Molecule.load(TEST_FILES["water.xyz"])
        return deepcopy(_WATER)

    def test_from_molecule_creates_correct_atoms(self):
        """Test that from_molecule preserves atomic numbers."""
        mol = self.load_water()
        mol.guess_bonds()

        graph = MolecularGraph.from_molecule(mol)

        np.testing.assert_array_equal(graph.atomic_numbers, mol.atomic_numbers)

    def test_from_molecule_creates_correct_positions(self):
        """Test that from_molecule preserves positions."""
        mol = self.load_water()
        mol.guess_bonds()

        graph = MolecularGraph.from_molecule(mol)

        np.testing.assert_allclose(graph.positions, mol.positions)

    def test_from_molecule_creates_bonds(self):
        """Test that from_molecule creates correct bonds."""
        mol = self.load_water()
        mol.guess_bonds()

        graph = MolecularGraph.from_molecule(mol)

        # Water has 2 O-H bonds
        self.assertEqual(graph.n_bonds, 2)


if __name__ == "__main__":
    unittest.main()

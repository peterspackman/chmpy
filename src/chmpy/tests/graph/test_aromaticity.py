"""Tests for aromaticity perception."""

import logging
import unittest

import numpy as np

from chmpy.graph import MolecularGraph
from chmpy.graph.aromaticity import (
    count_pi_electrons,
    get_aromatic_rings,
    is_aromatic_atom,
    is_aromatic_bond,
    is_aromatic_ring,
    perceive_aromaticity,
)

LOG = logging.getLogger(__name__)


def make_benzene_graph() -> MolecularGraph:
    """Create benzene (C6H6) - aromatic 6-membered ring."""
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


def make_pyridine_graph() -> MolecularGraph:
    """Create pyridine (C5H5N) - aromatic with pyridine nitrogen."""
    # N at position 0, rest are C
    atomic_numbers = np.array([7, 6, 6, 6, 6, 6])
    positions = np.zeros((6, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_pyrrole_graph() -> MolecularGraph:
    """Create pyrrole (C4H5N) - aromatic 5-membered ring with NH."""
    # N at position 0 with H (degree 3), rest are C
    # Ring: N-C-C-C-C-N
    atomic_numbers = np.array([7, 6, 6, 6, 6, 1])  # N, 4C, H on N
    positions = np.zeros((6, 3))
    # Ring edges + N-H
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_furan_graph() -> MolecularGraph:
    """Create furan (C4H4O) - aromatic 5-membered ring with O."""
    # O at position 0
    atomic_numbers = np.array([8, 6, 6, 6, 6])
    positions = np.zeros((5, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_thiophene_graph() -> MolecularGraph:
    """Create thiophene (C4H4S) - aromatic 5-membered ring with S."""
    # S at position 0
    atomic_numbers = np.array([16, 6, 6, 6, 6])
    positions = np.zeros((5, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_imidazole_graph() -> MolecularGraph:
    """Create imidazole (C3H4N2) - aromatic 5-membered ring with 2 nitrogens."""
    # Imidazole: N1-C2-N3-C4-C5-N1
    # N1 is pyrrole-type (NH, contributes 2), N3 is pyridine-type (contributes 1)
    atomic_numbers = np.array([7, 6, 7, 6, 6, 1])  # N, C, N, C, C, H on N1
    positions = np.zeros((6, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)]  # Ring + N-H
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_naphthalene_graph() -> MolecularGraph:
    """Create naphthalene (C10H8) - fused aromatic system."""
    atomic_numbers = np.array([6] * 10)
    positions = np.zeros((10, 3))
    # Two fused 6-membered rings
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 9), (9, 0),  # First ring
        (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),  # Second ring
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_cyclobutadiene_graph() -> MolecularGraph:
    """Create cyclobutadiene (C4H4) - antiaromatic (4 electrons)."""
    atomic_numbers = np.array([6, 6, 6, 6])
    positions = np.zeros((4, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_cyclohexane_graph() -> MolecularGraph:
    """Create cyclohexane (C6H12) - non-aromatic saturated ring."""
    # All sp3 carbons with 4 bonds each
    atomic_numbers = np.array([6, 6, 6, 6, 6, 6] + [1] * 12)  # 6C + 12H
    positions = np.zeros((18, 3))
    # Ring edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    # Add C-H bonds (2 per carbon)
    for i in range(6):
        edges.append((i, 6 + 2 * i))
        edges.append((i, 6 + 2 * i + 1))
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_cyclopentadienyl_anion_graph() -> MolecularGraph:
    """Create cyclopentadienyl anion (C5H5-) - aromatic (6 electrons)."""
    # 5 carbons, each contributes 1.2 electrons on average (6 total)
    # Actually, one carbon has negative charge contributing 2, others 1 each
    # For simplicity, model as 5 sp2 carbons
    atomic_numbers = np.array([6, 6, 6, 6, 6])
    positions = np.zeros((5, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


class AromaticityPerceptionTestCase(unittest.TestCase):
    """Test cases for aromaticity perception."""

    def test_benzene_aromatic(self):
        """Test that benzene is recognized as aromatic."""
        graph = make_benzene_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # All 6 atoms should be aromatic
        self.assertTrue(all(aromatic_atoms))
        self.assertEqual(len(aromatic_bonds), 6)

    def test_pyridine_aromatic(self):
        """Test that pyridine is recognized as aromatic."""
        graph = make_pyridine_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # All 6 atoms should be aromatic
        self.assertTrue(all(aromatic_atoms))

    def test_pyrrole_aromatic(self):
        """Test that pyrrole is recognized as aromatic."""
        graph = make_pyrrole_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # Ring atoms (0-4) should be aromatic, H (5) is not
        for i in range(5):
            self.assertTrue(aromatic_atoms[i], f"Atom {i} should be aromatic")
        self.assertFalse(aromatic_atoms[5])  # H is not aromatic

    def test_furan_aromatic(self):
        """Test that furan is recognized as aromatic."""
        graph = make_furan_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # All 5 atoms should be aromatic
        self.assertTrue(all(aromatic_atoms))

    def test_thiophene_aromatic(self):
        """Test that thiophene is recognized as aromatic."""
        graph = make_thiophene_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # All 5 atoms should be aromatic
        self.assertTrue(all(aromatic_atoms))

    def test_imidazole_aromatic(self):
        """Test that imidazole is recognized as aromatic."""
        graph = make_imidazole_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # Ring atoms (0-4) should be aromatic
        for i in range(5):
            self.assertTrue(aromatic_atoms[i], f"Atom {i} should be aromatic")

    def test_naphthalene_aromatic(self):
        """Test that naphthalene is recognized as aromatic."""
        graph = make_naphthalene_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # All 10 atoms should be aromatic
        self.assertTrue(all(aromatic_atoms))

    def test_cyclobutadiene_not_aromatic(self):
        """Test that cyclobutadiene is not aromatic (4 electrons, antiaromatic)."""
        graph = make_cyclobutadiene_graph()

        # 4-membered ring is excluded by size check
        rings = get_aromatic_rings(graph)
        self.assertEqual(len(rings), 0)

    def test_cyclohexane_not_aromatic(self):
        """Test that cyclohexane is not aromatic (sp3 carbons)."""
        graph = make_cyclohexane_graph()
        aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)

        # sp3 carbons with degree 4 are not aromatic
        # Ring atoms have degree 4 (2 C-C + 2 C-H)
        ring_atoms_aromatic = aromatic_atoms[:6]
        self.assertFalse(any(ring_atoms_aromatic))


class PiElectronCountTestCase(unittest.TestCase):
    """Test cases for π electron counting."""

    def test_benzene_pi_electrons(self):
        """Benzene should have 6 π electrons."""
        graph = make_benzene_graph()
        ring = tuple(range(6))
        pi_electrons = count_pi_electrons(graph, ring)

        self.assertEqual(pi_electrons, 6)

    def test_pyridine_pi_electrons(self):
        """Pyridine should have 6 π electrons."""
        graph = make_pyridine_graph()
        ring = tuple(range(6))
        pi_electrons = count_pi_electrons(graph, ring)

        # N (degree 2) contributes 1, 5 carbons contribute 1 each = 6
        self.assertEqual(pi_electrons, 6)

    def test_pyrrole_pi_electrons(self):
        """Pyrrole should have 6 π electrons."""
        graph = make_pyrrole_graph()
        ring = tuple(range(5))  # Ring atoms only
        pi_electrons = count_pi_electrons(graph, ring)

        # N (degree 3 with H) contributes 2, 4 carbons contribute 1 each = 6
        self.assertEqual(pi_electrons, 6)

    def test_furan_pi_electrons(self):
        """Furan should have 6 π electrons."""
        graph = make_furan_graph()
        ring = tuple(range(5))
        pi_electrons = count_pi_electrons(graph, ring)

        # O (degree 2) contributes 2, 4 carbons contribute 1 each = 6
        self.assertEqual(pi_electrons, 6)

    def test_thiophene_pi_electrons(self):
        """Thiophene should have 6 π electrons."""
        graph = make_thiophene_graph()
        ring = tuple(range(5))
        pi_electrons = count_pi_electrons(graph, ring)

        # S (degree 2) contributes 2, 4 carbons contribute 1 each = 6
        self.assertEqual(pi_electrons, 6)


class AromaticRingQueryTestCase(unittest.TestCase):
    """Test cases for aromatic ring queries."""

    def test_get_aromatic_rings_benzene(self):
        """Test getting aromatic rings from benzene."""
        graph = make_benzene_graph()
        rings = get_aromatic_rings(graph)

        self.assertEqual(len(rings), 1)
        self.assertEqual(len(rings[0]), 6)

    def test_get_aromatic_rings_naphthalene(self):
        """Test getting aromatic rings from naphthalene."""
        graph = make_naphthalene_graph()
        rings = get_aromatic_rings(graph)

        self.assertEqual(len(rings), 2)

    def test_is_aromatic_atom_benzene(self):
        """Test is_aromatic_atom for benzene."""
        graph = make_benzene_graph()

        for i in range(6):
            self.assertTrue(is_aromatic_atom(graph, i))

    def test_is_aromatic_bond_benzene(self):
        """Test is_aromatic_bond for benzene."""
        graph = make_benzene_graph()

        # All ring bonds are aromatic
        self.assertTrue(is_aromatic_bond(graph, 0, 1))
        self.assertTrue(is_aromatic_bond(graph, 1, 2))

    def test_is_aromatic_ring_benzene(self):
        """Test is_aromatic_ring directly."""
        graph = make_benzene_graph()
        ring = tuple(range(6))

        self.assertTrue(is_aromatic_ring(graph, ring))

    def test_is_aromatic_ring_cyclobutadiene(self):
        """Test that cyclobutadiene ring is not aromatic."""
        graph = make_cyclobutadiene_graph()
        ring = tuple(range(4))

        # 4-membered ring excluded by size
        self.assertFalse(is_aromatic_ring(graph, ring))


class HuckelRuleTestCase(unittest.TestCase):
    """Test cases for Hückel's 4n+2 rule."""

    def test_valid_electron_counts(self):
        """Test that valid electron counts are accepted."""
        from chmpy.graph.aromaticity import _satisfies_huckel_rule

        # 4n+2: 2, 6, 10, 14, 18
        self.assertTrue(_satisfies_huckel_rule(2))
        self.assertTrue(_satisfies_huckel_rule(6))
        self.assertTrue(_satisfies_huckel_rule(10))
        self.assertTrue(_satisfies_huckel_rule(14))
        self.assertTrue(_satisfies_huckel_rule(18))

    def test_invalid_electron_counts(self):
        """Test that invalid electron counts are rejected."""
        from chmpy.graph.aromaticity import _satisfies_huckel_rule

        # 4n: 4, 8, 12, 16 (antiaromatic)
        self.assertFalse(_satisfies_huckel_rule(4))
        self.assertFalse(_satisfies_huckel_rule(8))
        self.assertFalse(_satisfies_huckel_rule(12))

        # Odd numbers
        self.assertFalse(_satisfies_huckel_rule(3))
        self.assertFalse(_satisfies_huckel_rule(5))
        self.assertFalse(_satisfies_huckel_rule(7))


if __name__ == "__main__":
    unittest.main()

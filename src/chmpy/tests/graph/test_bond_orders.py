"""Tests for bond order perception."""

import logging
import unittest

import numpy as np

from chmpy.graph import MolecularGraph
from chmpy.graph.bond_orders import (
    get_bond_order,
    hydrogen_count,
    implicit_hydrogen_count,
    is_double_bond,
    is_single_bond,
    is_triple_bond,
    perceive_bond_orders,
    total_bond_order,
)

LOG = logging.getLogger(__name__)


def make_ethane_graph() -> MolecularGraph:
    """Create ethane (C2H6) with C-C single bond (~1.54 Å)."""
    atomic_numbers = np.array([6, 6, 1, 1, 1, 1, 1, 1])  # 2C + 6H
    positions = np.array([
        [0.0, 0.0, 0.0],      # C1
        [1.54, 0.0, 0.0],     # C2 (C-C single bond)
        [-0.5, 0.9, 0.0],     # H
        [-0.5, -0.9, 0.0],    # H
        [-0.5, 0.0, 0.9],     # H
        [2.04, 0.9, 0.0],     # H
        [2.04, -0.9, 0.0],    # H
        [2.04, 0.0, 0.9],     # H
    ])
    edges = [
        (0, 1),  # C-C
        (0, 2), (0, 3), (0, 4),  # C-H
        (1, 5), (1, 6), (1, 7),  # C-H
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_ethene_graph() -> MolecularGraph:
    """Create ethene (C2H4) with C=C double bond (~1.34 Å)."""
    atomic_numbers = np.array([6, 6, 1, 1, 1, 1])  # 2C + 4H
    positions = np.array([
        [0.0, 0.0, 0.0],      # C1
        [1.34, 0.0, 0.0],     # C2 (C=C double bond)
        [-0.5, 0.9, 0.0],     # H
        [-0.5, -0.9, 0.0],    # H
        [1.84, 0.9, 0.0],     # H
        [1.84, -0.9, 0.0],    # H
    ])
    edges = [
        (0, 1),  # C=C
        (0, 2), (0, 3),  # C-H
        (1, 4), (1, 5),  # C-H
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_ethyne_graph() -> MolecularGraph:
    """Create ethyne/acetylene (C2H2) with C≡C triple bond (~1.20 Å)."""
    atomic_numbers = np.array([6, 6, 1, 1])  # 2C + 2H
    positions = np.array([
        [0.0, 0.0, 0.0],      # C1
        [1.20, 0.0, 0.0],     # C2 (C≡C triple bond)
        [-1.06, 0.0, 0.0],    # H
        [2.26, 0.0, 0.0],     # H
    ])
    edges = [
        (0, 1),  # C≡C
        (0, 2),  # C-H
        (1, 3),  # C-H
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_benzene_graph() -> MolecularGraph:
    """Create benzene (C6H6) with aromatic C-C bonds (~1.40 Å)."""
    # Benzene carbons at aromatic distance
    r = 1.40
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    carbon_pos = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(6)])

    # Hydrogens
    h_r = 2.48  # Slightly further out
    hydrogen_pos = np.column_stack([
        h_r * np.cos(angles), h_r * np.sin(angles), np.zeros(6)
    ])

    atomic_numbers = np.array([6] * 6 + [1] * 6)
    positions = np.vstack([carbon_pos, hydrogen_pos])

    edges = []
    # C-C ring bonds
    for i in range(6):
        edges.append((i, (i + 1) % 6))
    # C-H bonds
    for i in range(6):
        edges.append((i, i + 6))

    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_formaldehyde_graph() -> MolecularGraph:
    """Create formaldehyde (CH2O) with C=O double bond (~1.21 Å)."""
    atomic_numbers = np.array([6, 8, 1, 1])  # C, O, 2H
    positions = np.array([
        [0.0, 0.0, 0.0],      # C
        [1.21, 0.0, 0.0],     # O (C=O)
        [-0.5, 0.9, 0.0],     # H
        [-0.5, -0.9, 0.0],    # H
    ])
    edges = [
        (0, 1),  # C=O
        (0, 2), (0, 3),  # C-H
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_methanol_graph() -> MolecularGraph:
    """Create methanol (CH3OH) with C-O single bond (~1.43 Å)."""
    atomic_numbers = np.array([6, 8, 1, 1, 1, 1])  # C, O, 4H
    positions = np.array([
        [0.0, 0.0, 0.0],      # C
        [1.43, 0.0, 0.0],     # O (C-O)
        [-0.5, 0.9, 0.0],     # H on C
        [-0.5, -0.9, 0.0],    # H on C
        [-0.5, 0.0, 0.9],     # H on C
        [1.93, 0.9, 0.0],     # H on O
    ])
    edges = [
        (0, 1),  # C-O
        (0, 2), (0, 3), (0, 4),  # C-H
        (1, 5),  # O-H
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


class BondOrderHeuristicTestCase(unittest.TestCase):
    """Test cases for bond order perception from geometry."""

    def test_ethane_single_bond(self):
        """Test single C-C bond detection in ethane."""
        graph = make_ethane_graph()
        order = get_bond_order(graph, 0, 1)

        self.assertAlmostEqual(order, 1.0, places=0)

    def test_ethene_double_bond(self):
        """Test double C=C bond detection in ethene."""
        graph = make_ethene_graph()
        order = get_bond_order(graph, 0, 1)

        self.assertAlmostEqual(order, 2.0, places=0)

    def test_ethyne_triple_bond(self):
        """Test triple C≡C bond detection in ethyne."""
        graph = make_ethyne_graph()
        order = get_bond_order(graph, 0, 1)

        self.assertAlmostEqual(order, 3.0, places=0)

    def test_formaldehyde_double_bond(self):
        """Test double C=O bond detection in formaldehyde."""
        graph = make_formaldehyde_graph()
        order = get_bond_order(graph, 0, 1)

        self.assertAlmostEqual(order, 2.0, places=0)

    def test_methanol_single_bond(self):
        """Test single C-O bond detection in methanol."""
        graph = make_methanol_graph()
        order = get_bond_order(graph, 0, 1)

        self.assertAlmostEqual(order, 1.0, places=0)


class BondTypePredicatesTestCase(unittest.TestCase):
    """Test cases for bond type predicates."""

    def test_is_single_bond(self):
        """Test is_single_bond predicate."""
        graph = make_ethane_graph()
        self.assertTrue(is_single_bond(graph, 0, 1))

        graph = make_ethene_graph()
        self.assertFalse(is_single_bond(graph, 0, 1))

    def test_is_double_bond(self):
        """Test is_double_bond predicate."""
        graph = make_ethene_graph()
        self.assertTrue(is_double_bond(graph, 0, 1))

        graph = make_ethane_graph()
        self.assertFalse(is_double_bond(graph, 0, 1))

    def test_is_triple_bond(self):
        """Test is_triple_bond predicate."""
        graph = make_ethyne_graph()
        self.assertTrue(is_triple_bond(graph, 0, 1))

        graph = make_ethene_graph()
        self.assertFalse(is_triple_bond(graph, 0, 1))


class BondOrderPerceptionTestCase(unittest.TestCase):
    """Test cases for full bond order perception."""

    def test_perceive_ethane(self):
        """Test perceive_bond_orders on ethane."""
        graph = make_ethane_graph()
        bond_orders = perceive_bond_orders(graph)

        # C-C bond
        self.assertAlmostEqual(bond_orders[0, 1], 1.0, places=0)
        # C-H bonds
        self.assertAlmostEqual(bond_orders[0, 2], 1.0, places=0)

    def test_perceive_ethene(self):
        """Test perceive_bond_orders on ethene."""
        graph = make_ethene_graph()
        bond_orders = perceive_bond_orders(graph)

        # C=C bond
        self.assertAlmostEqual(bond_orders[0, 1], 2.0, places=0)

    def test_perceive_benzene_aromatic(self):
        """Test perceive_bond_orders recognizes aromatic bonds."""
        graph = make_benzene_graph()
        bond_orders = perceive_bond_orders(graph)

        # C-C aromatic bonds should be 1.5
        for i in range(6):
            j = (i + 1) % 6
            order = bond_orders[i, j]
            self.assertAlmostEqual(order, 1.5, places=1)


class ValenceCalculationTestCase(unittest.TestCase):
    """Test cases for valence calculations."""

    def test_total_bond_order_carbon(self):
        """Test total bond order calculation for carbon."""
        graph = make_ethane_graph()
        valence = total_bond_order(graph, 0)

        # Carbon in ethane: 1 C-C + 3 C-H = 4
        self.assertAlmostEqual(valence, 4.0, places=0)

    def test_hydrogen_count(self):
        """Test explicit hydrogen counting."""
        graph = make_ethane_graph()
        h_count = hydrogen_count(graph, 0)

        # Each carbon in ethane has 3 hydrogens
        self.assertEqual(h_count, 3)

    def test_implicit_hydrogen_count(self):
        """Test implicit hydrogen calculation."""
        # Create graph without explicit hydrogens
        atomic_numbers = np.array([6, 6])
        positions = np.array([[0, 0, 0], [1.54, 0, 0]])
        edges = [(0, 1)]
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, edges)

        implicit_h = implicit_hydrogen_count(graph, 0)

        # Carbon with one single bond should have 3 implicit H
        self.assertEqual(implicit_h, 3)


class EdgeCasesTestCase(unittest.TestCase):
    """Test edge cases for bond order perception."""

    def test_no_bond(self):
        """Test get_bond_order returns 0 for non-bonded atoms."""
        graph = make_ethane_graph()
        order = get_bond_order(graph, 0, 5)  # Non-adjacent atoms

        self.assertEqual(order, 0.0)

    def test_hydrogen_always_single(self):
        """Test that hydrogen bonds are always single."""
        graph = make_ethane_graph()

        # C-H bonds
        for h_idx in range(2, 8):
            c_idx = 0 if h_idx < 5 else 1
            order = get_bond_order(graph, c_idx, h_idx)
            self.assertAlmostEqual(order, 1.0, places=0)


if __name__ == "__main__":
    unittest.main()

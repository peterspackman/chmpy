"""Tests for ring detection algorithms."""

import logging
import unittest

import numpy as np

from chmpy.graph import MolecularGraph
from chmpy.graph.rings import (
    find_all_rings,
    find_sssr,
    fused_ring_systems,
    is_in_ring,
    is_ring_bond,
    ring_membership,
    ring_sizes,
    smallest_ring_containing_atom,
)

LOG = logging.getLogger(__name__)


def make_benzene_graph() -> MolecularGraph:
    """Create a benzene molecular graph (6-membered ring)."""
    # C6H6 - just the carbons for simplicity
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


def make_naphthalene_graph() -> MolecularGraph:
    """Create a naphthalene molecular graph (2 fused 6-membered rings)."""
    # C10H8 - just the carbons
    atomic_numbers = np.array([6] * 10)
    positions = np.zeros((10, 3))  # Positions not critical for ring detection
    # Naphthalene: atoms 0-5 form first ring, atoms 4-9 form second ring
    # Shared bond is 4-5 (renumbered: 4-9 and 5-0 of second ring)
    #   1 - 2
    #  /     \
    # 0       3 - 8
    #  \     / \ /
    #   5 - 4   9
    #        \ /
    #         7 - 6
    # Actually simpler numbering:
    # Ring 1: 0-1-2-3-4-9-0
    # Ring 2: 4-5-6-7-8-9-4
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 9), (9, 0),  # First ring
        (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),  # Second ring (shared edge 4-9)
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_cubane_graph() -> MolecularGraph:
    """Create a cubane molecular graph (8 vertices, 12 edges, forms cube)."""
    # C8H8 - just the carbons (cube vertices)
    atomic_numbers = np.array([6] * 8)
    # Cube vertices
    positions = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    # Cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_cyclopropane_graph() -> MolecularGraph:
    """Create a cyclopropane graph (3-membered ring)."""
    atomic_numbers = np.array([6, 6, 6])
    positions = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
    edges = [(0, 1), (1, 2), (2, 0)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_acyclic_graph() -> MolecularGraph:
    """Create a simple acyclic (chain) graph."""
    # Propane: C-C-C
    atomic_numbers = np.array([6, 6, 6])
    positions = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    edges = [(0, 1), (1, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_spiro_graph() -> MolecularGraph:
    """Create a spiro compound (two rings sharing one atom)."""
    # Spiro[4.4]nonane-like: two 5-membered rings sharing one carbon
    atomic_numbers = np.array([6] * 9)
    positions = np.zeros((9, 3))
    # Ring 1: 0-1-2-3-4-0
    # Ring 2: 0-5-6-7-8-0
    # Atom 0 is shared
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # First ring
        (0, 5), (5, 6), (6, 7), (7, 8), (8, 0),  # Second ring
    ]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


class RingDetectionTestCase(unittest.TestCase):
    """Test cases for ring detection."""

    def test_benzene_sssr(self):
        """Test SSSR for benzene (one 6-membered ring)."""
        graph = make_benzene_graph()
        rings = find_sssr(graph)

        self.assertEqual(len(rings), 1)
        self.assertEqual(len(rings[0]), 6)

    def test_naphthalene_sssr(self):
        """Test SSSR for naphthalene (two fused 6-membered rings)."""
        graph = make_naphthalene_graph()
        rings = find_sssr(graph)

        # SSSR should have 2 rings
        self.assertEqual(len(rings), 2)
        # Both should be 6-membered
        sizes = sorted(len(r) for r in rings)
        self.assertEqual(sizes, [6, 6])

    def test_cubane_sssr(self):
        """Test SSSR for cubane (cube has 5 independent faces for SSSR)."""
        graph = make_cubane_graph()
        rings = find_sssr(graph)

        # Cubane: V=8, E=12, SSSR = E - V + 1 = 12 - 8 + 1 = 5
        self.assertEqual(len(rings), 5)
        # All rings should be 4-membered (square faces)
        for ring in rings:
            self.assertEqual(len(ring), 4)

    def test_cyclopropane_sssr(self):
        """Test SSSR for cyclopropane (one 3-membered ring)."""
        graph = make_cyclopropane_graph()
        rings = find_sssr(graph)

        self.assertEqual(len(rings), 1)
        self.assertEqual(len(rings[0]), 3)

    def test_acyclic_sssr(self):
        """Test SSSR for acyclic molecule (no rings)."""
        graph = make_acyclic_graph()
        rings = find_sssr(graph)

        self.assertEqual(len(rings), 0)

    def test_spiro_sssr(self):
        """Test SSSR for spiro compound (two rings sharing one atom)."""
        graph = make_spiro_graph()
        rings = find_sssr(graph)

        # Should have 2 rings
        self.assertEqual(len(rings), 2)
        # Both 5-membered
        sizes = sorted(len(r) for r in rings)
        self.assertEqual(sizes, [5, 5])

    def test_find_all_rings_benzene(self):
        """Test finding all rings in benzene."""
        graph = make_benzene_graph()
        rings = find_all_rings(graph, max_size=12)

        # Benzene has only one unique ring
        self.assertEqual(len(rings), 1)

    def test_find_all_rings_cubane(self):
        """Test finding all rings in cubane."""
        graph = make_cubane_graph()
        rings = find_all_rings(graph, max_size=12)

        # Cubane has 6 faces (4-membered rings)
        # Plus larger rings from combining faces
        self.assertGreaterEqual(len(rings), 6)

    def test_is_in_ring_benzene(self):
        """Test is_in_ring for benzene atoms."""
        graph = make_benzene_graph()

        # All atoms should be in a ring
        for i in range(6):
            self.assertTrue(is_in_ring(graph, i))

    def test_is_in_ring_acyclic(self):
        """Test is_in_ring for acyclic molecule."""
        graph = make_acyclic_graph()

        # No atoms in rings
        for i in range(3):
            self.assertFalse(is_in_ring(graph, i))

    def test_ring_membership_naphthalene(self):
        """Test ring membership counts for naphthalene."""
        graph = make_naphthalene_graph()
        membership = ring_membership(graph)

        # Atoms 4 and 9 are shared between rings (membership = 2)
        # Others have membership = 1
        self.assertEqual(membership[4], 2)
        self.assertEqual(membership[9], 2)

        # Non-shared atoms
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            self.assertEqual(membership[i], 1)

    def test_ring_sizes_cubane(self):
        """Test ring sizes for cubane."""
        graph = make_cubane_graph()
        sizes = ring_sizes(graph)

        # All 5 SSSR rings should be 4-membered
        self.assertEqual(sizes, [4, 4, 4, 4, 4])

    def test_is_ring_bond_benzene(self):
        """Test is_ring_bond for benzene."""
        graph = make_benzene_graph()

        # All bonds in benzene are ring bonds
        self.assertTrue(is_ring_bond(graph, 0, 1))
        self.assertTrue(is_ring_bond(graph, 1, 2))
        self.assertTrue(is_ring_bond(graph, 5, 0))

        # Non-existent bond
        self.assertFalse(is_ring_bond(graph, 0, 3))

    def test_smallest_ring_containing_atom(self):
        """Test finding smallest ring containing an atom."""
        graph = make_benzene_graph()
        ring = smallest_ring_containing_atom(graph, 0)

        self.assertIsNotNone(ring)
        self.assertEqual(len(ring), 6)
        self.assertIn(0, ring)

    def test_smallest_ring_acyclic(self):
        """Test smallest_ring for atom not in any ring."""
        graph = make_acyclic_graph()
        ring = smallest_ring_containing_atom(graph, 0)

        self.assertIsNone(ring)

    def test_fused_ring_systems_naphthalene(self):
        """Test fused ring system detection for naphthalene."""
        graph = make_naphthalene_graph()
        systems = fused_ring_systems(graph)

        # Naphthalene has one fused ring system containing all atoms
        self.assertEqual(len(systems), 1)
        self.assertEqual(len(systems[0]), 10)

    def test_fused_ring_systems_spiro(self):
        """Test fused ring system detection for spiro compound."""
        graph = make_spiro_graph()
        systems = fused_ring_systems(graph)

        # Spiro compound: rings share only one atom, not fused
        # So should have 2 separate "systems" (each ring is its own system)
        self.assertEqual(len(systems), 2)

    def test_fused_ring_systems_acyclic(self):
        """Test fused ring systems for acyclic molecule."""
        graph = make_acyclic_graph()
        systems = fused_ring_systems(graph)

        self.assertEqual(len(systems), 0)


class RingEdgeCasesTestCase(unittest.TestCase):
    """Test edge cases for ring detection."""

    def test_single_atom(self):
        """Test graph with single atom (no rings)."""
        atomic_numbers = np.array([6])
        positions = np.array([[0, 0, 0]])
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, [])

        rings = find_sssr(graph)
        self.assertEqual(len(rings), 0)

    def test_two_atoms_bonded(self):
        """Test graph with two bonded atoms (no rings)."""
        atomic_numbers = np.array([6, 6])
        positions = np.array([[0, 0, 0], [1.5, 0, 0]])
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, [(0, 1)])

        rings = find_sssr(graph)
        self.assertEqual(len(rings), 0)

    def test_disconnected_rings(self):
        """Test two disconnected rings."""
        # Two separate cyclopropanes
        atomic_numbers = np.array([6] * 6)
        positions = np.zeros((6, 3))
        edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        graph = MolecularGraph.from_edge_list(atomic_numbers, positions, edges)

        rings = find_sssr(graph)
        self.assertEqual(len(rings), 2)


if __name__ == "__main__":
    unittest.main()

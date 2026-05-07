"""Tests for substructure matching."""

import logging
import unittest

import numpy as np

from chmpy.graph import MolecularGraph
from chmpy.graph.substructure import (
    count_substructures,
    find_functional_groups,
    find_substructure,
    functional_group_matches,
    has_substructure,
    list_functional_groups,
)

LOG = logging.getLogger(__name__)


def make_benzene_graph() -> MolecularGraph:
    """Create benzene C6."""
    atomic_numbers = np.array([6, 6, 6, 6, 6, 6])
    positions = np.zeros((6, 3))
    edges = [(i, (i + 1) % 6) for i in range(6)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_propane_graph() -> MolecularGraph:
    """Create propane C-C-C."""
    atomic_numbers = np.array([6, 6, 6])
    positions = np.zeros((3, 3))
    edges = [(0, 1), (1, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_ethane_graph() -> MolecularGraph:
    """Create ethane C-C."""
    atomic_numbers = np.array([6, 6])
    positions = np.zeros((2, 3))
    edges = [(0, 1)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_methanol_graph() -> MolecularGraph:
    """Create methanol C-O-H."""
    atomic_numbers = np.array([6, 8, 1])
    positions = np.zeros((3, 3))
    edges = [(0, 1), (1, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_ethanol_graph() -> MolecularGraph:
    """Create ethanol C-C-O-H."""
    atomic_numbers = np.array([6, 6, 8, 1])
    positions = np.zeros((4, 3))
    edges = [(0, 1), (1, 2), (2, 3)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_water_graph() -> MolecularGraph:
    """Create water O-H, O-H."""
    atomic_numbers = np.array([8, 1, 1])
    positions = np.zeros((3, 3))
    edges = [(0, 1), (0, 2)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


def make_single_carbon() -> MolecularGraph:
    """Create single carbon atom."""
    atomic_numbers = np.array([6])
    positions = np.zeros((1, 3))
    return MolecularGraph.from_edge_list(atomic_numbers, positions, [])


def make_ammonia_graph() -> MolecularGraph:
    """Create ammonia N-H with 3 hydrogens."""
    atomic_numbers = np.array([7, 1, 1, 1])
    positions = np.zeros((4, 3))
    edges = [(0, 1), (0, 2), (0, 3)]
    return MolecularGraph.from_edge_list(atomic_numbers, positions, edges)


class BasicSubstructureTestCase(unittest.TestCase):
    """Test cases for basic substructure matching."""

    def test_find_ethane_in_propane(self):
        """Ethane should be found in propane (as C1-C2 and C2-C3)."""
        target = make_propane_graph()
        query = make_ethane_graph()

        matches = find_substructure(target, query)

        # Propane C1-C2-C3 contains C1-C2 and C2-C3
        # Each bond can be matched in 2 directions, so 2 bonds * 2 = 4 matches
        self.assertEqual(len(matches), 4)

    def test_has_ethane_in_propane(self):
        """has_substructure should find ethane in propane."""
        target = make_propane_graph()
        query = make_ethane_graph()

        self.assertTrue(has_substructure(target, query))

    def test_no_match_larger_query(self):
        """Query larger than target should not match."""
        target = make_ethane_graph()
        query = make_propane_graph()

        matches = find_substructure(target, query)

        self.assertEqual(len(matches), 0)
        self.assertFalse(has_substructure(target, query))

    def test_empty_query_matches_everything(self):
        """Empty query should match with empty mapping."""
        target = make_propane_graph()
        atomic_numbers = np.array([], dtype=np.int32)
        positions = np.zeros((0, 3))
        query = MolecularGraph.from_edge_list(atomic_numbers, positions, [])

        matches = find_substructure(target, query)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], {})

    def test_single_atom_query(self):
        """Single atom query should match each same-element atom."""
        target = make_propane_graph()  # 3 carbons
        query = make_single_carbon()

        matches = find_substructure(target, query)

        # Each carbon should match
        self.assertEqual(len(matches), 3)


class ElementMatchingTestCase(unittest.TestCase):
    """Test cases for element-aware matching."""

    def test_element_matching(self):
        """Different elements should not match."""
        target = make_methanol_graph()  # C-O-H
        query = make_ethane_graph()  # C-C

        # With element matching, C-C should not match C-O
        matches = find_substructure(target, query, match_element=True)
        self.assertEqual(len(matches), 0)

    def test_element_matching_disabled(self):
        """Without element matching, topology should match."""
        target = make_methanol_graph()  # C-O-H
        query = make_ethane_graph()  # C-C

        # Without element matching, C-C matches C-O
        matches = find_substructure(target, query, match_element=False)
        self.assertGreater(len(matches), 0)


class RingSubstructureTestCase(unittest.TestCase):
    """Test cases for ring substructure matching."""

    def test_find_chain_in_ring(self):
        """Linear chain should be found in ring."""
        target = make_benzene_graph()  # 6-membered ring
        query = make_propane_graph()  # C-C-C chain

        matches = find_substructure(target, query)

        # Should find multiple chains in the ring
        self.assertGreater(len(matches), 0)

    def test_find_edge_in_ring(self):
        """Edge should be found in ring."""
        target = make_benzene_graph()
        query = make_ethane_graph()

        matches = find_substructure(target, query)

        # Benzene has 6 edges, each matchable in 2 directions = 12 matches
        self.assertEqual(len(matches), 12)


class CountSubstructuresTestCase(unittest.TestCase):
    """Test cases for counting substructures."""

    def test_count_cc_bonds_in_propane(self):
        """Count C-C bonds (embeddings) in propane."""
        target = make_propane_graph()
        query = make_ethane_graph()

        count = count_substructures(target, query)

        # 2 bonds * 2 directions = 4 embeddings
        self.assertEqual(count, 4)

    def test_count_cc_bonds_in_benzene(self):
        """Count C-C bonds (embeddings) in benzene."""
        target = make_benzene_graph()
        query = make_ethane_graph()

        count = count_substructures(target, query)

        # 6 bonds * 2 directions = 12 embeddings
        self.assertEqual(count, 12)


class FunctionalGroupTestCase(unittest.TestCase):
    """Test cases for functional group matching."""

    def test_hydroxyl_in_methanol(self):
        """Find O-H (hydroxyl) in methanol."""
        target = make_methanol_graph()
        matches = functional_group_matches(
            target,
            pattern_atoms=[8, 1],  # O-H
            pattern_bonds=[(0, 1)],
        )

        self.assertEqual(len(matches), 1)

    def test_hydroxyl_in_water(self):
        """Find O-H (hydroxyl) in water."""
        target = make_water_graph()
        matches = functional_group_matches(
            target,
            pattern_atoms=[8, 1],
            pattern_bonds=[(0, 1)],
        )

        # Water has 2 O-H bonds
        self.assertEqual(len(matches), 2)

    def test_amine_in_ammonia(self):
        """Find N-H in ammonia."""
        target = make_ammonia_graph()
        matches = functional_group_matches(
            target,
            pattern_atoms=[7, 1],  # N-H
            pattern_bonds=[(0, 1)],
        )

        # Ammonia has 3 N-H bonds
        self.assertEqual(len(matches), 3)

    def test_find_hydroxyl_by_name(self):
        """Find hydroxyl using named functional group."""
        target = make_ethanol_graph()
        matches = find_functional_groups(target, "hydroxyl")

        self.assertEqual(len(matches), 1)

    def test_list_functional_groups(self):
        """Test listing available functional groups."""
        groups = list_functional_groups()

        self.assertIn("hydroxyl", groups)
        self.assertIn("amine", groups)

    def test_unknown_functional_group(self):
        """Unknown functional group should raise KeyError."""
        target = make_ethanol_graph()

        with self.assertRaises(KeyError):
            find_functional_groups(target, "nonexistent_group")


class MappingValidityTestCase(unittest.TestCase):
    """Test cases for mapping validity."""

    def test_mapping_preserves_bonds(self):
        """Mapping should preserve bond structure."""
        target = make_propane_graph()
        query = make_ethane_graph()

        matches = find_substructure(target, query)

        for mapping in matches:
            # Check that matched atoms are bonded in target
            query_atoms = list(mapping.keys())
            target_atoms = [mapping[q] for q in query_atoms]

            # Query atoms 0 and 1 are bonded
            self.assertTrue(target.has_bond(target_atoms[0], target_atoms[1]))

    def test_mapping_is_injective(self):
        """Mapping should be one-to-one (no repeated target atoms)."""
        target = make_benzene_graph()
        query = make_propane_graph()

        matches = find_substructure(target, query)

        for mapping in matches:
            target_atoms = list(mapping.values())
            # All target atoms should be unique
            self.assertEqual(len(target_atoms), len(set(target_atoms)))


if __name__ == "__main__":
    unittest.main()

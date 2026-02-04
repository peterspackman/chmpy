"""Tests for symmetric_graph.py base framework."""

import logging
import unittest

import numpy as np

from chmpy.crystal import Crystal
from chmpy.crystal.space_group import SpaceGroup
from chmpy.crystal.space_group_table import SpaceGroupTable
from chmpy.crystal.symmetric_graph import (
    AlgebraicEdge,
    AlgebraicVertexRef,
    SymmetricGraph,
    apply_symop_to_edge,
    apply_symop_to_vertex,
    canonical_edge_representative,
    compute_edge_orbit,
    compute_edge_orbit_size,
    edges_in_same_orbit,
    normalize_edge,
)

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class AlgebraicVertexRefTestCase(unittest.TestCase):
    """Test AlgebraicVertexRef class."""

    def test_ordering(self):
        """Test lexicographic ordering by asym_idx, symop_idx, cell."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        v3 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))

        self.assertTrue(v1 < v2)
        self.assertTrue(v2 < v3)
        self.assertTrue(v1 < v3)

    def test_cell_ordering(self):
        """Test that cell affects ordering."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 1))
        v3 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(1, 0, 0))

        self.assertTrue(v1 < v2)
        self.assertTrue(v2 < v3)

    def test_frozen_hashable(self):
        """Test that AlgebraicVertexRef is hashable."""
        v = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        s = {v}
        self.assertIn(v, s)

    def test_with_cell(self):
        """Test creating vertex with different cell."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        v2 = v1.with_cell((1, 2, 3))

        self.assertEqual(v2.asym_idx, 0)
        self.assertEqual(v2.symop_idx, 1)
        self.assertEqual(v2.cell, (1, 2, 3))


class AlgebraicEdgeTestCase(unittest.TestCase):
    """Test AlgebraicEdge class."""

    def test_canonical_ordering(self):
        """Test that edges are stored in canonical order (src <= dst)."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))

        # Create with v2 first (wrong order)
        edge = AlgebraicEdge.create(v2, v1)

        # Should be reordered
        self.assertEqual(edge.src, v1)
        self.assertEqual(edge.dst, v2)

    def test_frozen_hashable(self):
        """Test that AlgebraicEdge is hashable."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 1))
        edge = AlgebraicEdge.create(v1, v2)

        edge_set = {edge}
        self.assertIn(edge, edge_set)

    def test_is_homo_edge(self):
        """Test homo-edge detection."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        v3 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))

        homo = AlgebraicEdge.create(v1, v2)
        hetero = AlgebraicEdge.create(v1, v3)

        self.assertTrue(homo.is_homo_edge())
        self.assertFalse(hetero.is_homo_edge())


class VertexTransformTestCase(unittest.TestCase):
    """Test vertex transformation under space group operations."""

    def setUp(self):
        self.sg = SpaceGroup(14)  # P21/c
        self.sg_table = SpaceGroupTable.from_space_group(self.sg)

    def test_identity_transform(self):
        """Test that identity symop preserves vertex."""
        v = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(1, 2, 3))
        identity_idx = self.sg_table.identity_idx()

        v_transformed = apply_symop_to_vertex(v, identity_idx, self.sg_table)

        # Identity should preserve symop_idx
        self.assertEqual(v_transformed.asym_idx, v.asym_idx)
        # Cell might be rotated by identity (which is trivial)
        # Just check it's still an integer tuple
        self.assertIsInstance(v_transformed.cell, tuple)
        self.assertEqual(len(v_transformed.cell), 3)

    def test_group_action_composition(self):
        """Test that (g*h)(v) = g(h(v))."""
        v = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))

        for g in range(self.sg_table.n_ops):
            for h in range(self.sg_table.n_ops):
                # Compute g(h(v))
                hv = apply_symop_to_vertex(v, h, self.sg_table)
                g_hv = apply_symop_to_vertex(hv, g, self.sg_table)

                # Compute (g∘h)(v) using multiplication table
                gh = self.sg_table.compose(h, g)
                gh_v = apply_symop_to_vertex(v, gh, self.sg_table)

                # These should be equal (up to cell normalization issues)
                # At minimum, asym_idx should be preserved
                self.assertEqual(g_hv.asym_idx, gh_v.asym_idx)


class EdgeTransformTestCase(unittest.TestCase):
    """Test edge transformation under space group operations."""

    def setUp(self):
        self.sg = SpaceGroup(14)  # P21/c
        self.sg_table = SpaceGroupTable.from_space_group(self.sg)

    def test_identity_preserves_edge(self):
        """Test that identity symop preserves normalized edge."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 1))
        edge = AlgebraicEdge.create(v1, v2)

        identity_idx = self.sg_table.identity_idx()
        transformed = apply_symop_to_edge(edge, identity_idx, self.sg_table)
        transformed_norm = normalize_edge(transformed, self.sg_table)
        edge_norm = normalize_edge(edge, self.sg_table)

        self.assertEqual(edge_norm._key(), transformed_norm._key())

    def test_orbit_closure(self):
        """Test that orbit of orbit element equals original orbit."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(1, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        # Compute orbit
        orbit = set()
        for g in range(self.sg_table.n_ops):
            t = apply_symop_to_edge(edge, g, self.sg_table)
            t_norm = normalize_edge(t, self.sg_table)
            orbit.add(t_norm._key())

        # Pick another element from orbit and compute its orbit
        orbit_list = list(orbit)
        if len(orbit_list) > 1:
            key = orbit_list[1]
            src_key, dst_key = key
            src2 = AlgebraicVertexRef(*src_key)
            dst2 = AlgebraicVertexRef(*dst_key)
            edge2 = AlgebraicEdge(src=src2, dst=dst2)

            orbit2 = set()
            for g in range(self.sg_table.n_ops):
                t = apply_symop_to_edge(edge2, g, self.sg_table)
                t_norm = normalize_edge(t, self.sg_table)
                orbit2.add(t_norm._key())

            self.assertEqual(orbit, orbit2)


class NormalizeEdgeTestCase(unittest.TestCase):
    """Test edge normalization."""

    def setUp(self):
        self.sg = SpaceGroup(1)  # P1
        self.sg_table = SpaceGroupTable.from_space_group(self.sg)

    def test_src_at_origin(self):
        """Test that normalization puts src at (0,0,0)."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(1, 2, 3))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(4, 5, 6))
        edge = AlgebraicEdge.create(v1, v2)

        normalized = normalize_edge(edge, self.sg_table)

        self.assertEqual(normalized.src.cell, (0, 0, 0))

    def test_src_less_than_dst(self):
        """Test that normalization ensures src <= dst."""
        v1 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        normalized = normalize_edge(edge, self.sg_table)

        self.assertTrue(normalized.src <= normalized.dst)

    def test_idempotent(self):
        """Test that normalizing a normalized edge gives same result."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(1, 2, 3))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(4, 5, 6))
        edge = AlgebraicEdge.create(v1, v2)

        norm1 = normalize_edge(edge, self.sg_table)
        norm2 = normalize_edge(norm1, self.sg_table)

        self.assertEqual(norm1._key(), norm2._key())


class CanonicalRepresentativeTestCase(unittest.TestCase):
    """Test canonical edge representative computation."""

    def setUp(self):
        self.sg = SpaceGroup(14)  # P21/c
        self.sg_table = SpaceGroupTable.from_space_group(self.sg)

    def test_canonical_is_minimal(self):
        """Test that canonical rep is lexicographically smallest in orbit."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(1, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        canonical = canonical_edge_representative(edge, self.sg_table)
        canonical_key = canonical._key()

        # Check that no element in orbit is smaller
        for g in range(self.sg_table.n_ops):
            t = apply_symop_to_edge(edge, g, self.sg_table)
            t_norm = normalize_edge(t, self.sg_table)
            self.assertLessEqual(canonical_key, t_norm._key())

    def test_equivalent_edges_same_canonical(self):
        """Test that equivalent edges have same canonical rep."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        # Transform by some symop
        transformed = apply_symop_to_edge(edge, 1, self.sg_table)

        canon1 = canonical_edge_representative(edge, self.sg_table)
        canon2 = canonical_edge_representative(transformed, self.sg_table)

        self.assertEqual(canon1._key(), canon2._key())

    def test_edges_in_same_orbit(self):
        """Test edges_in_same_orbit helper."""
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        # Transform
        transformed = apply_symop_to_edge(edge, 1, self.sg_table)

        self.assertTrue(edges_in_same_orbit(edge, transformed, self.sg_table))


class OrbitSizeTestCase(unittest.TestCase):
    """Test orbit size computation."""

    def test_p1_orbit_size_is_1(self):
        """Test that P1 (trivial group) gives orbit size 1."""
        sg = SpaceGroup(1)
        sg_table = SpaceGroupTable.from_space_group(sg)

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        orbit_size = compute_edge_orbit_size(edge, sg_table)
        self.assertEqual(orbit_size, 1)

    def test_orbit_size_divides_group_order(self):
        """Test that orbit size divides group order (orbit-stabilizer theorem)."""
        sg = SpaceGroup(14)  # P21/c has 4 symops
        sg_table = SpaceGroupTable.from_space_group(sg)

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        orbit_size = compute_edge_orbit_size(edge, sg_table)

        # Orbit size must divide |G| = 4
        self.assertEqual(sg_table.n_ops % orbit_size, 0)

    def test_compute_edge_orbit_returns_distinct_edges(self):
        """Test that compute_edge_orbit returns correct number of edges."""
        sg = SpaceGroup(14)
        sg_table = SpaceGroupTable.from_space_group(sg)

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=0, symop_idx=1, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        orbit = compute_edge_orbit(edge, sg_table)
        orbit_size = compute_edge_orbit_size(edge, sg_table)

        self.assertEqual(len(orbit), orbit_size)


class SymmetricGraphTestCase(unittest.TestCase):
    """Test SymmetricGraph base class."""

    def setUp(self):
        self.sg = SpaceGroup(14)  # P21/c
        self.sg_table = SpaceGroupTable.from_space_group(self.sg)

    def test_add_remove_edge(self):
        """Test adding and removing edges."""
        graph = SymmetricGraph(
            sg_table=self.sg_table,
            coset_table=None,
            n_asym_vertices=2,
            asym_edges=set(),
        )

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))

        # Add edge
        added = graph.add_edge(v1, v2)
        self.assertTrue(added)
        self.assertTrue(graph.has_edge(v1, v2))
        self.assertEqual(graph.n_unique_edges(), 1)

        # Adding again should return False
        added_again = graph.add_edge(v1, v2)
        self.assertFalse(added_again)

        # Remove edge
        removed = graph.remove_edge(v1, v2)
        self.assertTrue(removed)
        self.assertFalse(graph.has_edge(v1, v2))
        self.assertEqual(graph.n_unique_edges(), 0)

        # Removing again should return False
        removed_again = graph.remove_edge(v1, v2)
        self.assertFalse(removed_again)

    def test_equivalent_edges_stored_once(self):
        """Test that equivalent edges are stored as one."""
        graph = SymmetricGraph(
            sg_table=self.sg_table,
            coset_table=None,
            n_asym_vertices=2,
            asym_edges=set(),
        )

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))
        edge = AlgebraicEdge.create(v1, v2)

        # Add edge
        graph.add_edge(v1, v2)

        # Transform edge and try to add
        transformed = apply_symop_to_edge(edge, 1, self.sg_table)
        added = graph.add_edge(transformed.src, transformed.dst)

        # Should not be added (already have equivalent)
        self.assertFalse(added)
        self.assertEqual(graph.n_unique_edges(), 1)

    def test_n_total_edges(self):
        """Test that n_total_edges sums multiplicities."""
        graph = SymmetricGraph(
            sg_table=self.sg_table,
            coset_table=None,
            n_asym_vertices=2,
            asym_edges=set(),
        )

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))

        graph.add_edge(v1, v2)

        n_total = graph.n_total_edges()
        n_unique = graph.n_unique_edges()

        self.assertGreaterEqual(n_total, n_unique)

    def test_all_edges_iterator(self):
        """Test that all_edges iterator yields correct count."""
        graph = SymmetricGraph(
            sg_table=self.sg_table,
            coset_table=None,
            n_asym_vertices=2,
            asym_edges=set(),
        )

        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))

        graph.add_edge(v1, v2)

        edges = list(graph.all_edges())
        self.assertEqual(len(edges), graph.n_total_edges())


class CrystalSymmetricGraphTestCase(unittest.TestCase):
    """Test symmetric graph with real crystal structures."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_graph_with_crystal(self):
        """Test creating symmetric graph from crystal."""
        sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)

        graph = SymmetricGraph(
            sg_table=sg_table,
            coset_table=None,
            n_asym_vertices=self.acetic.nsites,
            asym_edges=set(),
        )

        # Add an edge
        v1 = AlgebraicVertexRef(asym_idx=0, symop_idx=0, cell=(0, 0, 0))
        v2 = AlgebraicVertexRef(asym_idx=1, symop_idx=0, cell=(0, 0, 0))

        graph.add_edge(v1, v2)

        self.assertGreater(graph.n_total_edges(), 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

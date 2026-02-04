"""Tests for SpaceGroupTable."""

import logging
import unittest

import numpy as np

from chmpy.crystal.space_group import SpaceGroup
from chmpy.crystal.space_group_table import SpaceGroupTable

LOG = logging.getLogger(__name__)


class SpaceGroupTableTestCase(unittest.TestCase):
    """Test SpaceGroupTable construction and group axioms."""

    def test_construction_p1(self):
        """Test SpaceGroupTable construction for P1."""
        sg = SpaceGroup(1)  # P1
        table = SpaceGroupTable.from_space_group(sg)

        self.assertEqual(table.n_ops, len(sg.symmetry_operations))
        self.assertEqual(table.mult_table.shape, (table.n_ops, table.n_ops))
        self.assertEqual(table.inverse_table.shape, (table.n_ops,))

    def test_construction_p21c(self):
        """Test SpaceGroupTable construction for P21/c."""
        sg = SpaceGroup(14)  # P21/c
        table = SpaceGroupTable.from_space_group(sg)

        self.assertEqual(table.n_ops, len(sg.symmetry_operations))
        self.assertEqual(table.mult_table.shape, (table.n_ops, table.n_ops))

    def test_group_axioms_p1(self):
        """Test group axioms for P1."""
        sg = SpaceGroup(1)
        table = SpaceGroupTable.from_space_group(sg)

        results = table.verify_group_axioms()

        self.assertTrue(results["closure"], "P1 closure failed")
        self.assertTrue(results["identity"], "P1 identity failed")
        self.assertTrue(results["inverses"], "P1 inverses failed")
        self.assertTrue(results["associativity"], "P1 associativity failed")

    def test_group_axioms_p21c(self):
        """Test group axioms for P21/c (monoclinic, 4 symops)."""
        sg = SpaceGroup(14)
        table = SpaceGroupTable.from_space_group(sg)

        results = table.verify_group_axioms()

        self.assertTrue(results["closure"], "P21/c closure failed")
        self.assertTrue(results["identity"], "P21/c identity failed")
        self.assertTrue(results["inverses"], "P21/c inverses failed")
        self.assertTrue(results["associativity"], "P21/c associativity failed")

    def test_group_axioms_p41212(self):
        """Test group axioms for P41212 (tetragonal, 8 symops)."""
        sg = SpaceGroup(92)  # P41212
        table = SpaceGroupTable.from_space_group(sg)

        results = table.verify_group_axioms()

        self.assertTrue(results["closure"], "P41212 closure failed")
        self.assertTrue(results["identity"], "P41212 identity failed")
        self.assertTrue(results["inverses"], "P41212 inverses failed")
        self.assertTrue(results["associativity"], "P41212 associativity failed")

    def test_group_axioms_fm3m(self):
        """Test group axioms for Fm-3m (cubic, 192 symops)."""
        sg = SpaceGroup(225)  # Fm-3m
        table = SpaceGroupTable.from_space_group(sg)

        results = table.verify_group_axioms()

        self.assertTrue(results["closure"], "Fm-3m closure failed")
        self.assertTrue(results["identity"], "Fm-3m identity failed")
        self.assertTrue(results["inverses"], "Fm-3m inverses failed")
        self.assertTrue(results["associativity"], "Fm-3m associativity failed")

    def test_identity_element(self):
        """Test that identity element is correctly identified."""
        sg = SpaceGroup(14)
        table = SpaceGroupTable.from_space_group(sg)

        e = table.identity_idx()

        # e * g = g for all g
        for g in range(table.n_ops):
            self.assertEqual(table.compose(g, e), g)
            self.assertEqual(table.compose(e, g), g)

    def test_inverse_property(self):
        """Test that inverse table is correct."""
        sg = SpaceGroup(14)
        table = SpaceGroupTable.from_space_group(sg)

        e = table.identity_idx()

        # g * g^-1 = e for all g
        for g in range(table.n_ops):
            g_inv = table.inv(g)
            self.assertEqual(table.compose(g, g_inv), e)
            self.assertEqual(table.compose(g_inv, g), e)

    def test_composition_matches_symop(self):
        """Test that table composition matches SymmetryOperation composition."""
        sg = SpaceGroup(14)
        table = SpaceGroupTable.from_space_group(sg)
        symops = sg.symmetry_operations

        for i in range(table.n_ops):
            for j in range(table.n_ops):
                # Compose using table
                k = table.compose(i, j)

                # Compose using SymmetryOperation
                g_i = symops[i]
                g_j = symops[j]
                g_result = g_j.compose(g_i)

                # Result should match symop at index k (up to lattice translation)
                g_k = symops[k]
                self.assertTrue(
                    np.allclose(g_k.rotation, g_result.rotation, atol=1e-10),
                    f"Rotation mismatch for compose({i}, {j})"
                )
                self.assertTrue(
                    np.allclose(g_k.translation % 1, g_result.translation % 1, atol=1e-10),
                    f"Translation mismatch for compose({i}, {j})"
                )

    def test_symop_to_idx_mapping(self):
        """Test that symop_to_idx and idx_to_symop are consistent."""
        sg = SpaceGroup(14)
        table = SpaceGroupTable.from_space_group(sg)

        # Round-trip: idx -> code -> idx
        for i in range(table.n_ops):
            code = table.idx_to_symop[i]
            self.assertEqual(table.symop_to_idx[code], i)

    def test_rotations_array(self):
        """Test that rotations array matches symops."""
        sg = SpaceGroup(14)
        table = SpaceGroupTable.from_space_group(sg)
        symops = sg.symmetry_operations

        for i, s in enumerate(symops):
            self.assertTrue(
                np.allclose(table.rotations[i], s.rotation),
                f"Rotation mismatch at index {i}"
            )


class SymmetryOperationComposeTestCase(unittest.TestCase):
    """Test SymmetryOperation compose and inverse methods."""

    def test_compose_identity(self):
        """Test composition with identity."""
        from chmpy.crystal.symmetry_operation import SymmetryOperation

        e = SymmetryOperation.identity()
        g = SymmetryOperation.from_string_code("-x,y+1/2,-z")

        # e * g = g
        self.assertEqual((e * g).integer_code, g.integer_code)
        # g * e = g
        self.assertEqual((g * e).integer_code, g.integer_code)

    def test_compose_inverse(self):
        """Test composition with inverse gives identity."""
        from chmpy.crystal.symmetry_operation import SymmetryOperation

        g = SymmetryOperation.from_string_code("-x,y+1/2,-z")
        g_inv = g.inverse()

        result = g * g_inv
        e = SymmetryOperation.identity()

        # Result should be identity (up to lattice translation)
        self.assertTrue(np.allclose(result.rotation, e.rotation))
        self.assertTrue(np.allclose(result.translation % 1, e.translation % 1))

    def test_compose_associativity(self):
        """Test that composition is associative."""
        from chmpy.crystal.symmetry_operation import SymmetryOperation

        a = SymmetryOperation.from_string_code("-x,-y,-z")
        b = SymmetryOperation.from_string_code("-x,y+1/2,-z+1/2")
        c = SymmetryOperation.from_string_code("x,-y+1/2,z+1/2")

        # (a * b) * c = a * (b * c)
        ab_c = (a * b) * c
        a_bc = a * (b * c)

        self.assertTrue(np.allclose(ab_c.rotation, a_bc.rotation))
        self.assertTrue(np.allclose(ab_c.translation % 1, a_bc.translation % 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

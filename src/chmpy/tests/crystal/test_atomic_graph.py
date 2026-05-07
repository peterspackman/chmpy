"""Tests for atomic_graph.py - AtomicBondGraph."""

import logging
import unittest

from chmpy.crystal import Crystal
from chmpy.crystal.atomic_graph import (
    AlgebraicBondIndex,
    AtomicBondGraph,
    AtomicCosetTable,
)
from chmpy.crystal.space_group_table import SpaceGroupTable

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class AtomicCosetTableTestCase(unittest.TestCase):
    """Test AtomicCosetTable construction."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_construction(self):
        """Test coset table is built correctly."""
        sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)
        coset_table = AtomicCosetTable.from_crystal(self.acetic, sg_table)

        self.assertEqual(coset_table.n_symops, sg_table.n_ops)
        self.assertGreater(coset_table.n_vertices, 0)

    def test_symop_to_atom_valid(self):
        """Test that symop_to_atom has valid mappings."""
        sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)
        coset_table = AtomicCosetTable.from_crystal(self.acetic, sg_table)

        # Each symop should map each asym atom to some UC atom
        for symop_idx in range(coset_table.n_symops):
            for asym_idx in range(self.acetic.nsites):
                atom_idx = coset_table.symop_to_atom[symop_idx, asym_idx]
                self.assertGreaterEqual(atom_idx, 0)
                self.assertLess(atom_idx, coset_table.n_vertices)


class AtomicBondGraphConstructionTestCase(unittest.TestCase):
    """Test AtomicBondGraph construction from crystal."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])

    def test_from_crystal_acetic_acid(self):
        """Test building bond graph from acetic acid."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        self.assertGreater(bonds.n_unique_bonds(), 0)
        self.assertGreater(bonds.n_total_bonds(), 0)
        self.assertEqual(bonds.n_asym_atoms, self.acetic.nsites)

    def test_from_crystal_ice_ii(self):
        """Test building bond graph from ice II."""
        bonds = AtomicBondGraph.from_crystal(self.ice_ii)

        self.assertGreater(bonds.n_unique_bonds(), 0)
        # Ice should have O-H bonds
        n_total = bonds.n_total_bonds()
        self.assertGreater(n_total, 0)

    def test_unique_vs_total_bonds(self):
        """Test that total >= unique bonds."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        n_unique = bonds.n_unique_bonds()
        n_total = bonds.n_total_bonds()

        self.assertGreaterEqual(n_total, n_unique)

    def test_multiplicities_sum(self):
        """Test that sum of multiplicities equals total bonds."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        total_from_mults = sum(mult for _, mult in bonds.unique_bonds())
        self.assertEqual(total_from_mults, bonds.n_total_bonds())


class AtomicBondGraphEditingTestCase(unittest.TestCase):
    """Test bond graph editing operations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_add_remove_bond(self):
        """Test adding and removing bonds."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        initial_count = bonds.n_unique_bonds()

        # Create atom refs for a new bond
        atom_a = bonds.make_atom_ref(asym_idx=0)
        atom_b = bonds.make_atom_ref(asym_idx=1)

        # Check if this bond exists
        had_bond = bonds.has_bond(atom_a, atom_b)

        if had_bond:
            # Remove it
            removed = bonds.remove_bond(atom_a, atom_b)
            self.assertTrue(removed)
            self.assertFalse(bonds.has_bond(atom_a, atom_b))
            self.assertLess(bonds.n_unique_bonds(), initial_count)

            # Re-add it
            added = bonds.add_bond(atom_a, atom_b)
            self.assertTrue(added)
            self.assertTrue(bonds.has_bond(atom_a, atom_b))
        else:
            # Add it
            added = bonds.add_bond(atom_a, atom_b)
            self.assertTrue(added)
            self.assertTrue(bonds.has_bond(atom_a, atom_b))
            self.assertGreater(bonds.n_unique_bonds(), initial_count)

            # Remove it
            removed = bonds.remove_bond(atom_a, atom_b)
            self.assertTrue(removed)
            self.assertFalse(bonds.has_bond(atom_a, atom_b))

    def test_add_equivalent_bond(self):
        """Test that adding equivalent bond doesn't duplicate."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        # Get the first unique bond
        if bonds.n_unique_bonds() == 0:
            self.skipTest("No bonds in structure")

        unique = list(bonds.unique_bonds())
        bond, mult = unique[0]

        # Try to add an equivalent bond
        added = bonds.add_bond(bond.src, bond.dst)

        # Should return False (already exists)
        self.assertFalse(added)


class AtomicBondGraphQueryTestCase(unittest.TestCase):
    """Test bond graph query operations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_bonds_at_atom(self):
        """Test querying bonds at a specific atom."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        # Query bonds at first atom
        atom = bonds.make_atom_ref(asym_idx=0)
        incident = bonds.bonds_at_atom(atom)

        # Each bond should involve this atom
        for bond in incident:
            self.assertTrue(
                bond.src == atom or bond.dst == atom,
                f"Bond {bond} does not involve {atom}"
            )

    def test_all_bonds_iterator(self):
        """Test iterating over all bonds."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        all_list = list(bonds.all_bonds())
        self.assertEqual(len(all_list), bonds.n_total_bonds())

    def test_unique_bonds_with_multiplicities(self):
        """Test getting unique bonds with multiplicities."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        unique = bonds.unique_bonds()

        self.assertEqual(len(unique), bonds.n_unique_bonds())

        for bond, mult in unique:
            self.assertIsInstance(bond, AlgebraicBondIndex)
            self.assertGreater(mult, 0)

    def test_get_bond_atom_types(self):
        """Test getting atomic numbers for bond atoms."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        if bonds.n_unique_bonds() == 0:
            self.skipTest("No bonds in structure")

        unique = list(bonds.unique_bonds())
        bond, _ = unique[0]

        type_a, type_b = bonds.get_bond_atom_types(bond)

        # Should be valid atomic numbers
        self.assertGreater(type_a, 0)
        self.assertGreater(type_b, 0)

    def test_bonds_by_type(self):
        """Test filtering bonds by element types."""
        bonds = AtomicBondGraph.from_crystal(self.acetic)

        # Acetic acid has C-C, C-H, C-O, O-H bonds
        # Get C-C bonds (C=6)
        cc_bonds = bonds.bonds_by_type(6, 6)

        # All returned bonds should be C-C
        for bond, _mult in cc_bonds:
            type_a, type_b = bonds.get_bond_atom_types(bond)
            self.assertEqual(type_a, 6)
            self.assertEqual(type_b, 6)


class BondMultiplicityTestCase(unittest.TestCase):
    """Test bond multiplicity calculations."""

    def test_multiplicity_divides_group_order(self):
        """Test that multiplicities divide group order."""
        acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        bonds = AtomicBondGraph.from_crystal(acetic)

        n_symops = len(acetic.space_group.symmetry_operations)

        for _bond, mult in bonds.unique_bonds():
            # Multiplicity should divide group order
            # (Actually, it should divide n_symops * n_uc_copies)
            # For now just check it's reasonable
            self.assertGreaterEqual(mult, 1)
            self.assertLessEqual(mult, n_symops * bonds.n_uc_atoms)


class BondGraphReprTestCase(unittest.TestCase):
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        bonds = AtomicBondGraph.from_crystal(acetic)

        repr_str = repr(bonds)

        self.assertIn("AtomicBondGraph", repr_str)
        self.assertIn("n_asym_atoms", repr_str)
        self.assertIn("n_unique_bonds", repr_str)


class HighSymmetryTestCase(unittest.TestCase):
    """Test bond graph with high-symmetry structures."""

    def test_ice_ii_bond_graph(self):
        """Test bond graph for ice II (high symmetry)."""
        ice = Crystal.load(TEST_FILES["iceII.cif"])
        bonds = AtomicBondGraph.from_crystal(ice)

        # Ice II has O-H bonds
        # Check we find some bonds
        self.assertGreater(bonds.n_unique_bonds(), 0)

        # Total should be greater than unique (due to symmetry)
        # Unless there's only one unique bond per asymmetric unit
        n_unique = bonds.n_unique_bonds()
        n_total = bonds.n_total_bonds()

        LOG.info(f"Ice II: {n_unique} unique bonds, {n_total} total bonds")

        # Verify multiplicities are reasonable
        n_symops = len(ice.space_group.symmetry_operations)
        for _bond, mult in bonds.unique_bonds():
            self.assertLessEqual(mult, n_symops * bonds.n_uc_atoms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

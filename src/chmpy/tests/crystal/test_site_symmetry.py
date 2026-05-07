"""Tests for SiteSymmetryTable."""

import logging
import unittest

import numpy as np

from chmpy.core.element import Element
from chmpy.crystal import Crystal, SpaceGroup, UnitCell
from chmpy.crystal.asymmetric_unit import AsymmetricUnit
from chmpy.crystal.site_symmetry import SiteSymmetry, SiteSymmetryTable

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class SiteSymmetryTestCase(unittest.TestCase):
    """Test SiteSymmetry class."""

    def test_general_position(self):
        """Test site symmetry detection for general position."""
        # P1 space group - all positions are general
        sg = SpaceGroup(1)
        symops = sg.symmetry_operations

        position = np.array([0.123, 0.456, 0.789])
        site_sym = SiteSymmetry.from_position(position, symops)

        # Only identity in stabilizer
        self.assertEqual(site_sym.n_stabilizer, 1)
        self.assertTrue(site_sym.is_general_position)
        self.assertEqual(site_sym.multiplicity, 1)

    def test_special_position_inversion(self):
        """Test site symmetry for a site on inversion center."""
        # P-1 (space group 2) has inversion centers at (0,0,0), (0.5,0,0), etc.
        sg = SpaceGroup(2)
        symops = sg.symmetry_operations

        # Origin - should have inversion as stabilizer
        position = np.array([0.0, 0.0, 0.0])
        site_sym = SiteSymmetry.from_position(position, symops)

        # Should have both identity and inversion in stabilizer
        self.assertEqual(site_sym.n_stabilizer, 2)
        self.assertFalse(site_sym.is_general_position)
        self.assertEqual(site_sym.multiplicity, 1)  # |G|/|stabilizer| = 2/2 = 1

    def test_special_position_half_coords(self):
        """Test site symmetry for site at (0.5, 0, 0) in P-1."""
        sg = SpaceGroup(2)
        symops = sg.symmetry_operations

        # (0.5, 0, 0) is also an inversion center in P-1
        position = np.array([0.5, 0.0, 0.0])
        site_sym = SiteSymmetry.from_position(position, symops)

        # Should have both identity and inversion in stabilizer
        self.assertEqual(site_sym.n_stabilizer, 2)
        self.assertEqual(site_sym.multiplicity, 1)


class SiteSymmetryTableTestCase(unittest.TestCase):
    """Test SiteSymmetryTable class."""

    def setUp(self):
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.r3c_example = Crystal.load(TEST_FILES["r3c_example.cif"])

    def test_from_crystal_ice_ii(self):
        """Test SiteSymmetryTable for iceII (P1)."""
        table = SiteSymmetryTable.from_crystal(self.ice_ii)

        # P1 has only 1 symop - all positions are general
        self.assertEqual(table.n_symops, 1)
        self.assertEqual(table.n_asym, len(self.ice_ii.asymmetric_unit))

        # All sites should be general positions in P1
        self.assertTrue(np.all(table.general_position_mask))
        self.assertEqual(table.n_general_positions(), table.n_asym)

        # All multiplicities should be 1
        for site_sym in table.site_symmetries:
            self.assertEqual(site_sym.multiplicity, 1)

    def test_from_crystal_acetic_acid(self):
        """Test SiteSymmetryTable for acetic acid (Pna21)."""
        table = SiteSymmetryTable.from_crystal(self.acetic)

        # Pna21 has 4 symops
        self.assertEqual(table.n_symops, 4)

        # All atoms in acetic acid are on general positions (multiplicity 4)
        for site_sym in table.site_symmetries:
            self.assertEqual(site_sym.multiplicity, 4)
            self.assertTrue(site_sym.is_general_position)

    def test_total_multiplicity_matches_uc(self):
        """Test that total multiplicity equals number of UC atoms."""
        for name, crystal in [
            ("ice_ii", self.ice_ii),
            ("acetic", self.acetic),
            ("r3c_example", self.r3c_example),
        ]:
            with self.subTest(crystal=name):
                table = SiteSymmetryTable.from_crystal(crystal)
                uc_atoms = crystal.unit_cell_atoms()

                # Total multiplicity should equal number of UC atoms
                self.assertEqual(
                    table.total_multiplicity(),
                    len(uc_atoms["element"]),
                    f"Total multiplicity mismatch for {name}",
                )

    def test_multiplicity_formula(self):
        """Test that multiplicity = |G| / |stabilizer|."""
        table = SiteSymmetryTable.from_crystal(self.acetic)

        for site_sym in table.site_symmetries:
            expected_mult = table.n_symops // site_sym.n_stabilizer
            self.assertEqual(site_sym.multiplicity, expected_mult)


class SiteSymmetrySpecialPositionTestCase(unittest.TestCase):
    """Test special position handling."""

    def test_atom_on_special_position(self):
        """Test atom explicitly placed on a special position."""
        # Create P21/c structure with atom on inversion center
        uc = UnitCell.from_lengths_and_angles(
            [10.0, 10.0, 10.0], [90.0, 90.0, 90.0], unit="degrees"
        )
        sg = SpaceGroup(2)  # P-1

        # Put atom exactly at origin (inversion center)
        elements = [Element["C"]]
        positions = np.array([[0.0, 0.0, 0.0]])
        labels = np.array(["C1"])

        asym = AsymmetricUnit(elements=elements, positions=positions, labels=labels)
        crystal = Crystal(uc, sg, asym)

        table = SiteSymmetryTable.from_crystal(crystal)

        # Atom should be on special position (multiplicity 1, not 2)
        self.assertEqual(table.site_symmetries[0].multiplicity, 1)
        self.assertFalse(table.site_symmetries[0].is_general_position)
        self.assertEqual(table.site_symmetries[0].n_stabilizer, 2)

    def test_atom_on_general_position(self):
        """Test atom on general position."""
        uc = UnitCell.from_lengths_and_angles(
            [10.0, 10.0, 10.0], [90.0, 90.0, 90.0], unit="degrees"
        )
        sg = SpaceGroup(2)  # P-1

        # Put atom at a general position (not on any special position)
        elements = [Element["C"]]
        positions = np.array([[0.1, 0.2, 0.3]])
        labels = np.array(["C1"])

        asym = AsymmetricUnit(elements=elements, positions=positions, labels=labels)
        crystal = Crystal(uc, sg, asym)

        table = SiteSymmetryTable.from_crystal(crystal)

        # Atom should be on general position (multiplicity 2)
        self.assertEqual(table.site_symmetries[0].multiplicity, 2)
        self.assertTrue(table.site_symmetries[0].is_general_position)
        self.assertEqual(table.site_symmetries[0].n_stabilizer, 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

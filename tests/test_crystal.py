import logging
import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.crystal import Crystal, AsymmetricUnit, UnitCell
from shmolecule.space_group import SpaceGroup
from shmolecule.cif import Cif
from tempfile import TemporaryDirectory
from shmolecule.element import Element
from .test_asymmetric_unit import ice_ii_asym

LOG = logging.getLogger(__name__)
_ICE_II = join(dirname(__file__), "iceII.cif")
_ICE_II_CELL = UnitCell.rhombohedral(7.78, 113.1, unit="degrees")
_ICE_II_SG = SpaceGroup(1)


class CrystalTestCase(unittest.TestCase):
    ice_ii = Crystal(_ICE_II_CELL, _ICE_II_SG, ice_ii_asym())

    def test_crystal_load(self):
        c = Crystal.load(_ICE_II)
        self.assertTrue(len(c.asymmetric_unit) == len(self.ice_ii.asymmetric_unit))
        self.assertTrue(c.space_group == self.ice_ii.space_group)
        self.assertTrue(len(c.symmetry_operations) == 1)

    def test_crystal_save(self):
        c = Crystal.load(_ICE_II)
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            c.save(join(tmpdirname, "tmp.cif"))

    def test_crystal_molecules(self):
        c = Crystal.load(_ICE_II)
        mols = c.symmetry_unique_molecules()
        self.assertTrue(len(mols) == 12, "Expect 12 water molecules in unit cell")
        formulae = [x.molecular_formula for x in mols]
        LOG.debug("Formulae = %s", formulae)
        self.assertTrue(
            all(f == "H2O" for f in formulae), "Expect molecular formula to be H2O"
        )

    def test_unit_conversions(self):
        pos = self.ice_ii.asymmetric_unit.positions
        np.testing.assert_allclose(
            self.ice_ii.to_fractional(self.ice_ii.to_cartesian(pos)), pos, atol=1e-8
        )

    def test_unit_cell_atoms(self):
        atom_calc = self.ice_ii.unit_cell_atoms()
        atoms = self.ice_ii.unit_cell_atoms()
        self.assertEqual(atom_calc, atoms)
        self.assertTrue(len(atoms["element"]) == len(self.ice_ii.asymmetric_unit))
        np.testing.assert_allclose(
            atoms["frac_pos"], self.ice_ii.asymmetric_unit.positions
        )


class CifTestCase(unittest.TestCase):
    def test_needs_quote(self):
        from shmolecule.cif import needs_quote

        self.assertTrue(needs_quote("this string will need quoting"))
        self.assertFalse(needs_quote("thiswon't"))
        self.assertFalse(needs_quote(3.45))

    def test_is_scalar(self):
        from shmolecule.cif import is_scalar

        self.assertTrue(is_scalar(3))
        self.assertTrue(is_scalar("test string"))
        self.assertFalse(is_scalar([3, 4, 5]))
        self.assertFalse(is_scalar((3, 4)))

    def test_format_field(self):
        from shmolecule.cif import format_field

        self.assertTrue(len(format_field(5.4)) == 20)
        self.assertTrue(format_field("3.4") == "3.4")

    def test_cif_funcs(self):
        c = Cif({})
        self.assertTrue(c.is_empty_line(""))
        self.assertFalse(c.is_data_line(""))
        self.assertFalse(c.is_data_line("_data"))

        c.content_lines = [";", "unmatched quote"]
        c.line_index = 0
        with self.assertRaises(ValueError):
            c.parse_quoted_block()
        c.content_lines = ["loop_", "_unmatched", "2"]
        c.line_index = 0
        c.parse_loop_block()

    def test_uncertainty(self):
        c = Cif({})
        c.content_lines = ["loop_", "_unmatched", "2"]
        c.line_index = 0
        with self.assertRaises(NotImplementedError):
            c.parse(ignore_uncertainty=False)

    def test_to_string(self):
        c = Cif({"crystal": {"loop_block": [1, 2], "quote_block": "test this"}})
        s = c.to_string()
        c2 = Cif.from_string(s)
        self.assertDictEqual(c.data, c2.data)

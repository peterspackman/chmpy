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
_ACETIC = join(dirname(__file__), "acetic_acid.cif")
_ICE_II_CELL = UnitCell.rhombohedral(7.78, 113.1, unit="degrees")
_ICE_II_SG = SpaceGroup(1)


class CrystalTestCase(unittest.TestCase):
    def setUp(self):
        self.ice_ii = Crystal(_ICE_II_CELL, _ICE_II_SG, ice_ii_asym())
        self.acetic = Crystal.load(_ACETIC)

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

    def test_handles_higher_occupation(self):
        from copy import deepcopy

        asym = deepcopy(ice_ii_asym())
        natom = len(asym) + 1
        natom_old = len(asym)
        asym.positions = np.vstack((asym.positions, asym.positions[0, :]))
        asym.labels = np.hstack((asym.labels, asym.labels[0] + "X"))
        asym.atomic_numbers = np.hstack((asym.atomic_numbers, asym.atomic_numbers[0]))
        asym.elements.append(asym.elements[0])
        asym.properties["occupation"] = np.ones(natom)
        asym.properties["occupation"][0] = 0.5
        asym.properties["occupation"][-1] = 0.5
        x = Crystal(_ICE_II_CELL, _ICE_II_SG, asym)
        atoms = x.unit_cell_atoms()
        # should have merged the sites
        self.assertEqual(len(atoms["element"]), natom_old)

    def test_cached_calls(self):
        for c in ("ice_ii", "acetic"):
            x = getattr(self, c)
            g1 = x.unit_cell_connectivity()
            g2 = x.unit_cell_connectivity()
            self.assertEqual(g1, g2)
            m1 = x.unit_cell_molecules()
            m2 = x.unit_cell_molecules()
            self.assertEqual(m1, m2)
            m1 = x.symmetry_unique_molecules()
            m2 = x.symmetry_unique_molecules()
            self.assertEqual(m1, m2)

    def test_surroundings_functions(self):
        for c in ("ice_ii", "acetic"):
            x = getattr(self, c)
            atoms = x.atoms_in_radius(5.0)
            atoms = x.atomic_surroundings()
            atoms = x.atom_group_surroundings([0, 1, 2])
            surroundings = x.molecule_surroundings()


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

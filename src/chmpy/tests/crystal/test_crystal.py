import logging
import unittest
import numpy as np
from copy import deepcopy
from chmpy.crystal import Crystal, AsymmetricUnit, UnitCell, SpaceGroup
from chmpy.fmt.cif import Cif
from tempfile import TemporaryDirectory
from chmpy import Element
from .test_asymmetric_unit import ice_ii_asym
from .. import TEST_FILES
from pathlib import Path

LOG = logging.getLogger(__name__)
_ICE_II_CELL = UnitCell.rhombohedral(7.78, 113.1, unit="degrees")
_ICE_II_SG = SpaceGroup(1)


_NONSTANDARD_SG2 = """data_c1
_symmetry_cell_setting           triclinic
_symmetry_space_group_name_H-M   'C -1'
_symmetry_Int_Tables_number      2
_space_group_name_Hall           '-C 1'
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 x,y,z
2 1/2+x,1/2+y,z
3 -x,-y,-z
4 1/2-x,1/2-y,-z
_cell_length_a     13.68
_cell_length_b     13.67
_cell_length_c     25.04
_cell_angle_alpha  89.98
_cell_angle_beta   99.20
_cell_angle_gamma  89.98
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.5842 0.1055 0.03779
#END
data_c2
_symmetry_cell_setting           triclinic
_symmetry_space_group_name_H-M   'C -1'
_symmetry_Int_Tables_number      2
_space_group_name_Hall           '-C 1'
_cell_length_a     13.68
_cell_length_b     13.67
_cell_length_c     25.04
_cell_angle_alpha  89.98
_cell_angle_beta   99.20
_cell_angle_gamma  89.98
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.5842 0.1055 0.03779"""


class CrystalTestCase(unittest.TestCase):
    def setUp(self):
        self.ice_ii = Crystal(_ICE_II_CELL, _ICE_II_SG, ice_ii_asym())
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.acetic_res = Crystal.load(TEST_FILES["acetic_acid.res"])
        self.r3c_example = Crystal.load(TEST_FILES["r3c_example.cif"])

    def test_crystal_load(self):
        c = Crystal.load(TEST_FILES["iceII.cif"])
        self.assertTrue(len(c.asymmetric_unit) == len(self.ice_ii.asymmetric_unit))
        self.assertTrue(c.space_group == self.ice_ii.space_group)
        self.assertTrue(len(c.symmetry_operations) == 1)
        np.testing.assert_equal(c.site_labels, c.asymmetric_unit.labels)

        with self.assertRaises(ValueError):
            contents = "LATT 4klj1klj\n"
            c = Crystal.from_shelx_string(contents)

        from chmpy.fmt.shelx import parse_shelx_file

        shelx_data = parse_shelx_file(TEST_FILES["acetic_acid.res"])

        c = Crystal.from_cif_file(
            TEST_FILES["acetic_acid.cif"], data_block_name="acetic_acid"
        )

    def test_bad_occupations(self):
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
        self.assertEqual(x.titl, "H24O13")

    def test_repr(self):
        self.assertEqual(repr(self.acetic), "<Crystal C2H4O2 Pna2_1>")
        self.acetic.properties["density"] = 0.5
        self.acetic.properties["lattice_energy"] = 0.0
        self.assertEqual(repr(self.acetic), "<Crystal C2H4O2 Pna2_1 (0.500, 0.000)>")

    def test_nonstandard_spacegroups(self):
        crystals = Crystal.from_cif_string(_NONSTANDARD_SG2)
        for c in crystals.values():
            self.assertEqual(c.space_group.international_tables_number, 2)

    def test_density(self):
        self.assertAlmostEqual(self.acetic.density, 1.271208154)
        self.acetic.properties["density"] = 0.5
        self.assertAlmostEqual(self.acetic.density, 0.5)

    def test_crystal_save(self):
        c = Crystal.load(TEST_FILES["iceII.cif"])
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            c.save(Path(tmpdirname, "tmp.cif"))
            c.save(Path(tmpdirname, "tmp.res"))

        c = Crystal(_ICE_II_CELL, _ICE_II_SG, ice_ii_asym())
        c.properties["titl"] = "iceII"
        s = c.to_cif_string()
        c2 = Crystal.from_cif_string(s)
        c2 = Crystal.from_cif_string(s, data_block_name="iceII")

    def test_crystal_molecules(self):
        c = Crystal.load(TEST_FILES["iceII.cif"])
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
        for c in ("ice_ii", "acetic", "r3c_example"):
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

    def test_environments_functions(self):
        for c in ("ice_ii", "acetic", "r3c_example"):
            x = getattr(self, c)
            atoms = x.atoms_in_radius(5.0)
            atoms = x.atomic_surroundings()
            atoms = x.atom_group_surroundings([0, 1, 2])
            environments = x.molecule_environments()

    def test_cartesian_symmetry_operations(self):
        for c in ("ice_ii", "acetic", "r3c_example"):
            x = getattr(self, c)
            mol = x.symmetry_unique_molecules()[0]
            pos = mol.positions
            pos_frac = x.to_fractional(pos)
            symops_cart = x.cartesian_symmetry_operations()
            symops_frac = x.symmetry_operations
            for (r, t), sf in zip(symops_cart, symops_frac):
                pos_a = np.dot(pos, r) + t
                pos_b = x.to_cartesian(sf.apply(pos_frac))
                np.testing.assert_allclose(pos_a, pos_b)
                mol_t = mol.transformed(rotation=r, translation=t)
                np.testing.assert_allclose(mol_t.positions, pos_b)


class CifTestCase(unittest.TestCase):
    def test_needs_quote(self):
        from chmpy.fmt.cif import needs_quote

        self.assertTrue(needs_quote("this string will need quoting"))
        self.assertFalse(needs_quote("thiswon't"))
        self.assertFalse(needs_quote(3.45))

    def test_is_scalar(self):
        from chmpy.fmt.cif import is_scalar

        self.assertTrue(is_scalar(3))
        self.assertTrue(is_scalar("test string"))
        self.assertFalse(is_scalar([3, 4, 5]))
        self.assertFalse(is_scalar((3, 4)))

    def test_format_field(self):
        from chmpy.fmt.cif import format_field

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

import logging
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chmpy.crystal import Crystal, SpaceGroup, UnitCell
from chmpy.fmt.cif import Cif

from .. import TEST_FILES
from .test_asymmetric_unit import ice_ii_asym

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

        parse_shelx_file(TEST_FILES["acetic_acid.res"])

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
        Crystal.from_cif_string(s)
        Crystal.from_cif_string(s, data_block_name="iceII")

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
            x.atoms_in_radius(5.0)
            x.atomic_surroundings()
            x.atom_group_surroundings([0, 1, 2])
            x.molecule_environments()

    def test_cartesian_symmetry_operations(self):
        for c in ("ice_ii", "acetic", "r3c_example"):
            x = getattr(self, c)
            mol = x.symmetry_unique_molecules()[0]
            pos = mol.positions
            pos_frac = x.to_fractional(pos)
            symops_cart = x.cartesian_symmetry_operations()
            symops_frac = x.symmetry_operations
            for (r, t), sf in zip(symops_cart, symops_frac, strict=False):
                pos_a = np.dot(pos, r) + t
                pos_b = x.to_cartesian(sf.apply(pos_frac))
                np.testing.assert_allclose(pos_a, pos_b)
                mol_t = mol.transformed(rotation=r, translation=t)
                np.testing.assert_allclose(mol_t.positions, pos_b)


class CifTestCase(unittest.TestCase):
    """Crystals read from and written back to CIF.

    The CIF reader and writer themselves are tested in tests/fmt/test_cif.py.
    """

    def test_crystal_survives_a_cif_round_trip(self):
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        again = Crystal.from_cif_string(crystal.to_cif_string())
        np.testing.assert_allclose(
            again.unit_cell.parameters, crystal.unit_cell.parameters
        )
        np.testing.assert_allclose(
            again.asymmetric_unit.positions, crystal.asymmetric_unit.positions
        )
        self.assertEqual(
            again.space_group.international_tables_number,
            crystal.space_group.international_tables_number,
        )

    def test_mmcif_names_and_cartesian_coordinates(self):
        "mmCIF spells its data names differently and stores orthogonal coordinates"
        crystal = Crystal.from_cif_string(
            """data_mm
_cell.length_a      10.0
_cell.length_b      10.0
_cell.length_c      10.0
_cell.angle_alpha   90.0
_cell.angle_beta    90.0
_cell.angle_gamma   90.0
_symmetry.Int_Tables_number  1
loop_
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
O1 O 1.0 2.0 3.0
H1 H 4.0 5.0 6.0
"""
        )
        self.assertEqual(
            [x.symbol for x in crystal.asymmetric_unit.elements], ["O", "H"]
        )
        np.testing.assert_allclose(
            crystal.asymmetric_unit.positions,
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

    def test_to_cif_data_does_not_touch_the_crystal(self):
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        before = {k: str(v) for k, v in crystal.properties["cif_data"].items()}
        crystal.to_cif_data()
        after = {k: str(v) for k, v in crystal.properties["cif_data"].items()}
        self.assertEqual(before, after)

    def test_moved_atoms_are_written_out(self):
        "the crystal, not the file it came from, says where the atoms are"
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        crystal.normalize_hydrogen_bondlengths()
        again = Crystal.from_cif_string(crystal.to_cif_string())
        np.testing.assert_allclose(
            again.asymmetric_unit.positions, crystal.asymmetric_unit.positions
        )

    def test_changed_unit_cell_is_written_out(self):
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        crystal.unit_cell = UnitCell.cubic(10.0)
        again = Crystal.from_cif_string(crystal.to_cif_string())
        np.testing.assert_allclose(again.unit_cell.parameters, [10, 10, 10, 90, 90, 90])

    def test_stale_per_site_columns_are_dropped(self):
        "a column that no longer has one value per site cannot be carried over"
        crystal = Crystal.load(TEST_FILES["r3c_example.cif"])
        asym = crystal.asymmetric_unit
        self.assertIn("atom_site_U_iso_or_equiv", crystal.to_cif_data()[crystal.titl])

        keep = slice(0, len(asym) - 2)
        asym.positions = asym.positions[keep]
        asym.labels = asym.labels[keep]
        asym.elements = asym.elements[keep]
        asym.atomic_numbers = asym.atomic_numbers[keep]
        asym.properties["occupation"] = asym.properties["occupation"][keep]

        cif_data = crystal.to_cif_data()[crystal.titl]
        self.assertNotIn("atom_site_U_iso_or_equiv", cif_data)
        again = Crystal.from_cif_string(crystal.to_cif_string())
        self.assertEqual(len(again.asymmetric_unit), len(asym))
        self.assertEqual(list(again.asymmetric_unit.labels), list(asym.labels))

    def test_metadata_is_carried_through(self):
        "what the file says that is not about the structure survives a rewrite"
        crystal = Crystal.from_cif_string(
            """data_meta
_cell_length_a    5.0
_cell_length_b    5.0
_cell_length_c    5.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_cell_formula_units_Z             4
_symmetry_Int_Tables_number       1
_chemical_name_common             'a made up thing'
_diffrn_ambient_temperature       100(2)
_refine_ls_R_factor_gt            0.0321
loop_
_publ_author_name
'Bloggs, J.'
'Doe, J.'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.0 0.0 0.0
"""
        )
        crystal.normalize_hydrogen_bondlengths()
        written = crystal.to_cif_data()["meta"]
        self.assertEqual(written["chemical_name_common"], "a made up thing")
        self.assertEqual(written["diffrn_ambient_temperature"], 100.0)
        self.assertEqual(written["refine_ls_R_factor_gt"], 0.0321)
        self.assertEqual(written["cell_formula_units_Z"], 4)
        self.assertEqual(written["publ_author_name"], ["Bloggs, J.", "Doe, J."])
        # and they survive being written out and read back
        again = Cif.from_string(crystal.to_cif_string())["meta"]
        self.assertEqual(again["publ_author_name"], ["Bloggs, J.", "Doe, J."])
        self.assertEqual(again["chemical_name_common"], "a made up thing")

    def test_space_group_data_comes_from_the_crystal(self):
        crystal = Crystal.load(TEST_FILES["r3c_example.cif"])
        written = crystal.to_cif_data()[crystal.titl]
        self.assertEqual(
            written["symmetry_Int_Tables_number"],
            crystal.space_group.international_tables_number,
        )
        self.assertNotIn("symmetry_space_group_name_Hall", written)

    def test_source_data_can_be_left_out(self):
        crystal = Crystal.load(TEST_FILES["r3c_example.cif"])
        written = crystal.to_cif_data(source_data=False)[crystal.titl]
        self.assertNotIn("atom_site_U_iso_or_equiv", written)
        again = Crystal.from_cif_string(crystal.to_cif_string(source_data=False))
        np.testing.assert_allclose(
            again.asymmetric_unit.positions, crystal.asymmetric_unit.positions
        )


class OccupancyTestCase(unittest.TestCase):
    """Partial occupancies and special positions, which interact.

    A site on a special position is generated once per symmetry operation
    that leaves it alone, so those copies have to be merged -- but they are
    the same atom, not more of it.
    """

    CELL = """data_{name}
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_Int_Tables_number {number}
loop_
_symmetry_equiv_pos_as_xyz
{symops}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
"""
    P1 = {"number": 1, "symops": "x,y,z"}
    P_1 = {"number": 2, "symops": "x,y,z\n-x,-y,-z"}

    def crystal(self, name, rows, setting=None):
        setting = setting or self.P1
        return Crystal.from_cif_string(
            self.CELL.format(name=name, **setting) + "\n".join(rows) + "\n"
        )

    def expected_density(self, n_carbon):
        "g/cm^3 for n carbons in a 1000 cubic angstrom cell"
        from chmpy.core.element import Element

        return n_carbon * Element["C"].mass / 1000.0 / 0.6022

    def test_density_weights_by_occupancy(self):
        whole = self.crystal("whole", ["C1 C 0.1 0.2 0.3 1.0"])
        half = self.crystal("half", ["C1 C 0.1 0.2 0.3 0.5"])
        self.assertAlmostEqual(whole.density, self.expected_density(1))
        self.assertAlmostEqual(half.density, self.expected_density(0.5))

    def test_two_alternatives_weigh_one_atom(self):
        "a group modelled in two places is still one group per unit cell"
        crystal = self.crystal(
            "split", ["C1A C 0.10 0.20 0.30 0.7", "C1B C 0.12 0.22 0.30 0.3"]
        )
        self.assertAlmostEqual(crystal.density, self.expected_density(1))

    def test_special_position_is_not_counted_twice(self):
        "an atom on the inversion centre is one atom, not two"
        crystal = self.crystal("special", ["C1 C 0.0 0.0 0.0 1.0"], setting=self.P_1)
        atoms = crystal.unit_cell_atoms()
        self.assertEqual(len(atoms["element"]), 1)
        np.testing.assert_allclose(atoms["occupation"], [1.0])
        self.assertAlmostEqual(crystal.density, self.expected_density(1))

    def test_general_position_is_counted_once_per_symop(self):
        crystal = self.crystal("general", ["C1 C 0.1 0.2 0.3 1.0"], setting=self.P_1)
        atoms = crystal.unit_cell_atoms()
        self.assertEqual(len(atoms["element"]), 2)
        np.testing.assert_allclose(atoms["occupation"], [1.0, 1.0])
        self.assertAlmostEqual(crystal.density, self.expected_density(2))

    def test_two_sites_sharing_one_position_do_add_up(self):
        "substitutional disorder is different: two atoms, one place"
        crystal = self.crystal(
            "shared", ["NA1 Na 0.1 0.2 0.3 0.6", "K1 K 0.1 0.2 0.3 0.4"]
        )
        atoms = crystal.unit_cell_atoms()
        self.assertEqual(len(atoms["element"]), 1)
        np.testing.assert_allclose(atoms["occupation"], [1.0])

    def test_site_multiplicity_agrees_with_the_unit_cell(self):
        """The stabilizer subgroup gives the Wyckoff multiplicity directly.

        No table of Wyckoff positions is needed to get this right.
        """
        from chmpy.crystal.site_symmetry import SiteSymmetryTable

        crystal = self.crystal(
            "mixed",
            ["C1 C 0.0 0.0 0.0 1.0", "C2 C 0.1 0.2 0.3 1.0"],
            setting=self.P_1,
        )
        table = SiteSymmetryTable.from_crystal(crystal)
        atoms = crystal.unit_cell_atoms()
        self.assertEqual(table.site_symmetries[0].multiplicity, 1)
        self.assertEqual(table.site_symmetries[1].multiplicity, 2)
        self.assertEqual(table.total_multiplicity(), len(atoms["element"]))

    def test_structure_factors_scale_with_occupancy(self):
        from chmpy.crystal.powder import generate_hkl, structure_factors

        whole = self.crystal("whole", ["C1 C 0.1 0.2 0.3 1.0"])
        half = self.crystal("half", ["C1 C 0.1 0.2 0.3 0.5"])
        hkl, d = generate_hkl(whole.unit_cell, 2.0)
        np.testing.assert_allclose(
            structure_factors(half, hkl, d),
            0.5 * structure_factors(whole, hkl, d),
            rtol=1e-10,
        )

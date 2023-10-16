import logging
import unittest
import numpy as np
from os.path import join, dirname
from chmpy.crystal import SymmetryOperation, SpaceGroup
from chmpy.crystal.space_group import expanded_symmetry_list, reduced_symmetry_list

LOG = logging.getLogger(__name__)

SG225_REDUCED = (
    "-x,-y,+z",
    "-x,-z,-y",
    "-x,-z,+y",
    "-x,+z,-y",
    "-x,+z,+y",
    "-x,+y,-z",
    "-x,+y,+z",
    "-y,-x,-z",
    "-y,-x,+z",
    "-y,-z,-x",
    "-y,-z,+x",
    "-y,+z,-x",
    "-y,+z,+x",
    "-y,+x,-z",
    "-y,+x,+z",
    "-z,-x,-y",
    "-z,-x,+y",
    "-z,-y,-x",
    "-z,-y,+x",
    "-z,+y,-x",
    "-z,+y,+x",
    "-z,+x,-y",
    "-z,+x,+y",
)
SG33_REDUCED = ("1/2-x,1/2+y,1/2+z", "1/2+x,1/2-y,z", "-x,-y,1/2+z")


class SymmetryOperationTestCase(unittest.TestCase):
    identity = SymmetryOperation.from_integer_code(16484)

    def test_seitz(self):
        s = self.identity.seitz_matrix
        expected = np.eye(4)
        np.testing.assert_allclose(s, expected)

    def test_cif_form(self):
        self.assertEqual(self.identity.cif_form, "+x,+y,+z")
        inv = self.identity.inverted()
        self.assertEqual(inv.cif_form, "-x,-y,-z")
        inv += (0.5, 0.0, 0.0)
        self.assertEqual(inv.cif_form, "1/2-x,-y,-z")
        inv -= (0.5, 0.0, 0.0)
        self.assertEqual(inv.cif_form, "-x,-y,-z")

    def test_apply(self):
        pts_seitz = np.random.rand(100, 4)
        pts_seitz[:, 3] = 1
        np.testing.assert_allclose(pts_seitz, self.identity.apply(pts_seitz))

    def test_ordering(self):
        inv = self.identity.inverted()
        self.assertTrue(inv < self.identity)
        self.assertTrue(self.identity > inv)

    def test_repr(self):
        self.assertEqual(repr(self.identity), "<SymmetryOperation: +x,+y,+z>")

    def test_expanded_and_reduced_symops(self):
        for symops, sg, LATT in ((SG33_REDUCED, 33, -1), (SG225_REDUCED, 225, 4)):
            reduced_symops = [SymmetryOperation.from_string_code(x) for x in symops]
            expanded_symops = sorted(SpaceGroup(sg).symmetry_operations)
            np.testing.assert_equal(
                sorted(expanded_symmetry_list(reduced_symops, LATT)), expanded_symops
            )
            reduced_symops_2 = reduced_symmetry_list(expanded_symops, LATT)
            np.testing.assert_equal(sorted(reduced_symops_2), sorted(reduced_symops))


class SpaceGroupTestCase(unittest.TestCase):
    sg_1 = SpaceGroup(1)
    sg_14 = SpaceGroup(14)
    sg_33 = SpaceGroup(33)
    sg_76 = SpaceGroup(76)
    sg_148 = SpaceGroup(148)
    sg_169 = SpaceGroup(169)
    sg_225 = SpaceGroup(225)

    def test_construction(self):
        for invalid_num in (-1, 0, 231, 1000):
            with self.assertRaises(ValueError):
                sg = SpaceGroup(invalid_num)

        for invalid_choice in ("a", 1, "b"):
            with self.assertRaises(ValueError):
                sg = SpaceGroup(148, choice=invalid_choice)

        sg_148_h = SpaceGroup(148, choice="H")
        sg_148_r = SpaceGroup(148, choice="R")
        for invalid_latt in (-8, 10, -4):
            with self.assertRaises(ValueError):
                sg = SpaceGroup.from_symmetry_operations(
                    [SymmetryOperation.identity()], invalid_latt
                )

    def test_cif_section(self):
        expected = "1 +x,+y,+z"
        self.assertEqual(self.sg_1.cif_section, expected)

    def test_crystal_system(self):
        sgs = (
            self.sg_1,
            self.sg_14,
            self.sg_33,
            self.sg_76,
            self.sg_148,
            self.sg_169,
            self.sg_225,
        )
        systems = (
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        )
        for s, sys in zip(sgs, systems):
            self.assertEqual(s.crystal_system, sys)

        sg_bad = SpaceGroup(1)
        sg_bad.international_tables_number = -1
        with self.assertRaises(ValueError):
            sg_bad.crystal_system

    def test_lattice_type(self):
        sgs = (
            SpaceGroup(15),
            SpaceGroup(169),
            SpaceGroup(148),
            SpaceGroup(148, choice="R"),
        )
        latt = ("monoclinic", "hexagonal", "hexagonal", "rhombohedral")
        for s, sys in zip(sgs, latt):
            self.assertEqual(s.lattice_type, sys)

    def test_ordered_symmetry_operations(self):
        ordered = self.sg_225.ordered_symmetry_operations()
        self.assertEqual(ordered[0], SymmetryOperation.identity())
        np.testing.assert_equal(sorted(ordered[1:]), ordered[1:])

        # Test corrupted space group
        sg = SpaceGroup(2)
        sg.symmetry_operations = [SymmetryOperation.from_string_code("-x,+y,+z")]
        with self.assertRaises(ValueError):
            sg.ordered_symmetry_operations()

    def test_repr(self):
        self.assertEqual(repr(self.sg_1), "<SpaceGroup 1: P1>")

    def test_hash(self):
        self.assertEqual(
            set(
                (
                    SpaceGroup(148, choice="R"),
                    SpaceGroup(148),
                    SpaceGroup(148),
                    SpaceGroup(1),
                )
            ),
            set((SpaceGroup(148, choice="R"), SpaceGroup(148), SpaceGroup(1))),
        )

    def test_reduced_expand(self):
        for i in range(1, 231):
            sg = SpaceGroup(i)
            self.assertEqual(
                SpaceGroup.from_symmetry_operations(
                    sg.reduced_symmetry_operations(), sg.latt
                ),
                sg,
            )

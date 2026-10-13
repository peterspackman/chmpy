import unittest
from fractions import Fraction

import numpy as np

from chmpy.crystal import SpaceGroup, UnitCell
from chmpy.crystal.surface_symmetry import SurfaceSymmetry, primitive_lattice

CUBIC = UnitCell.cubic(4.0)
HEXAGONAL = UnitCell.hexagonal(3.2, 5.2, unit="degrees")
MONOCLINIC = UnitCell.monoclinic(5.0, 6.0, 7.0, 100.0, unit="degrees")


def surface(sg, cell, hkl):
    return SurfaceSymmetry.from_space_group(SpaceGroup(sg), cell, hkl)


class SurfaceSymmetryTestCase(unittest.TestCase):
    def assert_plane_groups(self, sg, cell, expected):
        for hkl, symbol in expected.items():
            self.assertEqual(surface(sg, cell, hkl).plane_group.symbol, symbol, hkl)

    def test_fcc(self):
        self.assert_plane_groups(
            225,
            CUBIC,
            {(1, 0, 0): "p4mm", (1, 1, 0): "p2mm", (1, 1, 1): "p3m1", (2, 1, 0): "cm"},
        )

    def test_bcc(self):
        self.assert_plane_groups(
            229, CUBIC, {(1, 0, 0): "p4mm", (1, 1, 0): "c2mm", (1, 1, 1): "p3m1"}
        )

    def test_hcp(self):
        self.assert_plane_groups(194, HEXAGONAL, {(0, 0, 1): "p3m1", (1, 0, 0): "p2mm"})

    def test_monoclinic_glide_and_screw(self):
        self.assert_plane_groups(
            14, MONOCLINIC, {(1, 0, 0): "pg", (0, 1, 0): "p1", (1, 1, 0): "p1"}
        )

    def test_screw_along_normal_relates_terminations(self):
        s = surface(14, MONOCLINIC, (0, 1, 0))
        self.assertEqual(
            [op.translation[2] for op in s.shift_operations], [Fraction(1, 2)]
        )
        self.assertEqual(s.equivalent_offsets(0.1), [0.1, 0.6])

    def test_layer_spacing_of_centred_lattices(self):
        self.assertAlmostEqual(surface(225, CUBIC, (1, 0, 0)).layer_spacing, 2.0)
        self.assertAlmostEqual(
            surface(229, CUBIC, (1, 1, 1)).layer_spacing, 4.0 / (2 * np.sqrt(3))
        )

    def test_surface_cell_is_primitive(self):
        s = surface(225, CUBIC, (1, 1, 1))
        volume = abs(np.linalg.det(np.array(s.surface_vectors, dtype=float)))
        self.assertAlmostEqual(volume, 0.25)

    def test_polar_direction(self):
        self.assertTrue(surface(4, MONOCLINIC, (0, 1, 0)).is_polar_direction)
        self.assertFalse(surface(4, MONOCLINIC, (1, 0, 0)).is_polar_direction)

    def test_primitive_lattice_of_primitive_group_is_identity(self):
        np.testing.assert_array_equal(
            np.array(primitive_lattice(SpaceGroup(14)), dtype=int), np.eye(3)
        )

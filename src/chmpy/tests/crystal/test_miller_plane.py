import unittest
from itertools import product

import numpy as np

from chmpy.crystal import SpaceGroup, UnitCell
from chmpy.crystal.miller_plane import (
    MillerPlane,
    face_images,
    faces_are_equivalent,
    friedel_mate_is_equivalent,
    reduce_hkl,
    surface_basis,
    unique_faces,
)

TRICLINIC = UnitCell.triclinic(5.1, 6.3, 7.7, 81.0, 97.0, 104.0, unit="degrees")
HEXAGONAL = UnitCell.hexagonal(3.2, 5.2, unit="degrees")
CUBIC = UnitCell.cubic(4.0)


class SurfaceBasisTestCase(unittest.TestCase):
    def assert_valid_basis(self, hkl, cell):
        basis = surface_basis(hkl, cell)
        h = np.array(reduce_hkl(hkl))
        self.assertEqual(round(np.linalg.det(basis)), 1, f"{hkl} not unimodular")
        np.testing.assert_array_equal(basis @ h, [0, 0, 1], err_msg=str(hkl))
        metric = cell.metric_tensor
        a, b = basis[0], basis[1]
        self.assertLessEqual(a @ metric @ a, b @ metric @ b + 1e-9)
        self.assertLessEqual(abs(a @ metric @ b), 0.5 * (a @ metric @ a) + 1e-9)

    def test_basis_all_small_indices_triclinic(self):
        for hkl in product(range(-3, 4), repeat=3):
            if hkl != (0, 0, 0):
                self.assert_valid_basis(hkl, TRICLINIC)

    def test_basis_large_indices(self):
        for hkl in [(7, -5, 11), (13, 0, 2), (0, 9, -4)]:
            self.assert_valid_basis(hkl, TRICLINIC)

    def test_reduce_hkl(self):
        self.assertEqual(reduce_hkl((2, 0, -4)), (1, 0, -2))
        with self.assertRaises(ValueError):
            reduce_hkl((0, 0, 0))

    def test_cubic_100_cell(self):
        plane = MillerPlane((1, 0, 0), CUBIC)
        self.assertAlmostEqual(plane.d_spacing, 4.0)
        self.assertAlmostEqual(plane.area, 16.0)
        np.testing.assert_allclose(plane.normal, [1, 0, 0], atol=1e-12)

    def test_w_steps_one_spacing_along_normal(self):
        plane = MillerPlane((2, -1, 3), TRICLINIC)
        w = plane.vectors[2]
        self.assertAlmostEqual(w @ plane.normal, plane.d_spacing)

    def test_cell_volume_is_area_times_spacing(self):
        plane = MillerPlane((1, 2, -1), TRICLINIC)
        self.assertAlmostEqual(plane.area * plane.d_spacing, TRICLINIC.volume())

    def test_surface_frame_is_orthonormal_and_right_handed(self):
        frame = MillerPlane((1, 1, 0), HEXAGONAL).surface_frame()
        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(frame), 1.0)

    def test_hexagonal_basal_plane_is_hexagonal(self):
        g = MillerPlane((0, 0, 1), HEXAGONAL).metric_2d
        self.assertAlmostEqual(g[0, 0], g[1, 1])
        self.assertAlmostEqual(abs(g[0, 1]), 0.5 * g[0, 0])


class FaceSymmetryTestCase(unittest.TestCase):
    def form_size(self, sg, hkl):
        return len(face_images(SpaceGroup(sg), hkl))

    def test_cubic_form_sizes(self):
        sizes = {
            (1, 0, 0): 6,
            (1, 1, 0): 12,
            (1, 1, 1): 8,
            (2, 1, 0): 24,
            (3, 2, 1): 48,
        }
        for hkl, n in sizes.items():
            self.assertEqual(self.form_size(225, hkl), n, hkl)

    def test_hexagonal_form_sizes(self):
        sizes = {(0, 0, 1): 2, (1, 0, 0): 6, (1, 1, 0): 6, (1, 0, 1): 12}
        for hkl, n in sizes.items():
            self.assertEqual(self.form_size(194, hkl), n, hkl)

    def test_hexagonal_prism_faces_are_equivalent(self):
        sg = SpaceGroup(194)
        self.assertTrue(faces_are_equivalent(sg, (1, 0, 0), (0, 1, 0)))
        self.assertTrue(faces_are_equivalent(sg, (1, 0, 0), (1, -1, 0)))
        self.assertFalse(faces_are_equivalent(sg, (1, 0, 0), (1, 1, 0)))

    def test_friedel_mates(self):
        self.assertTrue(friedel_mate_is_equivalent(SpaceGroup(14), (0, 1, 0)))
        self.assertFalse(friedel_mate_is_equivalent(SpaceGroup(4), (0, 1, 0)))
        self.assertTrue(friedel_mate_is_equivalent(SpaceGroup(4), (1, 0, 0)))

    def test_unique_faces_cubic(self):
        faces = unique_faces(SpaceGroup(225), CUBIC, d_min=4.0 / np.sqrt(3) - 1e-6)
        self.assertEqual([f.hkl for f in faces], [(1, 0, 0), (1, 1, 0), (1, 1, 1)])

    def test_unique_faces_hexagonal_keeps_prism(self):
        faces = unique_faces(SpaceGroup(194), HEXAGONAL, d_min=2.0)
        hkls = [f.hkl for f in faces]
        self.assertIn((1, 0, 0), hkls)
        for i, a in enumerate(hkls):
            for b in hkls[i + 1 :]:
                self.assertFalse(faces_are_equivalent(SpaceGroup(194), a, b))

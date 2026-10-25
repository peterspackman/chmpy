import unittest

import numpy as np

from chmpy.ext.elastic_tensor import ElasticTensor
from chmpy.util.optional import have

TENSOR = """
    48.137	11.411	12.783	0.000	-3.654	0.000
    11.411	34.968	14.749	0.000	-0.094	0.000
    12.783	14.749	26.015	0.000	-4.528	0.000
    0.000	0.000	0.000	14.545	0.000	0.006
    -3.654	-0.094	-4.528	0.000	10.771	0.000
    0.000	0.000	0.000	0.006	0.000	11.947
"""


class ElasticTensorTestCase(unittest.TestCase):
    def setUp(self):
        self.elastic = ElasticTensor.from_string(TENSOR)

    def test_youngs_modulus(self):
        ym = self.elastic.youngs_modulus([[0, 0, 1]])
        self.assertAlmostEqual(ym[0], 16.95754579335107)

    def test_linear_compressibility(self):
        lc = self.elastic.linear_compressibility([[0, 0, 1]])
        self.assertAlmostEqual(lc[0], 28.214314777188065)

    def test_second_direction_is_perpendicular_unit(self):
        from chmpy.ext.elastic_tensor import angles_to_cartesian, angles_to_cartesian_2

        th, ph, ch = np.array([0.7]), np.array([1.1]), np.array([0.4])
        a = angles_to_cartesian(th, ph)
        b = angles_to_cartesian_2(th, ph, ch)
        self.assertAlmostEqual(np.linalg.norm(b), 1.0)
        self.assertAlmostEqual((a @ b.T).item(), 0.0)

    def test_isotropic_shear_and_poisson(self):
        c11, c12 = 300.0, 100.0
        c44 = (c11 - c12) / 2
        c = np.zeros((6, 6))
        c[:3, :3] = c12
        np.fill_diagonal(c, [c11] * 3 + [c44] * 3)
        t = ElasticTensor(c)
        th, ph, ch = np.array([0.7]), np.array([1.1]), np.array([0.4])
        self.assertAlmostEqual(t.shear_modulus_angular(th, ph, ch)[0], c44)
        self.assertAlmostEqual(
            t.poisson_ratio_angular(th, ph, ch)[0], c12 / (c11 + c12)
        )

    def test_triangular_input(self):
        c = self.elastic.c_voigt
        upper = [list(c[i, i:]) for i in range(6)]
        lower = [list(c[i, : i + 1]) for i in range(6)]
        np.testing.assert_allclose(ElasticTensor(upper).c_voigt, c)
        np.testing.assert_allclose(ElasticTensor(lower).c_voigt, c)
        upper_str = "\n".join(" ".join(str(x) for x in row) for row in upper)
        np.testing.assert_allclose(ElasticTensor.from_string(upper_str).c_voigt, c)

    # averages() integrates over a trimesh icosphere
    @unittest.skipUnless(have("trimesh"), "needs chmpy[mesh]")
    def test_reorientation_to_standard_frame(self):
        """Test reorientation of elastic tensor from arbitrary to standard crystallographic frame.

        Uses MALNAC06 example data from GULP optimization to verify that:
        1. Rotation matrix is orthogonal
        2. Unit cell parameters are preserved
        3. Tensor invariants (bulk and shear moduli) are preserved
        4. 4th-order tensor contractions are invariant
        """
        # MALNAC06 example data - lattice vectors from GULP optimization
        gulp_vectors = np.array([
            [ 4.777394,  0.096445, -0.149209],
            [-0.330401,  5.723170,  0.439075],
            [-1.709427, -2.386116,  6.715339]
        ])

        # Elastic tensor from GULP in GPa
        gulp_tensor_data = np.array([
            [38.9166, 19.2288, 21.8648,  2.7327, -10.0968,   0.3838],
            [19.2288, 45.1814, 20.8091,  7.5240,  -7.8010, -11.8334],
            [21.8648, 20.8091, 30.4442,  2.4606, -15.8000,  -2.7215],
            [ 2.7327,  7.5240,  2.4606, 17.6056,  -7.3482,  -5.2410],
            [-10.0968, -7.8010, -15.8000, -7.3482, 16.9020,   4.4631],
            [ 0.3838, -11.8334, -2.7215, -5.2410,  4.4631,  16.0395]
        ])

        # Create elastic tensor from GULP data
        et_original = ElasticTensor(gulp_tensor_data)

        # Test reorientation to standard frame
        et_reoriented = et_original.reoriented_into_standard_frame(gulp_vectors)

        # Test 1: Check that rotation matrix is orthogonal
        from chmpy.crystal.unit_cell import UnitCell
        from chmpy.util.num import kabsch_rotation_matrix

        original_uc = UnitCell(gulp_vectors)
        standard_uc = UnitCell(np.eye(3))
        standard_uc.set_lengths_and_angles(
            [original_uc.a, original_uc.b, original_uc.c],
            [original_uc.alpha, original_uc.beta, original_uc.gamma]
        )

        R = kabsch_rotation_matrix(original_uc.direct, standard_uc.direct)

        # Rotation matrix should be orthogonal (R @ R.T = I) and have determinant = 1
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)
        orthogonality_error = np.max(np.abs(R @ R.T - np.eye(3)))
        self.assertLess(orthogonality_error, 1e-10)

        # Test 2: Unit cell parameters should be preserved
        self.assertAlmostEqual(original_uc.a, standard_uc.a, places=10)
        self.assertAlmostEqual(original_uc.b, standard_uc.b, places=10)
        self.assertAlmostEqual(original_uc.c, standard_uc.c, places=10)
        self.assertAlmostEqual(original_uc.alpha, standard_uc.alpha, places=10)
        self.assertAlmostEqual(original_uc.beta, standard_uc.beta, places=10)
        self.assertAlmostEqual(original_uc.gamma, standard_uc.gamma, places=10)
        self.assertAlmostEqual(original_uc.volume(), standard_uc.volume(), places=10)

        # Test 3: Tensor rotation should preserve physical invariants
        averages_orig = et_original.averages()
        averages_reor = et_reoriented.averages()

        # Bulk modulus should be invariant under rotation
        self.assertAlmostEqual(
            averages_orig['bulk_modulus_avg']['hill'],
            averages_reor['bulk_modulus_avg']['hill'],
            places=10
        )

        # Shear modulus should be invariant under rotation
        self.assertAlmostEqual(
            averages_orig['shear_modulus_avg']['hill'],
            averages_reor['shear_modulus_avg']['hill'],
            places=10
        )

        # Young's modulus average should be invariant
        self.assertAlmostEqual(
            averages_orig['youngs_modulus_avg']['hill'],
            averages_reor['youngs_modulus_avg']['hill'],
            places=10
        )

        # Test 4: True 4th-order tensor contractions should be invariant
        # Full contraction: S_iijj (trace of 4th-order tensor)
        full_trace_orig = np.trace(np.trace(et_original.elasticity_tensor, axis1=0, axis2=1), axis1=0, axis2=1)
        full_trace_reor = np.trace(np.trace(et_reoriented.elasticity_tensor, axis1=0, axis2=1), axis1=0, axis2=1)
        self.assertAlmostEqual(full_trace_orig, full_trace_reor, places=10)

        # Test 5: Coordinate transformation accuracy with random fractional coordinates
        np.random.seed(42)  # For reproducible tests
        n_points = 50
        random_frac_coords = np.random.rand(n_points, 3)

        # Convert to Cartesian in both frames
        gulp_cartesian = original_uc.to_cartesian(random_frac_coords)
        standard_cartesian = standard_uc.to_cartesian(random_frac_coords)

        # Apply rotation to GULP Cartesian coordinates
        rotated_gulp_cartesian = gulp_cartesian @ R

        # Should match standard frame coordinates to machine precision
        max_diff = np.max(np.abs(rotated_gulp_cartesian - standard_cartesian))
        self.assertLess(max_diff, 1e-14)

        # Test 6: Verify the reoriented tensor is symmetric
        symmetry_error = np.max(np.abs(et_reoriented.c_voigt - et_reoriented.c_voigt.T))
        self.assertLess(symmetry_error, 1e-10)

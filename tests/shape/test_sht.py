import logging
import unittest
import numpy as np

def function_to_test(theta, phi):
    return (1.0 + 0.01 * np.cos(theta) +
                0.1 * (3.0 * np.cos(theta) * np.cos(theta) - 1.0) +
                (np.cos(phi) + 0.3 * np.sin(phi)) * np.sin(theta) +
                (np.cos(2.0 * phi) + 0.1 * np.sin(2.0 * phi)) * 
                 np.sin(theta) * np.sin(theta) * (7.0 * np.cos(theta) * np.cos(theta) - 1.0) * 3.0/8.0
           )



class SHTTestCase(unittest.TestCase):
    def setUp(self):
        from chmpy.shape import SHT
        self.sht = SHT(4)

    def test_construction(self):
        from chmpy.shape import SHT

        sht = SHT(20)
        self.assertEqual(sht.lmax, 20)

    def test_angular_grid(self):
        xs, ys = self.sht.grid
        self.assertEqual(xs.shape, (16, 9))
        self.assertEqual(ys.shape, (16, 9))

    def test_cartesian_grid(self):
        cartesian_grid = self.sht.grid_cartesian
        self.assertEqual(cartesian_grid[0].shape, (16, 9))

    def test_transform_real(self):
        x, y, z = self.sht.grid_cartesian
        vals = z ** 2 * np.exp(-np.ones(z.shape)) * 0.5
        coeffs = self.sht.analysis(vals)
        reconstructed = self.sht.synthesis(coeffs)
        np.testing.assert_allclose(vals, reconstructed.real)

    def test_transform_cplx(self):
        x, y, z = self.sht.grid_cartesian
        vals = z ** 2 * np.exp(-np.ones(z.shape)) * 0.5
        vals_cplx = vals.astype(np.complex128)
        coeffs = self.sht.analysis(vals_cplx)
        reconstructed = self.sht.synthesis(coeffs)
        np.testing.assert_allclose(vals, reconstructed)

    def test_plot(self):
        x, y, z = self.sht.grid_cartesian
        vals = z ** 2 * np.exp(-np.ones(z.shape)) * 0.5
        from tempfile import TemporaryDirectory
        from chmpy.shape.sht import plot_sphere
        from os.path import join

        with TemporaryDirectory() as tmpdirname:
            plot_sphere(join(tmpdirname, "test.png"), self.sht.grid, vals)

    def test_assoc_legendre(self):
        plm = self.sht.plm
        expected_plm = np.array([
            0.28209479177387814,
            0.24430125595145993,
            -0.07884789131313,
            -0.326529291016351,
            -0.24462907724141,
            0.2992067103010745,
            0.33452327177864455,
            0.06997056236064664,
            -0.25606603842001846,
            0.2897056515173922,
            0.3832445536624809,
            0.18816934037548755,
            0.27099482274755193,
            0.4064922341213279,
            0.24892463950030275,
        ])

        np.testing.assert_allclose(expected_plm, plm.evaluate_batch(0.5))

    def test_analysis_and_reconstruction(self):
        values = self.sht.compute_on_grid(function_to_test)
        values = values.astype(np.complex128)
        expected = np.array([
            np.sqrt(4 * np.pi), # 0 0 
            -np.sqrt(2 * np.pi / 3) - 0.3 * np.sqrt(2 * np.pi / 3) *1j, # 1 -1
            0.01 * np.sqrt(4 * np.pi / 3.0), # 1 0
            -np.sqrt(2 * np.pi / 3) + 0.3 * np.sqrt(2 * np.pi / 3) *1j, # 1 1
            0.0, # 2 -2
            0.0, # 2 -1
            0.1 * np.sqrt(16 * np.pi / 5.0), # 2  0
            0.0, # 2  1
            0.0, # 2  2
            0.0, # 3 -3
            0.0, # 3 -2
            0.0, # 3 -1
            0.0, # 3  0
            0.0, # 3  1
            0.0, # 3  2
            0.0, # 3  3
            0.0, # 4 -4
            0.0, # 4 -3
            0.5 * np.sqrt(2 * np.pi / 5.0) - 0.05 * np.sqrt(2.0 * np.pi / 5.0) * 1j, # 4  2
            0.0, # 4 -1
            0.0, # 4  0
            0.0, # 4  1
            0.5 * np.sqrt(2 * np.pi / 5.0) + 0.05 * np.sqrt(2.0 * np.pi / 5.0) * 1j, # 4  2
            0.0, # 4  3
            0.0, # 4  4
        ], dtype=np.complex128)

        coeffs = self.sht.analysis(values)
        np.testing.assert_allclose(expected, coeffs, atol=1e-12)

        values_synth = self.sht.synthesis(coeffs)
        np.testing.assert_allclose(values, values_synth.real, atol=1e-12)
        np.testing.assert_allclose(np.zeros_like(values), values_synth.imag, atol=1e-12)




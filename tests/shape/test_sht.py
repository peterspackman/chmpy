import logging
import unittest
import numpy as np

try:
    import shtns

    HAVE_SHTNS = True
except ImportError as e:
    HAVE_SHTNS = False


@unittest.skipUnless(HAVE_SHTNS, "requires SHTns library")
class SHTTestCase(unittest.TestCase):
    def setUp(self):
        from chmpy.shape import SHT

        self.sht = SHT(l_max=2)

    def test_construction(self):
        from chmpy.shape import SHT

        sht = SHT(l_max=20)
        self.assertEqual(sht._shtns.lmax, 20)

    def test_mgrid(self):
        xs, ys = self.sht.mgrid
        self.assertEqual(xs.shape, (5, 16))
        self.assertEqual(ys.shape, (5, 16))

    def test_angular_grid(self):
        angular_grid = self.sht.grid
        self.assertEqual(angular_grid.shape, (80, 2))

    def test_cartesian_grid(self):
        cartesian_grid = self.sht.grid_cartesian
        self.assertEqual(cartesian_grid.shape, (80, 3))

    def test_transform_real(self):
        grid = self.sht.grid_cartesian
        vals = grid[:, 2] ** 2 * np.exp(-np.linalg.norm(grid, axis=1)) * 0.5
        coeffs = self.sht.analyse(vals)
        reconstructed = self.sht.synth_real(coeffs)
        np.testing.assert_allclose(vals, reconstructed)

    def test_transform_cplx(self):
        grid = self.sht.grid_cartesian
        vals = grid[:, 2] ** 2 * np.exp(-np.linalg.norm(grid, axis=1)) * 0.5
        vals_cplx = vals.astype(np.complex128)
        coeffs = self.sht.analyse(vals_cplx)
        reconstructed = self.sht.synth_cplx(coeffs)
        np.testing.assert_allclose(vals, reconstructed)

    def test_plot(self):
        grid = self.sht.grid_cartesian
        vals = grid[:, 2] ** 2 * np.exp(-np.linalg.norm(grid, axis=1)) * 0.5
        from tempfile import TemporaryDirectory
        from chmpy.shape.sht import plot_sphere
        from os.path import join

        with TemporaryDirectory() as tmpdirname:
            plot_sphere(join(tmpdirname, "test.png"), self.sht.mgrid, vals)

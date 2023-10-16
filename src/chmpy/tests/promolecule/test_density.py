import unittest
import numpy as np
from os.path import join, dirname
from chmpy import PromoleculeDensity, StockholderWeight
from .. import TEST_FILES


class PromoleculeDensityTestCase(unittest.TestCase):
    pos = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    els = np.ones(2, dtype=int)

    def setUp(self):
        self.dens = PromoleculeDensity((self.els, self.pos))

    def test_construction(self):
        self.assertEqual(self.dens.natoms, 2)
        with self.assertRaises(ValueError):
            dens = PromoleculeDensity((np.ones(2) * 300, self.pos))

    def test_rho(self):
        pts = np.array(self.pos) + (1.0, 0.0, 0.0)
        rho = self.dens.rho(pts)
        # this will only be true if the interpolator data remains the same
        expected = np.array((0.375174, 0.005646))
        np.testing.assert_allclose(rho, expected, atol=1e-5)

    def test_bb(self):
        from chmpy import Element

        bbox = self.dens.bb()
        buff = Element[1].vdw + 3.8
        expected = np.array(((-4.89, -4.89, -4.89), (5.89, 4.89, 4.89)))
        np.testing.assert_allclose(bbox, expected, atol=1e-5)

    def test_repr(self):
        self.assertEqual(
            self.dens.__repr__(),
            "<PromoleculeDensity: 2 atoms, centre=(0.5, 0.0, 0.0)>",
        )

    def test_d_norm(self):
        pts = np.array(self.pos) + (1.0, 0.0, 0.0)
        d, d_norm, vecs = self.dens.d_norm(pts)
        expected = np.array((-1.0, -0.082569))
        expected_d = np.array((0.0, 1.0))
        expected_vecs = np.array(((-0.917431, 0, 0), (-1.834862, 0, 0)))
        np.testing.assert_allclose(d_norm, expected, atol=1e-5)
        np.testing.assert_allclose(d, expected_d, atol=1e-5)
        np.testing.assert_allclose(vecs, expected_vecs, atol=1e-5)

    def test_from_xyz_file(self):
        dens = PromoleculeDensity.from_xyz_file(TEST_FILES["water.xyz"])


class StockholderWeightTestCase(unittest.TestCase):
    pos = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    els = np.ones(2, dtype=int)

    def setUp(self):
        self.stock = StockholderWeight(
            PromoleculeDensity((self.els[:1], self.pos[:1, :])),
            PromoleculeDensity((self.els[1:], self.pos[1:, :])),
        )

    def test_construction(self):
        np.testing.assert_allclose(self.stock.positions, self.pos)
        np.testing.assert_allclose(self.stock.vdw_radii, [1.09, 1.09])
        self.stock = StockholderWeight.from_arrays(
            self.els[:1], self.pos[:1, :], self.els[1:], self.pos[1:, :]
        )
        np.testing.assert_allclose(self.stock.positions, self.pos)
        np.testing.assert_allclose(self.stock.vdw_radii, [1.09, 1.09])

    def test_weights(self):
        pts = np.array(((0.5, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, -1.0, 0.0)))
        np.testing.assert_allclose(self.stock.weights(pts), 0.5)

    def test_d_norm(self):
        pts = np.array(((0.5, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, -1.0, 0.0)))
        d_a, d_b, d_norm_a, d_norm_b, dp, angles = self.stock.d_norm(pts)
        expected_d_norm = (-0.541284, 0.025719, 0.025719)
        expected_d = (0.5, 1.118034, 1.118034)
        np.testing.assert_allclose(d_norm_a, expected_d_norm, atol=1e-5)
        np.testing.assert_allclose(d_norm_b, expected_d_norm, atol=1e-5)
        np.testing.assert_allclose(d_a, expected_d, atol=1e-5)
        np.testing.assert_allclose(d_b, expected_d, atol=1e-5)

    def test_from_xyz_files(self):
        stock = StockholderWeight.from_xyz_files(
            TEST_FILES["water.xyz"], TEST_FILES["water.xyz"]
        )
        pts = np.array(
            (
                (-0.7021961, -0.0560603, 0.0099423),
                (-1.0221932, 0.8467758, -0.0114887),
                (0.2575211, 0.0421215, 0.0052190),
            )
        )
        pts = np.vstack((pts, pts))
        np.testing.assert_allclose(stock.positions, pts)
        np.testing.assert_allclose(stock.vdw_radii, [1.52, 1.09, 1.09] * 2)

    def test_bb(self):
        bbox = self.stock.bb()
        expected = np.array(((-4.89, -4.89, -4.89), (4.89, 4.89, 4.89)))
        np.testing.assert_allclose(bbox, expected, atol=1e-5)

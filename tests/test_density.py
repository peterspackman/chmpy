import logging
import unittest
import numpy as np
from os.path import join, dirname
from tempfile import TemporaryDirectory
from shmolecule.density import PromoleculeDensity, StockholderWeight


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
        expected = np.array((0.076777, 0.00074))
        np.testing.assert_allclose(rho, expected, atol=1e-5)

    def test_bb(self):
        from shmolecule.element import Element

        bbox = self.dens.bb()
        buff = Element[1].vdw + 2.5
        expected = np.array(((-buff, -buff, -buff), (1.0 + buff, buff, buff)))
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
        dens = PromoleculeDensity.from_xyz_file(join(dirname(__file__), "water.xyz"))

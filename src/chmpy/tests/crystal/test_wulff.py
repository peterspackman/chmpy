import logging
import unittest
import numpy as np
from os.path import join, dirname
from chmpy.crystal.wulff import WulffConstruction, WulffSHT

LOG = logging.getLogger(__name__)

class WulffConstructionTestCase(unittest.TestCase):
    def test_cube(self):
        facets = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))
        facet_energies = np.ones(6)
        w = WulffConstruction(facets, facet_energies)
        self.assertEqual(len(w.wulff_facets), 6)
        self.assertEqual(len(w.wulff_vertices), 8)

        mesh = w.to_trimesh()
        self.assertAlmostEqual(mesh.volume, 8.0)
        self.assertAlmostEqual(mesh.area, 24.0)

    def test_cube_sht(self):
        facets = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))
        facet_energies = np.ones(6)
        s = WulffSHT(facets, facet_energies, l_max=2)
        coeffs_real_expected = np.array(
            (4.27895414e+00, -5.55111512e-17,
             -7.04135032e-03,  2.05657853e-01,
             0.00000000e+00,  3.79056240e-02))


        np.testing.assert_allclose(s.coeffs.real, coeffs_real_expected, rtol=1e-6, atol=1e-7)
        
    def test_cube_sht_invariants(self):
        from scipy.spatial.transform import Rotation
        facets = np.array((
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))

        facet_energies = np.ones(6)
        facet_energies[0] = 5

        l_max = 20
        s = WulffSHT(facets, facet_energies, l_max=l_max)
        invariants_expected = s.power_spectrum()
        coeffs = None

        for rot in Rotation.random(100):
            facets_r = facets @ rot.as_matrix()
            s = WulffSHT(facets_r, facet_energies, l_max=l_max)
            inv = s.power_spectrum()
            coeffs = s.coeffs
            np.testing.assert_allclose(inv, invariants_expected, rtol=1e-3, atol=1e-1)




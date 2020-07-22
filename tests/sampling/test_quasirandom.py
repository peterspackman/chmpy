import logging
import unittest
import numpy as np
from chmpy import Element
from chmpy.sampling import (
    quasirandom_kgf as kgf,
    quasirandom_sobol as sobol,
    quasirandom_kgf_batch as kgf_b,
    quasirandom_sobol_batch as sobol_b,
)


class QuasirandomTestCase(unittest.TestCase):
    def test_batch_single_equivalent(self):
        start, end = 1, 1000
        for dims in range(1, 10):
            for single, batch in ((kgf, kgf_b), (sobol, sobol_b)):
                pts_single = np.vstack(
                    [single(seed, dims) for seed in range(start, end + 1)]
                )
                pts_batch = batch(start, end, dims)
                np.testing.assert_allclose(pts_single, pts_batch)

    def test_approx_circle_area(self):
        start, end = 1, 1000
        for method in (kgf_b, sobol_b):
            pts = method(start, end, 2)
            norms = np.linalg.norm(pts, axis=1)
            ratio_inside = np.sum(norms <= 1.0) / norms.shape[0]
            estimated_area = 4 * ratio_inside
            self.assertAlmostEqual(estimated_area, np.pi, places=2)

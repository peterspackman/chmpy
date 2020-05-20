import logging
import unittest
import numpy as np
from os.path import join, dirname
from chmpy.interpolate import InterpolatorLog1D


class InterpolatorLog1DTestCase(unittest.TestCase):
    def test_interp(self):
        x = np.logspace(0, 1, 1000, dtype=np.float32)
        y = np.exp(-2 * x)
        interp = InterpolatorLog1D(x, y)
        np.testing.assert_allclose(y, interp(x))
        v = 9 * np.random.rand(100).astype(np.float32) + 1
        np.testing.assert_allclose(np.exp(-2 * v), interp(v), atol=1e-5, rtol=0.01)

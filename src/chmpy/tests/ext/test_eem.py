from chmpy import Molecule
from chmpy.ext.charges import EEM
import unittest
from .. import TEST_FILES
import numpy as np


class EEMTestCase(unittest.TestCase):
    def setUp(self):
        self.eem = EEM()
        self.water = Molecule.load(TEST_FILES["water.xyz"])

    def test_calculate_charges(self):
        charges = self.eem.calculate_charges(self.water)
        np.testing.assert_allclose(charges, [-0.64, 0.32, 0.32], atol=1e-2, rtol=1e-2)

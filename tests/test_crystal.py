import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.crystal import Crystal

_ICE_II = join(dirname(__file__), "iceII.cif")

class CrystalTestCase(unittest.TestCase):
    def test_crystal_load(self):
        c = Crystal.load(_ICE_II)

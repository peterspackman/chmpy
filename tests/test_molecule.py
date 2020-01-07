import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.molecule import Molecule

_WATER = join(dirname(__file__), "water.xyz")


class MoleculeTestCase(unittest.TestCase):
    def test_xyz_file_read(self):
        mol = Molecule.load(_WATER)
        assert len(mol) == 3
        assert mol.positions.shape == (3, 3)

    def test_molecule_centroid(self):
        mol = Molecule.load(_WATER)
        cent = mol.centroid
        np.testing.assert_allclose(
            cent, (-0.488956, 0.277612, 0.001224), rtol=1e-3, atol=1e-5
        )
        com = mol.center_of_mass
        np.testing.assert_allclose(
            com, (-0.6664043, -0.0000541773, 0.008478989), rtol=1e-3, atol=1e-5
        )

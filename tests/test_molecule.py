import unittest
from shmolecule.molecule import Molecule
from shmolecule.xyz_file import parse_xyz_file

_WATER = """3
0 1
O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190"""

class MoleculeTestCase(unittest.TestCase):
    def test_xyz_file_read(self):
        mol = Molecule.from_xyz_string(_WATER)
        assert len(mol) == 3
        assert mol.positions.shape == (3, 3)

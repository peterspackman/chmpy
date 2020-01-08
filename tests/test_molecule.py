import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.molecule import Molecule
from tempfile import TemporaryDirectory
import logging

_WATER = join(dirname(__file__), "water.xyz")
LOG = logging.getLogger(__name__)


class MoleculeTestCase(unittest.TestCase):
    pos = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    els = np.ones(2, dtype=int)

    def test_construction(self):
        bonds = np.diag(np.ones(2))
        labels = np.array(["H1", "H2"])
        m = Molecule.from_arrays(self.els, self.pos, bonds=bonds, labels=labels)

    def test_distances(self):
        m1 = Molecule.load(_WATER)
        m2 = Molecule.load(_WATER)
        m2.positions += (0, 3.0, 0)
        self.assertAlmostEqual(m1.distance_to(m2, method="center_of_mass"), 3.0)
        self.assertAlmostEqual(
            m1.distance_to(m2, method="nearest_atom"), 2.121545157481363
        )
        self.assertAlmostEqual(m1.distance_to(m2, method="centroid"), 3.0)
        with self.assertRaises(ValueError):
            m1.distance_to(m2, method="unjknaskldfj")

    def test_xyz_file_read(self):
        mol = Molecule.load(_WATER)
        self.assertTrue(len(mol) == 3)
        self.assertTrue(mol.positions.shape == (3, 3))
        self.assertTrue(mol.molecular_formula == "H2O")

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

    def test_repr(self):
        mol = Molecule.load(_WATER)
        expected = "<Molecule: H2O(-0.67,-0.00,0.01)>"
        self.assertEqual(repr(mol), expected)

    def test_save(self):
        c = Molecule.load(_WATER)
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            c.save(join(tmpdirname, "tmp.xyz"))
            c.save(join(tmpdirname, "tmp.xyz"), header=False)

    def test_bbox(self):
        mol = Molecule.load(_WATER)
        bbox = mol.bbox_corners
        expected = (np.min(mol.positions, axis=0), np.max(mol.positions, axis=0))
        np.testing.assert_allclose(bbox, expected, atol=1e-5)
        np.testing.assert_allclose(mol.bbox_size, expected[1] - expected[0])

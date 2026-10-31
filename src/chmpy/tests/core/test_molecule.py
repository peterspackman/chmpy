import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chmpy import Molecule

from .. import TEST_FILES

LOG = logging.getLogger(__name__)
_WATER = None


class MoleculeTestCase(unittest.TestCase):
    pos = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    els = np.ones(2, dtype=int)

    @staticmethod
    def load_water():
        global _WATER
        from copy import deepcopy

        if _WATER is None:
            _WATER = Molecule.load(TEST_FILES["water.xyz"])
        return deepcopy(_WATER)

    def test_construction(self):
        bonds = np.diag(np.ones(2))
        labels = np.array(["H1", "H2"])
        Molecule.from_arrays(self.els, self.pos, bonds=bonds, labels=labels)

    def test_distances(self):
        m1 = self.load_water()
        m2 = self.load_water()
        m2.positions += (0, 3.0, 0)
        self.assertAlmostEqual(m1.distance_to(m2, method="center_of_mass"), 3.0)
        self.assertAlmostEqual(
            m1.distance_to(m2, method="nearest_atom"), 2.121545157481363
        )
        self.assertAlmostEqual(m1.distance_to(m2, method="centroid"), 3.0)
        with self.assertRaises(ValueError):
            m1.distance_to(m2, method="unjknaskldfj")

    def test_xyz_file_read(self):
        mol = self.load_water()
        self.assertEqual(len(mol), 3)
        self.assertEqual(mol.positions.shape, (3, 3))
        self.assertEqual(mol.molecular_formula, "H2O")

    def test_sdf_file_read(self):
        mol = Molecule.load(TEST_FILES["DB09563.sdf"])
        self.assertEqual(len(mol), 21)
        self.assertEqual(mol.positions.shape, (21, 3))
        self.assertEqual(mol.molecular_formula, "C5H14NO")

    def test_molecule_centroid(self):
        mol = self.load_water()
        cent = mol.centroid
        np.testing.assert_allclose(
            cent, (-0.488956, 0.277612, 0.001224), rtol=1e-3, atol=1e-5
        )
        com = mol.center_of_mass
        np.testing.assert_allclose(
            com, (-0.6664043, -0.0000541773, 0.008478989), rtol=1e-3, atol=1e-5
        )

    def test_mass(self):
        self.assertAlmostEqual(self.load_water().mass, 18.015, places=2)

    def test_repr(self):
        mol = self.load_water()
        expected = "<Molecule (H2O)[-0.67 -0.00 0.01]>"
        self.assertEqual(repr(mol), expected)

    def test_save(self):
        mol = self.load_water()
        with TemporaryDirectory() as tmpdirname:
            LOG.debug("created temp directory: %s", tmpdirname)
            mol.save(Path(tmpdirname, "tmp.xyz"))
            mol.save(Path(tmpdirname, "tmp.xyz"), header=False)

    def test_bbox(self):
        mol = self.load_water()
        bbox = mol.bbox_corners
        expected = (np.min(mol.positions, axis=0), np.max(mol.positions, axis=0))
        np.testing.assert_allclose(bbox, expected, atol=1e-5)
        np.testing.assert_allclose(mol.bbox_size, expected[1] - expected[0])

    def test_to_sdf_string(self):
        from chmpy.fmt.sdf import parse_sdf_contents

        mol = self.load_water()
        sdf = parse_sdf_contents(mol.to_sdf_string() + "\n$$$$\n")[0]
        atoms = sdf["atoms"]
        np.testing.assert_allclose(
            np.c_[atoms["x"], atoms["y"], atoms["z"]], mol.positions, atol=1e-4
        )
        # bonds are guessed if missing, and each one written once
        self.assertEqual(len(sdf["bonds"]["left"]), 2)
        self.assertTrue(np.all(sdf["bonds"]["left"] < sdf["bonds"]["right"]))

    def test_rotational_constants(self):
        mol = self.load_water()
        mhz = mol.rotational_constants()
        self.assertTrue(np.isrealobj(mhz))
        self.assertTrue(np.all(np.diff(mhz) <= 0))
        np.testing.assert_allclose(mol.rotational_constants(unit="GHz"), mhz * 1e-3)
        np.testing.assert_allclose(
            mol.rotational_constants(unit="cm^-1"), mhz / 29979.2458
        )
        with self.assertRaises(ValueError):
            mol.rotational_constants(unit="eV")

    def test_atomic_shape_descriptors(self):
        mol = self.load_water()
        self.assertEqual(mol.atomic_shape_descriptors(l_max=3).shape[0], 3)

    def test_from_aims_string_kwargs(self):
        aims = "atom 0.0 0.0 0.0 O\natom 0.0 0.757 0.586 H\n"
        mol = Molecule.from_aims_string(aims, charge=-1)
        self.assertEqual(mol.charge, -1)

    def test_mol2_aromatic_bonds_and_charges(self):
        mol2 = """@<TRIPOS>MOLECULE
benz
 2 1 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.ar      1  BEN      -0.1000
      2 C2          1.3900    0.0000    0.0000 C.ar      1  BEN       0.1000
@<TRIPOS>BOND
     1     1     2   ar
"""
        mol = Molecule.from_mol2_string(mol2)
        self.assertEqual(mol.charge, 0)
        np.testing.assert_allclose(mol.partial_charges, [-0.1, 0.1])
        self.assertEqual(mol.bonds[0, 1], 1.5)
        self.assertEqual(mol.bonds[1, 0], 1.5)

    def test_pdb_file_read(self):
        mol = Molecule.from_pdb_file(TEST_FILES["HXACAN01.pdb"])
        self.assertEqual(len(mol), 20)


class MolecularAxesTestCase(unittest.TestCase):
    """`axes` returns the principal axes as rows, ordered by extent."""

    @staticmethod
    def bent_molecule():
        # deliberately not aligned with x, y or z, so a frame that merely
        # rotates the coordinates cannot pass by accident
        positions = np.array(
            [
                (0.0, 0.0, 0.0),
                (2.5, 0.4, 0.3),
                (5.0, -0.3, 0.6),
                (1.2, 1.1, -0.4),
            ]
        )
        return Molecule.from_arrays(np.array([6, 6, 6, 1]), positions)

    def test_axes_are_orthonormal_rows(self):
        axes = self.bent_molecule().axes()
        self.assertEqual(axes.shape, (3, 3))
        np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)

    def test_rows_are_the_principal_axes(self):
        """SVD puts them in the columns, so the rows are the transpose."""
        mol = self.bent_molecule()
        u, _, _ = np.linalg.svd((mol.positions - mol.center_of_mass).T)
        # sign is arbitrary; the direction is what matters
        np.testing.assert_allclose(np.abs(mol.axes()), np.abs(u.T), atol=1e-12)

    def test_frame_diagonalises_the_second_moments(self):
        """The defining property of the frame.

        Projecting onto a basis that is not the principal one leaves the
        moments coupled, which is exactly what a transposed frame does. The
        moments are taken about the centre of mass, not the mean position,
        since that is the point the axes are derived about.
        """
        frame = self.bent_molecule().positions_in_molecular_axis_frame()
        moments = frame.T @ frame
        off_diagonal = moments - np.diag(np.diag(moments))
        self.assertLess(
            np.abs(off_diagonal).max(), 1e-10, f"not a principal frame:\n{moments}"
        )

    def test_frame_is_ordered_by_extent(self):
        spread = self.bent_molecule().positions_in_molecular_axis_frame().var(axis=0)
        self.assertTrue(
            np.all(np.diff(spread) <= 1e-12),
            f"principal frame not ordered by extent: {spread}",
        )

    def test_homogeneous_matches_plain(self):
        mol = self.bent_molecule()
        homogeneous = (
            np.c_[mol.positions, np.ones(len(mol))] @ mol.axes(homogeneous=True).T
        )
        np.testing.assert_allclose(
            homogeneous[:, :3],
            mol.positions_in_molecular_axis_frame(),
            atol=1e-12,
        )

    def test_orientation_preserves_geometry(self):
        mol = self.bent_molecule()
        before = mol.distance_matrix
        after = mol.oriented().distance_matrix
        np.testing.assert_allclose(before, after, atol=1e-12)

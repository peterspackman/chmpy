import unittest
from os.path import dirname, join

import numpy as np
from scipy.sparse.csgraph import connected_components

from chmpy import Crystal, Molecule
from chmpy.calc import System
from chmpy.calc.potentials import LennardJones
from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.crystal.slab import Slab, is_molecular, terminations

TEST_FILES = join(dirname(__file__), "..", "test_files")


def rocksalt():
    return Crystal(
        UnitCell.cubic(5.64),
        SpaceGroup(225),
        AsymmetricUnit(
            [Element["Na"], Element["Cl"]], np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        ),
    )


def copper():
    return Crystal(
        UnitCell.cubic(3.61),
        SpaceGroup(225),
        AsymmetricUnit([Element["Cu"]], np.array([[0.0, 0.0, 0.0]])),
    )


class SlabTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nacl = rocksalt()
        cls.cu = copper()
        cls.acetic = Crystal.load(join(TEST_FILES, "acetic_acid.cif"))

    def test_is_molecular(self):
        self.assertFalse(is_molecular(self.nacl))
        self.assertFalse(is_molecular(self.cu))
        self.assertTrue(is_molecular(self.acetic))

    def test_rocksalt_111_has_two_terminations(self):
        result = terminations(self.nacl, (1, 1, 1))
        self.assertEqual(len(result), 2)
        for t in result:
            self.assertAlmostEqual(t.gap, 5.64 / (2 * np.sqrt(3)))

    def test_atom_count_per_layer(self):
        self.assertEqual(len(Slab.from_crystal(self.cu, (1, 0, 0), layers=5)), 5)
        self.assertEqual(len(Slab.from_crystal(self.nacl, (1, 1, 0), layers=3)), 6)

    def test_layers_from_thickness(self):
        slab = Slab.from_crystal(self.cu, (1, 1, 1), thickness=10.0)
        d = 3.61 / np.sqrt(3)
        self.assertEqual(slab.layers, int(np.ceil(10.0 / d)))

    def test_cell_is_in_surface_frame(self):
        slab = Slab.from_crystal(self.cu, (1, 1, 1), layers=3, vacuum=12.0)
        self.assertAlmostEqual(slab.cell[0, 1], 0.0)
        np.testing.assert_allclose(slab.cell[:2, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(slab.cell[2, :2], 0.0, atol=1e-12)
        self.assertAlmostEqual(slab.cell[2, 2], slab.thickness + 12.0)
        z = slab.positions[:, 2]
        self.assertAlmostEqual(z.min() + z.max(), slab.cell[2, 2])

    def test_slab_space_groups(self):
        cases = [
            (self.cu, (1, 0, 0), 3, 123),
            (self.cu, (1, 0, 0), 4, 129),
            (self.cu, (1, 1, 1), 3, 164),
            (self.nacl, (1, 1, 1), 4, 156),
        ]
        for crystal, hkl, layers, number in cases:
            slab = Slab.from_crystal(crystal, hkl, layers=layers)
            self.assertEqual(slab.space_group().sg_number, number, (hkl, layers))

    def test_polar_rocksalt_111(self):
        polar = Slab.from_crystal(self.nacl, (1, 1, 1), layers=4)
        nonpolar = Slab.from_crystal(self.nacl, (1, 0, 0), layers=4)
        self.assertFalse(polar.is_symmetric)
        self.assertTrue(nonpolar.is_symmetric)
        formal = np.array([1.0, -1.0])
        self.assertGreater(abs(polar.dipole(formal)[2]), 1.0)
        self.assertAlmostEqual(nonpolar.dipole(formal)[2], 0.0)

    def test_molecules_kept_whole(self):
        slab = Slab.from_crystal(self.acetic, (1, 1, 1), layers=2)
        n_units = 2 * len(self.acetic.unit_cell_molecules())
        mol = Molecule.from_arrays(slab.elements, slab.positions)
        mol.guess_bonds()
        n_fragments, _ = connected_components(mol.bonds)
        self.assertEqual(n_fragments, n_units)
        self.assertEqual(len(slab), 8 * n_units)

    def test_molecular_surface_symmetry(self):
        slab = Slab.from_crystal(self.acetic, (0, 0, 1), layers=2)
        self.assertEqual(slab.plane_group.symbol, "pg")
        self.assertFalse(slab.is_symmetric)

    def test_fixed_mask(self):
        slab = Slab.from_crystal(self.cu, (1, 0, 0), layers=6)
        self.assertEqual(slab.fixed_mask(2).sum(), 2)
        self.assertTrue(
            np.all(slab.positions[slab.fixed_mask(2), 2] < slab.cell[2, 2] / 2)
        )


class SymmetricSlabTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acetic = Crystal.load(join(TEST_FILES, "acetic_acid.cif"))
        cls.cuts = terminations(cls.acetic, (1, 1, 0))
        cls.calc = LennardJones(epsilon=0.005, sigma=3.0, cutoff=8.0)

    def slab(self, termination, symmetric):
        return Slab.from_crystal(
            self.acetic,
            (1, 1, 0),
            termination=self.cuts[termination],
            layers=5,
            vacuum=20.0,
            symmetric=symmetric,
        )

    def gamma(self, slab):
        bulk = slab.oriented_bulk()
        e_bulk = self.calc(System.from_crystal(bulk)).energy
        e_slab = self.calc(System(slab.elements, slab.positions, slab.cell)).energy
        per_molecule = e_bulk / len(bulk.unit_cell_molecules())
        return (e_slab - slab.n_units * per_molecule) / (2 * slab.area)

    def test_both_faces_share_the_top_termination(self):
        for termination in (1, 2):
            self.assertFalse(self.slab(termination, symmetric=False).is_symmetric)
            self.assertTrue(self.slab(termination, symmetric=True).is_symmetric)

    def test_self_paired_termination_is_unchanged(self):
        plain, symmetric = self.slab(0, False), self.slab(0, True)
        self.assertEqual(len(plain), len(symmetric))
        self.assertAlmostEqual(plain.thickness, symmetric.thickness)

    def test_stoichiometric_slab_averages_the_two_faces(self):
        "Exact for a pair potential, which also gives both faces the same energy."
        average = self.gamma(self.slab(1, symmetric=False))
        faces = [self.gamma(self.slab(t, symmetric=True)) for t in (1, 2)]
        self.assertAlmostEqual(average, self.gamma(self.slab(2, symmetric=False)))
        self.assertAlmostEqual(average, np.mean(faces), places=8)
        self.assertAlmostEqual(faces[0], faces[1], places=8)

    def test_no_symmetric_slab_without_a_face_exchanging_operation(self):
        with self.assertRaises(ValueError):
            Slab.from_crystal(self.acetic, (0, 0, 1), layers=2, symmetric=True)

import unittest
from os.path import dirname, join

import numpy as np

from chmpy import Crystal
from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.crystal.surface_cut import dimer_cut_counts

TEST_FILES = join(dirname(__file__), "..", "test_files")


class DimerCutCountsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.argon = Crystal(
            UnitCell.cubic(5.26),
            SpaceGroup(225),
            AsymmetricUnit([Element["Ar"]], np.array([[0.0, 0.0, 0.0]])),
        )
        cls.acetic = Crystal.load(join(TEST_FILES, "acetic_acid.cif"))

    def test_fcc_broken_nearest_neighbour_bonds(self):
        for hkl, broken in {(1, 1, 1): 3, (1, 0, 0): 4, (1, 1, 0): 6}.items():
            result = dimer_cut_counts(self.argon, hkl, radius=4.0)
            np.testing.assert_array_equal(result.counts, [broken], err_msg=str(hkl))

    def test_fcc_surface_energy_ordering(self):
        energies = {
            hkl: dimer_cut_counts(self.argon, hkl, radius=4.0).surface_energy([-1.0])
            for hkl in [(1, 1, 1), (1, 0, 0), (1, 1, 0)]
        }
        self.assertLess(energies[(1, 1, 1)], energies[(1, 0, 0)])
        self.assertLess(energies[(1, 0, 0)], energies[(1, 1, 0)])

    def test_opposite_faces_break_the_same_pairs(self):
        top = dimer_cut_counts(self.acetic, (0, 0, 1))
        bottom = dimer_cut_counts(self.acetic, (0, 0, -1))
        np.testing.assert_array_equal(top.counts, bottom.counts)

    def test_attachment_energy_per_molecule(self):
        result = dimer_cut_counts(self.argon, (1, 1, 1), radius=4.0)
        self.assertAlmostEqual(result.attachment_energy([-1.0]), -3.0)
        self.assertAlmostEqual(result.cleavage_energy([-1.0]), 3.0)

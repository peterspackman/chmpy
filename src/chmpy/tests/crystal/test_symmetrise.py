import unittest
from os.path import dirname, join

import numpy as np

from chmpy import Crystal
from chmpy.crystal import UnitCell
from chmpy.crystal.symmetrise import symmetrise_unit_cell

TEST_FILES = join(dirname(__file__), "..", "test_files")


class SymmetriseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(join(TEST_FILES, "acetic_acid.cif"))
        cls.frac = cls.crystal.unit_cell_atoms()["frac_pos"]

    def test_unchanged_structure_round_trips(self):
        result = symmetrise_unit_cell(self.crystal, self.crystal.unit_cell, self.frac)
        np.testing.assert_allclose(
            result.asymmetric_unit.positions,
            self.crystal.asymmetric_unit.positions,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result.unit_cell.parameters, self.crystal.unit_cell.parameters
        )

    def test_noise_is_averaged_out(self):
        rng = np.random.default_rng(1)
        noisy = self.frac + rng.normal(scale=1e-3, size=self.frac.shape)
        lengths = np.array(self.crystal.unit_cell.lengths) * 1.01
        angles = np.radians([90.2, 89.9, 90.1])
        cell = UnitCell.from_lengths_and_angles(lengths, angles)
        result = symmetrise_unit_cell(self.crystal, cell, noisy)
        self.assertEqual(result.space_group, self.crystal.space_group)
        np.testing.assert_allclose(
            result.unit_cell.angles, np.radians(90.0), atol=1e-12
        )
        np.testing.assert_allclose(result.unit_cell.lengths, lengths)
        shift = (
            result.asymmetric_unit.positions - self.crystal.asymmetric_unit.positions
        )
        self.assertLess(np.abs(shift).max(), 3e-3)

    def test_rejects_broken_symmetry(self):
        moved = self.frac.copy()
        moved[0] += 0.05
        with self.assertRaises(ValueError):
            symmetrise_unit_cell(self.crystal, self.crystal.unit_cell, moved)

    def test_rejects_wrong_atom_count(self):
        with self.assertRaises(ValueError):
            symmetrise_unit_cell(self.crystal, self.crystal.unit_cell, self.frac[:-1])

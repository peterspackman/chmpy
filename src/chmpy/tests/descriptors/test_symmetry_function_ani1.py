"""Tests for ANI-1 style atomic environment symmetry functions."""

import unittest
from itertools import combinations

import numpy as np

from chmpy import Crystal
from chmpy.descriptors import SymmetryFunctionsANI1
from chmpy.tests import TEST_FILES

RADIAL_CUTOFF = 6.0
ANGULAR_CUTOFF = 4.0


class SymmetryFunctionsANI1TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        cls.symf = SymmetryFunctionsANI1.from_crystal(
            cls.crystal, radial_cutoff=RADIAL_CUTOFF, angular_cutoff=ANGULAR_CUTOFF
        )
        cls.mol = cls.crystal.symmetry_unique_molecules()[0]
        slab = cls.crystal.slab(bounds=((-3, -3, -3), (3, 3, 3)))
        cls.slab_nums = slab["element"]
        cls.slab_pos = slab["cart_pos"]

    def neighbours(self, i, cutoff):
        d = np.linalg.norm(self.slab_pos - self.mol.positions[i], axis=1)
        mask = (d < cutoff) & (d > 1e-3)
        return self.slab_nums[mask], self.slab_pos[mask], d[mask]

    def test_shape(self):
        self.assertEqual(self.symf.n_atoms, len(self.mol))
        self.assertEqual(self.symf.unique_atoms, ["C", "H", "O"])

    def test_radial_matches_reference(self):
        rs = np.linspace(1, 9.3, 32)
        for i in range(len(self.mol)):
            nums, _, d = self.neighbours(i, RADIAL_CUTOFF)
            fc = 0.5 * np.cos(d * np.pi / RADIAL_CUTOFF) + 0.5
            for sym, z in (("C", 6), ("H", 1), ("O", 8)):
                ref = (fc * (nums == z)) @ np.exp(-50 * (d[:, None] - rs) ** 2)
                np.testing.assert_allclose(self.symf.get_radial(i, sym), ref, atol=1e-8)

    def test_angular_matches_reference(self):
        rs = np.linspace(1, 6.27, 8)
        theta = np.linspace(0, np.pi, 8)
        zeta, eta = 70, 5
        i = 0
        nums, pos, d = self.neighbours(i, ANGULAR_CUTOFF)
        vec = pos - self.mol.positions[i]
        fc = 0.5 * np.cos(d * np.pi / ANGULAR_CUTOFF) + 0.5
        for pair in (("C", "O"), ("H", "H"), ("O", "O")):
            z = {"C": 6, "H": 1, "O": 8}
            za, zb = z[pair[0]], z[pair[1]]
            ref = np.zeros((len(rs), len(theta)))
            for j, k in combinations(range(len(d)), 2):
                if sorted((nums[j], nums[k])) != sorted((za, zb)):
                    continue
                cos = np.dot(vec[j], vec[k]) / (d[j] * d[k])
                angle = np.arccos(np.clip(cos, -1, 1))
                radial = np.exp(-eta * (0.5 * (d[j] + d[k]) - rs) ** 2)
                angular = (1 + np.cos(angle - theta)) ** zeta
                ref += fc[j] * fc[k] * np.outer(radial, angular)
            ref *= 2 ** (1 - zeta)
            self.assertGreater(ref.sum(), 0)
            np.testing.assert_allclose(
                self.symf.get_angular(i, pair), ref.ravel(), rtol=1e-8, atol=1e-12
            )


if __name__ == "__main__":
    unittest.main()

import logging
import unittest
import numpy as np
from os.path import join, dirname
from chmpy.crystal import Crystal
from .. import TEST_FILES


class ShapeDescriptorTestCase(unittest.TestCase):
    def setUp(self):
        self.water = Crystal.load(TEST_FILES["iceII.cif"])
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_atomic_descriptors(self):
        desc = self.acetic.atomic_shape_descriptors(l_max=3, radius=3.8)
        self.assertEqual(desc.shape, (8, 8))

    def test_molecular_descriptors(self):
        desc = self.acetic.molecular_shape_descriptors(l_max=3, radius=3.8)
        self.assertEqual(desc.shape, (1, 8))

    def test_atom_group_shape_descriptors(self):
        desc = self.acetic.atom_group_shape_descriptors([0, 1, 2], l_max=3, radius=3.8)
        self.assertEqual(desc.shape, (8,))

    def test_invariants(self):
        from chmpy.shape.shape_descriptors import make_N_invariants, make_invariants

        coeffs = np.random.rand(16).astype(np.complex128)
        inv = make_N_invariants(coeffs)
        self.assertEqual(len(inv), 4)

        coeffs = np.random.rand(26 * 26).astype(np.complex128)
        inv = make_invariants(25, coeffs)
        self.assertEqual(len(inv), 1038)

import logging
import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.crystal import Crystal

_ACETIC = join(dirname(__file__), "acetic_acid.cif")
_ICE_II = join(dirname(__file__), "iceII.cif")

try:
    import shtns

    HAVE_SHTNS = True
except ImportError as e:
    HAVE_SHTNS = False


@unittest.skipUnless(HAVE_SHTNS, "requires SHTns library")
class ShapeDescriptorTestCase(unittest.TestCase):
    def setUp(self):
        self.water = Crystal.load(_ICE_II)
        self.acetic = Crystal.load(_ACETIC)

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
        from shmolecule.shape_descriptors import make_N_invariants, make_invariants

        coeffs = np.random.rand(16).astype(np.complex128)
        inv = make_N_invariants(coeffs, kind="complex")
        self.assertEqual(len(inv), 4)

        coeffs = np.random.rand(26 * 26).astype(np.complex128)
        inv = make_invariants(25, coeffs)
        self.assertEqual(len(inv), 1180)

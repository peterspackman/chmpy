import logging
import unittest
import numpy as np
from os.path import join, dirname
from tempfile import TemporaryDirectory
from chmpy.core.element import (
    Element,
    vdw_radii,
    chemical_formula,
    cov_radii,
    element_names,
    element_symbols,
)


class ElementTestCase(unittest.TestCase):
    def test_construction(self):
        for s in (1, "D", "H", "hydrogen", "1"):
            e = Element[s]
            self.assertEqual(e.atomic_number, 1)
            self.assertEqual(e.symbol, "H")
            self.assertEqual(e.name, "hydrogen")

        for s in ("blah", "32.141", None, 1.5):
            with self.assertRaises(ValueError):
                e = Element[s]

    def test_vdw(self):
        e = Element[1]
        self.assertEqual(e.vdw, 1.09)
        self.assertEqual(e.vdw_radius, 1.09)
        self.assertEqual(e.cov, 0.23)
        self.assertEqual(e.covalent_radius, 0.23)

    def test_comparison(self):
        e1 = Element[1]
        e2 = Element["H"]
        b = Element["B"]
        c = Element["C"]
        o = Element["O"]
        self.assertEqual(e1, e2)

        with self.assertRaises(NotImplementedError):
            e1 == 1.3

        with self.assertRaises(NotImplementedError):
            e1 < 1.3

        self.assertTrue(c < o)
        self.assertTrue(b < o)
        self.assertTrue(c < e1)
        self.assertFalse(o < c)

    def test_chemical_formulat(self):
        self.assertEqual(chemical_formula(("H", "O", "H")), "H2O")
        self.assertEqual(chemical_formula(("H", "O", "H"), subscript=True), "H\u2082O")

    def test_radii(self):
        nums_valid = np.array([1, 2, 3])
        nums_invalid = np.array([-1, 105])
        np.testing.assert_allclose(cov_radii(nums_valid), [0.23, 1.5, 1.28])
        np.testing.assert_allclose(vdw_radii(nums_valid), [1.09, 1.4, 1.82])
        np.testing.assert_equal(element_symbols(nums_valid), ["H", "He", "Li"])
        np.testing.assert_equal(
            element_names(nums_valid), ["hydrogen", "helium", "lithium"]
        )

        for m in (cov_radii, vdw_radii, element_symbols, element_names):
            with self.assertRaises(ValueError):
                x = m(nums_invalid)

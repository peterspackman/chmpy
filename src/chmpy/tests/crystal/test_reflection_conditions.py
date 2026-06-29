"""Tests for systematic-absence (reflection-condition) detection.

Cross-checks SpaceGroup.is_systematically_absent against the reflection
conditions printed in the International Tables for Crystallography, across all
crystal systems (centering, glide-plane and screw-axis conditions).
"""

import logging
import unittest
from itertools import product

import numpy as np

from chmpy.crystal import SpaceGroup

LOG = logging.getLogger(__name__)


class ReflectionConditionTestCase(unittest.TestCase):
    def assert_absent(self, sg_number, hkl, expected, msg=""):
        got = SpaceGroup(sg_number).is_systematically_absent(hkl)
        self.assertEqual(bool(got), expected, f"#{sg_number} {hkl}: {msg}")

    def test_no_conditions_triclinic(self):
        for hkl in [(1, 2, 3), (0, 0, 1), (1, 0, 0), (0, 1, 0)]:
            self.assert_absent(1, hkl, False, "P1 has no conditions")
            self.assert_absent(2, hkl, False, "P-1 has no conditions")

    def test_screw_axes(self):
        # 2_1 || b (#4): 0k0 present iff k even
        self.assert_absent(4, (0, 1, 0), True, "2_1||b: 0k0 k=2n")
        self.assert_absent(4, (0, 2, 0), False)
        # P2_12_12_1 (#19): h00 h=2n, 0k0 k=2n, 00l l=2n
        self.assert_absent(19, (1, 0, 0), True)
        self.assert_absent(19, (0, 1, 0), True)
        self.assert_absent(19, (0, 0, 1), True)
        self.assert_absent(19, (2, 0, 0), False)
        # 4_1 || c (#76): 00l l=4n
        self.assert_absent(76, (0, 0, 1), True)
        self.assert_absent(76, (0, 0, 2), True)
        self.assert_absent(76, (0, 0, 4), False)
        # 6_1 || c (#169): 00l l=6n
        for l in (1, 2, 3, 4, 5):
            self.assert_absent(169, (0, 0, l), True)
        self.assert_absent(169, (0, 0, 6), False)
        # 6_3 || c (#173): 00l l=2n
        self.assert_absent(173, (0, 0, 1), True)
        self.assert_absent(173, (0, 0, 2), False)

    def test_glide_planes(self):
        # Pc (#7): h0l present iff l even
        self.assert_absent(7, (1, 0, 1), True)
        self.assert_absent(7, (1, 0, 2), False)
        # Pbca (#61): 0kl k=2n, h0l l=2n, hk0 h=2n
        self.assert_absent(61, (0, 1, 1), True)
        self.assert_absent(61, (0, 2, 1), False)
        self.assert_absent(61, (1, 0, 1), True)
        self.assert_absent(61, (1, 0, 2), False)
        self.assert_absent(61, (1, 1, 0), True)
        self.assert_absent(61, (2, 1, 0), False)

    def test_centering(self):
        # I (#71, Immm): h+k+l = 2n
        self.assert_absent(71, (1, 0, 0), True)
        self.assert_absent(71, (1, 1, 0), False)
        # F (#69, Fmmm): h,k,l all even or all odd
        self.assert_absent(69, (1, 0, 0), True)
        self.assert_absent(69, (1, 1, 1), False)
        self.assert_absent(69, (2, 0, 0), False)
        # C (#65, Cmmm): h+k = 2n
        self.assert_absent(65, (1, 0, 0), True)
        self.assert_absent(65, (1, 1, 0), False)
        # R obverse on hexagonal axes (#146, R3): -h+k+l = 3n
        self.assert_absent(146, (1, 0, 0), True)  # -1+0+0 = -1, not 3n
        self.assert_absent(146, (1, 1, 1), True)  # -1+1+1 = 1, not 3n
        self.assert_absent(146, (0, 0, 3), False)  # 0+0+3 = 3n

    def test_cubic_centering(self):
        # Fm-3m (#225)
        self.assert_absent(225, (1, 0, 0), True)
        self.assert_absent(225, (1, 1, 1), False)
        self.assert_absent(225, (2, 0, 0), False)
        # Im-3m (#229)
        self.assert_absent(229, (1, 0, 0), True)
        self.assert_absent(229, (1, 1, 0), False)

    def test_scalar_and_array_inputs(self):
        sg = SpaceGroup(14)
        # scalar input -> python bool
        result = sg.is_systematically_absent((0, 1, 0))
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
        # array input -> (N,) bool array
        arr = sg.is_systematically_absent([[0, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [True, False, True])

    def test_origin_reflection_never_absent(self):
        for n in range(1, 231):
            self.assertFalse(
                SpaceGroup(n).is_systematically_absent((0, 0, 0)),
                f"#{n}: 000 must never be absent",
            )

    def test_absences_respect_point_group(self):
        """Symmetry-equivalent reflections share absence status.

        For every rotation R, h and hR have equal structure-factor magnitude,
        so they must be absent or present together.
        """
        hkls = np.array([h for h in product(range(-3, 4), repeat=3) if any(h)])
        for n in (4, 14, 61, 76, 167, 230):
            sg = SpaceGroup(n)
            absent = sg.is_systematically_absent(hkls)
            with self.subTest(sg=n):
                for op in sg.symmetry_operations:
                    rotation = np.round(op.rotation).astype(int)
                    # h and hR are symmetry-equivalent: same absence status
                    moved = sg.is_systematically_absent(hkls @ rotation)
                    self.assertTrue(
                        np.array_equal(absent, moved),
                        f"#{n}: absence not invariant under {op}",
                    )


class ReflectionConditionTableTestCase(unittest.TestCase):
    """Tests for the ITA-style reflection_conditions() table."""

    def conditions(self, sg_number):
        return [str(c) for c in SpaceGroup(sg_number).reflection_conditions()]

    def test_no_conditions(self):
        self.assertEqual(self.conditions(1), [])  # P1
        self.assertEqual(self.conditions(2), [])  # P-1

    def test_monoclinic_orthorhombic(self):
        # values match the International Tables reflection-condition blocks
        self.assertEqual(self.conditions(14), ["h0l: l=2n", "0k0: k=2n"])  # P2_1/c
        self.assertEqual(self.conditions(33), ["0kl: k+l=2n", "h0l: h=2n"])  # Pna2_1
        self.assertEqual(
            self.conditions(61), ["0kl: k=2n", "h0l: l=2n", "hk0: h=2n"]
        )  # Pbca
        self.assertEqual(self.conditions(62), ["0kl: k+l=2n", "hk0: h=2n"])  # Pnma

    def test_screw_axis_only(self):
        self.assertEqual(self.conditions(4), ["0k0: k=2n"])  # P2_1
        self.assertEqual(self.conditions(76), ["00l: l=4n"])  # P4_1
        self.assertEqual(self.conditions(169), ["00l: l=6n"])  # P6_1
        self.assertEqual(self.conditions(173), ["00l: l=2n"])  # P6_3

    def test_cubic_glides(self):
        # Pa-3: the textbook cyclic a/b/c glide conditions
        self.assertEqual(self.conditions(205), ["0kl: k=2n", "h0l: l=2n", "hk0: h=2n"])

    def test_centering_appears_in_general_class(self):
        # body-centered: general condition h+k+l=2n
        conds = SpaceGroup(229).reflection_conditions()  # Im-3m
        self.assertEqual(conds[0].reflection_class, "hkl")
        self.assertEqual(conds[0].condition, "h+k+l=2n")

    def test_diamond_glide(self):
        # Fdd2: F-centering on hkl, d-glide gives k+l=4n / h+l=4n on the zones
        by_class = {
            c.reflection_class: c.condition
            for c in SpaceGroup(43).reflection_conditions()
        }
        self.assertEqual(by_class["0kl"], "k+l=4n")
        self.assertEqual(by_class["h0l"], "h+l=4n")

    def test_condition_objects(self):
        (cond,) = SpaceGroup(4).reflection_conditions()  # P2_1: only 0k0: k=2n
        self.assertTrue(cond.applies_to((0, 2, 0)))
        self.assertFalse(cond.applies_to((1, 2, 0)))
        self.assertTrue(cond.is_present((0, 2, 0)))
        self.assertFalse(cond.is_present((0, 1, 0)))

    def test_general_class_listed_first(self):
        # the general (centering) condition on hkl must precede the special ones
        for n in (88, 142, 227, 230):
            conds = SpaceGroup(n).reflection_conditions()
            self.assertEqual(conds[0].reflection_class, "hkl")

    def test_table_reconstructs_engine_absences(self):
        """The listed conditions must reproduce exactly the engine's absences.

        For every space group and every reflection in a box, a reflection is
        absent iff it belongs to some listed class and violates that class's
        condition.
        """
        box = np.array([h for h in product(range(-3, 4), repeat=3) if any(h)])
        for n in range(1, 231):
            sg = SpaceGroup(n)
            engine_absent = sg.is_systematically_absent(box)
            reconstructed = np.zeros(len(box), dtype=bool)
            for cond in sg.reflection_conditions():
                basis = np.array(cond.basis, dtype=float)
                if basis.shape[1] == 3:
                    member = np.ones(len(box), dtype=bool)
                else:
                    pinv = np.linalg.pinv(basis)
                    params = np.rint(box @ pinv.T).astype(int)
                    member = np.all(params @ basis.T.astype(int) == box, axis=1)
                present = np.ones(len(box), dtype=bool)
                for coeffs, m in cond.forms:
                    present &= (box @ np.array(coeffs)) % m == 0
                reconstructed |= member & ~present
            with self.subTest(sg=n):
                self.assertTrue(
                    np.array_equal(engine_absent, reconstructed),
                    f"#{n} {sg.symbol}: table does not reconstruct engine absences",
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

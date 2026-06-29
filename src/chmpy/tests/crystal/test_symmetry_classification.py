"""Tests for symmetry-operation classification and symmorphic detection.

Covers the geometric classification of individual symmetry operations
(geometric_type, rotation_symbol, rotation_order, intrinsic_translation) and
the SpaceGroup.is_symmorphic predicate.
"""

import logging
import unittest
from collections import Counter

import numpy as np

from chmpy.crystal import SpaceGroup, SymmetryOperation

LOG = logging.getLogger(__name__)

# The 73 symmorphic space groups by ITA number (Hahn, International Tables Vol. A).
SYMMORPHIC_SPACE_GROUPS = frozenset(
    {
        1,
        2,
        3,
        5,
        6,
        8,
        10,
        12,
        16,
        21,
        22,
        23,
        25,
        35,
        38,
        42,
        44,
        47,
        65,
        69,
        71,
        75,
        79,
        81,
        82,
        83,
        87,
        89,
        97,
        99,
        107,
        111,
        115,
        119,
        121,
        123,
        139,
        143,
        146,
        147,
        148,
        149,
        150,
        155,
        156,
        157,
        160,
        162,
        164,
        166,
        168,
        174,
        175,
        177,
        183,
        187,
        189,
        191,
        195,
        196,
        197,
        200,
        202,
        204,
        207,
        209,
        211,
        215,
        216,
        217,
        221,
        225,
        229,
    }
)

GEOMETRIC_TYPES = {
    "identity",
    "translation",
    "rotation",
    "screw",
    "inversion",
    "mirror",
    "glide",
    "rotoinversion",
}
ROTATION_SYMBOLS = {"1", "2", "3", "4", "6", "-1", "m", "-3", "-4", "-6"}


def op(code):
    "Build a SymmetryOperation from a string code, e.g. '-x,1/2+y,-z'."
    return SymmetryOperation.from_string_code(code)


class GeometricTypeTestCase(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(op("x,y,z").geometric_type, "identity")

    def test_inversion(self):
        self.assertEqual(op("-x,-y,-z").geometric_type, "inversion")

    def test_pure_rotation(self):
        self.assertEqual(op("-x,-y,z").geometric_type, "rotation")  # 2-fold
        self.assertEqual(op("-y,x-y,z").geometric_type, "rotation")  # 3-fold

    def test_screw(self):
        self.assertEqual(op("-x,1/2+y,-z").geometric_type, "screw")  # 2_1

    def test_mirror(self):
        self.assertEqual(op("x,-y,z").geometric_type, "mirror")

    def test_glide(self):
        self.assertEqual(op("x,-y,1/2+z").geometric_type, "glide")  # c-glide

    def test_rotoinversion(self):
        # -4 about c: (y, -x, -z)
        self.assertEqual(op("y,-x,-z").geometric_type, "rotoinversion")

    def test_translation(self):
        # identity rotation part with a centering translation
        self.assertEqual(op("1/2+x,1/2+y,1/2+z").geometric_type, "translation")

    def test_p21c_operations(self):
        types = Counter(o.geometric_type for o in SpaceGroup(14).symmetry_operations)
        self.assertEqual(
            types, Counter({"identity": 1, "inversion": 1, "screw": 1, "glide": 1})
        )


class RotationSymbolTestCase(unittest.TestCase):
    def test_proper(self):
        self.assertEqual(op("x,y,z").rotation_symbol, "1")
        self.assertEqual(op("-x,-y,z").rotation_symbol, "2")
        self.assertEqual(op("-y,x-y,z").rotation_symbol, "3")
        self.assertEqual(op("y,-x,z").rotation_symbol, "4")
        self.assertEqual(op("x-y,x,z").rotation_symbol, "6")

    def test_improper(self):
        self.assertEqual(op("-x,-y,-z").rotation_symbol, "-1")
        self.assertEqual(op("x,-y,z").rotation_symbol, "m")
        self.assertEqual(op("y,-x,-z").rotation_symbol, "-4")

    def test_screw_has_rotation_symbol_of_rotation_part(self):
        # a 2_1 screw still has rotational part "2"
        self.assertEqual(op("-x,1/2+y,-z").rotation_symbol, "2")

    def test_glide_has_rotation_symbol_m(self):
        self.assertEqual(op("x,-y,1/2+z").rotation_symbol, "m")


class RotationOrderTestCase(unittest.TestCase):
    def test_orders(self):
        self.assertEqual(op("x,y,z").rotation_order, 1)
        self.assertEqual(op("-x,-y,z").rotation_order, 2)
        self.assertEqual(op("x,-y,z").rotation_order, 2)  # mirror
        self.assertEqual(op("-x,-y,-z").rotation_order, 2)  # inversion
        self.assertEqual(op("-y,x-y,z").rotation_order, 3)
        self.assertEqual(op("y,-x,z").rotation_order, 4)
        self.assertEqual(op("x-y,x,z").rotation_order, 6)

    def test_rotoinversion_order_is_matrix_order(self):
        self.assertEqual(op("y,-x,-z").rotation_order, 4)  # -4
        self.assertEqual(op("y,y-x,-z").rotation_order, 6)  # -3 (3-bar)


class IntrinsicTranslationTestCase(unittest.TestCase):
    def test_screw(self):
        np.testing.assert_allclose(
            op("-x,1/2+y,-z").intrinsic_translation, [0.0, 0.5, 0.0]
        )

    def test_glide(self):
        np.testing.assert_allclose(
            op("x,-y,1/2+z").intrinsic_translation, [0.0, 0.0, 0.5]
        )

    def test_zero_for_pure_rotation(self):
        np.testing.assert_allclose(op("-x,-y,z").intrinsic_translation, [0.0, 0.0, 0.0])

    def test_zero_for_inversion(self):
        np.testing.assert_allclose(
            op("-x,-y,-z").intrinsic_translation, [0.0, 0.0, 0.0]
        )

    def test_has_intrinsic_translation(self):
        self.assertTrue(op("-x,1/2+y,-z").has_intrinsic_translation)
        self.assertTrue(op("x,-y,1/2+z").has_intrinsic_translation)
        self.assertFalse(op("-x,-y,z").has_intrinsic_translation)
        self.assertFalse(op("-x,-y,-z").has_intrinsic_translation)

    def test_intrinsic_translation_removes_location_part(self):
        # Two 2_1 screws about b differing only by a removable location shift
        # along a have the same intrinsic translation.
        a = op("-x,1/2+y,-z")
        b = op("1/2-x,1/2+y,-z")
        np.testing.assert_allclose(a.intrinsic_translation, b.intrinsic_translation)


class IsSymmorphicTestCase(unittest.TestCase):
    def test_canonical_73(self):
        """is_symmorphic must reproduce exactly the 73 symmorphic space groups."""
        found = {n for n in range(1, 231) if SpaceGroup(n).is_symmorphic}
        self.assertEqual(found, set(SYMMORPHIC_SPACE_GROUPS))

    def test_i222_vs_i212121(self):
        """Distinguish #23 (symmorphic) from #24 (non-symmorphic).

        Both are body-centered with point group 222; a per-rotation test would
        wrongly accept #24, whose screw axes share no common origin.
        """
        self.assertTrue(SpaceGroup(23).is_symmorphic)
        self.assertFalse(SpaceGroup(24).is_symmorphic)

    def test_spot_checks(self):
        for n in (1, 2, 5, 47, 69, 225):  # P1, P-1, C2, Pmmm, Fmmm, Fm-3m
            self.assertTrue(SpaceGroup(n).is_symmorphic, f"#{n} should be symmorphic")
        for n in (4, 9, 14, 33, 76, 227):  # P2_1, Cc, P2_1/c, Pna2_1, P4_1, Fd-3m
            self.assertFalse(
                SpaceGroup(n).is_symmorphic, f"#{n} should be non-symmorphic"
            )


class ClassificationSweepTestCase(unittest.TestCase):
    """Consistency invariants across all 230 space groups."""

    def test_all_operations_classify(self):
        for n in range(1, 231):
            sg = SpaceGroup(n)
            with self.subTest(sg=n):
                identity_count = 0
                has_inversion = False
                for o in sg.symmetry_operations:
                    self.assertIn(o.geometric_type, GEOMETRIC_TYPES)
                    self.assertIn(o.rotation_symbol, ROTATION_SYMBOLS)
                    self.assertIn(o.rotation_order, {1, 2, 3, 4, 6})
                    tau = o.intrinsic_translation
                    self.assertEqual(tau.shape, (3,))
                    self.assertTrue(np.all((tau >= 0) & (tau < 1)))
                    if o.geometric_type == "identity":
                        identity_count += 1
                    if o.geometric_type == "inversion":
                        has_inversion = True
                # exactly one identity operation per group
                self.assertEqual(identity_count, 1)
                # an inversion operation is present iff the group is centrosymmetric
                self.assertEqual(has_inversion, sg.centrosymmetric)

    def test_primitive_symmorphic_has_no_screw_or_glide(self):
        """For primitive lattices, symmorphic <=> no screw/glide operations.

        (In centered cells a rotation combined with a centering translation can
        be reported as a screw/glide, so this only holds for primitive groups.)
        """
        for n in range(1, 231):
            sg = SpaceGroup(n)
            identity = np.eye(3)
            n_centering = sum(
                1 for o in sg.symmetry_operations if np.allclose(o.rotation, identity)
            )
            if n_centering != 1:
                continue  # centered lattice
            has_screw_or_glide = any(
                o.geometric_type in ("screw", "glide") for o in sg.symmetry_operations
            )
            with self.subTest(sg=n):
                self.assertEqual(sg.is_symmorphic, not has_screw_or_glide)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

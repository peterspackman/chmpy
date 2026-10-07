import unittest

from chmpy.crystal import SpaceGroup
from chmpy.crystal.hall_symbol import parse_hall_symbol
from chmpy.crystal.space_group import SG_FROM_NUMBER
from chmpy.crystal.symmetry_operation import SymmetryOperation
from chmpy.util.optional import have


def codes(ops):
    return {op.integer_code for op in ops}


def triplets(*strings):
    return {SymmetryOperation.from_string_code(s).integer_code for s in strings}


class HallSymbolTestCase(unittest.TestCase):
    def test_every_tabulated_setting(self):
        for number, settings in SG_FROM_NUMBER.items():
            for setting in settings:
                sg = SpaceGroup(int(number), choice=setting.choice)
                ops = parse_hall_symbol(sg.hall_symbol)
                self.assertEqual(
                    codes(ops), set(setting.symops), (number, setting.choice)
                )

    def test_from_hall_symbol_round_trips(self):
        # Some settings of Ccce (No. 68) share their operations, so the
        # setting found is compared by its operations, not its label.
        for number, settings in SG_FROM_NUMBER.items():
            for setting in settings:
                sg = SpaceGroup(int(number), choice=setting.choice)
                found = SpaceGroup.from_hall_symbol(sg.hall_symbol)
                self.assertEqual(found.international_tables_number, int(number))
                self.assertEqual(codes(found.symmetry_operations), set(setting.symops))

    def test_screw_axis(self):
        self.assertEqual(
            codes(parse_hall_symbol("P 2yb")), triplets("x,y,z", "-x,y+1/2,-z")
        )

    def test_centring_and_inversion(self):
        self.assertEqual(len(parse_hall_symbol("-P 1")), 2)
        self.assertEqual(len(parse_hall_symbol("R 3")), 9)
        self.assertEqual(len(parse_hall_symbol("-F 4 2 3")), 192)

    def test_default_axes(self):
        self.assertIn(triplets("x,-y,-z").pop(), codes(parse_hall_symbol("P 4 2")))
        self.assertIn(triplets("-y,-x,-z").pop(), codes(parse_hall_symbol("P 3 2")))
        self.assertIn(triplets("z,x,y").pop(), codes(parse_hall_symbol("P 2 2 3")))

    def test_origin_shift_and_change_of_basis(self):
        shifted = codes(parse_hall_symbol("-P 1 (1 0 0)"))
        self.assertEqual(shifted, triplets("x,y,z", "-x+1/6,-y,-z"))
        self.assertEqual(
            codes(parse_hall_symbol("P 2 (x,y,z+1/4)")),
            codes(parse_hall_symbol("P 2")),
        )

    def test_invalid_symbols(self):
        for symbol in [
            "",
            "Q 2",
            "PP 2",
            "P",
            "P 5",
            "P 2q",
            "P 3'",
            "P 2'1",
            "P 2 2 2 2 2",
            "P 2 (0 0)",
        ]:
            with self.assertRaises(ValueError, msg=symbol):
                parse_hall_symbol(symbol)

    def test_untabulated_setting(self):
        with self.assertRaises(ValueError):
            SpaceGroup.from_hall_symbol("P 2 2 (x,y,z+1/4)")


@unittest.skipUnless(have("gemmi"), "needs gemmi for validation")
class HallSymbolGemmiTestCase(unittest.TestCase):
    def gemmi_codes(self, ops):
        return {
            SymmetryOperation.from_string_code(op.triplet()).integer_code for op in ops
        }

    def test_matches_gemmi_table(self):
        import gemmi

        for entry in gemmi.spacegroup_table():
            self.assertEqual(
                codes(parse_hall_symbol(entry.hall)),
                self.gemmi_codes(entry.operations()),
                entry.hall,
            )

    def test_change_of_basis_matches_gemmi(self):
        import gemmi

        for symbol in [
            "P 2 2 (x,y,z+1/4)",
            "-P 2ybc (z,x,y)",
            "C 2y (x-z,y,x)",
            "P 31 2 (0 0 4)",
            "-I 4bd 2 (1 0 3)",
        ]:
            self.assertEqual(
                codes(parse_hall_symbol(symbol)),
                self.gemmi_codes(gemmi.symops_from_hall(symbol)),
                symbol,
            )

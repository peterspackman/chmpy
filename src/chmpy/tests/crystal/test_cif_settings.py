import unittest

from chmpy import Crystal
from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.crystal.space_group import SG_FROM_NUMBER


def codes(space_group):
    return {op.integer_code for op in space_group.symmetry_operations}


class CifSettingTestCase(unittest.TestCase):
    def test_monoclinic_settings_round_trip(self):
        cell = UnitCell.monoclinic(7.0, 8.0, 9.0, 105.0, unit="degrees")
        asym = AsymmetricUnit([Element["C"]], [[0.1, 0.2, 0.3]])
        for number in (4, 7, 9, 14, 15):
            for setting in SG_FROM_NUMBER[str(number)]:
                sg = SpaceGroup(number, choice=setting.choice)
                crystal = Crystal(cell, sg, asym)
                loaded = Crystal.from_cif_string(crystal.to_cif_string())
                self.assertEqual(
                    codes(loaded.space_group), codes(sg), (number, setting.choice)
                )

    def test_listed_operations_win_over_a_contradicting_symbol(self):
        cell = UnitCell.monoclinic(7.0, 8.0, 9.0, 105.0, unit="degrees")
        asym = AsymmetricUnit([Element["C"]], [[0.1, 0.2, 0.3]])
        p21a = SpaceGroup(14, choice="b3")
        lines = [
            "_symmetry_space_group_name_H-M 'P 1 21/c 1'"
            if line.startswith("_symmetry_space_group_name_H-M")
            else line
            for line in Crystal(cell, p21a, asym).to_cif_string().splitlines()
        ]
        loaded = Crystal.from_cif_string("\n".join(lines))
        self.assertEqual(codes(loaded.space_group), codes(p21a))

    def test_monoclinic_symbol_names_the_setting(self):
        cell = UnitCell.monoclinic(7.0, 8.0, 9.0, 105.0, unit="degrees")
        asym = AsymmetricUnit([Element["C"]], [[0.1, 0.2, 0.3]])
        expected = {"b1": "P 1 21/c 1", "b2": "P 1 21/n 1", "b3": "P 1 21/a 1"}
        for choice, symbol in expected.items():
            contents = Crystal(
                cell, SpaceGroup(14, choice=choice), asym
            ).to_cif_string()
            line = next(x for x in contents.splitlines() if "space_group_name_H-M" in x)
            self.assertIn(symbol, line)

    def test_every_tabulated_setting_round_trips(self):
        cell = UnitCell.triclinic(7.0, 8.0, 9.0, 80.0, 95.0, 105.0, unit="degrees")
        asym = AsymmetricUnit([Element["C"]], [[0.1, 0.2, 0.3]])
        for number in range(1, 231):
            for setting in SG_FROM_NUMBER[str(number)]:
                sg = SpaceGroup(number, choice=setting.choice)
                contents = Crystal(cell, sg, asym).to_cif_string()
                self.assertIn(sg.hall_symbol, contents, (number, setting.choice))
                loaded = Crystal.from_cif_string(contents)
                self.assertEqual(
                    codes(loaded.space_group), codes(sg), (number, setting.choice)
                )

    def test_hall_symbol_alone_gives_the_setting(self):
        contents = "\n".join(
            [
                "data_test",
                "_cell_length_a 7.0",
                "_cell_length_b 8.0",
                "_cell_length_c 9.0",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 105",
                "_cell_angle_gamma 90",
                "_symmetry_Int_Tables_number 14",
                "_symmetry_space_group_name_Hall '-P 2yab'",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "C1 C 0.1 0.2 0.3",
            ]
        )
        loaded = Crystal.from_cif_string(contents)
        self.assertEqual(codes(loaded.space_group), codes(SpaceGroup(14, choice="b3")))

import unittest

import numpy as np

from chmpy.calc.system import System
from chmpy.fmt.lammps import (
    PRESSURE_UNITS,
    atom_types,
    lammps_cell,
    parse_ave_time_string,
    parse_data_string,
    parse_dump_string,
    parse_log_string,
    stress_from_pressure,
    system_to_data_string,
    to_lammps_input,
    type_elements,
)

CELL = np.array([[7.1, 0.3, -0.4], [1.2, 9.4, 0.5], [-0.8, 1.1, 6.7]])


def _system():
    frac = np.random.default_rng(1).random((12, 3))
    numbers = [6] * 6 + [1] * 4 + [7] * 2
    return System(numbers, frac @ CELL, cell=CELL)


def _frac_distance(a, b):
    d = (a - b) % 1.0
    return np.minimum(d, 1.0 - d).max()


CHARGE_DATA = """LAMMPS data file, charge style, offset box

4 atoms
2 atom types

-1.0 4.0 xlo xhi
2.0 7.0 ylo yhi
0.5 6.5 zlo zhi
0.5 0.0 0.0 xy xz yz

Masses

1 22.98977
2 35.453

Atoms # charge

3 2 -1.0 1.0 4.0 1.5 0 0 0
1 1 1.0 -1.0 2.0 0.5 0 0 0
2 1 1.0 0.5 3.0 1.0 1 0 0
4 2 -1.0 2.0 5.0 2.5 0 0 -1

Velocities

1 0.0 0.0 0.0
2 0.0 0.0 0.0
3 0.0 0.0 0.0
4 0.0 0.0 0.0
"""

# box: lx 4, ly 5, lz 6, xy 0.5, xz -1.0, yz 2.0, origin (0, 0, 0)
DUMP = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
-1.0 4.5 0.5
0.0 7.0 -1.0
0.0 6.0 2.0
ITEM: ATOMS id type x y z fx
2 1 1.0 2.0 3.0 0.5
1 2 0.5 0.5 0.5 -0.5
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 4.0
0.0 5.0
0.0 6.0
ITEM: ATOMS id type xs ys zs
1 2 0.5 0.0 0.0
2 1 0.0 0.5 0.5
"""

LOG = """LAMMPS (29 Aug 2024)
units metal
Per MPI rank memory allocation (min/avg/max) = 3.1 | 3.1 | 3.1 Mbytes
   Step          Temp          Press
         0   94             120.5
        10   93.2          -40.25
WARNING: something harmless (src/foo.cpp:1)
        20   94.1           10
Loop time of 0.5 on 1 procs for 20 steps with 24 atoms

Per MPI rank memory allocation (min/avg/max) = 3.1 | 3.1 | 3.1 Mbytes
   Step          Temp          Press
        20   94.1           10
        30   95            -12
"""

AVE_TIME = """# Time-averaged data for fix stress
# TimeStep c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6]
1000 1.0 2.0 3.0 4.0 5.0 6.0
2000 1.5 2.5 3.5 4.5 5.5 6.5
"""


class LammpsCellTest(unittest.TestCase):
    def test_lower_triangular(self):
        lower, _ = lammps_cell(CELL)
        self.assertTrue(np.allclose(np.triu(lower, 1), 0.0))

    def test_rotation_is_proper(self):
        _, rotation = lammps_cell(CELL)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_rotation_maps_cell(self):
        lower, rotation = lammps_cell(CELL)
        np.testing.assert_allclose(CELL @ rotation.T, lower, atol=1e-12)

    def test_left_handed_refused(self):
        with self.assertRaises(ValueError):
            lammps_cell(CELL[[1, 0, 2]])


class AtomTypesTest(unittest.TestCase):
    def test_default_ascending(self):
        self.assertEqual(atom_types([8, 1, 6, 1]), [1, 6, 8])

    def test_species_order_kept(self):
        self.assertEqual(atom_types([1, 6], species=["C", "H", "O"]), [6, 1, 8])

    def test_species_missing_element(self):
        with self.assertRaises(ValueError):
            atom_types([1, 6, 7], species=["C", "H"])


class DataFileTest(unittest.TestCase):
    def test_round_trip_elements(self):
        system = _system()
        back = parse_data_string(system_to_data_string(system))
        np.testing.assert_array_equal(back.numbers, system.numbers)

    def test_round_trip_fractional_positions(self):
        system = _system()
        back = parse_data_string(system_to_data_string(system))
        self.assertLess(
            _frac_distance(back.scaled_positions, system.scaled_positions), 1e-9
        )

    def test_round_trip_cell_is_rotated(self):
        _, rotation = lammps_cell(CELL)
        back = parse_data_string(system_to_data_string(_system()))
        np.testing.assert_allclose(back.cell, CELL @ rotation.T, atol=1e-9)

    def test_species_sets_type_numbers(self):
        text = system_to_data_string(_system(), species=["N", "C", "H"])
        self.assertIn("1 14.006700 # N", text)

    def test_isolated_refused(self):
        with self.assertRaises(ValueError):
            system_to_data_string(System([1], [[0, 0, 0]]))

    def test_elements_from_masses(self):
        system = parse_data_string(CHARGE_DATA)
        self.assertEqual(list(system.numbers), [11, 11, 17, 17])

    def test_species_overrides_masses(self):
        system = parse_data_string(CHARGE_DATA, species=["K", "Br"])
        self.assertEqual(list(system.numbers), [19, 19, 35, 35])

    def test_charge_style_positions_from_origin(self):
        system = parse_data_string(CHARGE_DATA)
        np.testing.assert_allclose(system.positions[0], [0.0, 0.0, 0.0])

    def test_image_flags_applied(self):
        system = parse_data_string(CHARGE_DATA)
        np.testing.assert_allclose(system.positions[1], [1.5 + 5.0, 1.0, 0.5])

    def test_negative_image_flag(self):
        system = parse_data_string(CHARGE_DATA)
        np.testing.assert_allclose(system.positions[3], [3.0, 3.0, -4.0])

    def test_tilted_cell(self):
        system = parse_data_string(CHARGE_DATA)
        np.testing.assert_allclose(system.cell, [[5, 0, 0], [0.5, 5, 0], [0, 0, 6]])

    def test_style_inferred_from_columns(self):
        text = CHARGE_DATA.replace("Atoms # charge", "Atoms")
        system = parse_data_string(text)
        np.testing.assert_allclose(system.positions[0], [0.0, 0.0, 0.0])

    def test_unknown_style_refused(self):
        with self.assertRaises(ValueError):
            parse_data_string(CHARGE_DATA, atom_style="ellipsoid")


class DumpTest(unittest.TestCase):
    def setUp(self):
        self.frames = parse_dump_string(DUMP)

    def test_frame_count(self):
        self.assertEqual([f.timestep for f in self.frames], [100, 200])

    def test_triclinic_box_from_bounds(self):
        np.testing.assert_allclose(
            self.frames[0].cell, [[4, 0, 0], [0.5, 5, 0], [-1, 2, 6]]
        )

    def test_triclinic_origin(self):
        np.testing.assert_allclose(self.frames[0].origin, [0.0, 0.0, 0.0])

    def test_sorted_by_id(self):
        np.testing.assert_array_equal(self.frames[0].columns["type"], [2, 1])

    def test_extra_columns(self):
        np.testing.assert_allclose(self.frames[0].columns["fx"], [-0.5, 0.5])

    def test_scaled_positions(self):
        np.testing.assert_allclose(self.frames[1].positions, [[2, 0, 0], [0, 2.5, 3]])


class LogTest(unittest.TestCase):
    def setUp(self):
        self.runs = parse_log_string(LOG)

    def test_run_count(self):
        self.assertEqual(len(self.runs), 2)

    def test_warning_skipped(self):
        np.testing.assert_allclose(self.runs[0]["Step"], [0, 10, 20])

    def test_unfinished_run_kept(self):
        np.testing.assert_allclose(self.runs[1]["Press"], [10, -12])


class AveTimeTest(unittest.TestCase):
    def test_columns_named(self):
        data = parse_ave_time_string(AVE_TIME)
        np.testing.assert_allclose(data["c_p[4]"], [4.0, 4.5])


class StressTest(unittest.TestCase):
    def test_pressure_is_compressive(self):
        stress = stress_from_pressure([1e4, 1e4, 1e4, 0, 0, 0])
        self.assertLess(stress[0, 0], 0.0)

    def test_metal_units(self):
        stress = stress_from_pressure([-1.602176634e6, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(stress[0, 0], 1.0)

    def test_voigt_order(self):
        stress = stress_from_pressure([0, 0, 0, 1, 2, 3], units="real")
        self.assertAlmostEqual(stress[1, 2] / stress[0, 1], 3.0)

    def test_rotation_back_to_crystal_frame(self):
        _, rotation = lammps_cell(CELL)
        sigma = np.arange(9.0).reshape(3, 3)
        sigma = sigma + sigma.T
        pressure = -(rotation @ sigma @ rotation.T) / PRESSURE_UNITS["metal"]
        np.testing.assert_allclose(
            stress_from_pressure(pressure, rotation=rotation), sigma, atol=1e-12
        )

    def test_unknown_units_refused(self):
        with self.assertRaises(ValueError):
            stress_from_pressure(np.zeros(6), units="lj")


PAIR = "pair_style pet pet-mad-s d3 pbesol\npair_coeff * * {elements}"


class InputTest(unittest.TestCase):
    def test_elements_filled_in(self):
        text = to_lammps_input("cell.data", PAIR, [6, 1, 8])
        self.assertIn("pair_coeff * * C H O\n", text)

    def test_reads_data_file(self):
        text = to_lammps_input("strained/cell.data", PAIR, ["C"])
        self.assertIn("read_data       strained/cell.data", text)

    def test_sections_in_order(self):
        text = to_lammps_input("cell.data", PAIR, ["C"], sections=["run 0", "run 10"])
        self.assertLess(text.index("run 0"), text.index("run 10"))

    def test_named_sections_headed(self):
        text = to_lammps_input("cell.data", PAIR, ["C"], sections={"sample": "run 10"})
        self.assertIn("# sample\nrun 10", text)

    def test_preamble_before_read_data(self):
        text = to_lammps_input("cell.data", PAIR, ["C"], preamble="package omp 4")
        self.assertLess(text.index("package omp 4"), text.index("read_data"))

    def test_lammps_variables_untouched(self):
        text = to_lammps_input("cell.data", PAIR, ["C"], sections=["print ${T}"])
        self.assertIn("print ${T}", text)

    def test_custom_template_text(self):
        template = "read_data {{ data_file }}\n{{ pair }}\nrun {{ steps }}"
        text = to_lammps_input(
            "cell.data", PAIR, ["C", "H"], template=template, steps=5
        )
        self.assertEqual(text.splitlines()[-2:], ["pair_coeff * * C H", "run 5"])

    def test_type_elements_follow_species(self):
        self.assertEqual(
            type_elements([1, 6, 8], species=["O", "C", "H"]), ["O", "C", "H"]
        )

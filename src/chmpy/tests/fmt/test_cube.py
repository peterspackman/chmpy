import unittest

from chmpy.fmt.cube import CubeData

CUBE = """title line
subtitle line
{natom}    0.000000    0.000000    0.000000{extra}
    2    0.500000    0.000000    0.000000
    2    0.000000    0.500000    0.000000
    2    0.000000    0.000000    0.500000
    1    1.000000    0.000000    0.000000    0.000000
{mo_line}  1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
"""


class CubeTestCase(unittest.TestCase):
    def test_density_cube(self):
        cube = CubeData.from_string(CUBE.format(natom=1, extra="", mo_line=""))
        self.assertEqual(cube.subtitle, "subtitle line")
        self.assertEqual(cube.data.size, 8)

    def test_mo_cube(self):
        contents = CUBE.format(natom=-1, extra="    1", mo_line="    1    5\n")
        cube = CubeData.from_string(contents)
        self.assertEqual(cube.natom, 1)
        self.assertEqual(cube.volume_origin.shape, (3,))
        self.assertEqual(cube.mo_ids, [5])
        self.assertEqual(cube.data.size, 8)

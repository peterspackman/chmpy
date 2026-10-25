import unittest

import numpy as np

from chmpy.fmt.sdf import parse_sdf_contents, to_sdf_string

SDF = """mol
  test

  3  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
M  CHG  2   1   1   3  -1
M  END
>  <NAME>  (1)
zwitterion

> <NOTE>
line 1
line 2

$$$$
"""


class SdfTestCase(unittest.TestCase):
    def test_write_read_roundtrip(self):
        d = {
            "header": ["t", "c", ""],
            "atoms": {
                "x": np.array([1.2345]),
                "y": np.array([-2.3456]),
                "z": np.array([3.4567]),
                "symbol": np.array(["C"]),
            },
            "bonds": {
                "left": np.array([], dtype=int),
                "right": np.array([], dtype=int),
            },
        }
        atoms = parse_sdf_contents(to_sdf_string(d) + "\n$$$$\n")[0]["atoms"]
        np.testing.assert_allclose(
            [atoms["x"][0], atoms["y"][0], atoms["z"][0]], [1.2345, -2.3456, 3.4567]
        )
        self.assertEqual(atoms["symbol"][0], "C")

    def test_multiple_charges(self):
        result = parse_sdf_contents(SDF)[0]
        np.testing.assert_array_equal(result["atoms"]["charge"], [1, 0, -1])

    def test_data_items(self):
        data = parse_sdf_contents(SDF)[0]["data"]
        self.assertEqual(data, {"NAME": "zwitterion", "NOTE": "line 1\nline 2"})

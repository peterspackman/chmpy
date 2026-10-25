import unittest

from chmpy.fmt.xyz_file import parse_xyz_string

WATER = """3
water
O 0.0 0.0 0.1173
H 0.0 0.7572 -0.4692
H 0.0 -0.7572 -0.4692
"""


class XyzTestCase(unittest.TestCase):
    def test_only_first_frame_is_read(self):
        elements, positions = parse_xyz_string(WATER + WATER)
        self.assertEqual(len(elements), 3)
        self.assertEqual(positions.shape, (3, 3))

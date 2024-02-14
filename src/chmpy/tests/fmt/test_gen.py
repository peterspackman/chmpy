from chmpy.fmt.gen import parse_gen_string, parse_gen_file
import unittest
import numpy as np
from .. import TEST_FILES


class GenFileTestCase(unittest.TestCase):
    def test_parse_file(self):
        els, pos, vecs, frac = parse_gen_file(TEST_FILES["example.gen"])
        self.assertEqual(len(els), 76)
        self.assertEqual(els[0].atomic_number, 1)
        self.assertEqual(els[75].atomic_number, 8)

        expected_coords_46 = np.array((0.3623979304, 0.6637420302, 0.8677686357))
        np.testing.assert_allclose(pos[46, :], expected_coords_46, atol=1e-5)

        np.testing.assert_allclose(np.zeros(3), vecs[0, :])

        self.assertAlmostEqual(vecs[1, 0], 13.171)

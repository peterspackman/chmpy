import tempfile
import unittest
from pathlib import Path

import numpy as np

from chmpy.fmt.gmf import GMF

CONTENTS = """
 title: created by occ-cg
  name: test
 space: P 21/c
  cell: 5.000000 6.000000 7.000000  90.000000 100.000000 90.000000
 morph: unrelaxed equilibrium

miller:    0   1   1
 0.250000   1  1      0.1234     0.0000     0.0000     0.0000  0
miller:    1   0   0
 0.500000   1  1      0.2345     0.0000     0.0000     0.0000  0
"""


class GMFTestCase(unittest.TestCase):
    def test_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "test.gmf")
            path.write_text(CONTENTS)
            gmf = GMF.from_file(path)
        np.testing.assert_array_equal(gmf.hkl, [(0, 1, 1), (1, 0, 0)])
        np.testing.assert_allclose(gmf.cuts, [0.25, 0.5])
        np.testing.assert_allclose(gmf.energies, [0.1234, 0.2345])

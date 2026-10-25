import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from chmpy.fmt.fchk import FchkFile

FCHK = """water
SP        RHF                                                         STO-3G
Number of atoms                            I                3
Total Energy                               R     -7.496590466223640E+01
"""


class FchkTestCase(unittest.TestCase):
    def test_scf_energy(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "water.fchk")
            path.write_text(FCHK)
            self.assertAlmostEqual(FchkFile.scf_energy(path), -74.9659046622364)

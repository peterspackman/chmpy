import unittest

from chmpy.fmt.shelx import parse_shelx_file_content

from .. import TEST_FILES

RES = TEST_FILES["acetic_acid.res"].read_text()


class ShelxTestCase(unittest.TestCase):
    def test_instructions_are_not_atoms(self):
        contents = (
            RES.replace("FVAR 1.00", "L.S. 10\nWGHT 0.1\nREM\nFVAR 1.00")
            .replace("C1      1", "AFIX 137\nC1      1")
            .replace("O2      3", "AFIX 0\nPART 1\nO2      3")
            .replace("END", "Q1      1  0.1  0.2  0.3 11.0 0.05\nEND")
        )
        d = parse_shelx_file_content(contents)
        self.assertEqual(len(d["ATOM"]), 8)

    def test_continuation_lines(self):
        contents = RES.replace(
            "C1      1  0.165100  0.285800  0.170900 1.0",
            "C1      1  0.165100  0.285800  0.170900 11.00000  0.02 0.03 =\n"
            "         0.04 0.001 0.002 0.003",
        )
        d = parse_shelx_file_content(contents)
        self.assertEqual(len(d["ATOM"]), 8)
        self.assertEqual(d["ATOM"][0]["label"], "C1")
        self.assertEqual(d["ATOM"][0]["occupation"], 1.0)

    def test_free_variable_occupation(self):
        contents = RES.replace("FVAR 1.00", "FVAR 1.00 0.25").replace(
            "O2      3  0.252900  0.370300  0.176900 1.0",
            "O2      3  0.252900  0.370300  0.176900 21.0",
        )
        d = parse_shelx_file_content(contents)
        self.assertAlmostEqual(d["ATOM"][-1]["occupation"], 0.25)

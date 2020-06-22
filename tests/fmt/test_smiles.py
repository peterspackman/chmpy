import unittest
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
from .. import TEST_FILES
from chmpy.fmt.smiles import parse

LOG = logging.getLogger(__name__)
_WATER = "O"
_CUBANE = "C12C3C4C1C5C4C3C25"
_PYRIDINE = "c1cnccc1"


class SMILESParserTest(unittest.TestCase):
    def test_parse_valid(self):
        from pyparsing import ParseException

        with self.assertRaises(ParseException):
            parse("invalid")
        parse(_WATER)
        parse(_PYRIDINE)

    def test_cubane_bonds(self):
        atoms, bonds = parse(_CUBANE)
        self.assertEqual(len(bonds), 12)

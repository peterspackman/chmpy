import logging
import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.crystal import Crystal

LOG = logging.getLogger(__name__)
_ICE_II = join(dirname(__file__), "iceII.cif")


class CrystalTestCase(unittest.TestCase):
    def test_crystal_load(self):
        c = Crystal.load(_ICE_II)
        assert len(c.asymmetric_unit) == 36, "Expect 36 atoms in asymmetric unit"

    def test_crystal_molecules(self):
        c = Crystal.load(_ICE_II)
        mols = c.symmetry_unique_molecules()
        assert len(mols) == 12, "Expect 12 water molecules in unit cell"
        formulae = [x.molecular_formula for x in mols]
        LOG.debug("Formulae = %s", formulae)
        assert all(f == "H2O" for f in formulae), "Expect molecular formula to be H2O"

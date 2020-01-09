import logging
import unittest
import numpy as np
from os.path import join, dirname
from shmolecule.crystal import Crystal
from tempfile import TemporaryDirectory
from shmolecule.util import save_mesh

_ACETIC = join(dirname(__file__), "acetic_acid.cif")


class SurfaceTestCase(unittest.TestCase):
    def setUp(self):
        self.acetic_acid = Crystal.load(_ACETIC)

    def test_promolecule_surfaces(self):
        surfaces = self.acetic_acid.promolecule_density_isosurfaces(separation=1.0)

    def test_hirshfeld_surfaces(self):
        surfaces = self.acetic_acid.stockholder_weight_isosurfaces(
            separation=1.0, radius=3.8
        )

    def test_save(self):
        with TemporaryDirectory() as tmpdirname:
            surfaces = self.acetic_acid.promolecule_density_isosurfaces(separation=1.0)
            save_mesh(surfaces[0], join(tmpdirname, "tmp.ply"))

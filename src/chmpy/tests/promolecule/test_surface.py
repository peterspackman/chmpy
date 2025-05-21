import unittest
from os.path import join
from tempfile import TemporaryDirectory

from chmpy.crystal import Crystal
from chmpy.util.mesh import save_mesh

from .. import TEST_FILES


class SurfaceTestCase(unittest.TestCase):
    def setUp(self):
        self.acetic_acid = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_promolecule_surfaces(self):
        _ = self.acetic_acid.promolecule_density_isosurfaces(separation=1.0)

    def test_hirshfeld_surfaces(self):
        _ = self.acetic_acid.stockholder_weight_isosurfaces(separation=1.0, radius=3.8)
        _ = self.acetic_acid.hirshfeld_surfaces(separation=1.0, radius=3.8, kind="atom")

    def test_save(self):
        with TemporaryDirectory() as tmpdirname:
            surfaces = self.acetic_acid.promolecule_density_isosurfaces(separation=1.0)
            save_mesh(surfaces[0], join(tmpdirname, "tmp.ply"))

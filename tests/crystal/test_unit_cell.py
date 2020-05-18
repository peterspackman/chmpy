import logging
import unittest
import numpy as np
from os.path import join, dirname
from chmpy.crystal import Crystal, UnitCell
from tempfile import TemporaryDirectory

LOG = logging.getLogger(__name__)
_ICE_II = join(dirname(__file__), "iceII.cif")


class UnitCellTestCase(unittest.TestCase):
    def test_unit_cell_lattice(self):
        c = UnitCell.cubic(2.0)
        np.testing.assert_allclose(c.lattice, 2.0 * np.eye(3), atol=1e-8)
        np.testing.assert_allclose(c.reciprocal_lattice, 0.5 * np.eye(3), atol=1e-8)

    def test_coordinate_transforms(self):
        c = UnitCell.cubic(2.0)
        np.testing.assert_allclose(
            c.to_fractional(np.eye(3)), 0.5 * np.eye(3), atol=1e-8
        )
        np.testing.assert_allclose(c.to_cartesian(np.eye(3)), 2 * np.eye(3), atol=1e-8)

    def test_handle_bad_angles(self):
        # should warn
        c = UnitCell.from_lengths_and_angles([2.0] * 3, [90] * 3)

    def test_repr(self):
        c = UnitCell.cubic(2.0)
        self.assertTrue(str(c) == "<UnitCell: cubic (2.000)>")

    def test_cell_types(self):
        cubic = UnitCell.cubic(2.0)
        orthorhombic = UnitCell.orthorhombic(3.0, 4.0, 5.0)
        tetragonal = UnitCell.tetragonal(3.0, 4.0, unit="degrees")
        tetragonal = UnitCell.from_unique_parameters((3.0, 4.0), cell_type="tetragonal")
        monoclinic = UnitCell.monoclinic(3.0, 4.0, 5.0, 75, unit="degrees")
        monoclinic = UnitCell.monoclinic(3.0, 4.0, 5.0, 1.5)
        triclinic = UnitCell.triclinic(3.0, 4.0, 5.0, 45, 75, 90, unit="degrees")
        rhombohedral = UnitCell.rhombohedral(3.0, 97, unit="degrees")
        rhombohedral = UnitCell.rhombohedral(3.0, 1.35)
        hexagonal = UnitCell.hexagonal(3.0, 4.0)
        hexagonal = UnitCell.from_lengths_and_angles(
            [3.0, 3.0, 5.0], [90, 90, 120], unit="degrees"
        )
        self.assertTrue(cubic.cell_type == "cubic")
        self.assertTrue(orthorhombic.cell_type == "orthorhombic")
        self.assertTrue(tetragonal.cell_type == "tetragonal")
        self.assertTrue(monoclinic.cell_type == "monoclinic")
        self.assertTrue(triclinic.cell_type == "triclinic")
        self.assertTrue(rhombohedral.cell_type == "rhombohedral")
        self.assertTrue(hexagonal.cell_type == "hexagonal")

    def test_angles(self):
        c = UnitCell.cubic(2.0)
        self.assertFalse(c.angles_different)
        self.assertTrue(c.is_cubic)
        self.assertFalse(c.is_triclinic)
        self.assertFalse(c.is_monoclinic)
        self.assertFalse(c.is_tetragonal)
        self.assertFalse(c.is_rhombohedral)
        self.assertFalse(c.is_hexagonal)
        self.assertFalse(c.is_orthorhombic)

        np.testing.assert_allclose([c.alpha_deg, c.beta_deg, c.gamma_deg], 90.0)
        np.testing.assert_allclose(c.parameters, [2, 2, 2, 90, 90, 90])

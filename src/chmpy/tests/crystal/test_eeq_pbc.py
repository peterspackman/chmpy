import logging
import unittest

import numpy as np

from chmpy.core import Element, Molecule
from chmpy.core.eeq import calculate_eeq_charges
from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell
from chmpy.crystal.eeq_pbc import (
    calculate_coordination_numbers_crystal,
    calculate_eeq_charges_crystal,
)

from .test_asymmetric_unit import ice_ii_asym

LOG = logging.getLogger(__name__)
_ICE_II_CELL = UnitCell.rhombohedral(7.78, 113.1, unit="degrees")
_ICE_II_SG = SpaceGroup(1)


class CrystalTestCase(unittest.TestCase):
    def setUp(self):
        self.ice_ii = Crystal(_ICE_II_CELL, _ICE_II_SG, ice_ii_asym())
        self.water_positions = (
            np.array(
                (
                    -0.7021961,
                    -1.0221932,
                    0.2575211,
                    -0.0560603,
                    0.8467758,
                    0.0421215,
                    0.0099423,
                    -0.0114887,
                    0.0052190,
                )
            )
            .reshape(3, 3)
            .T
        )
        self.water_elements = [Element["O"], Element["H"], Element["H"]]
        self.water_atomic_numbers = np.array([8, 1, 1])
        self.water_molecule = Molecule(self.water_elements, self.water_positions + 3.0)
        self.water_isolated = Crystal.from_molecule(self.water_molecule)

        a = 5.64
        self.nacl_pos = np.array(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
        self.nacl_elements = [Element["Na"], Element["Cl"]]
        self.nacl = Crystal(
            UnitCell(
                np.array(([a / 2, a / 2, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2]))
            ),
            SpaceGroup(1),
            AsymmetricUnit(self.nacl_elements, self.nacl_pos),
        )
        self.nacl_cubic = Crystal(
            UnitCell.cubic(5.64),
            SpaceGroup(225),
            AsymmetricUnit(self.nacl_elements, self.nacl_pos),
        )

    def test_eeq_coordination_numbers(self):
        cn_water = calculate_coordination_numbers_crystal(self.water_isolated)
        np.testing.assert_allclose(cn_water, [1.989382, 0.995133, 0.994268], 1e-5, 1e-5)

        cn = calculate_coordination_numbers_crystal(self.ice_ii)
        np.testing.assert_allclose(
            cn, np.tile([1.989382, 0.994269, 0.995114], 12), 1e-3, 1e-3
        )

        cn_prim = calculate_coordination_numbers_crystal(self.nacl)
        cn = calculate_coordination_numbers_crystal(self.nacl_cubic)
        np.testing.assert_allclose(
            cn,
            np.tile(cn_prim, 4),
            1e-3,
            1e-3,
        )

    def test_eeq_charges(self):
        calculate_eeq_charges(
            self.water_molecule.atomic_numbers, self.water_molecule.positions
        )
        q = calculate_eeq_charges_crystal(self.water_isolated, cutoff=12.0)
        qprim = calculate_eeq_charges_crystal(self.nacl, cutoff=12.0)
        q = calculate_eeq_charges_crystal(self.nacl_cubic, cutoff=12.0)
        np.testing.assert_allclose(
            q,
            np.tile(qprim, 4),
            1e-3,
            1e-3,
        )

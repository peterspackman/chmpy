"""
Tests for the EEQ (Electronegativity Equilibration Charge) implementation.
"""

import unittest

import numpy as np

from chmpy.core.eeq import (
    build_a_matrix,
    build_x_vector,
    calculate_coordination_numbers,
    calculate_eeq_charges,
)
from chmpy.core.element import Element
from chmpy.core.molecule import Molecule
from chmpy.util.unit import BOHR_TO_ANGSTROM


class TestEEQ(unittest.TestCase):
    def setUp(self):
        # Create a simple water molecule for testing
        # Coordinates in Angstroms
        self.water_positions = np.array(
            (
                [-0.7021961, -0.0560603, 0.0099423],
                [-1.0221932, 0.8467758, -0.0114887],
                [0.2575211, 0.0421215, 0.0052190],
            )
        )
        self.water_elements = [Element["O"], Element["H"], Element["H"]]
        self.water_atomic_numbers = np.array([8, 1, 1])

        self.MB16_43_01_positions = (
            np.array(
                (
                    [-1.85528263484662, 3.58670515364616, -2.41763729306344],
                    [4.40178023537845, 0.02338844412653, -4.95457749372945],
                    [-2.98706033463438, 4.76252065456814, 1.27043301573532],
                    [0.79980886075526, 1.41103455609189, -5.04655321620119],
                    [-4.20647469409936, 1.84275767548460, 4.55038084858449],
                    [-3.54356121843970, -3.18835665176557, 1.46240021785588],
                    [2.70032160109941, 1.06818452504054, -1.73234650374438],
                    [3.73114088824361, -2.07001543363453, 2.23160937604731],
                    [-1.75306819230397, 0.35951417150421, 1.05323406177129],
                    [5.41755788583825, -1.57881830078929, 1.75394002750038],
                    [-2.23462868255966, -2.13856505054269, 4.10922285746451],
                    [1.01565866207568, -3.21952154552768, -3.36050963020778],
                    [2.42119255723593, 0.26626435093114, -3.91862474360560],
                    [-3.02526098819107, 2.53667889095925, 2.31664984740423],
                    [-2.00438948664892, -2.29235136977220, 2.19782807357059],
                    [1.12226554109716, -1.36942007032045, 0.48455055461782],
                )
            )
            * BOHR_TO_ANGSTROM
        )
        self.MB16_43_01_elements = [
            Element[x]
            for x in [
                "Na",
                "H",
                "O",
                "H",
                "F",
                "H",
                "H",
                "O",
                "N",
                "H",
                "H",
                "Cl",
                "B",
                "B",
                "N",
                "Al",
            ]
        ]

        self.water = Molecule(self.water_elements, self.water_positions)
        self.MB16_43_01 = Molecule(self.MB16_43_01_elements, self.MB16_43_01_positions)

    def test_coordination_numbers(self):
        """Test calculation of coordination numbers."""
        cn_MB16_43_01 = calculate_coordination_numbers(
            self.MB16_43_01.atomic_numbers, self.MB16_43_01.positions
        )
        np.testing.assert_allclose(
            cn_MB16_43_01,
            [
                4.03670396918677e0,
                9.72798721502297e-1,
                1.98698465669657e0,
                1.47312608051590e0,
                9.97552155866795e-1,
                9.96862039916965e-1,
                1.45188437942218e0,
                1.99267278111197e0,
                3.84566220624764e0,
                1.00242959599510e0,
                9.96715113655073e-1,
                1.92505296745902e0,
                4.62015142034058e0,
                3.81973465175781e0,
                3.95710919750442e0,
                5.33862698412205e0,
            ],
            1e-5,
            1e-5,
        )
        cn_water = calculate_coordination_numbers(
            self.water_atomic_numbers, self.water_positions
        )
        np.testing.assert_allclose(cn_water, [1.989382, 0.995133, 0.994268], 1e-5, 1e-5)

    def test_eeq_charges(self):
        """Test calculation of coordination numbers."""

        q_water = calculate_eeq_charges(self.water_atomic_numbers, self.water_positions)
        np.testing.assert_allclose(q_water, [-0.592456, 0.297289, 0.295167], 1e-5, 1e-5)

    def test_build_a_matrix(self):
        """Test building of the A matrix."""
        # Build the A matrix for water
        A_water = build_a_matrix(self.water_atomic_numbers, self.water_positions)

        # Check dimensions
        self.assertEqual(A_water.shape, (4, 4))

        # Check symmetry
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(A_water[i, j], A_water[j, i])

        # Check last row and column are all 1's except A[3,3] = 0
        for i in range(3):
            self.assertEqual(A_water[3, i], 1.0)
            self.assertEqual(A_water[i, 3], 1.0)
        self.assertEqual(A_water[3, 3], 0.0)

    def test_build_x_vector(self):
        """Test building of the X vector."""
        # Calculate coordination numbers
        cn_water = calculate_coordination_numbers(
            self.water_atomic_numbers, self.water_positions
        )

        # Build the X vector for water
        X_water = build_x_vector(self.water_atomic_numbers, cn_water)

        # Check dimensions
        self.assertEqual(X_water.shape, (4,))

        # Check that the last element is 0.0 (default total charge)
        self.assertEqual(X_water[3], 0.0)

        # Check with non-zero total charge
        X_water_charged = build_x_vector(
            self.water_atomic_numbers, cn_water, charge=1.0
        )
        self.assertEqual(X_water_charged[3], 1.0)

        # Check first three elements match between both vectors
        for i in range(3):
            self.assertEqual(X_water[i], X_water_charged[i])


if __name__ == "__main__":
    unittest.main()

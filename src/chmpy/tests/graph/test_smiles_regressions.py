"""Regression tests for SMILES stereo, bond orders and element symbols."""

import unittest

import numpy as np

from chmpy import Molecule
from chmpy.graph import MolecularGraph, to_smiles
from chmpy.graph.bond_orders import perceive_bond_orders

# CHFClBr: C, H, F, Cl, Br on tetrahedral directions
_TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
CHFCLBR_Z = np.array([6, 1, 9, 17, 35])
CHFCLBR_POS = np.vstack(
    [[0, 0, 0]] + [_TET[i] * r for i, r in enumerate((1.09, 1.35, 1.77, 1.94))]
)

# 1-bromo-2-chloro-3-fluorocyclobutane, F[C@@H]1C[C@@H](Br)[C@@H]1Cl
CYCLOBUTANE_Z = np.array([9, 6, 6, 17, 6, 35, 6, 1, 1, 1, 1, 1])
CYCLOBUTANE_POS = np.array(
    [
        [-2.264, 0.164, 0.060],
        [-1.015, -0.292, 0.301],
        [-0.089, -0.502, -0.926],
        [-0.511, 0.337, -2.478],
        [1.067, 0.103, -0.088],
        [2.214, -1.193, 0.807],
        [-0.005, 0.757, 0.810],
        [-1.045, -1.180, 0.946],
        [0.037, -1.552, -1.208],
        [1.689, 0.827, -0.625],
        [-0.256, 1.783, 0.514],
        [0.179, 0.747, 1.888],
    ]
)

ACRYLONITRILE_Z = np.array([6, 6, 6, 1, 1, 1, 7])
ACRYLONITRILE_POS = np.array(
    [
        [-0.1616, -1.6386, 0.0],
        [0.5850, -0.5250, 0.0],
        [0.0, 0.7823, 0.0],
        [-1.2452, -1.5982, 0.0],
        [0.3060, -2.6164, 0.0],
        [1.6699, -0.5721, 0.0],
        [-0.4673, 1.8678, 0.0],
    ]
)

FURAN_Z = np.array([8, 6, 6, 6, 6, 1, 1, 1, 1])
FURAN_POS = np.array(
    [
        [0.0, 0.0, 1.1633],
        [0.0, 1.0947, 0.3480],
        [0.0, -1.0947, 0.3480],
        [0.0, 0.7132, -0.9622],
        [0.0, -0.7132, -0.9622],
        [0.0, 2.0494, 0.8511],
        [0.0, -2.0494, 0.8511],
        [0.0, 1.3708, -1.8197],
        [0.0, -1.3708, -1.8197],
    ]
)

SO2_Z = np.array([16, 8, 8])
SO2_POS = np.array(
    [[0.0, 0.0, 0.3703], [0.0, 1.2776, -0.3703], [0.0, -1.2776, -0.3703]]
)

KETENE_Z = np.array([6, 6, 1, 1, 8])
KETENE_POS = np.array(
    [
        [0.0, 0.0, -1.2193],
        [0.0, 0.0, 0.0989],
        [0.0, 0.9388, -1.7532],
        [0.0, -0.9388, -1.7532],
        [0.0, 0.0, 1.2786],
    ]
)


def difluoroethene(trans=True):
    c, s = np.cos(np.radians(60)), np.sin(np.radians(60))
    sgn = -1 if trans else 1
    pos = np.array(
        [
            [0, 0, 0],
            [1.33, 0, 0],
            [-1.35 * c, 1.35 * s, 0],
            [-1.08 * c, -1.08 * s, 0],
            [1.33 + 1.35 * c, sgn * 1.35 * s, 0],
            [1.33 + 1.08 * c, -sgn * 1.08 * s, 0],
        ]
    )
    return np.array([6, 6, 9, 1, 9, 1]), pos


def smiles(nums, pos, perm=None):
    if perm is not None:
        nums, pos = nums[perm], pos[perm]
    mol = Molecule.from_arrays(nums, pos)
    return to_smiles(MolecularGraph.from_molecule(mol))


def permutations(n, count=5, seed=0):
    rng = np.random.default_rng(seed)
    return [np.arange(n)] + [rng.permutation(n) for _ in range(count)]


class DoubleBondStereoTestCase(unittest.TestCase):
    def test_difluoroethene_keeps_double_bond(self):
        # trans: F/C=C/F, cis: F/C=C\F (equivalent forms as written)
        for trans, expected in ((True, "C(=C\\F)/F"), (False, "C(=C/F)/F")):
            nums, pos = difluoroethene(trans)
            for perm in permutations(len(nums)):
                self.assertEqual(smiles(nums, pos, perm), expected)

    def test_small_ring_double_bond_has_no_stereo(self):
        # cyclobutene
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.34, 0.0, 0.0],
                [1.34, 1.52, 0.0],
                [0.0, 1.52, 0.0],
                [-0.95, -0.55, 0.0],
                [2.29, -0.55, 0.0],
                [1.9, 1.9, 0.88],
                [1.9, 1.9, -0.88],
                [-0.56, 1.9, 0.88],
                [-0.56, 1.9, -0.88],
            ]
        )
        nums = np.array([6, 6, 6, 6, 1, 1, 1, 1, 1, 1])
        result = smiles(nums, pos)
        self.assertEqual(result, "C1=CCC1")


class TetrahedralStereoTestCase(unittest.TestCase):
    def test_chfclbr_both_enantiomers(self):
        # reference CIP: original geometry is S, mirror image is R
        for mirror, expected in ((1, "Cl[C@@H](Br)F"), (-1, "Cl[C@H](Br)F")):
            pos = CHFCLBR_POS * np.array([mirror, 1, 1])
            for perm in permutations(5):
                self.assertEqual(smiles(CHFCLBR_Z, pos, perm), expected)

    def test_ring_stereocenters(self):
        # equivalent to F[C@@H]1C[C@@H](Br)[C@@H]1Cl (checked with RDKit)
        results = {
            smiles(CYCLOBUTANE_Z, CYCLOBUTANE_POS, perm)
            for perm in permutations(len(CYCLOBUTANE_Z))
        }
        self.assertEqual(results, {"[C@@H]1(F)C[C@@H](Br)[C@@H]1Cl"})


class BondOrderSMILESRegressionTestCase(unittest.TestCase):
    def test_acrylonitrile(self):
        self.assertEqual(smiles(ACRYLONITRILE_Z, ACRYLONITRILE_POS), "C=CC#N")

    def test_furan_aromatic(self):
        self.assertEqual(smiles(FURAN_Z, FURAN_POS), "o1cccc1")

    def test_so2(self):
        self.assertEqual(smiles(SO2_Z, SO2_POS), "S(=O)=O")

    def test_ketene_order_independent(self):
        for perm in permutations(len(KETENE_Z)):
            self.assertEqual(smiles(KETENE_Z, KETENE_POS, perm), "O=C=C")

    def test_integer_kekule_orders(self):
        mol = Molecule.from_arrays(ACRYLONITRILE_Z, ACRYLONITRILE_POS)
        graph = MolecularGraph.from_molecule(mol)
        bo = perceive_bond_orders(graph)
        for i, j in graph.edges():
            self.assertEqual(bo[i, j], round(bo[i, j]))
        self.assertEqual(bo[0, 1], 2)
        self.assertEqual(bo[2, 6], 3)

    def test_bond_order_rounds(self):
        mol = Molecule.from_arrays(FURAN_Z, FURAN_POS)
        graph = MolecularGraph.from_molecule(mol)
        graph.bond_orders = perceive_bond_orders(graph)
        # aromatic C-C bonds are stored as 1.5
        self.assertEqual(graph.bond_order(1, 3), 2)
        self.assertEqual(graph.bond_order(0, 1), 2)


class ElementSymbolTestCase(unittest.TestCase):
    def test_metal_symbols_without_invented_hydrogens(self):
        nacl = smiles(np.array([11, 17]), np.array([[0, 0, 0], [2.36, 0, 0]]))
        self.assertNotIn("#", nacl)
        self.assertIn("[Na]", nacl)
        fe = smiles(np.array([26]), np.array([[0.0, 0.0, 0.0]]))
        self.assertEqual(fe, "[Fe]")
        alf3 = smiles(
            np.array([13, 9, 9, 9]),
            np.array([[0, 0, 0], [1.63, 0, 0], [-0.815, 1.41, 0], [-0.815, -1.41, 0]]),
        )
        self.assertEqual(alf3, "F[Al](F)F")


if __name__ == "__main__":
    unittest.main()

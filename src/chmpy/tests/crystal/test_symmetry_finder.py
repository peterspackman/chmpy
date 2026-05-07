"""Tests for native symmetry detection."""

import logging
import unittest

import numpy as np

from chmpy.core.element import Element
from chmpy.crystal import Crystal, SpaceGroup
from chmpy.crystal.asymmetric_unit import AsymmetricUnit
from chmpy.crystal.symmetry_finder import (
    _CRYSTALLOGRAPHIC_ROTATIONS,
    _metric_compatible_rotations,
    find_asymmetric_unit_indices,
    find_symmetry_operations,
)
from chmpy.crystal.unit_cell import UnitCell

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class TestRotationTable(unittest.TestCase):
    """Tests for the crystallographic rotation table."""

    def test_rotation_table_has_expected_count(self):
        """Should have 64 unique crystallographic rotations."""
        self.assertEqual(len(_CRYSTALLOGRAPHIC_ROTATIONS), 64)

    def test_identity_present(self):
        """Identity rotation should be in the table."""
        I = np.eye(3, dtype=int)
        found = any(np.array_equal(R, I) for R in _CRYSTALLOGRAPHIC_ROTATIONS)
        self.assertTrue(found, "Identity rotation not found in table")

    def test_inversion_present(self):
        """Inversion (-I) should be in the table."""
        neg_I = -np.eye(3, dtype=int)
        found = any(np.array_equal(R, neg_I) for R in _CRYSTALLOGRAPHIC_ROTATIONS)
        self.assertTrue(found, "Inversion not found in table")

    def test_all_entries_are_integer(self):
        """All rotation entries should be integers in {-1, 0, 1}."""
        for R in _CRYSTALLOGRAPHIC_ROTATIONS:
            self.assertTrue(np.all(np.abs(R) <= 1))

    def test_all_dets_are_pm1(self):
        """All rotation determinants should be +1 or -1."""
        for R in _CRYSTALLOGRAPHIC_ROTATIONS:
            det = int(round(np.linalg.det(R.astype(float))))
            self.assertIn(det, [-1, 1])


class TestMetricTensorFilter(unittest.TestCase):
    """Tests for metric tensor filtering of rotations."""

    def test_cubic_passes_many_rotations(self):
        """Cubic cell should allow 48 rotations."""
        uc = UnitCell.cubic(5.0)
        G = uc.metric_tensor
        compatible = _metric_compatible_rotations(G)
        self.assertEqual(len(compatible), 48)

    def test_monoclinic_passes_few_rotations(self):
        """Monoclinic cell should allow ~4 rotations."""
        uc = UnitCell.from_lengths_and_angles(
            (5.0, 6.0, 7.0), (90, 110, 90), unit="degrees"
        )
        G = uc.metric_tensor
        compatible = _metric_compatible_rotations(G)
        self.assertEqual(len(compatible), 4)

    def test_triclinic_passes_identity_and_inversion(self):
        """Triclinic cell should allow only 2 rotations (I and -I)."""
        uc = UnitCell.from_lengths_and_angles(
            (5.0, 6.0, 7.0), (80, 85, 95), unit="degrees"
        )
        G = uc.metric_tensor
        compatible = _metric_compatible_rotations(G)
        self.assertEqual(len(compatible), 2)

    def test_orthorhombic_passes_eight(self):
        """Orthorhombic cell (a!=b!=c, all 90) should allow 8 rotations."""
        uc = UnitCell.orthorhombic(5.0, 6.0, 7.0)
        G = uc.metric_tensor
        compatible = _metric_compatible_rotations(G)
        self.assertEqual(len(compatible), 8)

    def test_tetragonal_passes_sixteen(self):
        """Tetragonal cell (a=b!=c, all 90) should allow 16 rotations."""
        uc = UnitCell.tetragonal(5.0, 7.0)
        G = uc.metric_tensor
        compatible = _metric_compatible_rotations(G)
        self.assertEqual(len(compatible), 16)


class TestFindSymmetryOperations(unittest.TestCase):
    """Tests for the main symmetry detection pipeline."""

    def test_p1_two_different_atoms(self):
        """Two different-element atoms at unrelated positions should find P1."""
        uc = UnitCell.from_lengths_and_angles(
            (5.0, 6.0, 7.0), (80, 85, 95), unit="degrees"
        )
        # Two atoms of different elements at general positions
        positions = np.array([
            [0.123, 0.456, 0.789],
            [0.3, 0.1, 0.6],
        ])
        elements = np.array([6, 7])
        symops = find_symmetry_operations(uc, positions, elements)
        self.assertEqual(len(symops), 1)

    def test_p_minus1_inversion(self):
        """Two atoms related by inversion should find P-1 (2 symops)."""
        uc = UnitCell.from_lengths_and_angles(
            (5.0, 6.0, 7.0), (80, 85, 95), unit="degrees"
        )
        # Two atoms related by inversion through origin
        positions = np.array([
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],  # -0.1, -0.2, -0.3 mod 1
        ])
        elements = np.array([6, 6])
        symops = find_symmetry_operations(uc, positions, elements)
        self.assertEqual(len(symops), 2)

    def test_acetic_acid_detects_correct_symops(self):
        """Acetic acid (Pna2_1) should detect 4 symmetry operations."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        uc_dict = crystal.unit_cell_atoms()
        symops = find_symmetry_operations(
            crystal.unit_cell,
            uc_dict["frac_pos"],
            uc_dict["element"],
        )
        self.assertEqual(len(symops), 4)

    def test_different_elements_not_swapped(self):
        """Symmetry should not swap atoms of different elements."""
        uc = UnitCell.cubic(5.0)
        # Two different atoms at positions related by 4-fold rotation
        # but with different elements - should not find 4-fold symmetry
        positions = np.array([
            [0.1, 0.2, 0.3],
            [0.8, 0.1, 0.3],  # would be related by 4-fold
        ])
        elements = np.array([6, 7])  # C and N
        symops = find_symmetry_operations(uc, positions, elements)
        # Should find fewer ops than a cubic cell with same-element atoms
        self.assertLess(len(symops), 48)


class TestDetectSymmetry(unittest.TestCase):
    """Tests for Crystal.detect_symmetry() method."""

    def test_detect_symmetry_on_acetic_acid(self):
        """detect_symmetry on acetic acid should return same or self."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        result = crystal.detect_symmetry()
        # Should detect the same space group (Pna2_1 = SG 33)
        self.assertEqual(
            result.space_group.international_tables_number,
            crystal.space_group.international_tables_number,
        )

    def test_detect_symmetry_roundtrip_urea(self):
        """Urea lowered to P1 via as_P1 should recover higher symmetry."""
        uc = UnitCell.from_lengths_and_angles(
            (5.565, 5.565, 4.684), (90, 90, 90), unit="degrees"
        )
        sg = SpaceGroup(113)
        elements = [
            Element["C"], Element["O"], Element["N"],
            Element["H"], Element["H"],
        ]
        positions = np.array([
            [0.0000, 0.5000, 0.3284],
            [0.0000, 0.5000, 0.5976],
            [0.1459, 0.6459, 0.1782],
            [0.2556, 0.7556, 0.2843],
            [0.1430, 0.6430, 0.0362],
        ])
        asym = AsymmetricUnit(elements, positions)
        urea = Crystal(uc, sg, asym)

        # Lower to a subgroup
        lowered = urea.to_subgroup(subgroup_index=2)
        n_symops_lowered = len(lowered.symmetry_operations)

        # Detect symmetry should find more operations
        recovered = lowered.detect_symmetry()
        self.assertGreater(
            len(recovered.symmetry_operations),
            n_symops_lowered,
        )

    def test_detect_symmetry_returns_self_if_already_maximal(self):
        """If crystal already has maximal symmetry, return self."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        result = crystal.detect_symmetry()
        # Should return self (same object) if no higher symmetry found
        if result.space_group.international_tables_number == crystal.space_group.international_tables_number:
            self.assertIs(result, crystal)


class TestFindAsymmetricUnitIndices(unittest.TestCase):
    """Tests for orbit-based asymmetric unit identification."""

    def test_p1_all_atoms_are_independent(self):
        """In P1, every atom is its own orbit representative."""
        from chmpy.crystal.symmetry_operation import SymmetryOperation
        uc = UnitCell.cubic(5.0)
        positions = np.array([
            [0.1, 0.2, 0.3],
            [0.5, 0.5, 0.5],
        ])
        elements = np.array([6, 6])
        identity = SymmetryOperation(np.eye(3), np.zeros(3))
        indices = find_asymmetric_unit_indices(
            positions, elements, [identity], uc.direct
        )
        self.assertEqual(len(indices), 2)

    def test_inversion_groups_atoms_in_pairs(self):
        """With inversion, atoms related by inversion form one orbit."""
        from chmpy.crystal.symmetry_operation import SymmetryOperation
        uc = UnitCell.from_lengths_and_angles(
            (5.0, 6.0, 7.0), (80, 85, 95), unit="degrees"
        )
        positions = np.array([
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
        ])
        elements = np.array([6, 6])
        identity = SymmetryOperation(np.eye(3), np.zeros(3))
        inversion = SymmetryOperation(-np.eye(3), np.zeros(3))
        indices = find_asymmetric_unit_indices(
            positions, elements, [identity, inversion], uc.direct
        )
        self.assertEqual(len(indices), 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

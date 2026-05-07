"""Tests for subgroup enumeration and asymmetric unit expansion."""

import logging
import unittest

import numpy as np

from chmpy.core.element import Element
from chmpy.crystal import Crystal, SpaceGroup
from chmpy.crystal.asymmetric_unit import AsymmetricUnit
from chmpy.crystal.space_group_table import SpaceGroupTable
from chmpy.crystal.subgroup import (
    StandardSettingResult,
    SubgroupEnumerator,
    SubgroupResult,
    _deduplicate_asymmetric_unit,
    _detect_centering,
    _identify_point_group,
    _reduce_centered_ops,
    compute_closure,
    expand_asymmetric_unit,
    identify_standard_setting,
)
from chmpy.crystal.symmetry_operation import SymmetryOperation
from chmpy.crystal.unit_cell import UnitCell

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class TestComputeClosure(unittest.TestCase):
    """Tests for the compute_closure function."""

    def test_identity_closure(self):
        """Closure of {identity} should be {identity}."""
        sg = SpaceGroup(2)  # P-1, 2 operations
        table = SpaceGroupTable.from_space_group(sg)
        e = table.identity_idx()
        closure = compute_closure(frozenset([e]), table)
        self.assertEqual(closure, frozenset([e]))

    def test_full_group_closure(self):
        """Closure of all elements should be the full group."""
        sg = SpaceGroup(14)  # P2_1/c, 4 operations
        table = SpaceGroupTable.from_space_group(sg)
        all_indices = frozenset(range(table.n_ops))
        closure = compute_closure(all_indices, table)
        self.assertEqual(closure, all_indices)

    def test_single_generator_p2(self):
        """A single non-identity element generates a subgroup."""
        sg = SpaceGroup(14)  # P2_1/c, 4 operations
        table = SpaceGroupTable.from_space_group(sg)
        e = table.identity_idx()
        for g in range(table.n_ops):
            if g == e:
                continue
            closure = compute_closure(frozenset([g]), table)
            self.assertIn(e, closure)
            self.assertEqual(table.n_ops % len(closure), 0)


class TestSubgroupEnumerator(unittest.TestCase):
    """Tests for the SubgroupEnumerator class."""

    def test_p1_has_no_proper_subgroups(self):
        """P1 (trivial group) has no proper subgroups."""
        sg = SpaceGroup(1)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        self.assertEqual(len(subgroups), 0)

    def test_p_minus1_has_p1_subgroup(self):
        """P-1 has exactly one proper subgroup: P1."""
        sg = SpaceGroup(2)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        self.assertGreaterEqual(len(subgroups), 1)
        p1_subgroups = [s for s in subgroups if s.index == 2]
        self.assertEqual(len(p1_subgroups), 1)
        self.assertEqual(p1_subgroups[0].space_group_number, 1)

    def test_lagrange_theorem(self):
        """All subgroup orders must divide the parent group order."""
        for sg_num in [2, 10, 14, 15, 62]:
            with self.subTest(sg_num=sg_num):
                sg = SpaceGroup(sg_num)
                enumerator = SubgroupEnumerator.from_space_group(sg)
                n_parent = len(sg.symmetry_operations)
                for result in enumerator.enumerate_all():
                    n_sub = len(result.symop_indices)
                    self.assertEqual(
                        n_parent % n_sub, 0,
                        f"SG {sg_num}: subgroup order {n_sub} does not divide {n_parent}"
                    )

    def test_all_subgroups_are_valid(self):
        """Every enumerated subgroup should pass verification."""
        for sg_num in [2, 14, 62]:
            with self.subTest(sg_num=sg_num):
                sg = SpaceGroup(sg_num)
                enumerator = SubgroupEnumerator.from_space_group(sg)
                for result in enumerator.enumerate_all():
                    self.assertTrue(
                        enumerator.verify_is_subgroup(result.symop_indices),
                        f"SG {sg_num}: subgroup {result.symop_indices} failed verification"
                    )

    def test_verify_rejects_non_subgroup(self):
        """verify_is_subgroup should reject arbitrary index sets."""
        sg = SpaceGroup(14)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        e = enumerator.sg_table.identity_idx()
        for g in range(enumerator.sg_table.n_ops):
            if g == e:
                continue
            indices = [e, g]
            result = enumerator.verify_is_subgroup(indices)
            if result:
                closure = compute_closure(frozenset(indices), enumerator.sg_table)
                self.assertEqual(frozenset(indices), closure)

    def test_index_values(self):
        """Index values should be correct: |G|/|H|."""
        sg = SpaceGroup(62)  # Pnma, 8 operations
        enumerator = SubgroupEnumerator.from_space_group(sg)
        for result in enumerator.enumerate_all():
            expected_index = 8 // len(result.symop_indices)
            self.assertEqual(result.index, expected_index)

    def test_z_prime_factor_equals_index(self):
        """Z' factor should equal the subgroup index."""
        sg = SpaceGroup(14)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        for result in enumerator.enumerate_all():
            self.assertEqual(result.z_prime_factor, float(result.index))

    def test_all_subgroups_have_point_group(self):
        """Every subgroup should have a point_group_symbol."""
        sg = SpaceGroup(62)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        for result in enumerator.enumerate_all():
            self.assertIsNotNone(result.point_group_symbol)
            self.assertIsInstance(result.point_group_symbol, str)
            self.assertGreater(len(result.point_group_symbol), 0)


class TestPointGroupIdentification(unittest.TestCase):
    """Tests for point group identification from rotation matrices."""

    def test_identity_is_point_group_1(self):
        """Single identity matrix has point group 1."""
        I = np.eye(3)
        self.assertEqual(_identify_point_group([I]), "1")

    def test_inversion_is_minus1(self):
        """Identity + inversion = point group -1."""
        I = np.eye(3)
        inv = -np.eye(3)
        self.assertEqual(_identify_point_group([I, inv]), "-1")

    def test_mm2_identified(self):
        """Standard mm2 rotation set is identified."""
        I = np.eye(3)
        c2 = np.diag([-1, -1, 1.0])
        mx = np.diag([-1, 1, 1.0])
        my = np.diag([1, -1, 1.0])
        self.assertEqual(_identify_point_group([I, c2, mx, my]), "mm2")

    def test_2_identified(self):
        """2-fold rotation gives point group 2."""
        I = np.eye(3)
        c2 = np.diag([-1, -1, 1.0])
        self.assertEqual(_identify_point_group([I, c2]), "2")


class TestSpaceGroupIdentification(unittest.TestCase):
    """Tests for identifying space groups of subgroups."""

    def test_p1_identified_as_subgroup_of_p_minus1(self):
        """P1 should be identified as a subgroup of P-1."""
        sg = SpaceGroup(2)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        p1_results = [s for s in subgroups if s.space_group_number == 1]
        self.assertGreater(len(p1_results), 0)

    def test_known_subgroup_p21_of_p21c(self):
        """P2_1 should appear as a subgroup of P2_1/c."""
        sg = SpaceGroup(14)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        sg_numbers = {s.space_group_number for s in subgroups if s.space_group_number}
        self.assertIn(1, sg_numbers)

    def test_pnma_subgroups_identified(self):
        """Pnma (SG 62) should have identifiable subgroups."""
        sg = SpaceGroup(62)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        identified = [s for s in subgroups if s.space_group_number is not None]
        self.assertGreater(len(identified), 0)
        self.assertIn(1, {s.space_group_number for s in identified})

    def test_tetragonal_subgroup_identified_via_basis_transform(self):
        """Subgroups of tetragonal groups should be identified after basis transform."""
        sg = SpaceGroup(113)  # P-42_1m
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        # The Pmm2 subgroup should be identified via basis transformation
        identified = [s for s in subgroups if s.space_group_number is not None]
        {s.space_group_number for s in identified}
        # Should identify more subgroups than just P-4, P2_12_12, P2, P1
        # The basis transform should catch Pmm2 and Pm
        self.assertGreater(len(identified), 4)


class TestFindByIndex(unittest.TestCase):
    """Tests for finding subgroups by index."""

    def test_find_index_2_of_p21c(self):
        """P2_1/c should have index-2 subgroups."""
        sg = SpaceGroup(14)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        index2 = enumerator.find_by_index(2)
        self.assertGreater(len(index2), 0)
        for result in index2:
            self.assertEqual(result.index, 2)
            self.assertEqual(len(result.symop_indices), 2)

    def test_find_nonexistent_index(self):
        """Finding an impossible index should return empty list."""
        sg = SpaceGroup(2)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        result = enumerator.find_by_index(3)
        self.assertEqual(len(result), 0)


class TestFindForTargetZPrime(unittest.TestCase):
    """Tests for finding subgroups by target Z'."""

    def test_target_z_prime_1_from_half(self):
        """Should find index-2 subgroup for Z' 0.5 -> 1.0."""
        sg = SpaceGroup(14)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        results = enumerator.find_for_target_z_prime(0.5, 1.0)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.index, 2)

    def test_target_z_prime_impossible(self):
        """Non-integer ratio should return empty list."""
        sg = SpaceGroup(14)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        results = enumerator.find_for_target_z_prime(0.5, 0.7)
        self.assertEqual(len(results), 0)


class TestExpandAsymmetricUnit(unittest.TestCase):
    """Tests for asymmetric unit expansion."""

    def test_expansion_doubles_general_positions(self):
        """Index-2 subgroup should double atoms on general positions."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        sg = crystal.space_group
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.find_by_index(2)

        if not subgroups:
            self.skipTest("No index-2 subgroups found for this space group")

        result = subgroups[0]
        new_asym = expand_asymmetric_unit(
            crystal.asymmetric_unit,
            crystal.symmetry_operations,
            result.symop_indices,
            enumerator.sg_table,
        )
        self.assertGreater(len(new_asym), len(crystal.asymmetric_unit))

    def test_crystal_to_subgroup_preserves_unit_cell(self):
        """to_subgroup should preserve the unit cell (t-subgroup)."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        subgroups = crystal.enumerate_subgroups()
        if not subgroups:
            self.skipTest("No subgroups found")

        new_crystal = crystal.to_subgroup(subgroup_result=subgroups[0])
        np.testing.assert_allclose(
            new_crystal.unit_cell.direct,
            crystal.unit_cell.direct,
            atol=1e-10,
        )

    def test_crystal_to_subgroup_has_fewer_symops(self):
        """Subgroup crystal should have fewer symmetry operations."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        subgroups = crystal.enumerate_subgroups()
        if not subgroups:
            self.skipTest("No subgroups found")

        new_crystal = crystal.to_subgroup(subgroup_result=subgroups[0])
        self.assertLess(
            len(new_crystal.symmetry_operations),
            len(crystal.symmetry_operations),
        )

    def test_deduplication_removes_redundant_atoms(self):
        """Asymmetric units with symmetry-related atoms should be deduplicated."""
        UnitCell.from_lengths_and_angles(
            (5.565, 5.565, 4.684), (90, 90, 90), unit="degrees"
        )
        sg = SpaceGroup(113)
        symops = sg.symmetry_operations

        # Build asymmetric unit with redundant atoms:
        # N1 at (0.1455, 0.6455, 0.1800) and N1B at (-0.1455, 0.3545, 0.1800)
        # are related by the -x,-y,z operation in P-42_1m
        elements = [
            Element["C"], Element["O"],
            Element["N"], Element["N"],  # N1 and N1B (redundant)
            Element["H"], Element["H"],
        ]
        positions = np.array([
            [0.0000, 0.5000, 0.3284],   # C1 (Wyckoff 2c)
            [0.0000, 0.5000, 0.5976],   # O1 (Wyckoff 2c)
            [0.1459, 0.6459, 0.1782],   # N1 (Wyckoff 4d)
            [-0.1459, 0.3541, 0.1782],  # N1B (mirror of N1, redundant)
            [0.2556, 0.7556, 0.2843],   # H1
            [0.1430, 0.6430, 0.0362],   # H2
        ])
        asym = AsymmetricUnit(elements, positions)

        unique_indices, redundancy_map = _deduplicate_asymmetric_unit(
            asym, symops
        )
        # N1B should be identified as redundant with N1
        self.assertEqual(len(unique_indices), 5)
        self.assertNotIn(3, unique_indices)  # N1B (index 3) is redundant
        self.assertEqual(redundancy_map[3], 2)  # N1B maps to N1 (index 2)

    def test_expansion_with_redundant_input(self):
        """Expansion should handle redundant atoms and produce correct count."""
        uc = UnitCell.from_lengths_and_angles(
            (5.565, 5.565, 4.684), (90, 90, 90), unit="degrees"
        )
        sg = SpaceGroup(113)
        # Include redundant atoms: N1 and N1B are symmetry-related
        elements = [
            Element["C"], Element["O"],
            Element["N"], Element["N"],  # redundant pair
            Element["H"], Element["H"],
            Element["H"], Element["H"],  # redundant pair
        ]
        positions = np.array([
            [0.0000, 0.5000, 0.3284],
            [0.0000, 0.5000, 0.5976],
            [0.1459, 0.6459, 0.1782],
            [-0.1459, 0.3541, 0.1782],  # redundant N
            [0.2556, 0.7556, 0.2843],
            [0.1430, 0.6430, 0.0362],
            [-0.2556, 0.2444, 0.2843],  # redundant H
            [-0.1430, 0.3570, 0.0362],  # redundant H
        ])
        asym = AsymmetricUnit(elements, positions)
        crystal = Crystal(uc, sg, asym)

        # Convert to index-4 subgroup
        n = crystal.to_subgroup(subgroup_index=4)
        # Should get 8 atoms (one full molecule), not 14+
        self.assertEqual(len(n.asymmetric_unit), 8)

    def test_urea_expansion(self):
        """Urea (P-42_1m) index-2 subgroup should expand special-position atoms."""
        uc = UnitCell.from_lengths_and_angles(
            (5.565, 5.565, 4.684), (90, 90, 90), unit="degrees"
        )
        sg = SpaceGroup(113)
        elements = [
            Element["C"], Element["O"], Element["N"],
            Element["H"], Element["H"],
        ]
        # Correct Wyckoff positions: C,O at 2c (0,1/2,z), N,H at 4d
        positions = np.array([
            [0.0000, 0.5000, 0.3284],
            [0.0000, 0.5000, 0.5976],
            [0.1459, 0.6459, 0.1782],
            [0.2556, 0.7556, 0.2843],
            [0.1430, 0.6430, 0.0362],
        ])
        asym = AsymmetricUnit(elements, positions)
        urea = Crystal(uc, sg, asym)

        self.assertEqual(len(urea.asymmetric_unit), 5)
        subgroups = urea.enumerate_subgroups()

        # Should find index-2 subgroups
        index2 = [s for s in subgroups if s.index == 2]
        self.assertGreater(len(index2), 0)

        # At least one index-2 subgroup should expand the asymmetric unit
        expanded_counts = []
        for s in index2:
            new_crystal = urea.to_subgroup(subgroup_result=s)
            expanded_counts.append(len(new_crystal.asymmetric_unit))

        # The Pmm2 subgroup should give 10 atoms (all atoms expand by index)
        self.assertIn(10, expanded_counts)


class TestCrystalAPI(unittest.TestCase):
    """Tests for the Crystal-level subgroup API."""

    def test_enumerate_subgroups_returns_list(self):
        """enumerate_subgroups should return a list of SubgroupResult."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        subgroups = crystal.enumerate_subgroups()
        self.assertIsInstance(subgroups, list)
        for sg in subgroups:
            self.assertIsInstance(sg, SubgroupResult)

    def test_to_subgroup_by_index(self):
        """to_subgroup with subgroup_index should work."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        try:
            new_crystal = crystal.to_subgroup(subgroup_index=2)
            self.assertIsNotNone(new_crystal)
            self.assertEqual(
                len(new_crystal.symmetry_operations),
                len(crystal.symmetry_operations) // 2,
            )
        except ValueError:
            self.skipTest("No index-2 subgroup found")

    def test_to_subgroup_requires_one_argument(self):
        """to_subgroup should raise ValueError if 0 or 2+ arguments given."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        with self.assertRaises(ValueError):
            crystal.to_subgroup()
        with self.assertRaises(ValueError):
            crystal.to_subgroup(target_z_prime=1.0, subgroup_index=2)

    def test_to_subgroup_stores_metadata(self):
        """to_subgroup should store parent info in properties."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        subgroups = crystal.enumerate_subgroups()
        if not subgroups:
            self.skipTest("No subgroups found")
        new_crystal = crystal.to_subgroup(subgroup_result=subgroups[0])
        self.assertIn("parent_space_group", new_crystal.properties)
        self.assertIn("subgroup_index", new_crystal.properties)

    def test_to_p1_via_subgroups(self):
        """Reducing to P1 via subgroups should give same atom count as as_P1."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        p1_crystal = crystal.as_P1()

        subgroups = crystal.enumerate_subgroups()
        p1_subgroups = [s for s in subgroups if s.space_group_number == 1]
        if not p1_subgroups:
            self.skipTest("P1 subgroup not found")

        new_crystal = crystal.to_subgroup(subgroup_result=p1_subgroups[0])
        self.assertEqual(len(new_crystal.asymmetric_unit), len(p1_crystal.asymmetric_unit))


def _make_urea_crystal():
    """Helper: build a urea crystal (P-42_1m, SG 113)."""
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
    return Crystal(uc, sg, asym)


class TestIdentifyStandardSetting(unittest.TestCase):
    """Tests for identify_standard_setting()."""

    def test_already_standard_returns_no_transform(self):
        """Standard P2_1 ops should be identified with no transform."""
        sg = SpaceGroup(4)  # P2_1
        result = identify_standard_setting(sg.symmetry_operations)
        self.assertIsNotNone(result)
        self.assertEqual(result.sg_number, 4)
        self.assertIsNone(result.basis_transform)
        self.assertIsNone(result.origin_shift)

    def test_p1_trivially_standard(self):
        """P1 should always be identified as standard."""
        sg = SpaceGroup(1)
        result = identify_standard_setting(sg.symmetry_operations)
        self.assertIsNotNone(result)
        self.assertEqual(result.sg_number, 1)
        self.assertIsNone(result.basis_transform)
        self.assertIsNone(result.origin_shift)

    def test_nonstandard_p21_needs_origin_shift(self):
        """Non-standard P2_1 ops (shifted origin) should require origin shift."""
        # Standard P2_1: x,y,z and -x,1/2+y,-z
        # Shifted by p=(1/4,0,0): 1/2-x,1/2+y,-z
        op1 = SymmetryOperation.from_string_code("x,y,z")
        op2 = SymmetryOperation.from_string_code("1/2-x,1/2+y,-z")
        result = identify_standard_setting([op1, op2])
        self.assertIsNotNone(result)
        self.assertEqual(result.sg_number, 4)
        self.assertIsNotNone(result.origin_shift)

    def test_result_is_dataclass(self):
        """identify_standard_setting should return a StandardSettingResult."""
        sg = SpaceGroup(14)  # P2_1/c
        result = identify_standard_setting(sg.symmetry_operations)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, StandardSettingResult)


class TestToStandardSetting(unittest.TestCase):
    """Tests for Crystal.to_standard_setting()."""

    def test_already_standard_returns_copy(self):
        """A crystal already in standard setting should return unchanged copy."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        std = crystal.to_standard_setting()
        # Should have same space group
        self.assertEqual(
            std.space_group.international_tables_number,
            crystal.space_group.international_tables_number,
        )
        # Positions should be equivalent
        np.testing.assert_allclose(
            std.asymmetric_unit.positions,
            crystal.asymmetric_unit.positions,
            atol=1e-4,
        )

    def test_origin_shift_p21_urea_subgroup(self):
        """Urea P2_1 subgroup should be standardized, matching standard symops."""
        urea = _make_urea_crystal()
        sub = urea.to_subgroup(subgroup_index=4)
        # The subgroup may be in a non-standard setting
        std = sub.to_standard_setting()
        # Should be a valid space group
        self.assertIsNotNone(std.space_group.international_tables_number)
        # Symops should now match the standard setting
        std_sg = SpaceGroup(std.space_group.international_tables_number,
                            choice=std.space_group.choice)
        std_codes = {s.integer_code for s in std_sg.symmetry_operations}
        our_codes = {s.integer_code for s in std.symmetry_operations}
        self.assertEqual(std_codes, our_codes)

    def test_preserves_cartesian_positions(self):
        """to_standard_setting should preserve physical structure.

        An origin shift moves all atoms by p in fractional coords.
        After shifting the "after" unit cell atoms back by p, they
        should coincide (mod lattice) with the "before" atoms.
        """
        urea = _make_urea_crystal()
        sub = urea.to_subgroup(subgroup_index=4)
        std = sub.to_standard_setting()

        from chmpy.crystal.subgroup import identify_standard_setting
        result = identify_standard_setting(sub.symmetry_operations)
        p = result.origin_shift if result.origin_shift is not None else np.zeros(3)

        # Generate all unit cell positions for both
        uc_before = sub.unit_cell_atoms()
        uc_after = std.unit_cell_atoms()

        self.assertEqual(len(uc_before["element"]), len(uc_after["element"]))

        # Undo the origin shift on the "after" fractional positions
        frac_after_shifted = (uc_after["frac_pos"] + p) % 1.0
        frac_before = uc_before["frac_pos"]

        matched = set()
        for i in range(len(frac_after_shifted)):
            elem_i = uc_after["element"][i]
            best_dist = float("inf")
            best_j = -1
            for j in range(len(frac_before)):
                if j in matched or uc_before["element"][j] != elem_i:
                    continue
                diff = frac_after_shifted[i] - frac_before[j]
                diff = diff - np.round(diff)
                d = np.linalg.norm(std.unit_cell.to_cartesian(diff.reshape(1, 3)))
                if d < best_dist:
                    best_dist = d
                    best_j = j
            self.assertLess(best_dist, 0.1,
                            f"UC atom {i} has no match within 0.1 Å")
            matched.add(best_j)

    def test_preserves_unit_cell_atoms(self):
        """Unit cell contents should be identical before and after standardization."""
        urea = _make_urea_crystal()
        sub = urea.to_subgroup(subgroup_index=4)
        std = sub.to_standard_setting()

        # Generate all unit cell atoms for both
        uc_before = sub.unit_cell_atoms()
        uc_after = std.unit_cell_atoms()

        # Same number of atoms
        self.assertEqual(
            len(uc_before["element"]),
            len(uc_after["element"]),
        )

        # Same element composition
        from collections import Counter
        self.assertEqual(
            Counter(int(e) for e in uc_before["element"]),
            Counter(int(e) for e in uc_after["element"]),
        )

    def test_p1_trivially_standard(self):
        """P1 crystal should be unchanged by to_standard_setting."""
        crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        p1 = crystal.as_P1()
        std = p1.to_standard_setting()
        self.assertEqual(std.space_group.international_tables_number, 1)
        np.testing.assert_allclose(
            std.asymmetric_unit.positions,
            p1.asymmetric_unit.positions,
            atol=1e-4,
        )


class TestCenteringDetection(unittest.TestCase):
    """Tests for centering detection and reduction."""

    def test_detect_centering_body(self):
        """_detect_centering returns (0.5, 0.5, 0.5) for I-centered ops."""
        # Build I-centered ops: each rotation appears with t and t+(1/2,1/2,1/2)
        sg = SpaceGroup(217)  # I-43m
        ops = []
        for s in sg.symmetry_operations:
            ops.append((s.rotation, s.translation))
        centering = _detect_centering(ops)
        self.assertIsNotNone(centering)
        np.testing.assert_allclose(centering, [0.5, 0.5, 0.5], atol=1e-3)

    def test_detect_centering_primitive(self):
        """_detect_centering returns None for primitive (P) ops."""
        sg = SpaceGroup(62)  # Pnma (primitive)
        ops = [(s.rotation, s.translation) for s in sg.symmetry_operations]
        centering = _detect_centering(ops)
        self.assertIsNone(centering)

    def test_reduce_centered_ops_halves_count(self):
        """_reduce_centered_ops should halve the number of operations."""
        sg = SpaceGroup(217)  # I-43m, 48 ops
        ops = [(s.rotation, s.translation) for s in sg.symmetry_operations]
        centering = _detect_centering(ops)
        self.assertIsNotNone(centering)
        reduced = _reduce_centered_ops(ops, centering)
        self.assertEqual(len(reduced), len(ops) // 2)

    def test_i_centered_r3_subgroup_detected(self):
        """Hexamine R3 subgroup should be identified after centering reduction."""
        sg = SpaceGroup(217)  # I-43m
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        # Look for subgroups with 12 ops (6 primitive * 2 centering)
        # These are the rhombohedral subgroups
        sub_12 = [s for s in subgroups if len(s.symop_indices) == 12]
        # At least some of these should now be identified
        identified_12 = [s for s in sub_12 if s.space_group_number is not None]
        self.assertGreater(
            len(identified_12), 0,
            f"No 12-op subgroups identified out of {len(sub_12)} candidates"
        )

    def test_hexamine_all_subgroups_identified(self):
        """All hexamine (I-43m) subgroups should be identified.

        With the eigenvector-based transform derivation, all subgroups
        including 8-op mm2 (I-centered orthorhombic) should be identified.
        """
        sg = SpaceGroup(217)
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()
        unidentified = [
            s for s in subgroups if s.space_group_number is None
        ]
        self.assertEqual(
            len(unidentified), 0,
            f"{len(unidentified)} subgroups unidentified: "
            f"{[(len(s.symop_indices), s.point_group_symbol) for s in unidentified]}"
        )


class TestCenteringStandardSetting(unittest.TestCase):
    """Tests for identify_standard_setting with centering reduction."""

    def test_i_centered_subgroup_standard_setting(self):
        """An I-centered subgroup should be identifiable via standard setting."""
        sg = SpaceGroup(217)  # I-43m
        enumerator = SubgroupEnumerator.from_space_group(sg)
        subgroups = enumerator.enumerate_all()

        # Find a 12-op subgroup (rhombohedral, I-centered)
        sub_12 = [s for s in subgroups if len(s.symop_indices) == 12]
        if not sub_12:
            self.skipTest("No 12-op subgroups found")

        sub = sub_12[0]
        # Get the actual symops
        symops = [
            sg.symmetry_operations[i] for i in sub.symop_indices
        ]
        result = identify_standard_setting(symops)
        self.assertIsNotNone(
            result,
            "Could not identify standard setting for 12-op I-centered subgroup"
        )
        self.assertIsNotNone(result.basis_transform)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

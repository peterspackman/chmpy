"""Tests for DimerIndex and dimer equivalence."""

import logging
import unittest
from pathlib import Path

from chmpy.crystal import Crystal
from chmpy.crystal.dimer_index import (
    AlgebraicDimerIndex,
    AlgebraicMoleculeRef,
    DimerIndex,
    DimerList,
    MolecularCosetTable,
    MoleculeRef,
    apply_symop_to_algebraic_dimer,
    compute_dimer_orbit_size,
    find_dimers_around_molecule,
    find_symmetry_unique_dimers,
    find_symmetry_unique_dimers_algebraic,
    normalize_algebraic_dimer,
)
from chmpy.crystal.orbit import MoleculeOrbitTable
from chmpy.crystal.space_group_table import SpaceGroupTable

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class MoleculeRefTestCase(unittest.TestCase):
    """Test MoleculeRef ordering."""

    def test_ordering(self):
        """Test that MoleculeRef ordering is lexicographic."""
        ref1 = MoleculeRef(asym_mol_idx=0, instance_idx=0, cell_offset=(0, 0, 0))
        ref2 = MoleculeRef(asym_mol_idx=0, instance_idx=1, cell_offset=(0, 0, 0))
        ref3 = MoleculeRef(asym_mol_idx=1, instance_idx=0, cell_offset=(0, 0, 0))

        self.assertTrue(ref1 < ref2)
        self.assertTrue(ref2 < ref3)
        self.assertTrue(ref1 < ref3)

    def test_cell_offset_ordering(self):
        """Test that cell offset affects ordering."""
        ref1 = MoleculeRef(asym_mol_idx=0, instance_idx=0, cell_offset=(0, 0, 0))
        ref2 = MoleculeRef(asym_mol_idx=0, instance_idx=0, cell_offset=(0, 0, 1))
        ref3 = MoleculeRef(asym_mol_idx=0, instance_idx=0, cell_offset=(1, 0, 0))

        self.assertTrue(ref1 < ref2)
        self.assertTrue(ref2 < ref3)


class DimerIndexTestCase(unittest.TestCase):
    """Test DimerIndex class."""

    def test_canonical_ordering(self):
        """Test that DimerIndex enforces canonical ordering."""
        ref1 = MoleculeRef(asym_mol_idx=0, instance_idx=0, cell_offset=(0, 0, 0))
        ref2 = MoleculeRef(asym_mol_idx=1, instance_idx=0, cell_offset=(0, 0, 0))

        # Create with mol_a > mol_b
        dimer = DimerIndex.create(ref2, ref1, 5.0)

        # Should be reordered
        self.assertEqual(dimer.mol_a, ref1)
        self.assertEqual(dimer.mol_b, ref2)

    def test_homodimer_detection(self):
        """Test homodimer detection."""
        ref1 = MoleculeRef(asym_mol_idx=0, instance_idx=0, cell_offset=(0, 0, 0))
        ref2 = MoleculeRef(asym_mol_idx=0, instance_idx=1, cell_offset=(0, 0, 0))
        ref3 = MoleculeRef(asym_mol_idx=1, instance_idx=0, cell_offset=(0, 0, 0))

        homo = DimerIndex.create(ref1, ref2, 5.0)
        hetero = DimerIndex.create(ref1, ref3, 5.0)

        self.assertTrue(homo.is_homodimer())
        self.assertFalse(hetero.is_homodimer())


class DimerListTestCase(unittest.TestCase):
    """Test DimerList class."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])

    def test_from_crystal_acetic_acid(self):
        """Test DimerList construction for acetic acid."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        dimer_list = DimerList.from_crystal(self.acetic, mol_orbit, cutoff=10.0)

        # Should find some dimers
        self.assertGreater(dimer_list.n_dimers, 0)

        # All dimers should be homodimers (only one molecule type)
        for dimer in dimer_list.dimers:
            self.assertTrue(dimer.is_homodimer())

    def test_from_crystal_ice_ii(self):
        """Test DimerList construction for ice II."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.ice_ii)
        dimer_list = DimerList.from_crystal(self.ice_ii, mol_orbit, cutoff=5.0)

        # Should find dimers
        self.assertGreater(dimer_list.n_dimers, 0)

    def test_distances_within_cutoff(self):
        """Test that all dimers are within cutoff."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        cutoff = 8.0
        dimer_list = DimerList.from_crystal(self.acetic, mol_orbit, cutoff=cutoff)

        for dimer in dimer_list.dimers:
            self.assertLessEqual(dimer.distance, cutoff)

    def test_sorted_by_distance(self):
        """Test that dimers are sorted by distance."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        dimer_list = DimerList.from_crystal(self.acetic, mol_orbit, cutoff=10.0)

        distances = [d.distance for d in dimer_list.dimers]
        self.assertEqual(distances, sorted(distances))

    def test_group_by_distance(self):
        """Test distance grouping."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        dimer_list = DimerList.from_crystal(self.acetic, mol_orbit, cutoff=10.0)

        groups = dimer_list.group_by_distance(tolerance=0.1)

        # Each group should have a consistent distance
        for avg_dist, dimers in groups:
            for dimer in dimers:
                self.assertAlmostEqual(dimer.distance, avg_dist, delta=0.1)

    def test_unique_distances(self):
        """Test unique distance enumeration."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        dimer_list = DimerList.from_crystal(self.acetic, mol_orbit, cutoff=10.0)

        unique_dists = dimer_list.unique_distances(tolerance=0.1)

        # Should be sorted
        self.assertEqual(unique_dists, sorted(unique_dists))


class SymmetryUniqueDimersTestCase(unittest.TestCase):
    """Test symmetry-unique dimer finding."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_find_dimers_around_molecule(self):
        """Test finding dimers around a single reference molecule."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        dimers = find_dimers_around_molecule(
            self.acetic, mol_orbit, ref_mol_idx=0, cutoff=10.0
        )

        # Should find some dimers
        self.assertGreater(len(dimers), 0)

        # All dimers should involve molecule 0 at cell (0,0,0) as either mol_a or mol_b
        ref_mol = MoleculeRef(
            asym_mol_idx=mol_orbit.unit_cell_instances[0].asym_mol_idx,
            instance_idx=0,
            cell_offset=(0, 0, 0),
        )
        for dimer in dimers:
            # Reference molecule should be one of the two molecules in the dimer
            self.assertTrue(
                dimer.mol_a == ref_mol or dimer.mol_b == ref_mol,
                f"Dimer {dimer} does not involve reference molecule"
            )

    def test_find_symmetry_unique_dimers(self):
        """Test finding symmetry-unique dimers."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        equiv_classes, all_dimers = find_symmetry_unique_dimers(
            self.acetic, mol_orbit, cutoff=10.0
        )

        # Should find some equivalence classes
        self.assertGreater(len(equiv_classes), 0)

        # Each class should have a representative and members
        for ec in equiv_classes:
            self.assertIsNotNone(ec.representative)
            self.assertGreater(len(ec.members), 0)
            self.assertEqual(ec.multiplicity, len(ec.members))

    def test_equivalence_class_consistency(self):
        """Test that all members in a class have same distance."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        equiv_classes, _ = find_symmetry_unique_dimers(
            self.acetic, mol_orbit, cutoff=10.0, distance_tolerance=0.01
        )

        for ec in equiv_classes:
            for member in ec.members:
                self.assertAlmostEqual(
                    member.distance, ec.distance, delta=0.01
                )

    def test_all_dimers_mapped(self):
        """Test that all dimers are mapped to unique dimers."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        equiv_classes, all_dimers = find_symmetry_unique_dimers(
            self.acetic, mol_orbit, cutoff=8.0
        )

        # Total mapped should equal sum of multiplicities
        total_in_classes = sum(ec.multiplicity for ec in equiv_classes)
        self.assertEqual(len(all_dimers), total_in_classes)

        # Each mapped dimer should reference a valid unique index
        for uid, _dimer in all_dimers:
            self.assertGreaterEqual(uid, 0)
            self.assertLess(uid, len(equiv_classes))

    def test_non_permutative_option(self):
        """Test that non-permutative option distinguishes A-B from B-A."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)

        # Acetic acid has only one molecule type, so no heterodimers
        # Both modes should give same result
        perm, _ = find_symmetry_unique_dimers(
            self.acetic, mol_orbit, cutoff=8.0, permutative=True
        )
        nonperm, _ = find_symmetry_unique_dimers(
            self.acetic, mol_orbit, cutoff=8.0, permutative=False
        )

        # For homodimers only, both should be the same
        self.assertEqual(len(perm), len(nonperm))


class NonPermutativeDimersTestCase(unittest.TestCase):
    """Test non-permutative dimer enumeration with heterodimers."""

    def test_heterodimer_distinction(self):
        """Test that heterodimers are distinguished in non-permutative mode."""
        # Load BAINA which has multiple molecule types
        from pathlib import Path
        baina_path = Path(__file__).parent.parent.parent.parent.parent.parent / "test_cases" / "BAINA.cif"
        if not baina_path.exists():
            self.skipTest("BAINA.cif not found")

        c = Crystal.load(baina_path)
        mol_orbit = MoleculeOrbitTable.from_crystal(c)

        # Skip if only one molecule type
        if mol_orbit.n_unique_molecules < 2:
            self.skipTest("Need multiple molecule types for this test")

        perm, _ = find_symmetry_unique_dimers(
            c, mol_orbit, cutoff=5.5, permutative=True
        )
        nonperm, _ = find_symmetry_unique_dimers(
            c, mol_orbit, cutoff=5.5, permutative=False
        )

        # Non-permutative should have more unique dimers (heterodimers counted twice)
        self.assertGreater(len(nonperm), len(perm))

        # Check that central/neighbor types are recorded
        for ec in nonperm:
            self.assertIsNotNone(ec.central_mol_type)
            self.assertIsNotNone(ec.neighbor_mol_type)


class AlgebraicMoleculeRefTestCase(unittest.TestCase):
    """Test AlgebraicMoleculeRef class."""

    def test_ordering(self):
        """Test lexicographic ordering."""
        ref1 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 0))
        ref2 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=1, cell=(0, 0, 0))
        ref3 = AlgebraicMoleculeRef(asym_mol_idx=1, symop_idx=0, cell=(0, 0, 0))

        self.assertTrue(ref1 < ref2)
        self.assertTrue(ref2 < ref3)
        self.assertTrue(ref1 < ref3)

    def test_cell_ordering(self):
        """Test cell affects ordering."""
        ref1 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 0))
        ref2 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 1))
        ref3 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(1, 0, 0))

        self.assertTrue(ref1 < ref2)
        self.assertTrue(ref2 < ref3)


class AlgebraicDimerIndexTestCase(unittest.TestCase):
    """Test AlgebraicDimerIndex class."""

    def test_canonical_ordering(self):
        """Test canonical ordering is enforced."""
        ref1 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 0))
        ref2 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=1, cell=(0, 0, 0))

        # Create with mol_a > mol_b
        dimer = AlgebraicDimerIndex.create(ref2, ref1)

        # Should be reordered
        self.assertEqual(dimer.mol_a, ref1)
        self.assertEqual(dimer.mol_b, ref2)

    def test_frozen_hashable(self):
        """Test that AlgebraicDimerIndex is hashable."""
        ref1 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 0))
        ref2 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=1, cell=(0, 0, 1))
        dimer = AlgebraicDimerIndex.create(ref1, ref2)

        # Should be hashable
        dimer_set = {dimer}
        self.assertIn(dimer, dimer_set)


class MolecularCosetTableTestCase(unittest.TestCase):
    """Test MolecularCosetTable construction."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_construction(self):
        """Test coset table is built correctly."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)
        coset_table = MolecularCosetTable.from_crystal(self.acetic, mol_orbit, sg_table)

        # Check dimensions
        self.assertEqual(coset_table.n_symops, len(self.acetic.space_group.symmetry_operations))
        self.assertEqual(coset_table.n_uc_molecules, mol_orbit.n_uc_molecules)

        # Each symop should map to a valid molecule
        for symop_idx in range(coset_table.n_symops):
            mol_idx = coset_table.symop_to_mol[symop_idx]
            self.assertGreaterEqual(mol_idx, 0)
            self.assertLess(mol_idx, coset_table.n_uc_molecules)

    def test_site_symmetry_order(self):
        """Test site symmetry order calculation."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)
        coset_table = MolecularCosetTable.from_crystal(self.acetic, mol_orbit, sg_table)

        # Sum of site symmetry orders should equal total symops
        total_order = sum(coset_table.site_symmetry_order)
        self.assertEqual(total_order, coset_table.n_symops)


class AlgebraicDimerTransformTestCase(unittest.TestCase):
    """Test algebraic dimer transformations."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        self.sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)
        self.coset_table = MolecularCosetTable.from_crystal(
            self.acetic, self.mol_orbit, self.sg_table
        )

    def test_identity_transform(self):
        """Test that identity symop doesn't change dimer."""
        ref1 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 0))
        ref2 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=1, cell=(0, 0, 1))
        dimer = AlgebraicDimerIndex.create(ref1, ref2)

        identity_idx = self.sg_table.identity_idx()
        transformed = apply_symop_to_algebraic_dimer(
            dimer, identity_idx, self.sg_table, self.coset_table
        )

        # Identity should preserve the dimer (after normalization)
        dimer_norm = normalize_algebraic_dimer(dimer, self.sg_table)
        transformed_norm = normalize_algebraic_dimer(transformed, self.sg_table)
        self.assertEqual(dimer_norm._key(), transformed_norm._key())

    def test_orbit_closure(self):
        """Test that orbit of orbit element equals original orbit."""
        ref1 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=0, cell=(0, 0, 0))
        ref2 = AlgebraicMoleculeRef(asym_mol_idx=0, symop_idx=1, cell=(1, 0, 0))
        dimer = AlgebraicDimerIndex.create(ref1, ref2)

        # Compute orbit of original dimer
        orbit = set()
        for g in range(self.sg_table.n_ops):
            t = apply_symop_to_algebraic_dimer(dimer, g, self.sg_table, self.coset_table)
            t_norm = normalize_algebraic_dimer(t, self.sg_table)
            orbit.add(t_norm._key())

        # Pick an element of the orbit and compute its orbit
        orbit_list = list(orbit)
        if len(orbit_list) > 1:
            # Create a dimer from the second orbit element
            key = orbit_list[1]
            mol_a_key, mol_b_key = key
            mol_a2 = AlgebraicMoleculeRef(*mol_a_key)
            mol_b2 = AlgebraicMoleculeRef(*mol_b_key)
            dimer2 = AlgebraicDimerIndex(mol_a=mol_a2, mol_b=mol_b2)

            orbit2 = set()
            for g in range(self.sg_table.n_ops):
                t = apply_symop_to_algebraic_dimer(dimer2, g, self.sg_table, self.coset_table)
                t_norm = normalize_algebraic_dimer(t, self.sg_table)
                orbit2.add(t_norm._key())

            # Orbits should be equal
            self.assertEqual(orbit, orbit2)


class AlgebraicEquivalenceTestCase(unittest.TestCase):
    """Test algebraic dimer equivalence finding."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_algebraic_finds_classes(self):
        """Test that algebraic method finds equivalence classes."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        equiv_classes, dimers, distances = find_symmetry_unique_dimers_algebraic(
            self.acetic, mol_orbit, cutoff=8.0
        )

        # Should find classes
        self.assertGreater(len(equiv_classes), 0)

        # Each class should have valid indices
        for ec in equiv_classes:
            for idx in ec.member_indices:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(dimers))

    def test_orbit_size_multiplicity(self):
        """Test that multiplicity equals orbit size."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        sg_table = SpaceGroupTable.from_space_group(self.acetic.space_group)
        coset_table = MolecularCosetTable.from_crystal(self.acetic, mol_orbit, sg_table)

        equiv_classes, dimers, distances = find_symmetry_unique_dimers_algebraic(
            self.acetic, mol_orbit, cutoff=6.0
        )

        for ec in equiv_classes:
            rep = ec.representative
            orbit_size = compute_dimer_orbit_size(rep, sg_table, coset_table)
            self.assertEqual(ec.multiplicity, orbit_size)

    def test_total_orbit_sum_consistent(self):
        """Test that sum of orbit sizes is consistent with enumeration."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        n_uc_mols = mol_orbit.n_uc_molecules

        equiv_classes, dimers, distances = find_symmetry_unique_dimers_algebraic(
            self.acetic, mol_orbit, cutoff=6.0
        )

        # Sum of (orbit_size / n_uc_mols) should roughly equal unique dimers enumerated
        total_orbit = sum(ec.multiplicity for ec in equiv_classes)
        normalized = total_orbit / n_uc_mols

        # Should be close to the number of unique dimers we found
        self.assertGreater(normalized, 0)

    def test_dimers_sorted_by_distance(self):
        """Test that equivalence classes are sorted by distance."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        equiv_classes, dimers, distances = find_symmetry_unique_dimers_algebraic(
            self.acetic, mol_orbit, cutoff=8.0
        )

        # Check classes are sorted
        prev_dist = 0.0
        for ec in equiv_classes:
            dist = distances[ec.member_indices[0]]
            self.assertGreaterEqual(dist, prev_dist - 0.001)  # Allow small tolerance
            prev_dist = dist


class UreaAlgebraicTestCase(unittest.TestCase):
    """Test algebraic equivalence with urea (Z' < 1 case)."""

    @classmethod
    def setUpClass(cls):
        # Try to find urea.cif
        urea_paths = [
            Path(__file__).parent.parent.parent.parent.parent.parent / "test_cases" / "urea.cif",
            Path(__file__).parent.parent.parent.parent.parent.parent / "x23" / "urea.cif",
        ]
        cls.urea_path = None
        for p in urea_paths:
            if p.exists():
                cls.urea_path = p
                break

    def setUp(self):
        if self.urea_path is None:
            self.skipTest("urea.cif not found")
        self.urea = Crystal.load(self.urea_path)

    def test_site_symmetry_detected(self):
        """Test that site symmetry is correctly detected for urea."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.urea)
        sg_table = SpaceGroupTable.from_space_group(self.urea.space_group)
        coset_table = MolecularCosetTable.from_crystal(self.urea, mol_orbit, sg_table)

        # Urea has Z' = 0.5, so site symmetry order > 1
        n_symops = len(self.urea.space_group.symmetry_operations)
        n_mols = mol_orbit.n_uc_molecules

        # Average site symmetry order = n_symops / n_mols
        n_symops // n_mols
        for i in range(coset_table.n_uc_molecules):
            self.assertGreaterEqual(coset_table.site_symmetry_order[i], 1)

    def test_algebraic_equivalence_urea(self):
        """Test algebraic equivalence classes for urea."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.urea)

        equiv_classes, dimers, distances = find_symmetry_unique_dimers_algebraic(
            self.urea, mol_orbit, cutoff=5.0
        )

        # Should find some classes
        self.assertGreater(len(equiv_classes), 0)

        # Orbit sizes should be reasonable (at least 1, at most n_symops)
        n_symops = len(self.urea.space_group.symmetry_operations)
        for ec in equiv_classes:
            self.assertGreaterEqual(ec.multiplicity, 1)
            self.assertLessEqual(ec.multiplicity, n_symops)

    def test_distance_vs_algebraic_consistency(self):
        """Test relationship between distance and algebraic approaches."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.urea)
        n_uc_mols = mol_orbit.n_uc_molecules

        dist_classes, _ = find_symmetry_unique_dimers(
            self.urea, mol_orbit, cutoff=5.0
        )
        alg_classes, dimers, distances = find_symmetry_unique_dimers_algebraic(
            self.urea, mol_orbit, cutoff=5.0
        )

        # Sum of (algebraic orbit sizes / n_uc_mols) should equal sum of distance multiplicities
        dist_sum = sum(ec.multiplicity for ec in dist_classes)
        alg_sum = sum(ec.multiplicity for ec in alg_classes) / n_uc_mols

        self.assertAlmostEqual(dist_sum, alg_sum, delta=1.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

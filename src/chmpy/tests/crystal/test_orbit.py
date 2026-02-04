"""Tests for AtomOrbitTable and MoleculeOrbitTable."""

import logging
import unittest

import numpy as np

from chmpy.crystal import Crystal
from chmpy.crystal.orbit import AtomOrbitTable, MoleculeOrbitTable

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class AtomOrbitTableTestCase(unittest.TestCase):
    """Test AtomOrbitTable against existing unit_cell_atoms() implementation."""

    def setUp(self):
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.r3c_example = Crystal.load(TEST_FILES["r3c_example.cif"])

    def test_from_crystal_ice_ii(self):
        """Test AtomOrbitTable matches unit_cell_atoms for iceII (P1)."""
        crystal = self.ice_ii
        orbit = AtomOrbitTable.from_crystal(crystal)
        uc_atoms = crystal.unit_cell_atoms()

        # Check counts match
        self.assertEqual(orbit.n_uc, len(uc_atoms["element"]))
        self.assertEqual(orbit.n_asym, len(crystal.asymmetric_unit))
        self.assertEqual(orbit.n_symops, len(crystal.symmetry_operations))

        # For P1, n_uc should equal n_asym (no symmetry copies)
        self.assertEqual(orbit.n_uc, orbit.n_asym)

    def test_from_crystal_acetic_acid(self):
        """Test AtomOrbitTable matches unit_cell_atoms for acetic acid (Pna21)."""
        crystal = self.acetic
        orbit = AtomOrbitTable.from_crystal(crystal)
        uc_atoms = crystal.unit_cell_atoms()

        # Check counts match
        self.assertEqual(orbit.n_uc, len(uc_atoms["element"]))
        self.assertEqual(orbit.n_asym, len(crystal.asymmetric_unit))
        self.assertEqual(orbit.n_symops, len(crystal.symmetry_operations))

        # Pna21 has 4 symops, so expect 4x asym atoms in UC
        self.assertEqual(orbit.n_symops, 4)
        self.assertEqual(orbit.n_uc, orbit.n_asym * 4)

    def test_from_crystal_r3c(self):
        """Test AtomOrbitTable matches unit_cell_atoms for r3c_example (high symmetry)."""
        crystal = self.r3c_example
        orbit = AtomOrbitTable.from_crystal(crystal)
        uc_atoms = crystal.unit_cell_atoms()

        # Check counts match
        self.assertEqual(orbit.n_uc, len(uc_atoms["element"]))
        LOG.info(f"R3c: n_asym={orbit.n_asym}, n_symops={orbit.n_symops}, n_uc={orbit.n_uc}")

    def test_positions_match(self):
        """Test that positions match existing implementation."""
        for name, crystal in [
            ("ice_ii", self.ice_ii),
            ("acetic", self.acetic),
            ("r3c_example", self.r3c_example),
        ]:
            with self.subTest(crystal=name):
                orbit = AtomOrbitTable.from_crystal(crystal)
                uc_atoms = crystal.unit_cell_atoms()

                # Positions should match (after sorting to handle order differences)
                orbit_frac = orbit.uc_frac_positions
                existing_frac = uc_atoms["frac_pos"]

                # Sort both arrays for comparison (positions may be in different order)
                orbit_sorted = orbit_frac[np.lexsort(orbit_frac.T)]
                existing_sorted = existing_frac[np.lexsort(existing_frac.T)]

                np.testing.assert_allclose(
                    orbit_sorted, existing_sorted, atol=1e-6,
                    err_msg=f"Fractional positions don't match for {name}"
                )

    def test_atomic_numbers_match(self):
        """Test that atomic numbers match existing implementation."""
        for name, crystal in [
            ("ice_ii", self.ice_ii),
            ("acetic", self.acetic),
            ("r3c_example", self.r3c_example),
        ]:
            with self.subTest(crystal=name):
                orbit = AtomOrbitTable.from_crystal(crystal)
                uc_atoms = crystal.unit_cell_atoms()

                # Element counts should match
                orbit_counts = np.bincount(orbit.uc_atomic_numbers)
                existing_counts = np.bincount(uc_atoms["element"])

                np.testing.assert_array_equal(
                    orbit_counts, existing_counts,
                    err_msg=f"Atomic number counts don't match for {name}"
                )

    def test_occupations_match(self):
        """Test that occupations match existing implementation."""
        for name, crystal in [
            ("ice_ii", self.ice_ii),
            ("acetic", self.acetic),
        ]:
            with self.subTest(crystal=name):
                orbit = AtomOrbitTable.from_crystal(crystal)
                uc_atoms = crystal.unit_cell_atoms()

                # Total occupation should match
                np.testing.assert_allclose(
                    np.sum(orbit.uc_occupations),
                    np.sum(uc_atoms["occupation"]),
                    atol=1e-6,
                    err_msg=f"Total occupation doesn't match for {name}"
                )

    def test_to_dict_matches(self):
        """Test that to_dict() produces compatible output."""
        for name, crystal in [
            ("ice_ii", self.ice_ii),
            ("acetic", self.acetic),
        ]:
            with self.subTest(crystal=name):
                orbit = AtomOrbitTable.from_crystal(crystal)
                orbit_dict = orbit.to_dict()
                uc_atoms = crystal.unit_cell_atoms()

                # Check all expected keys exist
                for key in ["asym_atom", "frac_pos", "cart_pos", "element", "symop", "label", "occupation"]:
                    self.assertIn(key, orbit_dict, f"Missing key {key}")

                # Check shapes match
                self.assertEqual(orbit_dict["frac_pos"].shape, uc_atoms["frac_pos"].shape)
                self.assertEqual(orbit_dict["cart_pos"].shape, uc_atoms["cart_pos"].shape)
                self.assertEqual(len(orbit_dict["element"]), len(uc_atoms["element"]))

    def test_orbit_mapping_consistency(self):
        """Test that forward and reverse mappings are consistent."""
        orbit = AtomOrbitTable.from_crystal(self.acetic)

        # For each UC site, verify the mapping is consistent
        for uc_idx in range(orbit.n_uc):
            asym_idx, symop_idx = orbit.get_asym_atom_for_uc_site(uc_idx)

            # The orbit_uc_idx should map back to this site (or -1 if merged)
            mapped_uc = orbit.get_uc_site_for_asym_atom(asym_idx, symop_idx)
            if mapped_uc != -1:
                self.assertEqual(mapped_uc, uc_idx)

    def test_propagate_positions(self):
        """Test that position propagation works correctly."""
        crystal = self.acetic
        orbit = AtomOrbitTable.from_crystal(crystal)

        # Get original asym positions
        original_asym = crystal.asymmetric_unit.positions.copy()

        # Propagate with original positions - should match existing
        new_frac, new_cart = orbit.propagate_asym_positions(original_asym)

        # Positions should match the stored ones
        np.testing.assert_allclose(new_frac, orbit.uc_frac_positions, atol=1e-8)
        np.testing.assert_allclose(new_cart, orbit.uc_cart_positions, atol=1e-8)

    def test_propagate_perturbed_positions(self):
        """Test that perturbation propagates correctly."""
        crystal = self.acetic
        orbit = AtomOrbitTable.from_crystal(crystal)

        # Small perturbation to first atom
        perturbed_asym = crystal.asymmetric_unit.positions.copy()
        perturbation = np.array([0.001, 0.002, -0.001])
        perturbed_asym[0] += perturbation

        # Propagate
        new_frac, new_cart = orbit.propagate_asym_positions(perturbed_asym)

        # Find all UC sites that came from asym atom 0
        affected_uc = np.where(orbit.uc_to_asym == 0)[0]
        self.assertGreater(len(affected_uc), 0)

        # Those sites should have moved (under symop transformation of perturbation)
        for uc_idx in affected_uc:
            old_pos = orbit.uc_frac_positions[uc_idx]
            new_pos = new_frac[uc_idx]
            # Positions should differ
            self.assertFalse(np.allclose(old_pos, new_pos, atol=1e-6))

    def test_site_multiplicity(self):
        """Test site multiplicity tracking."""
        orbit = AtomOrbitTable.from_crystal(self.acetic)

        # For structures without special positions, all multiplicities should be 1
        # (Each UC site generated by exactly one (asym, symop) pair)
        np.testing.assert_array_equal(
            orbit.site_multiplicity, np.ones(orbit.n_uc, dtype=np.int32)
        )


class AtomOrbitTableSpecialPositionTestCase(unittest.TestCase):
    """Test AtomOrbitTable handling of special positions."""

    def test_merged_sites(self):
        """Test that sites on special positions are properly merged."""
        # Create a crystal with an atom on a special position
        from chmpy.crystal import Crystal, SpaceGroup, UnitCell
        from chmpy.crystal.asymmetric_unit import AsymmetricUnit
        from chmpy.core.element import Element

        # Create a P1 structure with two atoms at same position
        uc = UnitCell.cubic(10.0)
        sg = SpaceGroup(1)  # P1

        # Two atoms at same position with partial occupancy
        elements = [Element["C"], Element["C"]]
        positions = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        labels = np.array(["C1", "C2"])

        asym = AsymmetricUnit(
            elements=elements,
            positions=positions,
            labels=labels,
            occupation=np.array([0.5, 0.5]),
        )

        crystal = Crystal(uc, sg, asym)
        orbit = AtomOrbitTable.from_crystal(crystal)

        # Should have merged to 1 site
        self.assertEqual(orbit.n_uc, 1)
        # With combined occupation
        self.assertAlmostEqual(orbit.uc_occupations[0], 1.0)


class MoleculeOrbitTableTestCase(unittest.TestCase):
    """Test MoleculeOrbitTable class."""

    def setUp(self):
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_from_crystal_ice_ii(self):
        """Test MoleculeOrbitTable for iceII."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.ice_ii)

        # Ice II should have 12 water molecules in the unit cell
        # (P1 space group, 12 unique in asym unit)
        existing_mols = self.ice_ii.symmetry_unique_molecules()
        self.assertEqual(mol_orbit.n_unique_molecules, len(existing_mols))

    def test_from_crystal_acetic_acid(self):
        """Test MoleculeOrbitTable for acetic acid."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)

        # Acetic acid has 1 unique molecule, 4 symops -> 4 molecules in UC
        self.assertEqual(mol_orbit.n_unique_molecules, 1)
        self.assertEqual(mol_orbit.n_uc_molecules, 4)

    def test_molecule_count_matches_existing(self):
        """Test that molecule counts match existing implementation."""
        for name, crystal in [
            ("ice_ii", self.ice_ii),
            ("acetic", self.acetic),
        ]:
            with self.subTest(crystal=name):
                mol_orbit = MoleculeOrbitTable.from_crystal(crystal)
                existing_unique = crystal.symmetry_unique_molecules()
                existing_uc = crystal.unit_cell_molecules()

                self.assertEqual(
                    mol_orbit.n_unique_molecules,
                    len(existing_unique),
                    f"Unique molecule count mismatch for {name}",
                )
                self.assertEqual(
                    mol_orbit.n_uc_molecules,
                    len(existing_uc),
                    f"UC molecule count mismatch for {name}",
                )

    def test_instance_centroids(self):
        """Test that centroids are computed correctly."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)

        # Centroids should be in [0, 1) range
        self.assertTrue(np.all(mol_orbit.instance_centroids_frac >= 0))
        self.assertTrue(np.all(mol_orbit.instance_centroids_frac < 1))

        # Should have correct shape
        self.assertEqual(
            mol_orbit.instance_centroids_frac.shape,
            (mol_orbit.n_uc_molecules, 3),
        )
        self.assertEqual(
            mol_orbit.instance_centroids_cart.shape,
            (mol_orbit.n_uc_molecules, 3),
        )

    def test_get_instances_of_molecule(self):
        """Test retrieving instances of a specific molecule."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)

        # Acetic acid has 1 unique molecule with 4 instances
        instances = mol_orbit.get_instances_of_molecule(0)
        self.assertEqual(len(instances), 4)

        # All instances should reference the same unique molecule
        for inst in instances:
            self.assertEqual(inst.asym_mol_idx, 0)

    def test_instance_to_unique_mapping(self):
        """Test that instance_to_unique mapping is consistent."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)

        for i, inst in enumerate(mol_orbit.unit_cell_instances):
            self.assertEqual(
                mol_orbit.instance_to_unique[i],
                inst.asym_mol_idx,
            )

    def test_unique_molecule_atom_indices(self):
        """Test that atom indices for unique molecules are valid."""
        mol_orbit = MoleculeOrbitTable.from_crystal(self.acetic)
        uc_atoms = self.acetic.unit_cell_atoms()
        n_uc = len(uc_atoms["element"])

        for mol_idx in range(mol_orbit.n_unique_molecules):
            atom_indices = mol_orbit.get_unique_molecule_atoms(mol_idx)
            # All indices should be valid
            self.assertTrue(all(0 <= idx < n_uc for idx in atom_indices))
            # Should have non-zero atoms
            self.assertGreater(len(atom_indices), 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

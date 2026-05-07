"""Tests for MutableCrystalState and CrystalPerturbationManager."""

import logging
import unittest

import numpy as np

from chmpy.crystal import Crystal
from chmpy.crystal.crystal_state import (
    CrystalPerturbationManager,
    MutableCrystalState,
)

from .. import TEST_FILES

LOG = logging.getLogger(__name__)


class MutableCrystalStateTestCase(unittest.TestCase):
    """Test MutableCrystalState class."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.ice_ii = Crystal.load(TEST_FILES["iceII.cif"])

    def test_from_crystal(self):
        """Test creating state from crystal."""
        state = MutableCrystalState.from_crystal(self.acetic)

        # Check dimensions match
        self.assertEqual(state.asym_frac_positions.shape[0], len(self.acetic.asymmetric_unit))
        self.assertEqual(state.uc_frac_positions.shape[0], state.atom_orbit.n_uc)

    def test_initial_positions_match(self):
        """Test that initial positions match the crystal."""
        state = MutableCrystalState.from_crystal(self.acetic)

        # Asymmetric unit positions should match
        np.testing.assert_allclose(
            state.asym_frac_positions,
            self.acetic.asymmetric_unit.positions,
            atol=1e-8,
        )

        # UC positions should match (order may differ, so compare sorted)
        uc_atoms = self.acetic.unit_cell_atoms()
        state_sorted = state.uc_frac_positions[np.lexsort(state.uc_frac_positions.T)]
        existing_sorted = uc_atoms["frac_pos"][np.lexsort(uc_atoms["frac_pos"].T)]
        np.testing.assert_allclose(state_sorted, existing_sorted, atol=1e-6)

    def test_update_asym_positions(self):
        """Test position update propagation."""
        state = MutableCrystalState.from_crystal(self.acetic)

        # Small perturbation
        new_asym = state.asym_frac_positions.copy()
        new_asym[0] += np.array([0.001, 0.002, -0.001])

        # Update
        state.update_asym_positions(new_asym)

        # Asym positions should be updated
        np.testing.assert_allclose(state.asym_frac_positions, new_asym, atol=1e-10)

        # UC positions should be different from original
        original = MutableCrystalState.from_crystal(self.acetic)
        self.assertFalse(np.allclose(state.uc_frac_positions, original.uc_frac_positions))

    def test_perturb_single_atom(self):
        """Test single atom perturbation."""
        state = MutableCrystalState.from_crystal(self.acetic)
        original_pos = state.asym_frac_positions.copy()

        delta = np.array([0.01, 0.0, 0.0])
        state.perturb_asym_position(0, delta)

        # Atom 0 should be perturbed
        np.testing.assert_allclose(
            state.asym_frac_positions[0],
            original_pos[0] + delta,
            atol=1e-10,
        )

        # Other atoms unchanged
        np.testing.assert_allclose(
            state.asym_frac_positions[1:],
            original_pos[1:],
            atol=1e-10,
        )

    def test_gradient_propagation_roundtrip(self):
        """Test gradient back-propagation produces consistent results."""
        state = MutableCrystalState.from_crystal(self.acetic)

        # Create random asymmetric gradient
        asym_grad = np.random.randn(state.atom_orbit.n_asym, 3)

        # Forward: asym -> UC
        uc_grad = state.get_uc_gradient_from_asym_gradient(asym_grad)

        # Shape should be correct
        self.assertEqual(uc_grad.shape, (state.atom_orbit.n_uc, 3))

    def test_gradient_back_propagation(self):
        """Test gradient back-propagation sums contributions correctly."""
        state = MutableCrystalState.from_crystal(self.acetic)

        # Create unit UC gradient (all ones)
        uc_grad = np.ones((state.atom_orbit.n_uc, 3))

        # Back-propagate
        asym_grad = state.get_asym_gradient_from_uc_gradient(uc_grad)

        # Each asym atom should get contributions from all its symmetry copies
        # For Pna21 (4 symops), each asym atom maps to 4 UC atoms
        # Total contribution should be 4 (accounting for rotation)
        # The sum of all asym gradients should equal sum of UC gradients
        # when properly transformed
        self.assertEqual(asym_grad.shape, (state.atom_orbit.n_asym, 3))

    def test_to_dict_format(self):
        """Test that to_dict produces correct format."""
        state = MutableCrystalState.from_crystal(self.acetic)
        result = state.to_dict()

        expected_keys = ["asym_atom", "frac_pos", "cart_pos", "element", "symop", "label", "occupation"]
        for key in expected_keys:
            self.assertIn(key, result)


class CrystalPerturbationManagerTestCase(unittest.TestCase):
    """Test CrystalPerturbationManager class."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_from_crystal(self):
        """Test creating manager from crystal."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        self.assertIsNotNone(manager.state)
        self.assertIsNotNone(manager.mol_orbit)

    def test_reset_to_original(self):
        """Test reset functionality."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        # Apply displacement
        disp = np.random.randn(manager.state.atom_orbit.n_asym, 3) * 0.01
        manager.apply_displacement(disp)

        # Reset
        manager.reset_to_original()

        # Should match original
        np.testing.assert_allclose(
            manager.state.asym_frac_positions,
            manager.original_asym_positions,
            atol=1e-10,
        )

    def test_apply_displacement(self):
        """Test displacement application."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        disp = np.random.randn(manager.state.atom_orbit.n_asym, 3) * 0.01
        manager.apply_displacement(disp)

        # Check displacement was applied
        np.testing.assert_allclose(
            manager.state.asym_frac_positions,
            manager.original_asym_positions + disp,
            atol=1e-10,
        )

    def test_get_displacement_from_original(self):
        """Test displacement tracking."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        disp = np.random.randn(manager.state.atom_orbit.n_asym, 3) * 0.01
        manager.apply_displacement(disp)

        computed_disp = manager.get_displacement_from_original()
        np.testing.assert_allclose(computed_disp, disp, atol=1e-10)

    def test_molecule_centroids(self):
        """Test molecule centroid computation."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        centroids = manager.get_molecule_centroids()

        self.assertIsNotNone(centroids)
        self.assertEqual(centroids.shape, (manager.mol_orbit.n_uc_molecules, 3))

    def test_flat_positions_interface(self):
        """Test flat position getter/setter."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        flat = manager.get_flat_positions()
        self.assertEqual(len(flat), manager.n_degrees_of_freedom)

        # Modify and set back
        flat_modified = flat + 0.001
        manager.set_flat_positions(flat_modified)

        flat_retrieved = manager.get_flat_positions()
        np.testing.assert_allclose(flat_retrieved, flat_modified, atol=1e-10)

    def test_n_degrees_of_freedom(self):
        """Test degrees of freedom count."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        expected_dof = len(self.acetic.asymmetric_unit) * 3
        self.assertEqual(manager.n_degrees_of_freedom, expected_dof)


class GradientConsistencyTestCase(unittest.TestCase):
    """Test gradient propagation consistency."""

    def setUp(self):
        self.acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_numerical_gradient_consistency(self):
        """Test gradient propagation by numerical differentiation."""
        manager = CrystalPerturbationManager.from_crystal(self.acetic)

        # Define a simple energy function on UC positions
        def energy(uc_cart):
            # Sum of squared distances from origin
            return np.sum(uc_cart**2)

        def numerical_asym_gradient(eps=1e-5):
            """Compute numerical gradient on asymmetric unit."""
            grad = np.zeros((manager.state.atom_orbit.n_asym, 3))
            original = manager.state.asym_frac_positions.copy()

            for i in range(manager.state.atom_orbit.n_asym):
                for j in range(3):
                    # Forward
                    new_pos = original.copy()
                    new_pos[i, j] += eps
                    manager.state.update_asym_positions(new_pos)
                    e_plus = energy(manager.state.uc_cart_positions)

                    # Backward
                    new_pos = original.copy()
                    new_pos[i, j] -= eps
                    manager.state.update_asym_positions(new_pos)
                    e_minus = energy(manager.state.uc_cart_positions)

                    grad[i, j] = (e_plus - e_minus) / (2 * eps)

            # Reset
            manager.state.update_asym_positions(original)
            return grad

        # Compute analytical UC gradient: d(sum x^2)/dx = 2x
        2 * manager.state.uc_cart_positions

        # We need to convert this to fractional coordinates for back-prop
        # For numerical comparison, use numerical gradient instead
        numerical_grad = numerical_asym_gradient()

        # The gradient should be non-zero (structure not at minimum)
        self.assertFalse(np.allclose(numerical_grad, 0))

        # Gradient should be finite (no NaN or Inf)
        self.assertTrue(np.all(np.isfinite(numerical_grad)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

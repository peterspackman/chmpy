"""
Mutable crystal state for geometry optimization and perturbation propagation.

This module provides data structures that separate the immutable reference
structure from the mutable positions, enabling efficient updates during
geometry optimization while maintaining all symmetry relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .orbit import AtomOrbitTable, MoleculeOrbitTable
from .symmetry_operation import decode_symm_int

if TYPE_CHECKING:
    from .crystal import Crystal

LOG = logging.getLogger(__name__)


@dataclass
class MutableCrystalState:
    """
    Mutable state for a crystal structure during optimization.

    This class separates the immutable reference crystal from the mutable
    atomic positions, allowing efficient updates during geometry optimization
    while preserving symmetry relationships.

    Attributes:
        atom_orbit: AtomOrbitTable containing symmetry mappings
        asym_frac_positions: (N_asym, 3) current fractional positions of asym atoms
        uc_frac_positions: (N_uc, 3) current fractional positions of UC atoms
        uc_cart_positions: (N_uc, 3) current Cartesian positions of UC atoms
        _to_cartesian: Function to convert fractional to Cartesian
        _to_fractional: Function to convert Cartesian to fractional
    """

    atom_orbit: AtomOrbitTable
    asym_frac_positions: np.ndarray
    uc_frac_positions: np.ndarray
    uc_cart_positions: np.ndarray
    _to_cartesian: callable = field(default=None, repr=False)
    _to_fractional: callable = field(default=None, repr=False)

    @classmethod
    def from_crystal(cls, crystal: Crystal, tolerance: float = 1e-2) -> MutableCrystalState:
        """
        Create a MutableCrystalState from a Crystal object.

        Args:
            crystal: The crystal structure
            tolerance: Tolerance for site merging

        Returns:
            MutableCrystalState initialized with current positions
        """
        atom_orbit = AtomOrbitTable.from_crystal(crystal, tolerance=tolerance)

        return cls(
            atom_orbit=atom_orbit,
            asym_frac_positions=crystal.asymmetric_unit.positions.copy(),
            uc_frac_positions=atom_orbit.uc_frac_positions.copy(),
            uc_cart_positions=atom_orbit.uc_cart_positions.copy(),
            _to_cartesian=crystal.unit_cell.to_cartesian,
            _to_fractional=crystal.unit_cell.to_fractional,
        )

    def update_asym_positions(self, new_asym_frac: np.ndarray) -> None:
        """
        Update asymmetric unit positions and propagate to unit cell.

        This is the key operation for geometry optimization: given new
        asymmetric unit positions, efficiently recompute all unit cell
        positions using the stored symmetry mappings.

        Args:
            new_asym_frac: (N_asym, 3) new fractional positions
        """
        self.asym_frac_positions = new_asym_frac.copy()

        # Propagate to unit cell
        for uc_idx in range(self.atom_orbit.n_uc):
            asym_idx = self.atom_orbit.uc_to_asym[uc_idx]
            symop_code = self.atom_orbit.uc_symop_codes[uc_idx]

            # Decode and apply symop
            rotation, translation = decode_symm_int(symop_code)
            asym_pos = self.asym_frac_positions[asym_idx]
            new_pos = np.dot(asym_pos, rotation.T) + translation

            # Wrap to [0, 1)
            self.uc_frac_positions[uc_idx] = np.fmod(new_pos + 7.0, 1.0)

        # Update Cartesian positions
        if self._to_cartesian is not None:
            self.uc_cart_positions = self._to_cartesian(self.uc_frac_positions)

    def perturb_asym_position(self, asym_idx: int, delta_frac: np.ndarray) -> None:
        """
        Perturb a single asymmetric unit atom and propagate.

        Args:
            asym_idx: Index of atom to perturb
            delta_frac: (3,) perturbation in fractional coordinates
        """
        new_asym = self.asym_frac_positions.copy()
        new_asym[asym_idx] += delta_frac
        self.update_asym_positions(new_asym)

    def get_uc_gradient_from_asym_gradient(
        self, asym_gradient: np.ndarray
    ) -> np.ndarray:
        """
        Convert gradients on asymmetric unit to gradients on unit cell.

        For geometry optimization, we compute gradients on the independent
        (asymmetric unit) coordinates. This propagates those gradients to
        the full unit cell positions using the symmetry operations.

        Args:
            asym_gradient: (N_asym, 3) gradient on asymmetric unit positions

        Returns:
            (N_uc, 3) gradient on unit cell positions
        """
        uc_gradient = np.zeros((self.atom_orbit.n_uc, 3))

        for uc_idx in range(self.atom_orbit.n_uc):
            asym_idx = self.atom_orbit.uc_to_asym[uc_idx]
            symop_code = self.atom_orbit.uc_symop_codes[uc_idx]

            # Get rotation matrix
            rotation, _ = decode_symm_int(symop_code)

            # Transform gradient: g_uc = R @ g_asym
            uc_gradient[uc_idx] = np.dot(asym_gradient[asym_idx], rotation.T)

        return uc_gradient

    def get_asym_gradient_from_uc_gradient(
        self, uc_gradient: np.ndarray
    ) -> np.ndarray:
        """
        Back-propagate gradients from unit cell to asymmetric unit.

        This sums contributions from all symmetry-equivalent copies to
        get the effective gradient on the asymmetric unit positions.

        Args:
            uc_gradient: (N_uc, 3) gradient on unit cell positions

        Returns:
            (N_asym, 3) gradient on asymmetric unit positions
        """
        asym_gradient = np.zeros((self.atom_orbit.n_asym, 3))

        for uc_idx in range(self.atom_orbit.n_uc):
            asym_idx = self.atom_orbit.uc_to_asym[uc_idx]
            symop_code = self.atom_orbit.uc_symop_codes[uc_idx]

            # Get rotation matrix (transpose for back-propagation)
            rotation, _ = decode_symm_int(symop_code)

            # Back-propagate: g_asym += R^T @ g_uc
            asym_gradient[asym_idx] += np.dot(uc_gradient[uc_idx], rotation)

        return asym_gradient

    def to_dict(self) -> dict:
        """
        Convert current state to dictionary format compatible with
        Crystal.unit_cell_atoms().

        Returns:
            Dictionary with keys: asym_atom, frac_pos, cart_pos, element,
            symop, label, occupation
        """
        return {
            "asym_atom": self.atom_orbit.uc_to_asym.copy(),
            "frac_pos": self.uc_frac_positions.copy(),
            "cart_pos": self.uc_cart_positions.copy(),
            "element": self.atom_orbit.uc_atomic_numbers.copy(),
            "symop": self.atom_orbit.uc_symop_codes.copy(),
            "label": self.atom_orbit.uc_labels.copy(),
            "occupation": self.atom_orbit.uc_occupations.copy(),
        }


@dataclass
class CrystalPerturbationManager:
    """
    Manager for applying and tracking perturbations to crystal structure.

    This class provides a higher-level interface for geometry optimization,
    handling both atomic positions and (optionally) lattice parameters.

    Attributes:
        state: The MutableCrystalState being managed
        original_asym_positions: Original positions for reference
        mol_orbit: Optional MoleculeOrbitTable for molecular crystals
    """

    state: MutableCrystalState
    original_asym_positions: np.ndarray
    mol_orbit: MoleculeOrbitTable | None = None
    _molecule_centroids: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        tolerance: float = 1e-2,
        include_molecules: bool = True,
    ) -> CrystalPerturbationManager:
        """
        Create a CrystalPerturbationManager from a Crystal object.

        Args:
            crystal: The crystal structure
            tolerance: Tolerance for site merging
            include_molecules: Whether to also build MoleculeOrbitTable

        Returns:
            CrystalPerturbationManager ready for use
        """
        state = MutableCrystalState.from_crystal(crystal, tolerance=tolerance)

        mol_orbit = None
        if include_molecules:
            mol_orbit = MoleculeOrbitTable.from_crystal(crystal)

        return cls(
            state=state,
            original_asym_positions=state.asym_frac_positions.copy(),
            mol_orbit=mol_orbit,
        )

    def reset_to_original(self) -> None:
        """Reset positions to original values."""
        self.state.update_asym_positions(self.original_asym_positions)
        self._molecule_centroids = None

    def apply_displacement(self, displacement: np.ndarray) -> None:
        """
        Apply a displacement to the asymmetric unit positions.

        Args:
            displacement: (N_asym, 3) displacement in fractional coordinates
        """
        new_positions = self.state.asym_frac_positions + displacement
        self.state.update_asym_positions(new_positions)
        self._molecule_centroids = None

    def get_molecule_centroids(self) -> np.ndarray | None:
        """
        Get current molecule centroids (recomputed if positions changed).

        Returns:
            (N_uc_mol, 3) array of Cartesian centroids, or None if no mol_orbit
        """
        if self.mol_orbit is None:
            return None

        # Recompute centroids using current positions
        centroids = np.zeros((self.mol_orbit.n_uc_molecules, 3))

        for i, inst in enumerate(self.mol_orbit.unit_cell_instances):
            atom_cart = self.state.uc_cart_positions[list(inst.uc_atom_indices)]
            centroids[i] = np.mean(atom_cart, axis=0)

        return centroids

    def get_displacement_from_original(self) -> np.ndarray:
        """
        Get current displacement from original positions.

        Returns:
            (N_asym, 3) displacement in fractional coordinates
        """
        return self.state.asym_frac_positions - self.original_asym_positions

    @property
    def n_degrees_of_freedom(self) -> int:
        """Number of independent positional degrees of freedom."""
        return self.state.atom_orbit.n_asym * 3

    def get_flat_positions(self) -> np.ndarray:
        """Get asymmetric unit positions as flat array."""
        return self.state.asym_frac_positions.ravel()

    def set_flat_positions(self, flat_positions: np.ndarray) -> None:
        """Set asymmetric unit positions from flat array."""
        new_positions = flat_positions.reshape(-1, 3)
        self.state.update_asym_positions(new_positions)
        self._molecule_centroids = None

    def get_flat_gradient(self, uc_gradient: np.ndarray) -> np.ndarray:
        """
        Convert UC gradient to flat asymmetric unit gradient.

        Args:
            uc_gradient: (N_uc, 3) gradient on unit cell positions

        Returns:
            (N_asym * 3,) flat gradient on asymmetric unit positions
        """
        asym_gradient = self.state.get_asym_gradient_from_uc_gradient(uc_gradient)
        return asym_gradient.ravel()

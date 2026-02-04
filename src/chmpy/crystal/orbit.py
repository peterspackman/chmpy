"""
Orbit tables for crystal structures.

This module provides data structures for tracking the relationship between
asymmetric unit atoms/molecules and their symmetry-generated copies in the
unit cell and beyond.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy.spatial import cKDTree as KDTree

from .symmetry_operation import SymmetryOperation, decode_symm_int

if TYPE_CHECKING:
    from .crystal import Crystal

LOG = logging.getLogger(__name__)


@dataclass
class AtomOrbitTable:
    """
    Maps asymmetric unit atoms to all unit cell sites via symmetry operations.

    This table precomputes the relationship between asymmetric unit atoms and
    their symmetry-generated copies in the unit cell, supporting efficient:
    - Position propagation (update all UC positions from asym unit changes)
    - Site multiplicity tracking (for special positions)
    - Forward/reverse mapping between asym and UC indices

    Attributes:
        n_asym: Number of atoms in asymmetric unit
        n_symops: Number of symmetry operations
        n_uc: Number of unique sites in unit cell (may be < n_asym * n_symops
              due to special positions)

        uc_frac_positions: (N_uc, 3) fractional positions of UC sites
        uc_cart_positions: (N_uc, 3) Cartesian positions of UC sites
        uc_atomic_numbers: (N_uc,) atomic numbers
        uc_labels: (N_uc,) atom labels
        uc_occupations: (N_uc,) site occupancies

        orbit_uc_idx: (N_asym, N_symops) -> uc_site index or -1 if merged
        orbit_symop_code: (N_asym, N_symops) symop integer codes

        uc_to_asym: (N_uc,) -> asym_idx for each UC site
        uc_to_symop: (N_uc,) -> primary symop index for each UC site
        uc_symop_codes: (N_uc,) -> symop integer codes for each UC site
        site_multiplicity: (N_uc,) multiplicity for each site
    """

    n_asym: int
    n_symops: int
    n_uc: int

    # Unit cell site data
    uc_frac_positions: np.ndarray
    uc_cart_positions: np.ndarray
    uc_atomic_numbers: np.ndarray
    uc_labels: np.ndarray
    uc_occupations: np.ndarray

    # Forward mapping: (asym_idx, symop_idx) -> uc_site
    orbit_uc_idx: np.ndarray
    orbit_symop_code: np.ndarray

    # Reverse mapping: uc_site -> (asym_idx, symop_idx)
    uc_to_asym: np.ndarray
    uc_to_symop: np.ndarray
    uc_symop_codes: np.ndarray
    site_multiplicity: np.ndarray

    # Private: store reference to the to_cartesian function
    _to_cartesian: callable = field(default=None, repr=False)

    @classmethod
    def from_crystal(
        cls, crystal: Crystal, tolerance: float = 1e-2
    ) -> AtomOrbitTable:
        """
        Build an AtomOrbitTable from a Crystal object.

        This replicates the logic of Crystal.unit_cell_atoms() but stores
        the results in a structured format suitable for efficient updates.

        Args:
            crystal: The crystal structure to analyze
            tolerance: Minimum separation of sites below which atoms will be
                      merged (for special positions)

        Returns:
            AtomOrbitTable with all mappings computed
        """
        asym = crystal.asymmetric_unit
        sg = crystal.space_group
        uc = crystal.unit_cell

        n_asym = len(asym)
        n_symops = len(sg.symmetry_operations)

        # Get asymmetric unit data
        asym_positions = asym.positions
        asym_nums = asym.atomic_numbers
        asym_labels = asym.labels
        asym_occupations = asym.properties.get("occupation", np.ones(n_asym))

        # Apply all symmetry operations
        # apply_all_symops returns (symop_codes, transformed_positions)
        # where identity is always first
        symop_codes, transformed = sg.apply_all_symops(asym_positions)

        # Wrap to [0, 1) range
        uc_frac = np.fmod(transformed + 7.0, 1.0)

        # Tile auxiliary data
        all_nums = np.tile(asym_nums, n_symops)
        all_labels = np.tile(asym_labels, n_symops)
        all_occupations = np.tile(asym_occupations, n_symops)

        # Track which asymmetric unit atom each position came from
        asym_indices = np.tile(np.arange(n_asym), n_symops)

        # Track which symop index each position came from
        # (symop_codes are ordered: identity first, then rest)
        symop_indices = np.repeat(np.arange(n_symops), n_asym)

        # Build orbit mapping arrays (before merging)
        # orbit_uc_idx[asym_idx, symop_idx] will store the UC site index
        orbit_uc_idx = np.arange(n_asym * n_symops).reshape(n_symops, n_asym).T.copy()
        orbit_symop_code = np.zeros((n_asym, n_symops), dtype=np.int32)

        # Fill symop codes
        # The symop_codes array is ordered: first n_asym have identity,
        # next n_asym have symop 1, etc.
        for symop_idx in range(n_symops):
            start = symop_idx * n_asym
            orbit_symop_code[:, symop_idx] = symop_codes[start : start + n_asym]

        # Find sites to merge using KDTree
        tree = KDTree(uc_frac)
        dist_matrix = tree.sparse_distance_matrix(tree, max_distance=tolerance)

        # Use union-find to group merged sites
        n_sites = len(uc_frac)
        parent = np.arange(n_sites)

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Always merge to the lower index (keeps identity-generated first)
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py

        # Merge nearby sites (including zero distance - same position)
        for (i, j), d in dist_matrix.items():
            if i < j:
                union(i, j)

        # Find unique representatives and build new indices
        representatives = {}
        new_idx = 0
        old_to_new = np.full(n_sites, -1, dtype=np.int32)

        for i in range(n_sites):
            root = find(i)
            if root not in representatives:
                representatives[root] = new_idx
                new_idx += 1
            old_to_new[i] = representatives[root]

        n_uc = new_idx

        # Build final arrays
        uc_frac_positions = np.zeros((n_uc, 3))
        uc_atomic_numbers = np.zeros(n_uc, dtype=np.int32)
        uc_labels = np.empty(n_uc, dtype=object)
        uc_occupations = np.zeros(n_uc)
        uc_to_asym = np.zeros(n_uc, dtype=np.int32)
        uc_to_symop = np.zeros(n_uc, dtype=np.int32)
        uc_symop_codes = np.zeros(n_uc, dtype=np.int32)
        site_multiplicity = np.zeros(n_uc, dtype=np.int32)

        # Track which sites have been assigned
        assigned = np.zeros(n_uc, dtype=bool)

        for old_i in range(n_sites):
            new_i = old_to_new[old_i]
            site_multiplicity[new_i] += 1

            if not assigned[new_i]:
                # First encounter - set the data
                uc_frac_positions[new_i] = uc_frac[old_i]
                uc_atomic_numbers[new_i] = all_nums[old_i]
                uc_labels[new_i] = all_labels[old_i]
                uc_to_asym[new_i] = asym_indices[old_i]
                uc_to_symop[new_i] = symop_indices[old_i]
                uc_symop_codes[new_i] = symop_codes[old_i]
                assigned[new_i] = True

            # Always accumulate occupations
            uc_occupations[new_i] += all_occupations[old_i]

        # Update orbit_uc_idx with new indices
        flat_orbit = orbit_uc_idx.ravel()
        new_orbit = old_to_new[flat_orbit]
        orbit_uc_idx = new_orbit.reshape(n_asym, n_symops)

        # Mark merged sites with -1 (keep only the first occurrence for each UC site)
        seen_uc = {}
        for asym_idx in range(n_asym):
            for symop_idx in range(n_symops):
                uc_idx = orbit_uc_idx[asym_idx, symop_idx]
                key = uc_idx
                if key in seen_uc:
                    orbit_uc_idx[asym_idx, symop_idx] = -1
                else:
                    seen_uc[key] = (asym_idx, symop_idx)

        # Log warning if any occupations > 1
        if np.any(uc_occupations > 1.0 + 1e-6):
            LOG.debug("Some unit cell site occupations are > 1.0")

        # Convert to Cartesian
        uc_cart_positions = uc.to_cartesian(uc_frac_positions)

        return cls(
            n_asym=n_asym,
            n_symops=n_symops,
            n_uc=n_uc,
            uc_frac_positions=uc_frac_positions,
            uc_cart_positions=uc_cart_positions,
            uc_atomic_numbers=uc_atomic_numbers,
            uc_labels=uc_labels,
            uc_occupations=uc_occupations,
            orbit_uc_idx=orbit_uc_idx,
            orbit_symop_code=orbit_symop_code,
            uc_to_asym=uc_to_asym,
            uc_to_symop=uc_to_symop,
            uc_symop_codes=uc_symop_codes,
            site_multiplicity=site_multiplicity,
            _to_cartesian=uc.to_cartesian,
        )

    def propagate_asym_positions(
        self, asym_frac_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update all UC positions from new asymmetric unit positions.

        This is the key operation for geometry optimization: given updated
        asymmetric unit positions, efficiently recompute all unit cell positions.

        Args:
            asym_frac_positions: (N_asym, 3) new fractional positions for
                                 asymmetric unit atoms

        Returns:
            Tuple of (frac_positions, cart_positions) for unit cell
        """
        new_frac = np.zeros_like(self.uc_frac_positions)

        for uc_idx in range(self.n_uc):
            asym_idx = self.uc_to_asym[uc_idx]
            symop_code = self.uc_symop_codes[uc_idx]

            # Decode and apply symop
            rotation, translation = decode_symm_int(symop_code)
            asym_pos = asym_frac_positions[asym_idx]
            new_pos = np.dot(asym_pos, rotation.T) + translation

            # Wrap to [0, 1)
            new_frac[uc_idx] = np.fmod(new_pos + 7.0, 1.0)

        if self._to_cartesian is not None:
            new_cart = self._to_cartesian(new_frac)
        else:
            new_cart = None

        return new_frac, new_cart

    def to_dict(self) -> dict:
        """
        Convert to dictionary format compatible with Crystal.unit_cell_atoms().

        Returns:
            Dictionary with keys: asym_atom, frac_pos, cart_pos, element,
            symop, label, occupation
        """
        return {
            "asym_atom": self.uc_to_asym.copy(),
            "frac_pos": self.uc_frac_positions.copy(),
            "cart_pos": self.uc_cart_positions.copy(),
            "element": self.uc_atomic_numbers.copy(),
            "symop": self.uc_symop_codes.copy(),
            "label": self.uc_labels.copy(),
            "occupation": self.uc_occupations.copy(),
        }

    def get_uc_site_for_asym_atom(self, asym_idx: int, symop_idx: int) -> int:
        """
        Get the UC site index for a given (asym_idx, symop_idx) pair.

        Args:
            asym_idx: Index into asymmetric unit
            symop_idx: Index into symmetry operations list

        Returns:
            UC site index, or -1 if this combination was merged into another site
        """
        return self.orbit_uc_idx[asym_idx, symop_idx]

    def get_asym_atom_for_uc_site(self, uc_idx: int) -> tuple[int, int]:
        """
        Get the (asym_idx, symop_idx) for a given UC site.

        Args:
            uc_idx: Index into unit cell sites

        Returns:
            Tuple of (asym_idx, symop_idx)
        """
        return self.uc_to_asym[uc_idx], self.uc_to_symop[uc_idx]

    def is_special_position(self, uc_idx: int) -> bool:
        """
        Check if a UC site is on a special position.

        A site is on a special position if its multiplicity is greater than 1,
        meaning multiple (asym, symop) pairs map to the same position.

        Args:
            uc_idx: Index into unit cell sites

        Returns:
            True if site is on a special position
        """
        return self.site_multiplicity[uc_idx] > 1


@dataclass(frozen=True)
class MoleculeInstance:
    """
    A single molecule instance in the unit cell.

    Attributes:
        asym_mol_idx: Index of the unique molecule this instance belongs to
        uc_atom_indices: Tuple of UC atom indices that make up this molecule
        centroid_frac: (3,) fractional coordinates of molecule centroid
        centroid_cart: (3,) Cartesian coordinates of molecule centroid
        generator_symops: Tuple of symop codes that generated each atom
    """

    asym_mol_idx: int
    uc_atom_indices: tuple[int, ...]
    centroid_frac: np.ndarray
    centroid_cart: np.ndarray
    generator_symops: tuple[int, ...]

    def __hash__(self):
        return hash((self.asym_mol_idx, self.uc_atom_indices))

    def __eq__(self, other):
        if not isinstance(other, MoleculeInstance):
            return False
        return (
            self.asym_mol_idx == other.asym_mol_idx
            and self.uc_atom_indices == other.uc_atom_indices
        )


@dataclass
class MoleculeOrbitTable:
    """
    Maps unique molecules in the asymmetric unit to all their symmetry copies.

    This table tracks how molecules in the asymmetric unit are replicated
    throughout the unit cell by space group symmetry.

    Attributes:
        n_unique_molecules: Number of symmetry-unique molecules
        n_uc_molecules: Total number of molecules in unit cell
        unique_mol_atom_indices: List of tuples, each containing the UC atom
                                indices for one unique molecule
        unit_cell_instances: List of all MoleculeInstance objects
        instance_centroids_frac: (N_uc_mol, 3) fractional centroids
        instance_centroids_cart: (N_uc_mol, 3) Cartesian centroids
    """

    n_unique_molecules: int
    n_uc_molecules: int

    unique_mol_atom_indices: list[tuple[int, ...]]
    unique_mol_asym_indices: list[tuple[int, ...]]
    unit_cell_instances: list[MoleculeInstance]

    instance_centroids_frac: np.ndarray
    instance_centroids_cart: np.ndarray
    instance_to_unique: np.ndarray

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        bond_tolerance: float = 0.4,
    ) -> MoleculeOrbitTable:
        """
        Build a MoleculeOrbitTable from a Crystal object.

        Uses connectivity analysis to identify molecules, then tracks
        which molecules are symmetry-equivalent.

        Args:
            crystal: The crystal structure to analyze
            bond_tolerance: Tolerance for bond detection

        Returns:
            MoleculeOrbitTable with all molecule instances
        """
        # Get unit cell connectivity and atoms
        uc_graph, edge_cells = crystal.unit_cell_connectivity(tolerance=bond_tolerance)
        uc_dict = crystal.unit_cell_atoms()
        uc_frac = uc_dict["frac_pos"]
        uc_cart = uc_dict["cart_pos"]
        uc_asym = uc_dict["asym_atom"]
        uc_symop = uc_dict["symop"]

        # Find connected components (molecules)
        n_uc_mols, mol_labels = csgraph.connected_components(
            csgraph=uc_graph, directed=False, return_labels=True
        )

        # Build molecule instances with proper centroid calculation
        instances = []
        for mol_idx in range(n_uc_mols):
            atom_indices = tuple(np.where(mol_labels == mol_idx)[0])

            # Compute centroid properly handling PBC
            # Use breadth-first to build consistent positions
            nodes = np.array(atom_indices)
            root = nodes[0]
            n_uc = len(uc_frac)
            shifts = np.zeros((n_uc, 3))

            ordered, pred = csgraph.breadth_first_order(
                csgraph=uc_graph, i_start=root, directed=False
            )

            for j in ordered[1:]:
                i = pred[j]
                if j < i:
                    shifts[j, :] = shifts[i, :] - edge_cells.get((j, i), (0, 0, 0))
                else:
                    shifts[j, :] = shifts[i, :] + edge_cells.get((i, j), (0, 0, 0))

            mol_frac = (uc_frac + shifts)[nodes]
            mol_cart = crystal.to_cartesian(mol_frac)

            centroid_frac = np.mean(mol_frac, axis=0)
            centroid_cart = np.mean(mol_cart, axis=0)

            # Wrap centroid to [0, 1)
            centroid_frac_wrapped = np.fmod(centroid_frac + 7.0, 1.0)
            centroid_cart_wrapped = crystal.to_cartesian(centroid_frac_wrapped)

            generator_symops = tuple(uc_symop[list(atom_indices)])

            instances.append(
                MoleculeInstance(
                    asym_mol_idx=-1,  # Will be assigned later
                    uc_atom_indices=atom_indices,
                    centroid_frac=centroid_frac_wrapped,
                    centroid_cart=centroid_cart_wrapped,
                    generator_symops=generator_symops,
                )
            )

        # Identify unique molecules based on asymmetric unit atoms
        # (same algorithm as symmetry_unique_molecules)
        asym_atoms_covered = np.zeros(len(crystal.asymmetric_unit), dtype=bool)
        unique_molecules = []
        unique_mol_atom_indices = []
        unique_mol_asym_indices = []

        # Sort instances by percentage of identity symop (16484)
        def identity_fraction(inst):
            return sum(1 for s in inst.generator_symops if s == 16484) / len(
                inst.generator_symops
            )

        sorted_instances = sorted(instances, key=identity_fraction, reverse=True)

        for inst in sorted_instances:
            asym_atoms_in_mol = np.unique(uc_asym[list(inst.uc_atom_indices)])
            if np.all(asym_atoms_covered[asym_atoms_in_mol]):
                continue
            asym_atoms_covered[asym_atoms_in_mol] = True
            unique_molecules.append(inst)
            unique_mol_atom_indices.append(inst.uc_atom_indices)
            unique_mol_asym_indices.append(tuple(asym_atoms_in_mol))
            if np.all(asym_atoms_covered):
                break

        n_unique = len(unique_molecules)

        # Assign asym_mol_idx to all instances
        updated_instances = []
        instance_to_unique = np.zeros(len(instances), dtype=np.int32)

        for i, inst in enumerate(instances):
            inst_asym_atoms = tuple(sorted(np.unique(uc_asym[list(inst.uc_atom_indices)])))

            # Find matching unique molecule
            asym_mol_idx = -1
            for j, unique_asym in enumerate(unique_mol_asym_indices):
                if inst_asym_atoms == tuple(sorted(unique_asym)):
                    asym_mol_idx = j
                    break

            if asym_mol_idx == -1:
                LOG.warning(
                    f"Could not find matching unique molecule for instance {i}"
                )
                asym_mol_idx = 0  # Fallback

            instance_to_unique[i] = asym_mol_idx

            updated_instances.append(
                MoleculeInstance(
                    asym_mol_idx=asym_mol_idx,
                    uc_atom_indices=inst.uc_atom_indices,
                    centroid_frac=inst.centroid_frac,
                    centroid_cart=inst.centroid_cart,
                    generator_symops=inst.generator_symops,
                )
            )

        # Build centroid arrays
        instance_centroids_frac = np.array([inst.centroid_frac for inst in updated_instances])
        instance_centroids_cart = np.array([inst.centroid_cart for inst in updated_instances])

        return cls(
            n_unique_molecules=n_unique,
            n_uc_molecules=n_uc_mols,
            unique_mol_atom_indices=unique_mol_atom_indices,
            unique_mol_asym_indices=unique_mol_asym_indices,
            unit_cell_instances=updated_instances,
            instance_centroids_frac=instance_centroids_frac,
            instance_centroids_cart=instance_centroids_cart,
            instance_to_unique=instance_to_unique,
        )

    def get_instances_of_molecule(self, unique_mol_idx: int) -> list[MoleculeInstance]:
        """
        Get all instances of a specific unique molecule.

        Args:
            unique_mol_idx: Index of unique molecule (0 to n_unique_molecules-1)

        Returns:
            List of MoleculeInstance objects for this unique molecule
        """
        return [
            inst
            for inst in self.unit_cell_instances
            if inst.asym_mol_idx == unique_mol_idx
        ]

    def get_instance_count(self, unique_mol_idx: int) -> int:
        """Get how many instances of a unique molecule exist in the unit cell."""
        return sum(
            1 for inst in self.unit_cell_instances if inst.asym_mol_idx == unique_mol_idx
        )

    def get_unique_molecule_atoms(self, unique_mol_idx: int) -> tuple[int, ...]:
        """Get the UC atom indices for a unique molecule."""
        return self.unique_mol_atom_indices[unique_mol_idx]

    def get_unique_molecule_asym_atoms(self, unique_mol_idx: int) -> tuple[int, ...]:
        """Get the asymmetric unit atom indices for a unique molecule."""
        return self.unique_mol_asym_indices[unique_mol_idx]

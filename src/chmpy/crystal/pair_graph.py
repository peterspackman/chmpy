"""
Atom pair graph for efficient pairwise force calculations in crystals.

This module provides AtomPairGraph for computing pairwise interactions
(energy, forces, stress) using only symmetry-unique atom pairs. This
achieves O(N²/|G|) scaling instead of O(N²) for pairwise potentials.

Key features:
- Build pair list from crystal with distance cutoff
- Compute energies, forces, and stress tensors
- Automatic symmetry-based reduction
- Force symmetrization via orbit decomposition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from scipy.spatial import cKDTree as KDTree

from .symmetric_graph import (
    AlgebraicEdge,
    AlgebraicVertexRef,
    apply_symop_to_edge,
    canonical_edge_representative,
    compute_edge_orbit,
    compute_edge_orbit_size,
    normalize_edge,
)
from .atomic_graph import AtomicCosetTable, AlgebraicAtomRef

if TYPE_CHECKING:
    from .crystal import Crystal
    from .space_group_table import SpaceGroupTable

LOG = logging.getLogger(__name__)


# Type alias for clarity
AlgebraicAtomPairIndex = AlgebraicEdge


@dataclass
class AtomPairGraph:
    """
    Atom pairs within a cutoff distance for force calculations.

    This class enables efficient pairwise potential evaluation by
    storing only symmetry-unique pairs. Forces and stresses are
    computed by summing over unique pairs with appropriate multiplicities.

    Attributes:
        crystal: The crystal structure
        cutoff: Distance cutoff for pair interactions
        sg_table: SpaceGroupTable for the crystal's space group
        coset_table: AtomicCosetTable for atom site symmetry
        unique_pairs: List of unique atom pairs
        pair_vectors: (N, 3) separation vectors for each unique pair
        pair_distances: (N,) distances for each unique pair
        pair_multiplicities: (N,) orbit sizes for each unique pair
    """

    crystal: Crystal
    cutoff: float
    sg_table: SpaceGroupTable
    coset_table: AtomicCosetTable
    unique_pairs: list[AlgebraicAtomPairIndex]
    pair_vectors: np.ndarray  # (N, 3) r_ij vectors
    pair_distances: np.ndarray  # (N,) distances
    pair_multiplicities: np.ndarray  # (N,) orbit sizes
    _uc_atom_positions: np.ndarray  # Cached UC atom positions (Cartesian)
    _uc_atom_asym: np.ndarray  # UC atom -> asym mapping
    _uc_atom_symop: np.ndarray  # UC atom -> symop mapping

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        cutoff: float,
    ) -> AtomPairGraph:
        """
        Build a pair list from crystal structure within cutoff.

        Args:
            crystal: The crystal structure
            cutoff: Maximum distance for pair interactions

        Returns:
            AtomPairGraph with all unique pairs within cutoff
        """
        from .space_group_table import SpaceGroupTable

        sg_table = SpaceGroupTable.from_space_group(crystal.space_group)
        coset_table = AtomicCosetTable.from_crystal(crystal, sg_table)

        # Get unit cell atoms
        uc_dict = crystal.unit_cell_atoms()
        uc_frac = uc_dict["frac_pos"]
        uc_cart = uc_dict["cart_pos"]
        uc_asym = uc_dict["asym_atom"]
        uc_symop_codes = uc_dict["symop"]  # These are integer codes, not indices
        n_uc_atoms = len(uc_frac)

        # Convert symop integer codes to indices
        symops = crystal.space_group.symmetry_operations
        symop_code_to_idx = {s.integer_code: i for i, s in enumerate(symops)}
        uc_symop_idx = np.array(
            [symop_code_to_idx.get(code, 0) for code in uc_symop_codes],
            dtype=np.int32,
        )

        # Determine how many cells to search
        cell_lengths = np.array(crystal.unit_cell.lengths)
        n_cells = np.ceil(cutoff / cell_lengths).astype(int) + 1

        # Generate all cell translations
        h_range = np.arange(-n_cells[0], n_cells[0] + 1)
        k_range = np.arange(-n_cells[1], n_cells[1] + 1)
        l_range = np.arange(-n_cells[2], n_cells[2] + 1)

        # Build extended structure
        all_positions = []
        all_cells = []
        all_atom_indices = []

        for h in h_range:
            for k in k_range:
                for l in l_range:
                    cell_offset = np.array([h, k, l])
                    shifted_frac = uc_frac + cell_offset
                    shifted_cart = crystal.unit_cell.to_cartesian(shifted_frac)
                    all_positions.append(shifted_cart)
                    all_cells.extend([(h, k, l)] * n_uc_atoms)
                    all_atom_indices.extend(range(n_uc_atoms))

        all_positions = np.vstack(all_positions)
        all_atom_indices = np.array(all_atom_indices)

        # Build KDTree for neighbor search
        tree = KDTree(all_positions)

        # Find unique pairs
        unique_pairs = []
        pair_vectors = []
        pair_distances = []
        seen_pairs = set()

        # For each atom in the reference cell
        for i in range(n_uc_atoms):
            ref_pos = uc_cart[i]
            neighbors = tree.query_ball_point(ref_pos, cutoff)

            asym_i = uc_asym[i]
            symop_i = uc_symop_idx[i]

            atom_a = AlgebraicAtomRef(
                asym_idx=asym_i,
                symop_idx=symop_i,
                cell=(0, 0, 0),
            )

            for j_extended in neighbors:
                j = all_atom_indices[j_extended]
                cell_j = all_cells[j_extended]

                # Skip self-interaction
                if i == j and cell_j == (0, 0, 0):
                    continue

                # Compute distance
                r_ij = all_positions[j_extended] - ref_pos
                dist = np.linalg.norm(r_ij)

                if dist < 1e-10:
                    continue

                asym_j = uc_asym[j]
                symop_j = uc_symop_idx[j]

                atom_b = AlgebraicAtomRef(
                    asym_idx=asym_j,
                    symop_idx=symop_j,
                    cell=cell_j,
                )

                # Create pair and find canonical representative
                pair = AlgebraicEdge.create(atom_a, atom_b)
                canonical = canonical_edge_representative(pair, sg_table, coset_table)
                key = canonical._key()

                if key not in seen_pairs:
                    seen_pairs.add(key)
                    unique_pairs.append(canonical)

                    # Compute the separation vector for the canonical pair
                    # We need to recalculate based on canonical form
                    # For now, store the vector we computed
                    pair_vectors.append(r_ij)
                    pair_distances.append(dist)

        pair_vectors = np.array(pair_vectors) if pair_vectors else np.zeros((0, 3))
        pair_distances = np.array(pair_distances) if pair_distances else np.zeros(0)

        # Compute multiplicities
        pair_multiplicities = np.array(
            [
                compute_edge_orbit_size(pair, sg_table, coset_table)
                for pair in unique_pairs
            ]
        )

        return cls(
            crystal=crystal,
            cutoff=cutoff,
            sg_table=sg_table,
            coset_table=coset_table,
            unique_pairs=unique_pairs,
            pair_vectors=pair_vectors,
            pair_distances=pair_distances,
            pair_multiplicities=pair_multiplicities,
            _uc_atom_positions=uc_cart,
            _uc_atom_asym=uc_asym,
            _uc_atom_symop=uc_symop_idx,
        )

    def n_unique_pairs(self) -> int:
        """Return the number of unique pairs."""
        return len(self.unique_pairs)

    def n_total_pairs(self) -> int:
        """Return the total number of pairs (sum of multiplicities)."""
        return int(np.sum(self.pair_multiplicities))

    def compute_pairwise_energy(
        self,
        pair_potential: Callable[[float], float],
    ) -> float:
        """
        Compute total pairwise energy using unique pairs with multiplicities.

        For scalar quantities like energy, orbit-based reduction works correctly:
        all symmetry-equivalent pairs have the same distance, so we can compute
        E = sum_unique (multiplicity * V(r)).

        The result is divided by 2 to avoid double-counting (each pair contributes
        to both atoms).

        Args:
            pair_potential: Function V(r) returning energy for distance r

        Returns:
            Total energy per unit cell
        """
        total = 0.0
        n_uc = len(self._uc_atom_positions)

        for i, pair in enumerate(self.unique_pairs):
            dist = self.pair_distances[i]
            mult = self.pair_multiplicities[i]
            energy = pair_potential(dist)

            # Each pair contributes to both atoms, so divide by 2
            # Multiplicity counts all symmetry images
            total += mult * energy / 2.0

        return total

    def compute_pairwise_forces(
        self,
        force_function: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Compute forces on all UC atoms using unique pairs.

        The force function takes a separation vector r_ij and returns
        the force on atom i due to atom j (pointing from j to i).

        Forces are symmetrized by distributing contributions from each
        unique pair to all its symmetry images.

        Args:
            force_function: Function f(r_ij) -> force vector on atom i

        Returns:
            (n_uc_atoms, 3) array of forces on each UC atom
        """
        n_uc = len(self._uc_atom_positions)
        forces = np.zeros((n_uc, 3))

        for i, pair in enumerate(self.unique_pairs):
            r_ij = self.pair_vectors[i]
            f_ij = force_function(r_ij)  # Force on atom i from j

            # Get all symmetry images of this pair
            orbit = compute_edge_orbit(pair, self.sg_table, self.coset_table)

            for image in orbit:
                # Get the UC atom indices for this image
                # This requires mapping the algebraic refs back to UC indices
                idx_i, idx_j = self._get_uc_indices_for_pair(image)

                if idx_i is None or idx_j is None:
                    continue

                # Get the rotation for the symop that generated this image
                # We need to transform the force vector
                # For now, use a simplified approach: assume central force
                # (force along r_ij direction)
                r_image = self._get_separation_vector(image)
                if r_image is not None:
                    r_norm = np.linalg.norm(r_image)
                    if r_norm > 1e-10:
                        # Central force: f = f_mag * r_hat
                        f_mag = np.linalg.norm(f_ij)
                        f_image = f_mag * r_image / r_norm * np.sign(np.dot(f_ij, r_ij))

                        forces[idx_i] += f_image
                        forces[idx_j] -= f_image

        return forces

    def compute_pairwise_forces_simple(
        self,
        force_magnitude: Callable[[float], float],
    ) -> np.ndarray:
        """
        Compute forces assuming central force potential.

        For central potentials where F = -dV/dr * r_hat, this provides
        a simpler interface.

        This uses direct enumeration over all UC atom pairs within cutoff,
        rather than orbit-based calculation, to correctly handle atoms
        with different site symmetries.

        Args:
            force_magnitude: Function returning |F| = -dV/dr for distance r
                (positive for repulsive, negative for attractive)

        Returns:
            (n_uc_atoms, 3) array of forces on each UC atom
        """
        n_uc = len(self._uc_atom_positions)
        forces = np.zeros((n_uc, 3))

        uc_frac = self.crystal.unit_cell_atoms()["frac_pos"]

        # Determine cell search range
        cell_lengths = np.array(self.crystal.unit_cell.lengths)
        n_cells = np.ceil(self.cutoff / cell_lengths).astype(int) + 1

        # For each UC atom i, find all neighbors j within cutoff
        for i in range(n_uc):
            pos_i = self._uc_atom_positions[i]

            for j in range(n_uc):
                # Search over periodic images
                for h in range(-n_cells[0], n_cells[0] + 1):
                    for k in range(-n_cells[1], n_cells[1] + 1):
                        for l in range(-n_cells[2], n_cells[2] + 1):
                            # Skip self-interaction at origin
                            if i == j and (h, k, l) == (0, 0, 0):
                                continue

                            # Position of j in shifted cell
                            frac_j = uc_frac[j] + np.array([h, k, l])
                            pos_j = self.crystal.unit_cell.to_cartesian(frac_j)

                            r_ij = pos_j - pos_i
                            dist = np.linalg.norm(r_ij)

                            if dist < 1e-10 or dist > self.cutoff:
                                continue

                            # Compute force
                            f_mag = force_magnitude(dist)
                            f_ij = f_mag * r_ij / dist

                            # Add force to atom i (reaction on j is handled
                            # when we iterate with j as the central atom)
                            forces[i] += f_ij

        return forces

    def compute_pairwise_stress(
        self,
        force_magnitude: Callable[[float], float],
    ) -> np.ndarray:
        """
        Compute stress tensor from pairwise interactions.

        The stress tensor is computed using the virial formula:
            sigma_ab = (1/V) sum_pairs f_a * r_b

        Args:
            force_magnitude: Function returning |F| = -dV/dr for distance r

        Returns:
            (3, 3) stress tensor in energy/volume units
        """
        volume = self.crystal.unit_cell.volume()
        stress = np.zeros((3, 3))

        for i, pair in enumerate(self.unique_pairs):
            r_ij = self.pair_vectors[i]
            dist = self.pair_distances[i]
            f_mag = force_magnitude(dist)
            mult = self.pair_multiplicities[i]

            # Force vector
            f_ij = f_mag * r_ij / dist

            # Contribution to stress tensor: f_a * r_b
            # Factor of 1/2 because we count each pair once but it contributes
            # to both atoms
            stress += mult * np.outer(f_ij, r_ij) / 2.0

        return stress / volume

    def _get_uc_indices_for_pair(
        self, pair: AlgebraicAtomPairIndex
    ) -> tuple[int | None, int | None]:
        """
        Get UC atom indices for a pair.

        In a periodic system, the UC atom is identified by (asym_idx, symop_idx).
        The cell offset tells us which periodic image is involved, but forces
        still apply to the corresponding UC atom.

        For atoms on special positions, multiple symop indices can map to the
        same UC atom. We use the coset table to find the actual UC atom.

        Args:
            pair: The atom pair

        Returns:
            Tuple of (idx_i, idx_j) - UC atom indices for src and dst
        """
        n_uc = len(self._uc_atom_positions)

        idx_i = None
        idx_j = None

        # For each vertex, find the UC atom it corresponds to.
        # The coset table's symop_to_atom mapping tells us which UC atom
        # a given (symop_idx, asym_idx) generates.
        src_asym = pair.src.asym_idx
        src_symop = pair.src.symop_idx
        dst_asym = pair.dst.asym_idx
        dst_symop = pair.dst.symop_idx

        # Use coset table if available
        if self.coset_table is not None:
            # symop_to_atom[symop_idx, asym_idx] = uc_atom_idx
            if src_symop < self.coset_table.symop_to_atom.shape[0]:
                if src_asym < self.coset_table.symop_to_atom.shape[1]:
                    idx_i = self.coset_table.symop_to_atom[src_symop, src_asym]
                    if idx_i < 0:
                        idx_i = None

            if dst_symop < self.coset_table.symop_to_atom.shape[0]:
                if dst_asym < self.coset_table.symop_to_atom.shape[1]:
                    idx_j = self.coset_table.symop_to_atom[dst_symop, dst_asym]
                    if idx_j < 0:
                        idx_j = None
        else:
            # Fall back to direct lookup
            for k in range(n_uc):
                if (
                    self._uc_atom_asym[k] == src_asym
                    and self._uc_atom_symop[k] == src_symop
                ):
                    idx_i = k
                    break

            for k in range(n_uc):
                if (
                    self._uc_atom_asym[k] == dst_asym
                    and self._uc_atom_symop[k] == dst_symop
                ):
                    idx_j = k
                    break

        return idx_i, idx_j

    def _get_separation_vector(
        self, pair: AlgebraicAtomPairIndex
    ) -> np.ndarray | None:
        """
        Get the separation vector r_ij = r_j - r_i for a pair.

        Args:
            pair: The atom pair

        Returns:
            Separation vector or None if cannot be computed
        """
        uc_frac = self.crystal.unit_cell_atoms()["frac_pos"]

        # Get UC atom indices using coset table
        idx_i, idx_j = self._get_uc_indices_for_pair(pair)

        if idx_i is None or idx_j is None:
            return None

        # Get positions with cell offsets
        frac_i = uc_frac[idx_i] + np.array(pair.src.cell)
        frac_j = uc_frac[idx_j] + np.array(pair.dst.cell)

        pos_i = self.crystal.unit_cell.to_cartesian(frac_i)
        pos_j = self.crystal.unit_cell.to_cartesian(frac_j)

        return pos_j - pos_i

    def pairs_involving_atom(
        self, asym_idx: int
    ) -> list[tuple[AlgebraicAtomPairIndex, int, float]]:
        """
        Get unique pairs involving a specific atom type.

        Args:
            asym_idx: Asymmetric unit atom index

        Returns:
            List of (pair, multiplicity, distance) tuples
        """
        result = []
        for i, pair in enumerate(self.unique_pairs):
            if pair.src.asym_idx == asym_idx or pair.dst.asym_idx == asym_idx:
                result.append(
                    (pair, self.pair_multiplicities[i], self.pair_distances[i])
                )
        return result

    def unique_pairs_by_type(
        self, elem_a: int, elem_b: int
    ) -> list[tuple[AlgebraicAtomPairIndex, int, float]]:
        """
        Get unique pairs between specific element types.

        Args:
            elem_a: Atomic number of first element
            elem_b: Atomic number of second element

        Returns:
            List of (pair, multiplicity, distance) tuples
        """
        atoms = self.crystal.site_atoms
        result = []

        for i, pair in enumerate(self.unique_pairs):
            type_a = atoms[pair.src.asym_idx]
            type_b = atoms[pair.dst.asym_idx]

            if (type_a == elem_a and type_b == elem_b) or (
                type_a == elem_b and type_b == elem_a
            ):
                result.append(
                    (pair, self.pair_multiplicities[i], self.pair_distances[i])
                )

        return result

    def get_uc_atom_info(self) -> dict:
        """
        Get information about unit cell atoms (for interpreting force arrays).

        The returned arrays are indexed the same way as force arrays from
        compute_pairwise_forces().

        Returns:
            Dictionary with:
            - "element": (n_uc,) atomic numbers
            - "frac_pos": (n_uc, 3) fractional positions
            - "cart_pos": (n_uc, 3) Cartesian positions
            - "asym_atom": (n_uc,) asymmetric unit atom index
            - "label": (n_uc,) atom labels from CIF
        """
        return self.crystal.unit_cell_atoms()

    def speedup_factor(self) -> float:
        """
        Compute the speedup factor from using symmetry.

        Returns:
            Ratio of brute-force pairs to unique pairs
        """
        n_uc = len(self._uc_atom_positions)
        brute_force = n_uc * (n_uc - 1) // 2  # Approximate for large cutoff
        unique = self.n_unique_pairs()
        return brute_force / unique if unique > 0 else 1.0

    def __repr__(self) -> str:
        return (
            f"AtomPairGraph(cutoff={self.cutoff:.2f}, "
            f"n_unique={self.n_unique_pairs()}, "
            f"n_total={self.n_total_pairs()}, "
            f"speedup={self.speedup_factor():.1f}x)"
        )

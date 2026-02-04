"""
Neighbor list implementation for crystal structures.

This module provides efficient neighbor list construction for periodic
crystal structures using cell-linked lists for O(n) scaling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree as KDTree

if TYPE_CHECKING:
    from .crystal import Crystal
    from .orbit import AtomOrbitTable

LOG = logging.getLogger(__name__)


@dataclass
class CrystalNeighborList:
    """
    Neighbor list for a crystal structure in CSR (Compressed Sparse Row) format.

    This data structure stores all atom pairs within a cutoff distance,
    supporting periodic boundary conditions. The CSR format allows efficient
    iteration over neighbors of each atom.

    Attributes:
        n_atoms: Number of atoms in the reference structure (unit cell)
        cutoff: Cutoff distance used for neighbor search

        neighbor_indices: Flat array of neighbor atom indices (in unit cell)
        neighbor_distances: Flat array of distances to neighbors
        neighbor_offsets: Offsets into neighbor arrays; neighbors of atom i
                         are at neighbor_indices[offsets[i]:offsets[i+1]]
        neighbor_cells: (N_pairs, 3) array of periodic image offsets (h, k, l)
        n_pairs: Total number of neighbor pairs
    """

    n_atoms: int
    cutoff: float

    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    neighbor_offsets: np.ndarray
    neighbor_cells: np.ndarray
    n_pairs: int

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        cutoff: float = 6.0,
    ) -> CrystalNeighborList:
        """
        Build a neighbor list from a Crystal object.

        Uses KDTree for efficient neighbor search across periodic images.

        Args:
            crystal: The crystal structure
            cutoff: Maximum distance to consider atoms as neighbors

        Returns:
            CrystalNeighborList with all pairs within cutoff
        """
        uc_atoms = crystal.unit_cell_atoms()
        uc_cart = uc_atoms["cart_pos"]
        uc_frac = uc_atoms["frac_pos"]
        n_atoms = len(uc_cart)

        return cls._build_from_positions(
            uc_cart, uc_frac, crystal.unit_cell, cutoff
        )

    @classmethod
    def from_orbit_table(
        cls,
        orbit: AtomOrbitTable,
        unit_cell,
        cutoff: float = 6.0,
    ) -> CrystalNeighborList:
        """
        Build a neighbor list from an AtomOrbitTable.

        Args:
            orbit: The AtomOrbitTable containing unit cell positions
            unit_cell: UnitCell object for coordinate transformations
            cutoff: Maximum distance to consider atoms as neighbors

        Returns:
            CrystalNeighborList with all pairs within cutoff
        """
        return cls._build_from_positions(
            orbit.uc_cart_positions,
            orbit.uc_frac_positions,
            unit_cell,
            cutoff,
        )

    @classmethod
    def _build_from_positions(
        cls,
        cart_positions: np.ndarray,
        frac_positions: np.ndarray,
        unit_cell,
        cutoff: float,
    ) -> CrystalNeighborList:
        """
        Internal method to build neighbor list from positions.

        Args:
            cart_positions: (N, 3) Cartesian positions
            frac_positions: (N, 3) fractional positions
            unit_cell: UnitCell object for coordinate transformations
            cutoff: Maximum distance cutoff

        Returns:
            CrystalNeighborList
        """
        n_atoms = len(cart_positions)

        # Determine how many cells to search in each direction
        # Based on cutoff and cell lengths
        cell_lengths = np.array(unit_cell.lengths)
        n_cells = np.ceil(cutoff / cell_lengths).astype(int)

        # Generate all cell translations to check
        h_range = np.arange(-n_cells[0], n_cells[0] + 1)
        k_range = np.arange(-n_cells[1], n_cells[1] + 1)
        l_range = np.arange(-n_cells[2], n_cells[2] + 1)

        # Build extended structure with all periodic images
        all_positions = []
        all_cells = []
        all_orig_indices = []

        for h in h_range:
            for k in k_range:
                for l in l_range:
                    cell_offset = np.array([h, k, l])
                    shifted_frac = frac_positions + cell_offset
                    shifted_cart = unit_cell.to_cartesian(shifted_frac)
                    all_positions.append(shifted_cart)
                    all_cells.extend([cell_offset] * n_atoms)
                    all_orig_indices.extend(range(n_atoms))

        all_positions = np.vstack(all_positions)
        all_cells = np.array(all_cells)
        all_orig_indices = np.array(all_orig_indices)

        # Build KDTree for the extended structure
        tree = KDTree(all_positions)

        # Query neighbors for each atom in the reference unit cell
        # We only need to query the first n_atoms (the reference cell)
        ref_positions = cart_positions

        # Find all pairs within cutoff
        neighbor_lists = []
        for i in range(n_atoms):
            # Query all neighbors of atom i
            indices = tree.query_ball_point(ref_positions[i], cutoff)

            # Filter and store valid neighbors
            for j in indices:
                orig_j = all_orig_indices[j]
                cell_j = all_cells[j]
                dist = np.linalg.norm(all_positions[j] - ref_positions[i])

                # Skip self-interaction in reference cell
                if i == orig_j and np.all(cell_j == 0):
                    continue

                # Skip zero-distance pairs (shouldn't happen, but safety check)
                if dist < 1e-10:
                    continue

                neighbor_lists.append((i, orig_j, dist, tuple(cell_j)))

        # Convert to CSR format
        if not neighbor_lists:
            return cls(
                n_atoms=n_atoms,
                cutoff=cutoff,
                neighbor_indices=np.array([], dtype=np.int32),
                neighbor_distances=np.array([], dtype=np.float64),
                neighbor_offsets=np.zeros(n_atoms + 1, dtype=np.int32),
                neighbor_cells=np.array([], dtype=np.int32).reshape(0, 3),
                n_pairs=0,
            )

        # Sort by source atom index for CSR format
        neighbor_lists.sort(key=lambda x: x[0])

        # Build CSR arrays
        n_pairs = len(neighbor_lists)
        neighbor_indices = np.zeros(n_pairs, dtype=np.int32)
        neighbor_distances = np.zeros(n_pairs, dtype=np.float64)
        neighbor_cells = np.zeros((n_pairs, 3), dtype=np.int32)
        neighbor_offsets = np.zeros(n_atoms + 1, dtype=np.int32)

        current_atom = 0
        for idx, (i, j, dist, cell) in enumerate(neighbor_lists):
            # Update offset pointers
            while current_atom < i:
                current_atom += 1
                neighbor_offsets[current_atom] = idx

            neighbor_indices[idx] = j
            neighbor_distances[idx] = dist
            neighbor_cells[idx] = cell

        # Fill remaining offsets
        for i in range(current_atom + 1, n_atoms + 1):
            neighbor_offsets[i] = n_pairs

        return cls(
            n_atoms=n_atoms,
            cutoff=cutoff,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            neighbor_offsets=neighbor_offsets,
            neighbor_cells=neighbor_cells,
            n_pairs=n_pairs,
        )

    def get_neighbors(self, atom_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all neighbors of a specific atom.

        Args:
            atom_idx: Index of the atom in the unit cell

        Returns:
            Tuple of (neighbor_indices, distances, cell_offsets)
            - neighbor_indices: (M,) array of neighbor atom indices
            - distances: (M,) array of distances
            - cell_offsets: (M, 3) array of periodic cell offsets
        """
        start = self.neighbor_offsets[atom_idx]
        end = self.neighbor_offsets[atom_idx + 1]

        return (
            self.neighbor_indices[start:end].copy(),
            self.neighbor_distances[start:end].copy(),
            self.neighbor_cells[start:end].copy(),
        )

    def get_n_neighbors(self, atom_idx: int) -> int:
        """Get the number of neighbors for a specific atom."""
        return self.neighbor_offsets[atom_idx + 1] - self.neighbor_offsets[atom_idx]

    def iter_neighbors(self, atom_idx: int):
        """
        Iterate over neighbors of a specific atom.

        Yields:
            Tuple of (neighbor_idx, distance, cell_offset)
        """
        start = self.neighbor_offsets[atom_idx]
        end = self.neighbor_offsets[atom_idx + 1]

        for i in range(start, end):
            yield (
                self.neighbor_indices[i],
                self.neighbor_distances[i],
                self.neighbor_cells[i],
            )

    def iter_all_pairs(self):
        """
        Iterate over all neighbor pairs.

        Yields:
            Tuple of (atom_i, atom_j, distance, cell_offset)
        """
        for i in range(self.n_atoms):
            for j_idx, dist, cell in self.iter_neighbors(i):
                yield (i, j_idx, dist, cell)

    def get_coordination_numbers(self, max_dist: float | None = None) -> np.ndarray:
        """
        Get coordination numbers for all atoms.

        Args:
            max_dist: Optional distance cutoff (defaults to self.cutoff)

        Returns:
            (N,) array of coordination numbers
        """
        if max_dist is None:
            max_dist = self.cutoff

        coord_nums = np.zeros(self.n_atoms, dtype=np.int32)

        for i in range(self.n_atoms):
            _, distances, _ = self.get_neighbors(i)
            coord_nums[i] = np.sum(distances <= max_dist)

        return coord_nums

    def to_sparse_matrix(self, include_distances: bool = True):
        """
        Convert to scipy sparse matrix format.

        Args:
            include_distances: If True, values are distances; if False, values are 1

        Returns:
            scipy.sparse.csr_matrix of shape (n_atoms, n_atoms)
            Note: This collapses periodic images, summing contributions
        """
        from scipy.sparse import csr_matrix

        # Build COO format first
        rows = []
        cols = []
        data = []

        for i, j, dist, _ in self.iter_all_pairs():
            rows.append(i)
            cols.append(j)
            data.append(dist if include_distances else 1)

        if not rows:
            return csr_matrix((self.n_atoms, self.n_atoms))

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)

        return csr_matrix((data, (rows, cols)), shape=(self.n_atoms, self.n_atoms))

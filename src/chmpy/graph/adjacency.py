"""
Core graph representation for molecular structures.

Provides a MolecularGraph class that wraps a scipy sparse adjacency matrix
with vertex and edge attributes for atoms and bonds.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.csgraph import connected_components

if TYPE_CHECKING:
    from chmpy.core.molecule import Molecule

LOG = logging.getLogger(__name__)


class MolecularGraph:
    """
    Graph representation of a molecular structure.

    Stores connectivity as a sparse adjacency matrix with atomic numbers
    and positions as vertex attributes, and bond distances/orders as edge
    attributes.

    Attributes:
        n_atoms: Number of atoms (vertices) in the graph.
        atomic_numbers: Array of atomic numbers for each atom.
        positions: (N, 3) array of Cartesian coordinates (Angstroms).
        formal_charges: Array of formal charges for each atom.
        adjacency: Sparse CSR adjacency matrix (symmetric, undirected).
        bond_orders: Sparse matrix of bond orders (1=single, 2=double, etc).
        bond_distances: Sparse matrix of bond distances (Angstroms).
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        adjacency: csr_matrix,
        bond_orders: csr_matrix | None = None,
        bond_distances: csr_matrix | None = None,
        formal_charges: np.ndarray | None = None,
    ):
        """
        Initialize a MolecularGraph.

        Args:
            atomic_numbers: (N,) array of atomic numbers.
            positions: (N, 3) array of Cartesian coordinates.
            adjacency: Sparse adjacency matrix (N, N).
            bond_orders: Optional sparse matrix of bond orders.
            bond_distances: Optional sparse matrix of bond distances.
            formal_charges: Optional (N,) array of formal charges.
        """
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.adjacency = csr_matrix(adjacency, dtype=np.int8)
        self.bond_orders = bond_orders
        self.bond_distances = bond_distances

        # Formal charges (default to 0 if not provided)
        if formal_charges is not None:
            self.formal_charges = np.asarray(formal_charges, dtype=np.int32)
        else:
            self.formal_charges = np.zeros(len(atomic_numbers), dtype=np.int32)

        # Cached properties
        self._degree = None
        self._neighbors_list = None

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the graph."""
        return len(self.atomic_numbers)

    @property
    def n_bonds(self) -> int:
        """Number of bonds in the graph."""
        return self.adjacency.nnz // 2

    @property
    def degree(self) -> np.ndarray:
        """Degree (number of neighbors) for each atom."""
        if self._degree is None:
            self._degree = np.asarray(self.adjacency.sum(axis=1)).flatten()
        return self._degree

    def neighbors(self, atom_idx: int) -> np.ndarray:
        """
        Get indices of atoms bonded to the given atom.

        Args:
            atom_idx: Index of the atom.

        Returns:
            Array of neighbor atom indices.
        """
        return self.adjacency[atom_idx].indices

    def all_neighbors(self) -> list[np.ndarray]:
        """
        Get neighbor lists for all atoms.

        Returns:
            List of arrays, where each array contains neighbor indices.
        """
        if self._neighbors_list is None:
            self._neighbors_list = [self.neighbors(i) for i in range(self.n_atoms)]
        return self._neighbors_list

    def has_bond(self, atom_i: int, atom_j: int) -> bool:
        """Check if a bond exists between two atoms."""
        return self.adjacency[atom_i, atom_j] != 0

    def bond_distance(self, atom_i: int, atom_j: int) -> float | None:
        """
        Get the bond distance between two atoms.

        Returns:
            Bond distance in Angstroms, or None if no bond exists.
        """
        if self.bond_distances is None:
            return None
        if not self.has_bond(atom_i, atom_j):
            return None
        return self.bond_distances[atom_i, atom_j]

    def bond_order(self, atom_i: int, atom_j: int) -> int:
        """
        Get the bond order between two atoms.

        Returns:
            Bond order (1, 2, 3, etc), or 0 if no bond exists.
        """
        if self.bond_orders is None:
            return 1 if self.has_bond(atom_i, atom_j) else 0
        return int(self.bond_orders[atom_i, atom_j])

    def connected_components(self) -> tuple[int, np.ndarray]:
        """
        Find connected components (fragments) in the graph.

        Returns:
            Tuple of (n_components, labels) where labels assigns each
            atom to a component (0 to n_components-1).
        """
        return connected_components(self.adjacency, directed=False)

    def subgraph(self, atom_indices: np.ndarray) -> "MolecularGraph":
        """
        Extract a subgraph containing only the specified atoms.

        Args:
            atom_indices: Indices of atoms to include in the subgraph.

        Returns:
            New MolecularGraph containing only the specified atoms.
        """
        atom_indices = np.asarray(atom_indices)
        n = len(atom_indices)

        # Extract vertex attributes
        new_atomic_numbers = self.atomic_numbers[atom_indices]
        new_positions = self.positions[atom_indices]

        # Extract adjacency submatrix
        new_adj = self.adjacency[atom_indices][:, atom_indices]

        # Extract bond attributes if present
        new_bond_orders = None
        new_bond_distances = None
        if self.bond_orders is not None:
            new_bond_orders = self.bond_orders[atom_indices][:, atom_indices]
        if self.bond_distances is not None:
            new_bond_distances = self.bond_distances[atom_indices][:, atom_indices]

        return MolecularGraph(
            new_atomic_numbers,
            new_positions,
            new_adj,
            bond_orders=new_bond_orders,
            bond_distances=new_bond_distances,
        )

    def edges(self) -> list[tuple[int, int]]:
        """
        Get all edges (bonds) as a list of (i, j) tuples.

        Returns:
            List of unique (i, j) pairs where i < j.
        """
        rows, cols = self.adjacency.nonzero()
        return [(i, j) for i, j in zip(rows, cols) if i < j]

    @classmethod
    def from_molecule(cls, molecule: "Molecule") -> "MolecularGraph":
        """
        Create a MolecularGraph from a Molecule object.

        Uses the molecule's existing bonds if available, otherwise
        calls guess_bonds() to determine connectivity.

        Args:
            molecule: A Molecule object.

        Returns:
            MolecularGraph representation of the molecule.
        """
        if molecule.bonds is None:
            molecule.guess_bonds()

        n = len(molecule)
        atomic_numbers = molecule.atomic_numbers
        positions = molecule.positions

        # Convert bonds to adjacency matrix
        # molecule.bonds is a dok_matrix with bond distances
        bonds = molecule.bonds
        adjacency = dok_matrix((n, n), dtype=np.int8)
        bond_distances = dok_matrix((n, n), dtype=np.float64)

        for (i, j), dist in bonds.items():
            if dist > 0:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                bond_distances[i, j] = dist
                bond_distances[j, i] = dist

        return cls(
            atomic_numbers,
            positions,
            csr_matrix(adjacency),
            bond_distances=csr_matrix(bond_distances),
        )

    @classmethod
    def from_edge_list(
        cls,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        edges: list[tuple[int, int]],
        bond_orders: list[int] | None = None,
        formal_charges: np.ndarray | None = None,
    ) -> "MolecularGraph":
        """
        Create a MolecularGraph from an edge list.

        Args:
            atomic_numbers: (N,) array of atomic numbers.
            positions: (N, 3) array of Cartesian coordinates.
            edges: List of (i, j) pairs representing bonds.
            bond_orders: Optional list of bond orders for each edge.
            formal_charges: Optional (N,) array of formal charges.

        Returns:
            New MolecularGraph.
        """
        n = len(atomic_numbers)
        adjacency = dok_matrix((n, n), dtype=np.int8)
        bo_matrix = dok_matrix((n, n), dtype=np.int8) if bond_orders else None

        for idx, (i, j) in enumerate(edges):
            adjacency[i, j] = 1
            adjacency[j, i] = 1
            if bo_matrix is not None:
                bo_matrix[i, j] = bond_orders[idx]
                bo_matrix[j, i] = bond_orders[idx]

        return cls(
            atomic_numbers,
            positions,
            csr_matrix(adjacency),
            bond_orders=csr_matrix(bo_matrix) if bo_matrix is not None else None,
            formal_charges=formal_charges,
        )

    def __repr__(self) -> str:
        return f"<MolecularGraph(n_atoms={self.n_atoms}, n_bonds={self.n_bonds})>"

    def __len__(self) -> int:
        return self.n_atoms

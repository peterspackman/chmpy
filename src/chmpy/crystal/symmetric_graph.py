"""
Symmetric graph framework for crystal structures.

This module provides the foundation for representing any graph structure
(bonds, contacts, interactions) on a crystal with full symmetry awareness.
The key insight is that the asymmetric unit of the graph completely determines
the full periodic graph via the group action.

Core abstractions:
- AlgebraicVertexRef: Universal vertex reference (asym_idx, symop_idx, cell)
- AlgebraicEdge: Edge between two vertices with canonical form
- CosetTable: Abstract interface for handling site symmetry

The group action on vertices uses the Cayley table from SpaceGroupTable:
    g(v) = AlgebraicVertexRef(
        asym_idx = v.asym_idx,           # unchanged
        symop_idx = mult[v.symop_idx, g], # group multiplication
        cell = R_g @ v.cell              # rotate lattice vector
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from .space_group_table import SpaceGroupTable

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlgebraicVertexRef:
    """
    Universal vertex reference for any symmetric graph.

    This generalizes AlgebraicMoleculeRef and AlgebraicAtomRef into a single
    abstraction. The actual vertex is:
        g_{symop_idx}(asym_vertex[asym_idx]) + cell

    Attributes:
        asym_idx: Index in the asymmetric unit (unique vertex type)
        symop_idx: Index of generating space group operation
        cell: (h, k, l) lattice translation as a tuple of integers
    """

    asym_idx: int
    symop_idx: int
    cell: tuple[int, int, int]

    def _key(self) -> tuple:
        """Return comparison key for ordering."""
        return (self.asym_idx, self.symop_idx, self.cell)

    def __lt__(self, other: AlgebraicVertexRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() < other._key()

    def __le__(self, other: AlgebraicVertexRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() <= other._key()

    def __gt__(self, other: AlgebraicVertexRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() > other._key()

    def __ge__(self, other: AlgebraicVertexRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() >= other._key()

    def __repr__(self) -> str:
        return f"V({self.asym_idx}, s{self.symop_idx}, {self.cell})"

    def with_cell(self, new_cell: tuple[int, int, int]) -> AlgebraicVertexRef:
        """Return a copy with a different cell offset."""
        return AlgebraicVertexRef(
            asym_idx=self.asym_idx,
            symop_idx=self.symop_idx,
            cell=new_cell,
        )


@dataclass(frozen=True)
class AlgebraicEdge:
    """
    Edge in a symmetric graph.

    Stored in canonical form where:
    1. src <= dst lexicographically
    2. src.cell = (0, 0, 0) (normalized by lattice translation)

    This ensures each unique edge has exactly one representation.

    Attributes:
        src: Source vertex reference (always <= dst after normalization)
        dst: Destination vertex reference
    """

    src: AlgebraicVertexRef
    dst: AlgebraicVertexRef

    def __post_init__(self):
        # Verify canonical ordering
        if self.src > self.dst:
            old_src, old_dst = self.src, self.dst
            object.__setattr__(self, "src", old_dst)
            object.__setattr__(self, "dst", old_src)

    @classmethod
    def create(
        cls,
        v1: AlgebraicVertexRef,
        v2: AlgebraicVertexRef,
    ) -> AlgebraicEdge:
        """
        Create an AlgebraicEdge with automatic canonical ordering.

        Args:
            v1: First vertex
            v2: Second vertex

        Returns:
            AlgebraicEdge with src <= dst
        """
        if v1 <= v2:
            return cls(src=v1, dst=v2)
        else:
            return cls(src=v2, dst=v1)

    def _key(self) -> tuple:
        """Return comparison key."""
        return (self.src._key(), self.dst._key())

    def __hash__(self):
        return hash(self._key())

    def __repr__(self) -> str:
        return f"E({self.src} -- {self.dst})"

    def is_homo_edge(self) -> bool:
        """Check if both vertices are of the same asymmetric type."""
        return self.src.asym_idx == self.dst.asym_idx

    def involves_vertex_type(self, asym_idx: int) -> bool:
        """Check if this edge involves a specific vertex type."""
        return self.src.asym_idx == asym_idx or self.dst.asym_idx == asym_idx

    def is_intra_cell(self) -> bool:
        """Check if both vertices are in the same unit cell (after normalization)."""
        return self.dst.cell == (0, 0, 0)


class CosetTable(ABC):
    """
    Abstract interface for handling site symmetry (special positions).

    When vertices lie on special positions (Wyckoff sites with non-trivial
    site symmetry), multiple symops can generate the same vertex. This table
    tracks which vertices each symop generates and provides canonical
    representatives.

    Concrete implementations:
    - AtomicCosetTable: For atoms on special positions
    - MolecularCosetTable: For molecules on special positions (in dimer_index.py)
    """

    @property
    @abstractmethod
    def n_symops(self) -> int:
        """Number of space group operations."""
        pass

    @property
    @abstractmethod
    def n_vertices(self) -> int:
        """Number of vertices in the unit cell."""
        pass

    @abstractmethod
    def normalize_symop(self, symop_idx: int) -> tuple[int, int]:
        """
        Normalize a symop index to (vertex_idx, canonical_symop_idx).

        For vertices on special positions, multiple symops map to the same
        vertex. This returns the canonical (smallest index) symop that
        generates the vertex.

        Args:
            symop_idx: The symop index to normalize

        Returns:
            Tuple of (vertex_idx, canonical_symop_idx)
        """
        pass

    @abstractmethod
    def site_symmetry_order(self, vertex_idx: int) -> int:
        """
        Get the order of site symmetry for a vertex.

        For general positions, this is 1. For special positions on a
        symmetry element of order k, this is k.

        Args:
            vertex_idx: Index of vertex in unit cell

        Returns:
            Order of site symmetry (number of symops that fix this vertex)
        """
        pass


def apply_symop_to_vertex(
    vertex: AlgebraicVertexRef,
    symop_idx: int,
    sg_table: SpaceGroupTable,
    coset_table: CosetTable | None = None,
) -> AlgebraicVertexRef:
    """
    Apply a space group operation to a vertex reference.

    This is a purely algebraic operation using the group multiplication table:
    - The new symop index is computed via the Cayley table
    - The cell offset is transformed by the rotation part of the symop
    - If coset_table is provided, normalizes the symop to canonical

    Args:
        vertex: The vertex to transform
        symop_idx: Index of the symop to apply
        sg_table: The precomputed space group table
        coset_table: Optional coset table for symop normalization

    Returns:
        Transformed AlgebraicVertexRef
    """
    # New generator: g_new = g_symop ∘ g_vertex.symop
    # Using mult_table: mult[vertex.symop_idx, symop_idx] = new_symop_idx
    new_symop_idx = sg_table.mult_table[vertex.symop_idx, symop_idx]

    # Cell offset transforms by the rotation part of the applied symop
    R = sg_table.rotations[symop_idx]
    old_cell = np.array(vertex.cell, dtype=np.float64)
    new_cell_float = R @ old_cell

    # Cell offsets should be integers (lattice translations)
    new_cell = tuple(int(x) for x in np.round(new_cell_float))

    # Normalize symop if coset table is provided (for special positions)
    if coset_table is not None:
        _, canonical_symop = coset_table.normalize_symop(new_symop_idx)
        new_symop_idx = canonical_symop

    return AlgebraicVertexRef(
        asym_idx=vertex.asym_idx,
        symop_idx=new_symop_idx,
        cell=new_cell,
    )


def apply_symop_to_edge(
    edge: AlgebraicEdge,
    symop_idx: int,
    sg_table: SpaceGroupTable,
    coset_table: CosetTable | None = None,
) -> AlgebraicEdge:
    """
    Apply a space group operation to an edge.

    This transforms both vertices and returns the result in canonical form.

    Args:
        edge: The edge to transform
        symop_idx: Index of the symop to apply
        sg_table: The precomputed space group table
        coset_table: Optional coset table for symop normalization

    Returns:
        Transformed AlgebraicEdge in canonical form
    """
    new_src = apply_symop_to_vertex(edge.src, symop_idx, sg_table, coset_table)
    new_dst = apply_symop_to_vertex(edge.dst, symop_idx, sg_table, coset_table)
    return AlgebraicEdge.create(new_src, new_dst)


def normalize_edge(
    edge: AlgebraicEdge,
    sg_table: SpaceGroupTable,
) -> AlgebraicEdge:
    """
    Normalize an edge to canonical form.

    Normalization ensures:
    1. src <= dst (lexicographic ordering by asym_idx, symop_idx, then cell)
    2. The "smaller" vertex is at cell (0, 0, 0)

    This is necessary because edges that differ only by a lattice translation
    are equivalent.

    Args:
        edge: The edge to normalize
        sg_table: The precomputed space group table

    Returns:
        Normalized AlgebraicEdge
    """
    src = edge.src
    dst = edge.dst

    # Determine which vertex should be src (the smaller one)
    # Compare by (asym_idx, symop_idx) first, ignoring cell
    key_src = (src.asym_idx, src.symop_idx)
    key_dst = (dst.asym_idx, dst.symop_idx)

    if key_src > key_dst:
        src, dst = dst, src

    # If same (asym_idx, symop_idx), use cell to break tie
    if key_src == key_dst:
        if src.cell > dst.cell:
            src, dst = dst, src

    # Now translate so src is at (0, 0, 0)
    shift = src.cell

    new_src = AlgebraicVertexRef(
        asym_idx=src.asym_idx,
        symop_idx=src.symop_idx,
        cell=(0, 0, 0),
    )

    new_dst_cell = tuple(c - s for c, s in zip(dst.cell, shift))
    new_dst = AlgebraicVertexRef(
        asym_idx=dst.asym_idx,
        symop_idx=dst.symop_idx,
        cell=new_dst_cell,
    )

    # Use frozen dataclass directly to avoid re-swapping in __post_init__
    return AlgebraicEdge(src=new_src, dst=new_dst)


def canonical_edge_representative(
    edge: AlgebraicEdge,
    sg_table: SpaceGroupTable,
    coset_table: CosetTable | None = None,
) -> AlgebraicEdge:
    """
    Find the canonical representative of an edge under the group action.

    The canonical representative is the lexicographically smallest element
    of the orbit of the edge under all space group operations.

    Args:
        edge: The edge to canonicalize
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional coset table for symop normalization

    Returns:
        The canonical representative (smallest in orbit)
    """
    best = normalize_edge(edge, sg_table)

    for g in range(sg_table.n_ops):
        transformed = apply_symop_to_edge(edge, g, sg_table, coset_table)
        transformed = normalize_edge(transformed, sg_table)

        if transformed._key() < best._key():
            best = transformed

    return best


def compute_edge_orbit_size(
    edge: AlgebraicEdge,
    sg_table: SpaceGroupTable,
    coset_table: CosetTable | None = None,
) -> int:
    """
    Compute the orbit size of an edge under the space group action.

    This is the number of distinct edges obtained by applying all space
    group operations. This represents the multiplicity - how many
    symmetry-equivalent copies of this edge exist in the crystal.

    Args:
        edge: The edge to compute orbit size for
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional coset table for symop normalization

    Returns:
        Size of the orbit (number of distinct images)
    """
    orbit = set()
    for g in range(sg_table.n_ops):
        transformed = apply_symop_to_edge(edge, g, sg_table, coset_table)
        transformed = normalize_edge(transformed, sg_table)
        orbit.add(transformed._key())
    return len(orbit)


def edges_in_same_orbit(
    edge1: AlgebraicEdge,
    edge2: AlgebraicEdge,
    sg_table: SpaceGroupTable,
    coset_table: CosetTable | None = None,
) -> bool:
    """
    Check if two edges are in the same symmetry orbit.

    Two edges are equivalent if they have the same canonical representative.

    Args:
        edge1: First edge
        edge2: Second edge
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional coset table for symop normalization

    Returns:
        True if edges are symmetry-equivalent
    """
    canon1 = canonical_edge_representative(edge1, sg_table, coset_table)
    canon2 = canonical_edge_representative(edge2, sg_table, coset_table)
    return canon1._key() == canon2._key()


def compute_edge_orbit(
    edge: AlgebraicEdge,
    sg_table: SpaceGroupTable,
    coset_table: CosetTable | None = None,
) -> list[AlgebraicEdge]:
    """
    Compute the full orbit of an edge under the space group action.

    Args:
        edge: The edge to compute orbit for
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional coset table for symop normalization

    Returns:
        List of all distinct edges in the orbit
    """
    orbit_set = set()
    orbit_list = []

    for g in range(sg_table.n_ops):
        transformed = apply_symop_to_edge(edge, g, sg_table, coset_table)
        transformed = normalize_edge(transformed, sg_table)
        key = transformed._key()

        if key not in orbit_set:
            orbit_set.add(key)
            orbit_list.append(transformed)

    return orbit_list


@dataclass
class SymmetricGraph:
    """
    Base class for symmetric graphs on crystals.

    A SymmetricGraph stores only the asymmetric unit representatives of edges.
    The full graph is implicitly defined via the space group action.

    Attributes:
        sg_table: SpaceGroupTable for the crystal's space group
        coset_table: CosetTable for handling site symmetry (or None)
        n_asym_vertices: Number of vertices in the asymmetric unit
        asym_edges: Set of canonical edge representatives
    """

    sg_table: SpaceGroupTable
    coset_table: CosetTable | None
    n_asym_vertices: int
    asym_edges: set[AlgebraicEdge]

    def add_edge(self, v1: AlgebraicVertexRef, v2: AlgebraicVertexRef) -> bool:
        """
        Add an edge between v1 and v2 (and all equivalent edges).

        Only the canonical representative is stored; all symmetry-equivalent
        edges are implicitly added.

        Args:
            v1: First vertex
            v2: Second vertex

        Returns:
            True if the edge was newly added, False if already present
        """
        edge = AlgebraicEdge.create(v1, v2)
        canonical = canonical_edge_representative(edge, self.sg_table, self.coset_table)

        if canonical in self.asym_edges:
            return False

        self.asym_edges.add(canonical)
        return True

    def remove_edge(self, v1: AlgebraicVertexRef, v2: AlgebraicVertexRef) -> bool:
        """
        Remove the edge between v1 and v2 (and all equivalent edges).

        Args:
            v1: First vertex
            v2: Second vertex

        Returns:
            True if the edge was removed, False if not present
        """
        edge = AlgebraicEdge.create(v1, v2)
        canonical = canonical_edge_representative(edge, self.sg_table, self.coset_table)

        if canonical not in self.asym_edges:
            return False

        self.asym_edges.discard(canonical)
        return True

    def has_edge(self, v1: AlgebraicVertexRef, v2: AlgebraicVertexRef) -> bool:
        """
        Check if an edge exists between v1 and v2.

        Args:
            v1: First vertex
            v2: Second vertex

        Returns:
            True if the edge exists
        """
        edge = AlgebraicEdge.create(v1, v2)
        canonical = canonical_edge_representative(edge, self.sg_table, self.coset_table)
        return canonical in self.asym_edges

    def n_unique_edges(self) -> int:
        """Return the number of unique edges (asymmetric unit representatives)."""
        return len(self.asym_edges)

    def n_total_edges(self) -> int:
        """Return the total number of edges (sum of orbit sizes)."""
        total = 0
        for edge in self.asym_edges:
            total += compute_edge_orbit_size(edge, self.sg_table, self.coset_table)
        return total

    def unique_edges_with_multiplicities(self) -> list[tuple[AlgebraicEdge, int]]:
        """
        Get unique edges with their multiplicities.

        Returns:
            List of (edge, multiplicity) tuples
        """
        result = []
        for edge in self.asym_edges:
            mult = compute_edge_orbit_size(edge, self.sg_table, self.coset_table)
            result.append((edge, mult))
        return result

    def all_edges(self) -> Iterator[AlgebraicEdge]:
        """
        Iterate over all edges (full orbit expansion).

        Yields:
            All edges in all orbits
        """
        for edge in self.asym_edges:
            yield from compute_edge_orbit(edge, self.sg_table, self.coset_table)

    def edges_at_vertex(self, vertex: AlgebraicVertexRef) -> list[AlgebraicEdge]:
        """
        Find all edges incident to a specific vertex.

        Args:
            vertex: The vertex to query

        Returns:
            List of edges incident to this vertex
        """
        result = []
        for asym_edge in self.asym_edges:
            # Apply all symops to see if edge touches vertex
            for g in range(self.sg_table.n_ops):
                transformed = apply_symop_to_edge(
                    asym_edge, g, self.sg_table, self.coset_table
                )
                transformed = normalize_edge(transformed, self.sg_table)

                if transformed.src == vertex or transformed.dst == vertex:
                    result.append(transformed)

        return result

"""
Ring detection algorithms for molecular graphs.

Implements SSSR (Smallest Set of Smallest Rings) using a modified version
of Horton's algorithm, and all-rings enumeration.
"""

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)


def find_sssr(graph: "MolecularGraph") -> list[tuple[int, ...]]:
    """
    Find the Smallest Set of Smallest Rings (SSSR) in a molecular graph.

    Uses a modified Horton's algorithm:
    1. Find all fundamental cycles from shortest path trees
    2. Select a minimal linearly independent set

    The SSSR size equals M - N + C where:
    - M = number of edges (bonds)
    - N = number of vertices (atoms)
    - C = number of connected components

    Args:
        graph: A MolecularGraph instance.

    Returns:
        List of tuples, each containing atom indices forming a ring.
    """
    n = graph.n_atoms
    m = graph.n_bonds
    n_components, _ = graph.connected_components()

    # Calculate expected SSSR size (cyclomatic number)
    sssr_size = m - n + n_components

    if sssr_size <= 0:
        return []

    # Find all candidate rings using BFS from each atom
    candidate_rings = _find_candidate_rings(graph)

    if not candidate_rings:
        return []

    # Sort by ring size
    candidate_rings.sort(key=len)

    # Select linearly independent rings
    sssr = _select_independent_rings(candidate_rings, m, sssr_size, graph)

    return sssr


def _find_candidate_rings(graph: "MolecularGraph") -> list[tuple[int, ...]]:
    """
    Find candidate rings using BFS shortest path trees.

    For each atom, performs BFS and tracks back-edges that form cycles.
    """
    n = graph.n_atoms
    neighbors = graph.all_neighbors()
    candidate_rings = set()

    for start in range(n):
        # BFS from this atom
        parent = [-1] * n
        depth = [-1] * n
        depth[start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if depth[neighbor] == -1:
                    # Tree edge
                    depth[neighbor] = depth[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
                elif neighbor != parent[current] and depth[neighbor] <= depth[current]:
                    # Back edge - found a cycle
                    ring = _extract_ring(current, neighbor, parent)
                    if ring and len(ring) >= 3:
                        candidate_rings.add(ring)

    return list(candidate_rings)


def _extract_ring(atom1: int, atom2: int, parent: list[int]) -> tuple[int, ...] | None:
    """
    Extract a ring from two atoms meeting via a back edge.

    Traces back through parent pointers to find the common ancestor,
    forming a ring.
    """
    # Find paths from both atoms to their common ancestor
    path1 = [atom1]
    path2 = [atom2]

    # Build path from atom1 to root
    current = atom1
    visited = {atom1}
    while parent[current] != -1:
        current = parent[current]
        path1.append(current)
        visited.add(current)

    # Build path from atom2 until we hit path1
    current = atom2
    while current not in visited:
        if parent[current] == -1:
            return None  # No common ancestor (shouldn't happen in connected graph)
        current = parent[current]
        path2.append(current)

    # Find where path2 meets path1
    common_ancestor = current
    idx1 = path1.index(common_ancestor)

    # Construct the ring
    ring = path1[: idx1 + 1] + path2[-2::-1]  # Exclude duplicate common ancestor

    # Normalize: start from smallest index, canonical direction
    return _canonicalize_ring(ring)


def _canonicalize_ring(ring: list[int]) -> tuple[int, ...]:
    """
    Canonicalize a ring representation.

    Rotates to start at the smallest index and chooses the
    lexicographically smaller direction.
    """
    if not ring:
        return ()

    # Find position of minimum element
    min_idx = ring.index(min(ring))

    # Rotate to start at minimum
    rotated = ring[min_idx:] + ring[:min_idx]

    # Choose lexicographically smaller direction
    reversed_ring = [rotated[0]] + rotated[1:][::-1]

    if tuple(reversed_ring) < tuple(rotated):
        return tuple(reversed_ring)
    return tuple(rotated)


def _select_independent_rings(
    candidates: list[tuple[int, ...]],
    n_edges: int,
    sssr_size: int,
    graph: "MolecularGraph",
) -> list[tuple[int, ...]]:
    """
    Select linearly independent rings for SSSR.

    Uses Gaussian elimination on the edge-incidence matrix to find
    a minimal independent set.
    """
    if not candidates:
        return []

    # Build edge index mapping
    edges = graph.edges()
    edge_to_idx = {(min(e), max(e)): i for i, e in enumerate(edges)}

    # Build incidence matrix (ring x edge)
    n_candidates = len(candidates)
    incidence = np.zeros((n_candidates, n_edges), dtype=np.int8)

    for ring_idx, ring in enumerate(candidates):
        ring_size = len(ring)
        for i in range(ring_size):
            a, b = ring[i], ring[(i + 1) % ring_size]
            edge_key = (min(a, b), max(a, b))
            if edge_key in edge_to_idx:
                incidence[ring_idx, edge_to_idx[edge_key]] = 1

    # Gaussian elimination over GF(2) to find independent rows
    selected = []
    used_cols = set()

    for ring_idx in range(n_candidates):
        if len(selected) >= sssr_size:
            break

        row = incidence[ring_idx].copy()

        # Reduce by previously selected rows
        for prev_idx, pivot_col in selected:
            if row[pivot_col]:
                row ^= incidence[prev_idx]

        # Find pivot column
        pivot_col = -1
        for col in range(n_edges):
            if row[col] and col not in used_cols:
                pivot_col = col
                break

        if pivot_col != -1:
            selected.append((ring_idx, pivot_col))
            used_cols.add(pivot_col)

    return [candidates[idx] for idx, _ in selected]


def find_all_rings(
    graph: "MolecularGraph", max_size: int = 12
) -> list[tuple[int, ...]]:
    """
    Find all rings in a molecular graph up to a maximum size.

    Uses depth-limited DFS to enumerate all cycles.

    Args:
        graph: A MolecularGraph instance.
        max_size: Maximum ring size to consider (default 12).

    Returns:
        List of tuples, each containing atom indices forming a ring.
    """
    n = graph.n_atoms
    neighbors = graph.all_neighbors()
    all_rings = set()

    def dfs(start: int, current: int, path: list[int], visited: set[int]):
        """DFS to find rings starting from 'start'."""
        if len(path) > max_size:
            return

        for neighbor in neighbors[current]:
            if neighbor == start and len(path) >= 3:
                # Found a ring
                ring = _canonicalize_ring(path)
                all_rings.add(ring)
            elif neighbor not in visited and neighbor > start:
                # Continue searching (neighbor > start avoids duplicates)
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    for start in range(n):
        visited = {start}
        for neighbor in neighbors[start]:
            if neighbor > start:
                visited.add(neighbor)
                dfs(start, neighbor, [start, neighbor], visited)
                visited.remove(neighbor)

    return list(all_rings)


def is_in_ring(graph: "MolecularGraph", atom_idx: int) -> bool:
    """
    Check if an atom is part of any ring.

    Args:
        graph: A MolecularGraph instance.
        atom_idx: Index of the atom to check.

    Returns:
        True if the atom is in at least one ring.
    """
    rings = find_sssr(graph)
    return any(atom_idx in ring for ring in rings)


def ring_membership(graph: "MolecularGraph") -> np.ndarray:
    """
    Count ring membership for each atom.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        Array of shape (n_atoms,) with count of SSSR rings each atom belongs to.
    """
    rings = find_sssr(graph)
    counts = np.zeros(graph.n_atoms, dtype=np.int32)

    for ring in rings:
        for atom_idx in ring:
            counts[atom_idx] += 1

    return counts


def ring_sizes(graph: "MolecularGraph") -> list[int]:
    """
    Get the sizes of all rings in the SSSR.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        Sorted list of ring sizes.
    """
    rings = find_sssr(graph)
    return sorted(len(ring) for ring in rings)


def is_ring_bond(graph: "MolecularGraph", atom_i: int, atom_j: int) -> bool:
    """
    Check if a bond is part of any ring.

    Args:
        graph: A MolecularGraph instance.
        atom_i: First atom index.
        atom_j: Second atom index.

    Returns:
        True if the bond is part of at least one ring.
    """
    if not graph.has_bond(atom_i, atom_j):
        return False

    rings = find_sssr(graph)
    for ring in rings:
        ring_size = len(ring)
        for i in range(ring_size):
            a, b = ring[i], ring[(i + 1) % ring_size]
            if (a == atom_i and b == atom_j) or (a == atom_j and b == atom_i):
                return True
    return False


def smallest_ring_containing_atom(
    graph: "MolecularGraph", atom_idx: int
) -> tuple[int, ...] | None:
    """
    Find the smallest ring containing a given atom.

    Args:
        graph: A MolecularGraph instance.
        atom_idx: Index of the atom.

    Returns:
        Tuple of atom indices forming the smallest ring, or None if not in any ring.
    """
    rings = find_sssr(graph)
    containing = [ring for ring in rings if atom_idx in ring]

    if not containing:
        return None

    return min(containing, key=len)


def fused_ring_systems(graph: "MolecularGraph") -> list[set[int]]:
    """
    Find fused ring systems (groups of rings sharing bonds).

    Args:
        graph: A MolecularGraph instance.

    Returns:
        List of sets, each set containing atom indices of a fused ring system.
    """
    rings = find_sssr(graph)

    if not rings:
        return []

    # Build ring adjacency (rings that share atoms)
    n_rings = len(rings)
    ring_sets = [set(ring) for ring in rings]

    # Union-find to group connected rings
    parent = list(range(n_rings))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Connect rings that share at least 2 atoms (fused)
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            if len(ring_sets[i] & ring_sets[j]) >= 2:
                union(i, j)

    # Group rings by their root
    groups = {}
    for i in range(n_rings):
        root = find(i)
        if root not in groups:
            groups[root] = set()
        groups[root].update(ring_sets[i])

    return list(groups.values())

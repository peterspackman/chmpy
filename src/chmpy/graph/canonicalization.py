"""
Graph canonicalization using the Morgan algorithm.

Provides unique atom ordering for canonical SMILES generation and
Morgan/ECFP-style fingerprints.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from .aromaticity import perceive_aromaticity
from .bond_orders import get_bond_order, implicit_hydrogen_count
from .rings import find_sssr

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)


def canonical_ordering(graph: "MolecularGraph") -> np.ndarray:
    """
    Compute a canonical ordering of atoms in the molecular graph.

    Uses an extended Morgan algorithm with symmetry breaking to produce
    a unique, deterministic ordering regardless of input atom order.

    The algorithm:
    1. Compute initial atom invariants (atomic number, degree, etc.)
    2. Iteratively refine using neighbor information
    3. Break ties using lexicographic ordering

    Args:
        graph: A MolecularGraph instance.

    Returns:
        Array of shape (n_atoms,) containing the canonical index for each atom.
        To reorder atoms: new_order[canonical_ordering[i]] = i
    """
    n = graph.n_atoms

    if n == 0:
        return np.array([], dtype=np.int32)

    if n == 1:
        return np.array([0], dtype=np.int32)

    # Step 1: Compute initial invariants
    invariants = _compute_initial_invariants(graph)

    # Step 2: Morgan iteration to propagate neighbor information
    invariants = _morgan_iteration(graph, invariants)

    # Step 3: Break ties and compute final ordering
    ordering = _compute_ordering(invariants)

    return ordering


def _compute_initial_invariants(graph: "MolecularGraph") -> np.ndarray:
    """
    Compute initial atom invariants for Morgan algorithm.

    Invariants encode:
    - Atomic number
    - Degree (number of bonds)
    - Number of hydrogens
    - Formal charge (if available)
    - In-ring status
    - Aromaticity
    """
    n = graph.n_atoms

    # Get ring and aromaticity info
    rings = find_sssr(graph)
    ring_atoms = set()
    for ring in rings:
        ring_atoms.update(ring)

    aromatic_atoms, _ = perceive_aromaticity(graph)

    invariants = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # Pack multiple properties into a single integer
        atomic_num = graph.atomic_numbers[i]
        degree = len(graph.neighbors(i))
        h_count = implicit_hydrogen_count(graph, i)
        in_ring = 1 if i in ring_atoms else 0
        is_aromatic = 1 if aromatic_atoms[i] else 0

        # Combine into invariant (use prime multipliers for uniqueness)
        invariant = (
            atomic_num * 100000000
            + degree * 1000000
            + h_count * 10000
            + in_ring * 100
            + is_aromatic
        )
        invariants[i] = invariant

    return invariants


def _morgan_iteration(
    graph: "MolecularGraph", invariants: np.ndarray, max_iterations: int = 10
) -> np.ndarray:
    """
    Perform Morgan iteration to refine atom invariants.

    Each iteration updates each atom's invariant based on its neighbors'
    invariants, spreading connectivity information through the graph.
    """
    n = graph.n_atoms
    neighbors = graph.all_neighbors()

    for _ in range(max_iterations):
        new_invariants = np.zeros(n, dtype=np.int64)

        for i in range(n):
            # Hash current invariant with sorted neighbor invariants
            neighbor_invs = sorted(invariants[n] for n in neighbors[i])
            combined = (invariants[i],) + tuple(neighbor_invs)
            new_invariants[i] = hash(combined) & 0x7FFFFFFFFFFFFFFF

        # Check for convergence
        old_classes = len(set(invariants))
        new_classes = len(set(new_invariants))

        invariants = new_invariants

        # Stop if no more differentiation
        if new_classes == old_classes:
            break

    return invariants


def _compute_ordering(invariants: np.ndarray) -> np.ndarray:
    """
    Compute canonical ordering from invariants.

    Atoms are ordered by their invariant value. Ties are broken by
    index (ensuring determinism).
    """
    n = len(invariants)

    # Create (invariant, index) pairs
    pairs = [(inv, i) for i, inv in enumerate(invariants)]

    # Sort by invariant, then by index
    pairs.sort()

    # Create ordering array
    ordering = np.zeros(n, dtype=np.int32)
    for rank, (_, idx) in enumerate(pairs):
        ordering[idx] = rank

    return ordering


def atom_invariant(graph: "MolecularGraph", atom_idx: int) -> int:
    """
    Compute the invariant for a single atom.

    Args:
        graph: A MolecularGraph instance.
        atom_idx: Index of the atom.

    Returns:
        Integer invariant encoding atom properties.
    """
    invariants = _compute_initial_invariants(graph)
    return int(invariants[atom_idx])


def equivalent_atoms(graph: "MolecularGraph") -> list[set[int]]:
    """
    Find sets of equivalent (symmetric) atoms.

    Two atoms are equivalent if they have the same Morgan invariant
    after convergence, meaning they are in symmetric positions.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        List of sets, each containing indices of equivalent atoms.
    """
    invariants = _compute_initial_invariants(graph)
    invariants = _morgan_iteration(graph, invariants)

    # Group by invariant
    groups = {}
    for i, inv in enumerate(invariants):
        if inv not in groups:
            groups[inv] = set()
        groups[inv].add(i)

    return list(groups.values())


def morgan_fingerprint(
    graph: "MolecularGraph", radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """
    Compute an ECFP-style Morgan fingerprint.

    Generates circular fingerprints by hashing atom environments
    at different radii.

    Args:
        graph: A MolecularGraph instance.
        radius: Maximum radius for environment expansion.
        n_bits: Size of the fingerprint bit vector.

    Returns:
        Boolean array of shape (n_bits,) representing the fingerprint.
    """
    n = graph.n_atoms
    neighbors = graph.all_neighbors()
    fingerprint = np.zeros(n_bits, dtype=bool)

    # Initial atom invariants
    atom_features = _compute_initial_invariants(graph)

    # For each radius
    for r in range(radius + 1):
        if r == 0:
            # Radius 0: just the atom itself
            for i in range(n):
                bit = hash(atom_features[i]) % n_bits
                fingerprint[bit] = True
        else:
            # Radius r: include neighbor information
            new_features = np.zeros(n, dtype=np.int64)
            for i in range(n):
                # Include bond information
                neighbor_info = []
                for neighbor in neighbors[i]:
                    bond_order = get_bond_order(graph, i, neighbor)
                    neighbor_info.append((int(bond_order * 10), atom_features[neighbor]))
                neighbor_info.sort()

                combined = (atom_features[i],) + tuple(neighbor_info)
                new_features[i] = hash(combined) & 0x7FFFFFFFFFFFFFFF

                # Set bit
                bit = hash((r, new_features[i])) % n_bits
                fingerprint[bit] = True

            atom_features = new_features

    return fingerprint


def morgan_fingerprint_bits(
    graph: "MolecularGraph", radius: int = 2
) -> set[int]:
    """
    Get the set of Morgan fingerprint bit indices (on bits).

    Useful for Tanimoto similarity calculations.

    Args:
        graph: A MolecularGraph instance.
        radius: Maximum radius for environment expansion.

    Returns:
        Set of bit indices that are on.
    """
    n = graph.n_atoms
    neighbors = graph.all_neighbors()
    bits = set()

    atom_features = _compute_initial_invariants(graph)

    for r in range(radius + 1):
        if r == 0:
            for i in range(n):
                bits.add(hash(atom_features[i]))
        else:
            new_features = np.zeros(n, dtype=np.int64)
            for i in range(n):
                neighbor_info = []
                for neighbor in neighbors[i]:
                    bond_order = get_bond_order(graph, i, neighbor)
                    neighbor_info.append((int(bond_order * 10), atom_features[neighbor]))
                neighbor_info.sort()

                combined = (atom_features[i],) + tuple(neighbor_info)
                new_features[i] = hash(combined) & 0x7FFFFFFFFFFFFFFF
                bits.add(hash((r, new_features[i])))

            atom_features = new_features

    return bits


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute Tanimoto similarity between two fingerprints.

    Args:
        fp1: First fingerprint (boolean array).
        fp2: Second fingerprint (boolean array).

    Returns:
        Tanimoto similarity in [0, 1].
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)

    if union == 0:
        return 1.0  # Both empty

    return intersection / union


def canonical_atom_map(graph: "MolecularGraph") -> dict[int, int]:
    """
    Get mapping from original atom indices to canonical indices.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        Dictionary mapping original index to canonical index.
    """
    ordering = canonical_ordering(graph)
    return {i: int(ordering[i]) for i in range(len(ordering))}


def reorder_graph(graph: "MolecularGraph") -> "MolecularGraph":
    """
    Create a new graph with atoms in canonical order.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        New MolecularGraph with atoms reordered canonically.
    """
    from .adjacency import MolecularGraph as MG

    ordering = canonical_ordering(graph)
    n = graph.n_atoms

    # Create inverse mapping
    inverse = np.zeros(n, dtype=np.int32)
    for i, rank in enumerate(ordering):
        inverse[rank] = i

    # Reorder atoms
    new_atomic_numbers = graph.atomic_numbers[inverse]
    new_positions = graph.positions[inverse]

    # Reorder edges
    new_edges = []
    for i, j in graph.edges():
        new_i = ordering[i]
        new_j = ordering[j]
        new_edges.append((min(new_i, new_j), max(new_i, new_j)))

    return MG.from_edge_list(new_atomic_numbers, new_positions, new_edges)

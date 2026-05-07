"""
Bond order perception from molecular geometry.

Assigns bond orders (single, double, triple, aromatic) based on:
1. Bond length heuristics
2. Valence constraints
3. Aromaticity information
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import dok_matrix

from .aromaticity import perceive_aromaticity

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)


# Standard bond lengths (in Angstroms) for common element pairs
# Format: (element1, element2): {bond_order: (min, max)}
BOND_LENGTH_RANGES = {
    # C-C bonds
    (6, 6): {
        1: (1.45, 1.60),  # Single
        2: (1.30, 1.45),  # Double
        3: (1.15, 1.30),  # Triple
        1.5: (1.35, 1.45),  # Aromatic
    },
    # C-N bonds
    (6, 7): {
        1: (1.42, 1.55),
        2: (1.25, 1.42),
        3: (1.10, 1.25),
        1.5: (1.30, 1.42),
    },
    # C-O bonds
    (6, 8): {
        1: (1.38, 1.50),
        2: (1.18, 1.38),
        3: (1.10, 1.18),  # Rare
    },
    # C-S bonds
    (6, 16): {
        1: (1.75, 1.90),
        2: (1.60, 1.75),
    },
    # N-N bonds
    (7, 7): {
        1: (1.38, 1.50),
        2: (1.20, 1.38),
        3: (1.08, 1.20),
    },
    # N-O bonds
    (7, 8): {
        1: (1.35, 1.50),
        2: (1.18, 1.35),
    },
    # O-O bonds
    (8, 8): {
        1: (1.40, 1.55),
        2: (1.15, 1.40),
    },
    # C-H bonds (always single)
    (6, 1): {1: (0.95, 1.20)},
    (1, 6): {1: (0.95, 1.20)},
    # N-H bonds
    (7, 1): {1: (0.95, 1.15)},
    (1, 7): {1: (0.95, 1.15)},
    # O-H bonds
    (8, 1): {1: (0.90, 1.10)},
    (1, 8): {1: (0.90, 1.10)},
    # S-H bonds
    (16, 1): {1: (1.25, 1.45)},
    (1, 16): {1: (1.25, 1.45)},
}

# Standard valences for common elements
STANDARD_VALENCES = {
    1: [1],  # H
    6: [4],  # C
    7: [3],  # N (can also be 5 in certain cases)
    8: [2],  # O
    9: [1],  # F
    15: [3, 5],  # P
    16: [2, 4, 6],  # S
    17: [1],  # Cl
    35: [1],  # Br
    53: [1],  # I
}


def perceive_bond_orders(graph: "MolecularGraph") -> np.ndarray:
    """
    Perceive bond orders from molecular geometry.

    Uses a multi-step approach:
    1. Initial assignment from bond lengths
    2. Aromatic ring detection and assignment
    3. Valence constraint propagation

    Args:
        graph: A MolecularGraph with bond distance information.

    Returns:
        Sparse matrix of bond orders (N, N).
    """
    n = graph.n_atoms
    bond_orders = dok_matrix((n, n), dtype=np.float32)

    # Get edges
    edges = graph.edges()

    # Step 1: Initial assignment from bond lengths
    for i, j in edges:
        order = _bond_length_heuristic(graph, i, j)
        bond_orders[i, j] = order
        bond_orders[j, i] = order

    # Step 2: Detect and assign aromatic bonds
    aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)
    for i, j in aromatic_bonds:
        bond_orders[i, j] = 1.5
        bond_orders[j, i] = 1.5

    # Step 3: Valence constraint propagation
    bond_orders = _valence_propagation(graph, bond_orders)

    return bond_orders


def _bond_length_heuristic(
    graph: "MolecularGraph", atom_i: int, atom_j: int
) -> float:
    """
    Estimate bond order from bond length.

    Args:
        graph: A MolecularGraph.
        atom_i: First atom index.
        atom_j: Second atom index.

    Returns:
        Estimated bond order (1, 2, or 3).
    """
    # Get bond distance
    dist = graph.bond_distance(atom_i, atom_j)
    if dist is None:
        # Calculate from positions
        pos_i = graph.positions[atom_i]
        pos_j = graph.positions[atom_j]
        dist = np.linalg.norm(pos_i - pos_j)

    # Get element pair
    z1 = graph.atomic_numbers[atom_i]
    z2 = graph.atomic_numbers[atom_j]

    # Look up in bond length table
    key = (min(z1, z2), max(z1, z2))
    if key not in BOND_LENGTH_RANGES:
        # Default to single bond for unknown pairs
        return 1.0

    ranges = BOND_LENGTH_RANGES[key]

    # Check each bond order from highest to lowest
    for order in [3, 2, 1.5, 1]:
        if order not in ranges:
            continue
        min_len, max_len = ranges[order]
        if min_len <= dist <= max_len:
            return order

    # If no match, choose closest
    best_order = 1
    best_diff = float("inf")

    for order, (min_len, max_len) in ranges.items():
        mid = (min_len + max_len) / 2
        diff = abs(dist - mid)
        if diff < best_diff:
            best_diff = diff
            best_order = order

    return best_order


def _valence_propagation(
    graph: "MolecularGraph", bond_orders: dok_matrix
) -> dok_matrix:
    """
    Refine bond orders using valence constraints.

    Iteratively adjusts bond orders to satisfy valence rules for each atom.

    Args:
        graph: A MolecularGraph.
        bond_orders: Initial bond order matrix.

    Returns:
        Refined bond order matrix.
    """
    n = graph.n_atoms
    max_iterations = 10

    for iteration in range(max_iterations):
        changed = False

        for atom_idx in range(n):
            z = graph.atomic_numbers[atom_idx]
            if z not in STANDARD_VALENCES:
                continue

            target_valences = STANDARD_VALENCES[z]
            neighbors = graph.neighbors(atom_idx)

            # Calculate current valence (sum of bond orders)
            current_valence = sum(bond_orders[atom_idx, n] for n in neighbors)

            # Check if valence is satisfied
            if any(abs(current_valence - v) < 0.1 for v in target_valences):
                continue

            # Try to adjust
            target = min(target_valences, key=lambda v: abs(v - current_valence))
            diff = target - current_valence

            if abs(diff) < 0.1:
                continue

            # Adjust bonds proportionally
            adjustable_neighbors = [
                n for n in neighbors if _can_adjust_bond(graph, atom_idx, n, diff > 0)
            ]

            if not adjustable_neighbors:
                continue

            adjustment = diff / len(adjustable_neighbors)
            for neighbor in adjustable_neighbors:
                new_order = bond_orders[atom_idx, neighbor] + adjustment
                new_order = max(1, min(3, new_order))  # Clamp to valid range

                if abs(new_order - bond_orders[atom_idx, neighbor]) > 0.01:
                    bond_orders[atom_idx, neighbor] = new_order
                    bond_orders[neighbor, atom_idx] = new_order
                    changed = True

        if not changed:
            break

    return bond_orders


def _can_adjust_bond(
    graph: "MolecularGraph", atom_i: int, atom_j: int, increase: bool
) -> bool:
    """
    Check if a bond order can be adjusted in the specified direction.

    Args:
        graph: A MolecularGraph.
        atom_i: First atom index.
        atom_j: Second atom index.
        increase: True to increase, False to decrease.

    Returns:
        True if the bond can be adjusted.
    """
    # Hydrogen bonds are always single
    if graph.atomic_numbers[atom_i] == 1 or graph.atomic_numbers[atom_j] == 1:
        return False

    # Check if the other atom can accommodate the change
    z = graph.atomic_numbers[atom_j]
    if z not in STANDARD_VALENCES:
        return True

    target_valences = STANDARD_VALENCES[z]
    max_valence = max(target_valences)

    neighbors = graph.neighbors(atom_j)
    current_valence = sum(graph.bond_order(atom_j, n) for n in neighbors)

    if increase:
        return current_valence < max_valence
    else:
        return current_valence > 1


def get_bond_order(graph: "MolecularGraph", atom_i: int, atom_j: int) -> float:
    """
    Get the perceived bond order between two atoms.

    Args:
        graph: A MolecularGraph.
        atom_i: First atom index.
        atom_j: Second atom index.

    Returns:
        Bond order (1, 1.5, 2, or 3), or 0 if no bond.
    """
    if not graph.has_bond(atom_i, atom_j):
        return 0.0

    if graph.bond_orders is not None:
        order = graph.bond_orders[atom_i, atom_j]
        if order > 0:
            return float(order)

    # Perceive from geometry
    return _bond_length_heuristic(graph, atom_i, atom_j)


def is_single_bond(graph: "MolecularGraph", atom_i: int, atom_j: int) -> bool:
    """Check if a bond is a single bond."""
    order = get_bond_order(graph, atom_i, atom_j)
    return 0.9 <= order <= 1.1


def is_double_bond(graph: "MolecularGraph", atom_i: int, atom_j: int) -> bool:
    """Check if a bond is a double bond."""
    order = get_bond_order(graph, atom_i, atom_j)
    return 1.9 <= order <= 2.1


def is_triple_bond(graph: "MolecularGraph", atom_i: int, atom_j: int) -> bool:
    """Check if a bond is a triple bond."""
    order = get_bond_order(graph, atom_i, atom_j)
    return 2.9 <= order <= 3.1


def total_bond_order(graph: "MolecularGraph", atom_idx: int) -> float:
    """
    Calculate the total bond order (valence) for an atom.

    Args:
        graph: A MolecularGraph.
        atom_idx: Index of the atom.

    Returns:
        Sum of bond orders for all bonds to this atom.
    """
    neighbors = graph.neighbors(atom_idx)
    return sum(get_bond_order(graph, atom_idx, n) for n in neighbors)


def hydrogen_count(graph: "MolecularGraph", atom_idx: int) -> int:
    """
    Count explicit hydrogens bonded to an atom.

    Args:
        graph: A MolecularGraph.
        atom_idx: Index of the atom.

    Returns:
        Number of hydrogen atoms bonded to this atom.
    """
    neighbors = graph.neighbors(atom_idx)
    return sum(1 for n in neighbors if graph.atomic_numbers[n] == 1)


def implicit_hydrogen_count(graph: "MolecularGraph", atom_idx: int) -> int:
    """
    Calculate implicit hydrogen count for an atom.

    Based on standard valence minus current bond order sum.

    Args:
        graph: A MolecularGraph.
        atom_idx: Index of the atom.

    Returns:
        Number of implicit hydrogens.
    """
    z = graph.atomic_numbers[atom_idx]
    if z not in STANDARD_VALENCES:
        return 0

    max_valence = max(STANDARD_VALENCES[z])
    current = total_bond_order(graph, atom_idx)

    implicit = int(max_valence - current)
    return max(0, implicit)

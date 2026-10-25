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

    # Steps 1 and 3: integer (Kekule) assignment satisfying valences, using
    # bond lengths both to limit which bonds may be multiple and to choose
    # between alternative assignments
    for (i, j), order in kekule_bond_orders(graph).items():
        bond_orders[i, j] = order
        bond_orders[j, i] = order

    # Step 2: Detect and assign aromatic bonds
    aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)
    for i, j in aromatic_bonds:
        bond_orders[i, j] = 1.5
        bond_orders[j, i] = 1.5

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


# Allowed valences used when assigning bond orders. Elements with several
# entries may expand their valence (e.g. S in SO2, N in nitro groups) when
# that lets otherwise unsatisfied neighbours form multiple bonds.
_KEKULE_VALENCES = {
    1: [1],
    5: [3],
    6: [4],
    7: [3, 5],
    8: [2],
    9: [1],
    14: [4],
    15: [3, 5],
    16: [2, 4, 6],
    17: [1, 3, 5, 7],
    34: [2, 4, 6],
    35: [1, 3, 5, 7],
    53: [1, 3, 5, 7],
}


def _charge_adjusted_valence(z: int, valence: int, charge: int) -> int:
    """Valence of an atom with a given formal charge (e.g. N+ -> 4, O- -> 1)."""
    if charge == 0:
        return valence
    if z in (6, 14):
        return valence - abs(charge)
    if z == 5:
        return valence - charge
    return valence + charge


def kekule_bond_orders(graph: "MolecularGraph") -> dict[tuple[int, int], int]:
    """
    Assign integer (Kekule) bond orders satisfying valence constraints.

    Each atom's unsaturation (its standard valence minus its number of
    bonds) is distributed over bonds to other unsaturated atoms, so all
    atoms must be present (i.e. explicit hydrogens). Bonds whose length
    indicates a single bond are never promoted, and shorter bonds are
    preferred when several assignments are possible.

    Args:
        graph: A MolecularGraph.

    Returns:
        Dict mapping (i, j) with i < j to integer bond orders (1, 2 or 3).
    """
    from chmpy.core.element import cov_radii

    n = graph.n_atoms
    edges = [(int(i), int(j)) for i, j in graph.edges()]
    orders = dict.fromkeys(edges, 1)
    if not edges:
        return orders

    nums = graph.atomic_numbers
    degree = np.zeros(n, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    # Maximum extra bond order allowed on each bond, and a preference score
    # (shorter relative to the covalent radii sum = more likely multiple)
    radii = cov_radii(nums)
    cap = {}
    score = {}
    for i, j in edges:
        if nums[i] == 1 or nums[j] == 1:
            continue
        if (min(nums[i], nums[j]), max(nums[i], nums[j])) in BOND_LENGTH_RANGES:
            max_order = int(np.ceil(_bond_length_heuristic(graph, i, j)))
        else:
            max_order = 3
        if max_order <= 1:
            continue
        cap[(i, j)] = max_order - 1
        dist = graph.bond_distance(i, j)
        if not dist:
            dist = np.linalg.norm(graph.positions[i] - graph.positions[j])
        score[(i, j)] = float(dist) / (radii[i] + radii[j])

    adjacent = [[] for _ in range(n)]
    for i, j in cap:
        adjacent[i].append(j)
        adjacent[j].append(i)

    def key(a, b):
        return (a, b) if a < b else (b, a)

    # Choose the smallest valence that accommodates each atom's bonds
    level = np.zeros(n, dtype=int)
    deficit = np.zeros(n, dtype=int)

    def set_deficit(a):
        valences = _KEKULE_VALENCES.get(int(nums[a]))
        if valences is None:
            deficit[a] = 0
            return
        charge = int(graph.formal_charges[a])
        valences = [_charge_adjusted_valence(nums[a], v, charge) for v in valences]
        allowed = [v for v in valences if v >= degree[a]]
        if level[a] >= len(allowed):
            deficit[a] = 0
        else:
            deficit[a] = allowed[level[a]] - degree[a]

    for a in range(n):
        set_deficit(a)

    def solve():
        extra = dict.fromkeys(cap, 0)
        remaining = deficit.copy()

        def partners(a):
            return [
                b
                for b in adjacent[a]
                if remaining[b] > 0 and extra[key(a, b)] < cap[key(a, b)]
            ]

        # Greedy: always satisfy the most constrained atom first
        while True:
            best = None
            for a in range(n):
                if remaining[a] <= 0:
                    continue
                p = partners(a)
                if p and (best is None or len(p) < best[0]):
                    best = (len(p), a, p)
            if best is None:
                break
            _, a, p = best
            b = min(p, key=lambda x: (score[key(a, x)], len(partners(x)), x))
            extra[key(a, b)] += 1
            remaining[a] -= 1
            remaining[b] -= 1

        # Augmenting paths (alternately promoting and demoting bonds) between
        # pairs of atoms that are still unsaturated
        for start in range(n):
            while remaining[start] > 0 and _augment(start, extra, remaining):
                pass
        return extra, remaining

    def _augment(start, extra, remaining):
        # BFS over (atom, parity); parity 0 means the next bond is promoted
        parent = {(start, 0): None}
        queue = [(start, 0)]
        while queue:
            a, parity = queue.pop(0)
            for b in adjacent[a]:
                k = key(a, b)
                if parity == 0 and extra[k] >= cap[k]:
                    continue
                if parity == 1 and extra[k] <= 0:
                    continue
                state = (b, 1 - parity)
                if state in parent:
                    continue
                parent[state] = (a, parity)
                if parity == 0 and b != start and remaining[b] > 0:
                    path = [state]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    bonds = [
                        key(x[0], y[0]) for x, y in zip(path, path[1:], strict=False)
                    ]
                    if len(set(bonds)) != len(bonds):
                        continue
                    for idx, k2 in enumerate(reversed(bonds)):
                        extra[k2] += 1 if idx % 2 == 0 else -1
                    remaining[start] -= 1
                    remaining[b] -= 1
                    return True
                queue.append(state)
        return False

    extra, remaining = solve()

    # Expand valences of hypervalent-capable atoms (S, P, N, halogens) when
    # this satisfies more of their unsaturated neighbours (e.g. O=S=O)
    for _ in range(n):
        improved = False
        for a in range(n):
            if not adjacent[a]:
                continue
            unsatisfied = sum(remaining[b] for b in adjacent[a] if remaining[b] > 0)
            if unsatisfied == 0:
                continue
            old_level, old_deficit = level[a], deficit[a]
            level[a] += 1
            set_deficit(a)
            if deficit[a] <= old_deficit:
                level[a], deficit[a] = old_level, old_deficit
                continue
            new_extra, new_remaining = solve()
            if new_remaining.clip(min=0).sum() < remaining.clip(min=0).sum():
                extra, remaining = new_extra, new_remaining
                improved = True
            else:
                level[a], deficit[a] = old_level, old_deficit
        if not improved:
            break

    for k, e in extra.items():
        orders[k] += e
    return orders


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

    # Lazily perceive bond orders for the whole graph (with valence-constraint
    # correction) and cache the result. The bond-length heuristic alone gets
    # carboxylate-like sp2 carbons wrong because both C-O lengths are short
    # enough to be flagged as double; the valence pass in perceive_bond_orders
    # fixes that.
    if graph.bond_orders is None:
        graph.bond_orders = perceive_bond_orders(graph)

    order = graph.bond_orders[atom_i, atom_j]
    if order > 0:
        return float(order)

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

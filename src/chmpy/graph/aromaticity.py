"""
Aromaticity perception for molecular graphs.

Implements Hückel's 4n+2 rule for detecting aromatic rings, with support
for heteroatoms (N, O, S) in various bonding environments.

The detection flow is:
1. Get Kekulé bond orders from geometry (single/double/triple)
2. Find rings with conjugated bonds (alternating or equal unsaturation)
3. Check 4n+2 π electron count
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from .rings import find_sssr

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)

# Atomic numbers for common elements
C = 6
N = 7
O = 8
S = 16
P = 15
Se = 34

# Bond length thresholds for Kekulé assignment (in Angstroms)
# Used to determine if bonds are single vs double/aromatic
BOND_LENGTH_THRESHOLDS = {
    # (element1, element2): (single_min, double_max)
    # If length < double_max: could be double/aromatic
    # If length > single_min: likely single
    (6, 6): (1.45, 1.44),   # C-C: single > 1.45, double/arom < 1.44
    (6, 7): (1.40, 1.38),   # C-N
    (6, 8): (1.36, 1.32),   # C-O
    (6, 16): (1.78, 1.72),  # C-S
    (7, 7): (1.40, 1.30),   # N-N
    (7, 8): (1.36, 1.25),   # N-O
}


def perceive_aromaticity(graph: "MolecularGraph") -> tuple[np.ndarray, np.ndarray]:
    """
    Perceive aromaticity in a molecular graph.

    Identifies aromatic atoms and bonds based on ring membership and
    Hückel's 4n+2 rule for π electrons.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        Tuple of (aromatic_atoms, aromatic_bonds) where:
        - aromatic_atoms: boolean array (n_atoms,) marking aromatic atoms
        - aromatic_bonds: list of (i, j) tuples for aromatic bonds
    """
    n = graph.n_atoms
    aromatic_atoms = np.zeros(n, dtype=bool)
    aromatic_bonds = []

    rings = find_sssr(graph)
    if not rings:
        return aromatic_atoms, aromatic_bonds

    # Check each ring for aromaticity
    aromatic_rings = []
    for ring in rings:
        if is_aromatic_ring(graph, ring):
            aromatic_rings.append(ring)

    # Handle fused aromatic systems (e.g., naphthalene)
    # If individual rings aren't aromatic, check if they form an aromatic system
    aromatic_rings = _check_fused_systems(graph, rings, aromatic_rings)

    # Mark aromatic atoms and bonds
    for ring in aromatic_rings:
        for atom_idx in ring:
            aromatic_atoms[atom_idx] = True

        ring_size = len(ring)
        for i in range(ring_size):
            a, b = ring[i], ring[(i + 1) % ring_size]
            bond = (min(a, b), max(a, b))
            if bond not in aromatic_bonds:
                aromatic_bonds.append(bond)

    return aromatic_atoms, aromatic_bonds


def is_aromatic_ring(graph: "MolecularGraph", ring: tuple[int, ...]) -> bool:
    """
    Check if a ring is aromatic using Hückel's 4n+2 rule.

    A ring is aromatic if:
    1. It is planar (assumed for small rings)
    2. All atoms can participate in aromaticity (C, N, O, S, etc.)
    3. Ring is conjugated (bonds are unsaturated based on geometry)
    4. It has 4n+2 π electrons (n = 0, 1, 2, ...)

    Args:
        graph: A MolecularGraph instance.
        ring: Tuple of atom indices forming the ring.

    Returns:
        True if the ring is aromatic.
    """
    # Check ring size - aromatic rings are typically 5-7 membered
    ring_size = len(ring)
    if ring_size < 5 or ring_size > 7:
        return False

    # Check if all atoms can participate in aromaticity
    if not _all_atoms_can_be_aromatic(graph, ring):
        return False

    # Check that ring has conjugated bonds (from Kekulé structure)
    if not _ring_is_conjugated(graph, ring):
        return False

    # Count π electrons
    pi_electrons = count_pi_electrons(graph, ring)

    if pi_electrons is None:
        return False

    # Check 4n+2 rule
    return _satisfies_huckel_rule(pi_electrons)


def count_pi_electrons(
    graph: "MolecularGraph", ring: tuple[int, ...]
) -> int | None:
    """
    Count the number of π electrons contributed to a ring.

    Different atoms contribute different numbers based on their
    bonding environment:
    - Carbon (sp2): 1 electron
    - Nitrogen (pyridine-like, =N-): 1 electron
    - Nitrogen (pyrrole-like, -NH-): 2 electrons
    - Oxygen/Sulfur with lone pairs: 2 electrons

    Args:
        graph: A MolecularGraph instance.
        ring: Tuple of atom indices forming the ring.

    Returns:
        Number of π electrons, or None if the ring cannot be aromatic.
    """
    total = 0
    ring_set = set(ring)

    for atom_idx in ring:
        atomic_num = graph.atomic_numbers[atom_idx]
        neighbors = graph.neighbors(atom_idx)
        degree = len(neighbors)

        # Count neighbors in ring vs outside
        ring_neighbors = sum(1 for n in neighbors if n in ring_set)
        external_neighbors = degree - ring_neighbors

        contribution = _pi_electron_contribution(
            atomic_num, degree, ring_neighbors, external_neighbors, graph, atom_idx
        )

        if contribution is None:
            return None

        total += contribution

    return total


def _pi_electron_contribution(
    atomic_num: int,
    degree: int,
    ring_neighbors: int,
    external_neighbors: int,
    graph: "MolecularGraph",
    atom_idx: int,
) -> int | None:
    """
    Calculate π electron contribution for a single atom in a ring.

    Args:
        atomic_num: Atomic number of the atom.
        degree: Total degree (number of bonds) of the atom.
        ring_neighbors: Number of neighbors within the ring.
        external_neighbors: Number of neighbors outside the ring.
        graph: The molecular graph.
        atom_idx: Index of the atom.

    Returns:
        Number of π electrons contributed, or None if atom cannot be aromatic.
    """
    # Carbon
    if atomic_num == C:
        # sp2 carbon contributes 1 electron
        # Must have degree 2 or 3 (in ring or with H/substituent)
        if degree in (2, 3):
            return 1
        return None

    # Nitrogen
    if atomic_num == N:
        # Pyridine-type (=N-): degree 2, contributes 1 electron
        # Pyrrole-type (-NH-): degree 3 with H, contributes 2 electrons
        if degree == 2:
            # Pyridine nitrogen
            return 1
        elif degree == 3:
            # Could be pyrrole (NH) or substituted pyridine
            # If connected to H or has lone pair available, contributes 2
            # Check if external neighbor is likely H
            if external_neighbors >= 1:
                # Likely -NH- (pyrrole) or -NR- (substituted pyrrole)
                return 2
            # Otherwise, might be part of fused system
            return 1
        return None

    # Oxygen
    if atomic_num == O:
        # Furan-type: degree 2, contributes 2 electrons (lone pair)
        if degree == 2:
            return 2
        return None

    # Sulfur
    if atomic_num == S:
        # Thiophene-type: degree 2, contributes 2 electrons
        if degree == 2:
            return 2
        return None

    # Phosphorus (rare but possible)
    if atomic_num == P:
        if degree == 2:
            return 2
        return None

    # Selenium
    if atomic_num == Se:
        if degree == 2:
            return 2
        return None

    # Other elements - assume not aromatic
    return None


def _all_atoms_can_be_aromatic(graph: "MolecularGraph", ring: tuple[int, ...]) -> bool:
    """Check if all atoms in the ring can participate in aromaticity."""
    aromatic_elements = {C, N, O, S, P, Se}

    for atom_idx in ring:
        atomic_num = graph.atomic_numbers[atom_idx]
        if atomic_num not in aromatic_elements:
            return False

    return True


def _ring_is_conjugated(graph: "MolecularGraph", ring: tuple[int, ...]) -> bool:
    """
    Check if a ring is conjugated based on bond lengths (Kekulé structure).

    A ring is conjugated if all bonds are short enough to be unsaturated
    (double or aromatic character). This distinguishes benzene (~1.39 Å)
    from cyclohexane (~1.54 Å).

    Args:
        graph: A MolecularGraph instance.
        ring: Tuple of atom indices forming the ring.

    Returns:
        True if all ring bonds are consistent with conjugation.
    """
    ring_size = len(ring)

    for i in range(ring_size):
        atom_i = ring[i]
        atom_j = ring[(i + 1) % ring_size]

        # Get bond length
        dist = _get_bond_distance(graph, atom_i, atom_j)
        if dist is None:
            continue

        # Get element pair
        z1 = graph.atomic_numbers[atom_i]
        z2 = graph.atomic_numbers[atom_j]
        key = (min(z1, z2), max(z1, z2))

        # Check if bond is short enough for conjugation
        if key in BOND_LENGTH_THRESHOLDS:
            single_min, _ = BOND_LENGTH_THRESHOLDS[key]
            # If bond is clearly a single bond (long), ring is not conjugated
            if dist > single_min:
                return False
        elif z1 == 6 and z2 == 6:
            # Default C-C check
            if dist > 1.45:
                return False

    return True


def _get_bond_distance(graph: "MolecularGraph", atom_i: int, atom_j: int) -> float | None:
    """Get the distance between two bonded atoms."""
    # Try to get from graph's bond_distance method if available
    if hasattr(graph, 'bond_distance'):
        dist = graph.bond_distance(atom_i, atom_j)
        if dist is not None:
            return dist

    # Calculate from positions
    if graph.positions is not None:
        pos_i = graph.positions[atom_i]
        pos_j = graph.positions[atom_j]
        return float(np.linalg.norm(pos_i - pos_j))

    return None


def _satisfies_huckel_rule(n_electrons: int) -> bool:
    """
    Check if the number of electrons satisfies Hückel's 4n+2 rule.

    Valid values: 2, 6, 10, 14, 18, ... (n = 0, 1, 2, 3, 4, ...)
    """
    if n_electrons < 2:
        return False
    return (n_electrons - 2) % 4 == 0


def _check_fused_systems(
    graph: "MolecularGraph",
    all_rings: list[tuple[int, ...]],
    aromatic_rings: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """
    Check fused ring systems for aromaticity.

    For compounds like naphthalene where individual rings might not
    satisfy 4n+2 alone, but the fused system does.
    """
    if len(all_rings) < 2:
        return aromatic_rings

    # Find fused pairs (rings sharing at least 2 atoms)
    aromatic_set = set(map(tuple, aromatic_rings))
    ring_sets = [set(ring) for ring in all_rings]

    for i, ring1 in enumerate(all_rings):
        if tuple(ring1) in aromatic_set:
            continue

        for j, ring2 in enumerate(all_rings):
            if i >= j:
                continue

            shared = ring_sets[i] & ring_sets[j]
            if len(shared) >= 2:
                # Fused rings - check combined system
                combined = ring_sets[i] | ring_sets[j]
                combined_pi = _count_pi_electrons_for_set(graph, combined)

                if combined_pi is not None and _satisfies_huckel_rule(combined_pi):
                    # Add both rings as aromatic
                    if tuple(ring1) not in aromatic_set:
                        aromatic_rings.append(ring1)
                        aromatic_set.add(tuple(ring1))
                    if tuple(ring2) not in aromatic_set:
                        aromatic_rings.append(ring2)
                        aromatic_set.add(tuple(ring2))

    return aromatic_rings


def _count_pi_electrons_for_set(
    graph: "MolecularGraph", atom_set: set[int]
) -> int | None:
    """Count π electrons for an arbitrary set of atoms."""
    total = 0

    for atom_idx in atom_set:
        atomic_num = graph.atomic_numbers[atom_idx]
        neighbors = graph.neighbors(atom_idx)
        degree = len(neighbors)
        ring_neighbors = sum(1 for n in neighbors if n in atom_set)
        external_neighbors = degree - ring_neighbors

        contribution = _pi_electron_contribution(
            atomic_num, degree, ring_neighbors, external_neighbors, graph, atom_idx
        )

        if contribution is None:
            return None

        total += contribution

    return total


def get_aromatic_rings(graph: "MolecularGraph") -> list[tuple[int, ...]]:
    """
    Get all aromatic rings in the graph.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        List of tuples containing atom indices for each aromatic ring.
    """
    rings = find_sssr(graph)
    aromatic = []

    for ring in rings:
        if is_aromatic_ring(graph, ring):
            aromatic.append(ring)

    # Check fused systems
    return _check_fused_systems(graph, rings, aromatic)


def is_aromatic_atom(graph: "MolecularGraph", atom_idx: int) -> bool:
    """
    Check if an atom is aromatic.

    Args:
        graph: A MolecularGraph instance.
        atom_idx: Index of the atom to check.

    Returns:
        True if the atom is part of an aromatic ring.
    """
    aromatic_atoms, _ = perceive_aromaticity(graph)
    return aromatic_atoms[atom_idx]


def is_aromatic_bond(graph: "MolecularGraph", atom_i: int, atom_j: int) -> bool:
    """
    Check if a bond is aromatic.

    Args:
        graph: A MolecularGraph instance.
        atom_i: First atom index.
        atom_j: Second atom index.

    Returns:
        True if the bond is part of an aromatic ring.
    """
    _, aromatic_bonds = perceive_aromaticity(graph)
    bond = (min(atom_i, atom_j), max(atom_i, atom_j))
    return bond in aromatic_bonds

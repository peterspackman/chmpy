"""
Formal charge assignment for molecular graphs.

Assigns formal charges based on valence analysis and recognition of
common charged functional groups (nitro, carboxylate, ammonium, etc.).
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from .bond_orders import get_bond_order, total_bond_order

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)

# Standard valences for neutral atoms
STANDARD_VALENCES = {
    1: 1,   # H
    5: 3,   # B
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    14: 4,  # Si
    15: 3,  # P (can be 5)
    16: 2,  # S (can be 4, 6)
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}


def guess_formal_charges(graph: "MolecularGraph") -> np.ndarray:
    """
    Guess formal charges for atoms in a molecular graph.

    Uses valence analysis and pattern recognition to identify charged
    atoms. Handles common functional groups like:
    - Nitro groups: [N+](=O)[O-]
    - Carboxylates: C(=O)[O-]
    - Ammonium: [NH4+], [NR4+]
    - Sulfonates, phosphates, etc.

    Args:
        graph: A MolecularGraph instance.

    Returns:
        Array of formal charges (integers) for each atom.
    """
    n = graph.n_atoms
    charges = np.zeros(n, dtype=np.int32)

    # Track which atoms have been assigned charges
    assigned = set()

    # First pass: identify specific functional groups
    _assign_nitro_charges(graph, charges, assigned)
    _assign_carboxylate_charges(graph, charges, assigned)
    _assign_noxide_charges(graph, charges, assigned)

    # Second pass: check for atoms with unusual valence
    _assign_valence_based_charges(graph, charges, assigned)

    return charges


def _assign_nitro_charges(
    graph: "MolecularGraph",
    charges: np.ndarray,
    assigned: set[int],
) -> None:
    """
    Assign charges for nitro groups: -N(=O)=O → [N+](=O)[O-]

    Pattern: N bonded to exactly 2 O atoms (terminal) and 1 other atom (C or N).
    This covers both C-NO2 and N-NO2 cases.
    """
    for n_idx in range(graph.n_atoms):
        if graph.atomic_numbers[n_idx] != 7:  # Not nitrogen
            continue
        if n_idx in assigned:
            continue

        neighbors = list(graph.neighbors(n_idx))

        # Nitro N has exactly 3 neighbors: 2 O and 1 other (C or N)
        if len(neighbors) != 3:
            continue

        # Find oxygen neighbors
        o_neighbors = [i for i in neighbors if graph.atomic_numbers[i] == 8]

        if len(o_neighbors) != 2:
            continue

        # Check if both O atoms have only N as neighbor (terminal oxygens)
        terminal_oxygens = []
        for o_idx in o_neighbors:
            o_neighbors_of_o = list(graph.neighbors(o_idx))
            if len(o_neighbors_of_o) == 1 and o_neighbors_of_o[0] == n_idx:
                terminal_oxygens.append(o_idx)

        if len(terminal_oxygens) != 2:
            continue

        # This is a nitro group - N bonded to 2 terminal O atoms
        # Assign N+ and O- regardless of bond lengths
        charges[n_idx] = 1  # N+

        # Assign O- to the oxygen with longer bond (more single character)
        d1 = _bond_distance(graph, n_idx, terminal_oxygens[0])
        d2 = _bond_distance(graph, n_idx, terminal_oxygens[1])
        if d1 >= d2:
            charges[terminal_oxygens[0]] = -1
        else:
            charges[terminal_oxygens[1]] = -1

        # Also mark the connecting atom (ring N or C) as assigned so it doesn't
        # get a spurious charge from valence-based detection
        other_neighbor = [i for i in neighbors if i not in o_neighbors][0]
        assigned.add(other_neighbor)

        assigned.add(n_idx)
        assigned.update(terminal_oxygens)
        LOG.debug(f"Nitro group found at N={n_idx}, connected to atom {other_neighbor}")


def _assign_carboxylate_charges(
    graph: "MolecularGraph",
    charges: np.ndarray,
    assigned: set[int],
) -> None:
    """
    Assign charges for carboxylate groups: -C(=O)O- → -C(=O)[O-]

    Pattern: C bonded to exactly 2 O atoms, both terminal, with similar bonds
    (indicating resonance/delocalization)
    """
    for c_idx in range(graph.n_atoms):
        if graph.atomic_numbers[c_idx] != 6:  # Not carbon
            continue
        if c_idx in assigned:
            continue

        neighbors = list(graph.neighbors(c_idx))

        # Find oxygen neighbors
        o_neighbors = [i for i in neighbors if graph.atomic_numbers[i] == 8]

        if len(o_neighbors) != 2:
            continue

        # Check if both are terminal oxygens
        terminal_oxygens = []
        for o_idx in o_neighbors:
            o_neighbors_of_o = list(graph.neighbors(o_idx))
            if len(o_neighbors_of_o) == 1 and o_neighbors_of_o[0] == c_idx:
                terminal_oxygens.append(o_idx)

        if len(terminal_oxygens) != 2:
            continue

        # Check if bond lengths are similar (carboxylate) vs different (carboxylic acid)
        d1 = _bond_distance(graph, c_idx, terminal_oxygens[0])
        d2 = _bond_distance(graph, c_idx, terminal_oxygens[1])

        # Carboxylic acid: C=O ~1.21, C-OH ~1.36
        # Carboxylate: both C-O ~1.25-1.27
        # If difference is small, likely carboxylate
        if abs(d1 - d2) < 0.1 and d1 < 1.32 and d2 < 1.32:
            # Carboxylate - assign O- to one oxygen
            charges[terminal_oxygens[0]] = -1
            assigned.add(c_idx)
            assigned.update(terminal_oxygens)
            LOG.debug(f"Carboxylate found at C={c_idx}")


def _assign_noxide_charges(
    graph: "MolecularGraph",
    charges: np.ndarray,
    assigned: set[int],
) -> None:
    """
    Assign charges for N-oxide groups: R3N→O → [R3N+][O-]

    Pattern: N with 4 bonds including one to terminal O
    """
    for n_idx in range(graph.n_atoms):
        if graph.atomic_numbers[n_idx] != 7:
            continue
        if n_idx in assigned:
            continue

        neighbors = list(graph.neighbors(n_idx))

        # N-oxide has N with 4 substituents, one being terminal O
        if len(neighbors) != 4:
            continue

        # Find terminal oxygen
        for o_idx in neighbors:
            if graph.atomic_numbers[o_idx] != 8:
                continue
            o_neighbors = list(graph.neighbors(o_idx))
            if len(o_neighbors) == 1:
                # Terminal O bonded to tetravalent N → N-oxide
                charges[n_idx] = 1
                charges[o_idx] = -1
                assigned.add(n_idx)
                assigned.add(o_idx)
                LOG.debug(f"N-oxide found at N={n_idx}")
                break


def _assign_valence_based_charges(
    graph: "MolecularGraph",
    charges: np.ndarray,
    assigned: set[int],
) -> None:
    """
    Assign charges based on valence violations.

    If an atom has more bonds than its standard valence,
    assign a positive charge. If it could accept more bonds
    but doesn't, might be negative.
    """
    for atom_idx in range(graph.n_atoms):
        if atom_idx in assigned:
            continue

        z = graph.atomic_numbers[atom_idx]
        if z not in STANDARD_VALENCES:
            continue

        standard_valence = STANDARD_VALENCES[z]
        actual_valence = int(round(total_bond_order(graph, atom_idx)))

        # Quaternary nitrogen (4 bonds, normally 3)
        if z == 7 and actual_valence == 4:
            charges[atom_idx] = 1
            assigned.add(atom_idx)
            LOG.debug(f"Quaternary N+ at {atom_idx}")

        # Tetravalent boron (4 bonds, normally 3) → B-
        elif z == 5 and actual_valence == 4:
            charges[atom_idx] = -1
            assigned.add(atom_idx)
            LOG.debug(f"Tetravalent B- at {atom_idx}")


def _bond_distance(graph: "MolecularGraph", i: int, j: int) -> float:
    """Get bond distance between two atoms."""
    if hasattr(graph, 'bond_distance'):
        d = graph.bond_distance(i, j)
        if d is not None:
            return d
    return float(np.linalg.norm(graph.positions[i] - graph.positions[j]))


def assign_formal_charges(graph: "MolecularGraph") -> "MolecularGraph":
    """
    Return a new MolecularGraph with formal charges assigned.

    Args:
        graph: Input MolecularGraph.

    Returns:
        New MolecularGraph with formal_charges populated.
    """
    charges = guess_formal_charges(graph)

    # Create new graph with charges
    from .adjacency import MolecularGraph as MG
    return MG(
        graph.atomic_numbers,
        graph.positions,
        graph.adjacency,
        bond_orders=graph.bond_orders,
        bond_distances=graph.bond_distances,
        formal_charges=charges,
    )

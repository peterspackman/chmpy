"""
Stereochemistry detection and encoding.

Implements CIP (Cahn-Ingold-Prelog) rules for:
- R/S configuration at tetrahedral stereocenters
- E/Z configuration at double bonds
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from .bond_orders import get_bond_order, is_double_bond

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)


class StereoCenter:
    """Represents a tetrahedral stereocenter."""

    def __init__(
        self,
        atom_idx: int,
        neighbors: list[int],
        configuration: str | None = None,
    ):
        self.atom_idx = atom_idx
        self.neighbors = neighbors  # In priority order (high to low)
        self.configuration = configuration  # "R", "S", or None


class DoubleBondStereo:
    """Represents E/Z stereochemistry at a double bond."""

    def __init__(
        self,
        atom1: int,
        atom2: int,
        substituents1: list[int],
        substituents2: list[int],
        configuration: str | None = None,
    ):
        self.atom1 = atom1
        self.atom2 = atom2
        self.substituents1 = substituents1  # High priority first
        self.substituents2 = substituents2  # High priority first
        self.configuration = configuration  # "E", "Z", or None


def assign_stereochemistry(
    graph: "MolecularGraph",
) -> tuple[list[StereoCenter], list[DoubleBondStereo]]:
    """
    Detect and assign stereochemistry for a molecular graph.

    Args:
        graph: A MolecularGraph with 3D coordinates.

    Returns:
        Tuple of (stereocenters, double_bond_stereo) lists.
    """
    stereocenters = find_stereocenters(graph)
    double_bonds = find_double_bond_stereo(graph)

    # Assign configurations
    for sc in stereocenters:
        sc.configuration = _tetrahedral_stereo(graph, sc.atom_idx, sc.neighbors)

    for db in double_bonds:
        db.configuration = _double_bond_stereo(
            graph, db.atom1, db.atom2, db.substituents1, db.substituents2
        )

    return stereocenters, double_bonds


def find_stereocenters(graph: "MolecularGraph") -> list[StereoCenter]:
    """
    Find potential tetrahedral stereocenters in the graph.

    A stereocenter is an sp3 atom with 4 different substituents.

    Args:
        graph: A MolecularGraph.

    Returns:
        List of StereoCenter objects (without configuration assigned).
    """
    stereocenters = []

    for atom_idx in range(graph.n_atoms):
        neighbors = list(graph.neighbors(atom_idx))

        # Must have exactly 4 neighbors for tetrahedral center
        if len(neighbors) != 4:
            continue

        # Check if all substituents are different
        priorities = [_cip_priority(graph, n, atom_idx) for n in neighbors]

        # If any two have identical priority, not a stereocenter
        if len(set(priorities)) != 4:
            continue

        # Sort neighbors by CIP priority (high to low)
        sorted_neighbors = [
            n for _, n in sorted(zip(priorities, neighbors), reverse=True)
        ]

        stereocenters.append(StereoCenter(atom_idx, sorted_neighbors))

    return stereocenters


def find_double_bond_stereo(graph: "MolecularGraph") -> list[DoubleBondStereo]:
    """
    Find double bonds that can exhibit E/Z stereochemistry.

    Args:
        graph: A MolecularGraph.

    Returns:
        List of DoubleBondStereo objects (without configuration assigned).
    """
    double_bonds = []

    for i, j in graph.edges():
        if not is_double_bond(graph, i, j):
            continue

        # Get substituents on each carbon
        neighbors_i = [n for n in graph.neighbors(i) if n != j]
        neighbors_j = [n for n in graph.neighbors(j) if n != i]

        # Need at least 2 substituents total (one on each carbon)
        if len(neighbors_i) < 1 or len(neighbors_j) < 1:
            continue

        # Check for equivalent substituents on same carbon
        priorities_i = [_cip_priority(graph, n, i) for n in neighbors_i]
        priorities_j = [_cip_priority(graph, n, j) for n in neighbors_j]

        # If both substituents on one carbon are the same, no E/Z possible
        if len(neighbors_i) == 2 and priorities_i[0] == priorities_i[1]:
            continue
        if len(neighbors_j) == 2 and priorities_j[0] == priorities_j[1]:
            continue

        # Sort by priority
        sorted_i = [
            n for _, n in sorted(zip(priorities_i, neighbors_i), reverse=True)
        ]
        sorted_j = [
            n for _, n in sorted(zip(priorities_j, neighbors_j), reverse=True)
        ]

        double_bonds.append(DoubleBondStereo(i, j, sorted_i, sorted_j))

    return double_bonds


def _cip_priority(
    graph: "MolecularGraph",
    atom_idx: int,
    from_atom: int,
    depth: int = 0,
    max_depth: int = 6,
    visited: set | None = None,
) -> tuple:
    """
    Compute CIP priority for an atom.

    Uses recursive sphere-by-sphere expansion according to CIP rules:
    1. Higher atomic number = higher priority
    2. Higher mass isotope = higher priority (not implemented)
    3. Expand to next sphere if tied

    Args:
        graph: A MolecularGraph.
        atom_idx: Index of atom to compute priority for.
        from_atom: Atom we came from (excluded from neighbors).
        depth: Current recursion depth.
        max_depth: Maximum recursion depth.
        visited: Set of visited atoms.

    Returns:
        Tuple representing priority (higher = higher priority).
    """
    if visited is None:
        visited = set()

    if depth >= max_depth or atom_idx in visited:
        return (0,)

    visited = visited | {atom_idx}

    # Base priority: atomic number
    z = graph.atomic_numbers[atom_idx]

    if depth == max_depth - 1:
        return (z,)

    # Get neighbors excluding where we came from
    neighbors = [n for n in graph.neighbors(atom_idx) if n != from_atom and n not in visited]

    if not neighbors:
        return (z,)

    # Compute priorities of neighbors
    neighbor_priorities = []
    for neighbor in neighbors:
        # Account for bond order (double bond = 2 copies, triple = 3)
        bond_order = int(get_bond_order(graph, atom_idx, neighbor))
        for _ in range(bond_order):
            p = _cip_priority(graph, neighbor, atom_idx, depth + 1, max_depth, visited)
            neighbor_priorities.append(p)

    # Sort in descending order
    neighbor_priorities.sort(reverse=True)

    # Combine with own atomic number
    return (z,) + tuple(neighbor_priorities)


def _tetrahedral_stereo(
    graph: "MolecularGraph",
    center: int,
    neighbors: list[int],
) -> str | None:
    """
    Determine R/S configuration at a tetrahedral center.

    Uses 3D coordinates to determine handedness.

    Args:
        graph: A MolecularGraph with positions.
        center: Index of the stereocenter atom.
        neighbors: List of 4 neighbor indices in CIP priority order (high to low).

    Returns:
        "R", "S", or None if cannot be determined.
    """
    if len(neighbors) != 4:
        return None

    # Get positions
    center_pos = graph.positions[center]
    n1_pos = graph.positions[neighbors[0]]  # Highest priority
    n2_pos = graph.positions[neighbors[1]]
    n3_pos = graph.positions[neighbors[2]]
    n4_pos = graph.positions[neighbors[3]]  # Lowest priority (view from here)

    # Vectors from center to substituents
    v1 = n1_pos - center_pos
    v2 = n2_pos - center_pos
    v3 = n3_pos - center_pos
    v4 = n4_pos - center_pos

    # View from the lowest priority substituent toward the center
    # Determine if 1->2->3 is clockwise (R) or counterclockwise (S)

    # Project v1, v2, v3 onto plane perpendicular to v4
    normal = v4 / np.linalg.norm(v4)

    def project(v):
        return v - np.dot(v, normal) * normal

    p1 = project(v1)
    p2 = project(v2)
    p3 = project(v3)

    # Calculate signed angle from p1 to p2 and p1 to p3
    # Using cross product to determine handedness
    cross = np.cross(p1, p2)
    sign = np.dot(cross, normal)

    # Positive sign means counterclockwise (S), negative means clockwise (R)
    # But we're looking from n4 toward center, so reverse
    if sign > 0:
        return "R"
    elif sign < 0:
        return "S"
    else:
        return None


def _double_bond_stereo(
    graph: "MolecularGraph",
    atom1: int,
    atom2: int,
    substituents1: list[int],
    substituents2: list[int],
) -> str | None:
    """
    Determine E/Z configuration at a double bond.

    E (entgegen): High priority groups on opposite sides
    Z (zusammen): High priority groups on same side

    Args:
        graph: A MolecularGraph with positions.
        atom1: First carbon of double bond.
        atom2: Second carbon of double bond.
        substituents1: Substituents on atom1, high priority first.
        substituents2: Substituents on atom2, high priority first.

    Returns:
        "E", "Z", or None if cannot be determined.
    """
    if not substituents1 or not substituents2:
        return None

    # Get positions
    pos1 = graph.positions[atom1]
    pos2 = graph.positions[atom2]
    high1_pos = graph.positions[substituents1[0]]
    high2_pos = graph.positions[substituents2[0]]

    # Bond vector
    bond_vec = pos2 - pos1
    bond_vec = bond_vec / np.linalg.norm(bond_vec)

    # Vectors to high-priority substituents
    v1 = high1_pos - pos1
    v2 = high2_pos - pos2

    # Remove component along bond
    v1_perp = v1 - np.dot(v1, bond_vec) * bond_vec
    v2_perp = v2 - np.dot(v2, bond_vec) * bond_vec

    # Normalize
    if np.linalg.norm(v1_perp) < 1e-6 or np.linalg.norm(v2_perp) < 1e-6:
        return None

    v1_perp = v1_perp / np.linalg.norm(v1_perp)
    v2_perp = v2_perp / np.linalg.norm(v2_perp)

    # Dot product determines same side (Z) vs opposite (E)
    dot = np.dot(v1_perp, v2_perp)

    if dot > 0.1:  # Same side
        return "Z"
    elif dot < -0.1:  # Opposite sides
        return "E"
    else:
        return None


def get_stereocenter_config(
    graph: "MolecularGraph", atom_idx: int
) -> str | None:
    """
    Get the R/S configuration of a single atom if it's a stereocenter.

    Args:
        graph: A MolecularGraph.
        atom_idx: Index of the atom to check.

    Returns:
        "R", "S", or None if not a stereocenter.
    """
    neighbors = list(graph.neighbors(atom_idx))

    if len(neighbors) != 4:
        return None

    priorities = [_cip_priority(graph, n, atom_idx) for n in neighbors]
    if len(set(priorities)) != 4:
        return None

    sorted_neighbors = [
        n for _, n in sorted(zip(priorities, neighbors), reverse=True)
    ]

    return _tetrahedral_stereo(graph, atom_idx, sorted_neighbors)


def get_double_bond_config(
    graph: "MolecularGraph", atom1: int, atom2: int
) -> str | None:
    """
    Get the E/Z configuration of a double bond.

    Args:
        graph: A MolecularGraph.
        atom1: First atom of the double bond.
        atom2: Second atom of the double bond.

    Returns:
        "E", "Z", or None if not a stereogenic double bond.
    """
    if not is_double_bond(graph, atom1, atom2):
        return None

    neighbors1 = [n for n in graph.neighbors(atom1) if n != atom2]
    neighbors2 = [n for n in graph.neighbors(atom2) if n != atom1]

    if len(neighbors1) < 1 or len(neighbors2) < 1:
        return None

    priorities1 = [_cip_priority(graph, n, atom1) for n in neighbors1]
    priorities2 = [_cip_priority(graph, n, atom2) for n in neighbors2]

    if len(neighbors1) == 2 and priorities1[0] == priorities1[1]:
        return None
    if len(neighbors2) == 2 and priorities2[0] == priorities2[1]:
        return None

    sorted1 = [n for _, n in sorted(zip(priorities1, neighbors1), reverse=True)]
    sorted2 = [n for _, n in sorted(zip(priorities2, neighbors2), reverse=True)]

    return _double_bond_stereo(graph, atom1, atom2, sorted1, sorted2)


def is_chiral(graph: "MolecularGraph") -> bool:
    """
    Check if the molecule has any stereocenters.

    Args:
        graph: A MolecularGraph.

    Returns:
        True if the molecule has at least one stereocenter.
    """
    return len(find_stereocenters(graph)) > 0


def count_stereocenters(graph: "MolecularGraph") -> int:
    """
    Count the number of stereocenters in the molecule.

    Args:
        graph: A MolecularGraph.

    Returns:
        Number of tetrahedral stereocenters.
    """
    return len(find_stereocenters(graph))

"""
SMILES string generation from molecular graphs.

Generates canonical SMILES strings using DFS traversal with support for:
- Ring closures
- Aromaticity
- Stereochemistry (R/S, E/Z)
- Charges and isotopes
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from chmpy.core.element import Element

from .aromaticity import perceive_aromaticity
from .bond_orders import get_bond_order
from .canonicalization import canonical_ordering
from .rings import find_sssr
from .stereochemistry import (
    find_double_bond_stereo,
    find_stereocenters,
)

if TYPE_CHECKING:
    from .adjacency import MolecularGraph

LOG = logging.getLogger(__name__)

# Element symbols (atomic number -> symbol)
ELEMENT_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

# Organic subset elements that don't need brackets (with normal valence/no charge)
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}


def to_smiles(
    graph: "MolecularGraph",
    canonical: bool = True,
    explicit_h: bool = False,
    guess_charges: bool = False,
) -> str:
    """
    Generate a SMILES string from a molecular graph.

    Args:
        graph: A MolecularGraph instance.
        canonical: If True, generate canonical SMILES (default True).
        explicit_h: If True, include explicit hydrogens (default False).
        guess_charges: If True, automatically detect and assign formal charges
            for common functional groups (nitro, carboxylate, etc.).

    Returns:
        SMILES string representation of the molecule.
    """
    if graph.n_atoms == 0:
        return ""

    # Optionally guess formal charges
    if guess_charges:
        from .formal_charges import assign_formal_charges

        graph = assign_formal_charges(graph)

    # Get canonical ordering if requested
    if canonical:
        ordering = canonical_ordering(graph)
    else:
        ordering = np.arange(graph.n_atoms)

    # Detect aromaticity and rings
    aromatic_atoms, aromatic_bonds = perceive_aromaticity(graph)
    rings = find_sssr(graph)

    # Build ring bond set
    ring_bonds = _identify_ring_bonds(rings)

    # Find stereocenters
    stereocenters = {sc.atom_idx: sc for sc in find_stereocenters(graph)}
    # Double bonds in rings smaller than 8 cannot be E/Z in SMILES
    small_ring_bonds = _identify_ring_bonds([r for r in rings if len(r) < 8])
    double_bond_stereo = {}
    for db in find_double_bond_stereo(graph):
        if (min(db.atom1, db.atom2), max(db.atom1, db.atom2)) in small_ring_bonds:
            continue
        double_bond_stereo[(db.atom1, db.atom2)] = db
        double_bond_stereo[(db.atom2, db.atom1)] = db

    # Identify hydrogen atoms to skip (unless explicit_h is True)
    skip_atoms = set()
    if not explicit_h:
        for i in range(graph.n_atoms):
            if graph.atomic_numbers[i] == 1:
                skip_atoms.add(i)

    # DFS traversal
    writer = _SMILESWriter(
        graph,
        ordering,
        aromatic_atoms,
        aromatic_bonds,
        ring_bonds,
        stereocenters,
        double_bond_stereo,
        skip_atoms,
    )

    return writer.generate()


def _identify_ring_bonds(rings: list[tuple[int, ...]]) -> set[tuple[int, int]]:
    """Identify all bonds that are part of rings."""
    ring_bonds = set()
    for ring in rings:
        ring_size = len(ring)
        for i in range(ring_size):
            a, b = ring[i], ring[(i + 1) % ring_size]
            ring_bonds.add((min(a, b), max(a, b)))
    return ring_bonds


class _SMILESWriter:
    """Internal class for SMILES generation via DFS."""

    def __init__(
        self,
        graph: "MolecularGraph",
        ordering: np.ndarray,
        aromatic_atoms: np.ndarray,
        aromatic_bonds: list[tuple[int, int]],
        ring_bonds: set[tuple[int, int]],
        stereocenters: dict,
        double_bond_stereo: dict,
        skip_atoms: set[int],
    ):
        self.graph = graph
        self.ordering = ordering
        self.aromatic_atoms = aromatic_atoms
        self.aromatic_bonds = set(aromatic_bonds)
        self.ring_bonds = ring_bonds
        self.stereocenters = stereocenters
        self.double_bond_stereo = double_bond_stereo
        self.skip_atoms = skip_atoms

        self.visited = set()
        self.ring_openings = {}  # Maps atom_idx -> list of (ring_num, bond_symbol, other_atom)
        self.next_ring_num = 1
        # Maps (double bond atom, substituent) -> +1 ('/') or -1 ('\\') for the
        # single bond written in the direction double bond atom -> substituent
        self.bond_directions = {}

    def generate(self) -> str:
        """Generate the SMILES string."""
        # First pass: identify all ring closures and assign ring numbers
        self._identify_ring_closures()

        # Directional single bonds ('/' and '\\') encoding E/Z double bonds
        self._assign_bond_directions()

        # Handle multiple fragments
        fragments = []
        remaining = set(range(self.graph.n_atoms)) - self.skip_atoms

        while remaining:
            # Find lowest-ordered unvisited atom
            start = min(remaining, key=lambda x: self.ordering[x])
            smiles = self._dfs_to_smiles(start, -1)
            if smiles:
                fragments.append(smiles)
            remaining -= self.visited

        return ".".join(fragments)

    def _identify_ring_closures(self):
        """
        Pre-identify all ring closures using DFS.

        This assigns ring numbers and records which atoms need ring opening/closing digits.
        """
        visited = set()
        parent = {}

        # Get atoms to process (excluding skipped)
        atoms_to_process = set(range(self.graph.n_atoms)) - self.skip_atoms

        for start in sorted(atoms_to_process, key=lambda x: self.ordering[x]):
            if start in visited:
                continue

            # DFS from this start
            stack = [(start, -1)]
            while stack:
                atom, from_atom = stack.pop()
                if atom in visited:
                    continue

                visited.add(atom)
                parent[atom] = from_atom

                neighbors = [
                    n
                    for n in self.graph.neighbors(atom)
                    if n not in self.skip_atoms and n != from_atom
                ]
                neighbors.sort(key=lambda x: self.ordering[x], reverse=True)

                for neighbor in neighbors:
                    if neighbor in visited:
                        # Back edge - this is a ring closure
                        bond_key = (min(atom, neighbor), max(atom, neighbor))
                        if bond_key in self.ring_bonds:
                            ring_num = self.next_ring_num
                            self.next_ring_num += 1

                            bond_symbol = self._bond_symbol(atom, neighbor)

                            # Record ring opening at the earlier-visited atom (neighbor)
                            if neighbor not in self.ring_openings:
                                self.ring_openings[neighbor] = []
                            self.ring_openings[neighbor].append(
                                (ring_num, bond_symbol, atom)
                            )

                            # Record ring closing at current atom
                            if atom not in self.ring_openings:
                                self.ring_openings[atom] = []
                            self.ring_openings[atom].append((ring_num, "", neighbor))
                    else:
                        stack.append((neighbor, atom))

    def _assign_bond_directions(self):
        """
        Assign '/' and '\\' to single bonds adjacent to E/Z double bonds.

        The double bond itself is always written as '='. For each stereo
        double bond, every writable single bond to a substituent is marked
        according to which side of the double bond the substituent lies on.
        Single bonds shared between two conjugated stereo double bonds tie
        their markings together; inconsistent systems are left unmarked.
        """
        closure_bonds = set()
        for atom, entries in self.ring_openings.items():
            for _, _, other in entries:
                closure_bonds.add((min(atom, other), max(atom, other)))

        sides = {}
        for a1, a2 in self.double_bond_stereo:
            key = (min(a1, a2), max(a1, a2))
            if key in sides:
                continue
            bond_sides = self._substituent_sides(key[0], key[1], closure_bonds)
            if bond_sides:
                sides[key] = bond_sides

        owner = {}
        for key, bond_sides in sides.items():
            for end_sub in bond_sides:
                owner[end_sub] = key

        signs = {}
        for start in sides:
            if start in signs:
                continue
            signs[start] = 1
            queue = [start]
            component = [start]
            consistent = True
            while queue:
                key = queue.pop()
                for (end, sub), side in sides[key].items():
                    other = owner.get((sub, end))
                    if other is None:
                        continue
                    # the shared bond read from the other end flips direction
                    required = -signs[key] * side * sides[other][(sub, end)]
                    if other not in signs:
                        signs[other] = required
                        queue.append(other)
                        component.append(other)
                    elif signs[other] != required:
                        consistent = False
            if not consistent:
                for key in component:
                    signs[key] = None

        for key, bond_sides in sides.items():
            if signs[key] is None:
                continue
            for end_sub, side in bond_sides.items():
                self.bond_directions[end_sub] = signs[key] * side

    def _substituent_sides(self, atom1: int, atom2: int, closure_bonds: set) -> dict:
        """
        Side (+1/-1) of each markable substituent of the double bond atom1=atom2.

        Returns an empty dict if either end has no markable single bond or
        the geometry is ambiguous.
        """
        pos = self.graph.positions
        axis = pos[atom2] - pos[atom1]
        axis = axis / np.linalg.norm(axis)

        def perp(end, sub):
            v = pos[sub] - pos[end]
            v = v - np.dot(v, axis) * axis
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-6 else None

        substituents = []
        for end, other in ((atom1, atom2), (atom2, atom1)):
            subs = []
            for n in sorted(self.graph.neighbors(end), key=lambda x: self.ordering[x]):
                bond_key = (min(end, n), max(end, n))
                if (
                    n == other
                    or n in self.skip_atoms
                    or bond_key in closure_bonds
                    or bond_key in self.aromatic_bonds
                ):
                    continue
                order = get_bond_order(self.graph, end, n)
                order = self._adjust_bond_order_for_charges(end, n, order)
                if 0.9 <= order <= 1.1:
                    subs.append((end, int(n)))
            if not subs:
                return {}
            substituents.extend(subs)

        reference = perp(*substituents[0])
        if reference is None:
            return {}
        result = {}
        for end, sub in substituents:
            v = perp(end, sub)
            if v is None:
                return {}
            dot = np.dot(v, reference)
            if abs(dot) < 0.1:
                return {}
            result[(int(end), sub)] = 1 if dot > 0 else -1
        return result

    def _chiral_tag(self, atom_idx: int, neighbor_order: list[int]) -> str:
        """
        Tetrahedral chirality tag for the neighbours in SMILES output order.

        '@' means the remaining neighbours are anticlockwise when viewed from
        the first neighbour towards the centre, '@@' clockwise.
        """
        if len(neighbor_order) != 4:
            return ""
        pos = self.graph.positions
        first, a, b, c = (pos[i] for i in neighbor_order)
        volume = np.dot(pos[atom_idx] - first, np.cross(b - a, c - a))
        if abs(volume) < 1e-6:
            return ""
        return "@" if volume < 0 else "@@"

    def _dfs_to_smiles(self, atom_idx: int, from_atom: int) -> str:
        """Generate SMILES string via DFS from given atom."""
        if atom_idx in self.visited or atom_idx in self.skip_atoms:
            return ""

        self.visited.add(atom_idx)

        # Ring closure digits for this atom, and the atoms they connect to
        ring_str = ""
        ring_partners = []
        if atom_idx in self.ring_openings:
            for ring_num, bond_sym, other_atom in sorted(self.ring_openings[atom_idx]):
                if ring_num < 10:
                    ring_str += f"{bond_sym}{ring_num}"
                else:
                    ring_str += f"{bond_sym}%{ring_num}"
                ring_partners.append(other_atom)

        # Get neighbors (excluding where we came from and skipped atoms)
        neighbors = [
            n
            for n in self.graph.neighbors(atom_idx)
            if n != from_atom and n not in self.skip_atoms
        ]
        neighbors.sort(key=lambda x: self.ordering[x])

        # Get ring closure targets from this atom (these are handled via ring numbers, not branches)
        ring_closure_targets = set(ring_partners)

        # Filter out already-visited neighbors and ring closure targets
        branch_neighbors = [
            n
            for n in neighbors
            if n not in self.visited and n not in ring_closure_targets
        ]

        branches = []
        for neighbor in branch_neighbors:
            bond_symbol = self._bond_symbol(atom_idx, neighbor)
            branch_smiles = self._dfs_to_smiles(neighbor, atom_idx)
            if branch_smiles:
                branches.append((neighbor, bond_symbol + branch_smiles))

        # Neighbour order as written, for tetrahedral stereo: preceding atom,
        # implicit H, ring closures, then branches (OpenSMILES)
        neighbor_order = [from_atom] if from_atom >= 0 else []
        neighbor_order += [
            n
            for n in self.graph.neighbors(atom_idx)
            if n in self.skip_atoms and self.graph.atomic_numbers[n] == 1
        ]
        neighbor_order += ring_partners + [n for n, _ in branches]

        result = self._atom_symbol(atom_idx, neighbor_order) + ring_str

        # Last branch continues inline (no parentheses)
        # Earlier branches need parentheses so they're clearly attached to current atom
        # This produces C(N)N instead of CN(N) for urea
        for _, branch in branches[:-1]:
            result += "(" + branch + ")"
        if branches:
            result += branches[-1][1]

        return result

    def _atom_symbol(
        self, atom_idx: int, neighbor_order: list[int] | None = None
    ) -> str:
        """
        Generate the SMILES atom symbol.

        ``neighbor_order`` lists the atom's neighbours in the order they appear
        in the output, used to assign tetrahedral chirality tags.
        """
        z = self.graph.atomic_numbers[atom_idx]
        symbol = ELEMENT_SYMBOLS.get(z)
        if symbol is None:
            try:
                symbol = Element.from_atomic_number(int(z)).symbol
            except ValueError:
                symbol = f"#{z}"

        is_aromatic = self.aromatic_atoms[atom_idx]

        # Check for stereochemistry
        stereo_str = ""
        if atom_idx in self.stereocenters and neighbor_order is not None:
            stereo_str = self._chiral_tag(atom_idx, neighbor_order)

        # Get formal charge
        charge = self.graph.formal_charges[atom_idx]
        charge_str = ""
        if charge > 0:
            charge_str = "+" if charge == 1 else f"+{charge}"
        elif charge < 0:
            charge_str = "-" if charge == -1 else f"{charge}"

        # Determine if brackets are needed
        needs_brackets = False

        # Atoms not in organic subset need brackets
        if symbol not in ORGANIC_SUBSET:
            needs_brackets = True

        # Stereochemistry needs brackets
        if stereo_str:
            needs_brackets = True

        # Charged atoms need brackets
        if charge != 0:
            needs_brackets = True

        # Aromatic atoms - convert to lowercase
        if is_aromatic and symbol.upper() in ["C", "N", "O", "S", "P"]:
            symbol = symbol.lower()

        # Calculate hydrogen count and check if brackets needed
        if is_aromatic:
            # For aromatic atoms, use aromatic valence conventions
            # Aromatic carbon with 2 ring bonds expects 1 H, etc.
            aromatic_valence = self._aromatic_valence(z)
            actual_bonds = self._count_aromatic_bonds(atom_idx)
            expected_h = max(0, aromatic_valence - actual_bonds)
            # Count explicit H neighbors that were skipped
            explicit_h = sum(
                1
                for n in self.graph.neighbors(atom_idx)
                if self.graph.atomic_numbers[n] == 1 and n in self.skip_atoms
            )
            total_h = expected_h  # Use expected H for aromatic atoms
            # Only need brackets if we have explicit H that differs
            if explicit_h > 0 and explicit_h != expected_h:
                total_h = explicit_h
                needs_brackets = True
        else:
            # For non-aromatic atoms, use standard valence model
            # Count explicit H neighbors (hydrogens in skip_atoms that are bonded to this atom)
            explicit_h = sum(
                1
                for n in self.graph.neighbors(atom_idx)
                if self.graph.atomic_numbers[n] == 1 and n in self.skip_atoms
            )

            # Calculate expected H based on valence, charge, and non-H bonds
            # Charge affects effective valence: + charge adds capacity, - charge removes
            # N+ (charge +1) has valence 4, O- (charge -1) has valence 1
            default_valence = self._default_valence(z, is_aromatic)
            actual_non_h_bonds = self._count_bonds(
                atom_idx
            )  # this excludes H in skip_atoms

            # total_h is how many H are attached (explicit) + how many should be implicit
            # For atoms in brackets, we display the actual H count
            # For atoms outside brackets, we rely on implicit H conventions
            total_h = explicit_h  # start with explicit H from skip_atoms

            # Expected H based on valence (none invented for elements without
            # a defined valence, e.g. metals)
            if default_valence is None:
                default_valence = 0
                expected_h = 0
            else:
                # positive charge adds, negative removes
                effective_valence = default_valence + charge
                expected_h = max(0, effective_valence - actual_non_h_bonds - explicit_h)
            total_h += expected_h

            # Check if brackets needed due to unexpected H count
            standard_expected_h = max(
                0, default_valence - actual_non_h_bonds - explicit_h
            )
            if (
                symbol.upper() in ORGANIC_SUBSET
                and total_h != standard_expected_h + explicit_h
            ):
                needs_brackets = True

        # Build atom string
        if needs_brackets:
            h_str = ""
            if total_h == 1:
                h_str = "H"
            elif total_h > 1:
                h_str = f"H{total_h}"
            return f"[{symbol}{stereo_str}{h_str}{charge_str}]"

        return symbol

    def _default_valence(self, atomic_num: int, aromatic: bool = False) -> int | None:
        """Get default valence for an element (None if not in the organic subset)."""
        valences = {
            5: 3,  # B
            6: 4,  # C
            7: 3,  # N
            8: 2,  # O
            9: 1,  # F
            15: 3,  # P
            16: 2,  # S
            17: 1,  # Cl
            35: 1,  # Br
            53: 1,  # I
        }
        return valences.get(atomic_num)

    def _aromatic_valence(self, atomic_num: int) -> int:
        """Get aromatic valence for an element (bonds in aromatic system)."""
        # Aromatic valence = number of bonds expected in aromatic context
        aromatic_valences = {
            5: 2,  # b - boron in aromatic
            6: 3,  # c - aromatic carbon has 3 bonds (2 in ring + 1 H or substituent)
            7: 2,  # n - pyridine-like nitrogen
            8: 2,  # o - furan-like oxygen
            15: 2,  # p
            16: 2,  # s - thiophene-like sulfur
        }
        return aromatic_valences.get(atomic_num, 3)

    def _count_aromatic_bonds(self, atom_idx: int) -> int:
        """Count bonds for an aromatic atom (each aromatic bond counts as 1)."""
        count = 0
        for neighbor in self.graph.neighbors(atom_idx):
            if neighbor in self.skip_atoms:
                continue
            count += 1  # Each bond counts as 1 in aromatic context
        return count

    def _count_bonds(self, atom_idx: int) -> int:
        """Count total bond order for an atom (excluding H in skip_atoms)."""
        total = 0
        for neighbor in self.graph.neighbors(atom_idx):
            if neighbor in self.skip_atoms:
                continue
            order = get_bond_order(self.graph, atom_idx, neighbor)
            # Treat aromatic bonds as 1.5
            bond_key = (min(atom_idx, neighbor), max(atom_idx, neighbor))
            if bond_key in self.aromatic_bonds:
                total += 1.5
            else:
                total += order
        return int(round(total))

    def _bond_symbol(self, atom_i: int, atom_j: int) -> str:
        """Generate the SMILES bond symbol."""
        # Check for aromatic bond
        bond_key = (min(atom_i, atom_j), max(atom_i, atom_j))
        if bond_key in self.aromatic_bonds:
            # Aromatic bonds are implicit between aromatic atoms
            if self.aromatic_atoms[atom_i] and self.aromatic_atoms[atom_j]:
                return ""
            return ":"

        # Get bond order from geometry
        order = get_bond_order(self.graph, atom_i, atom_j)

        # Override bond order based on formal charges and valence rules
        # This handles cases like nitramines where N-N bond is short but should be single
        order = self._adjust_bond_order_for_charges(atom_i, atom_j, order)

        # E/Z stereo is encoded on the adjacent single bonds, not the double bond
        if 1.9 <= order <= 2.1:
            return "="

        if 2.9 <= order <= 3.1:
            return "#"

        # Directional single bond next to a stereo double bond
        direction = self.bond_directions.get((atom_i, atom_j))
        if direction is None and (atom_j, atom_i) in self.bond_directions:
            direction = -self.bond_directions[(atom_j, atom_i)]
        if direction is not None:
            return "/" if direction > 0 else "\\"

        # Single bond (implicit)
        return ""

    def _adjust_bond_order_for_charges(
        self, atom_i: int, atom_j: int, order: float
    ) -> float:
        """
        Adjust bond order based on formal charges and valence rules.

        For charged atoms, use valence constraints to determine correct bond order
        rather than relying solely on geometry.
        """
        charge_i = self.graph.formal_charges[atom_i]
        charge_j = self.graph.formal_charges[atom_j]

        # If neither atom is charged, use geometry-based order
        if charge_i == 0 and charge_j == 0:
            return order

        z_i = self.graph.atomic_numbers[atom_i]
        z_j = self.graph.atomic_numbers[atom_j]

        # N-N+ bond (nitramine): should be single
        # Neutral N (valence 3) bonded to N+ (valence 4)
        if z_i == 7 and z_j == 7:
            if (charge_i == 0 and charge_j == 1) or (charge_i == 1 and charge_j == 0):
                return 1.0

        # N+ to O- bond: should be single
        # N+ to neutral O: should be double
        if (z_i == 7 and z_j == 8) or (z_i == 8 and z_j == 7):
            n_idx = atom_i if z_i == 7 else atom_j
            o_idx = atom_j if z_i == 7 else atom_i
            n_charge = self.graph.formal_charges[n_idx]
            o_charge = self.graph.formal_charges[o_idx]

            if n_charge == 1:  # N+
                if o_charge == -1:  # O-
                    return 1.0  # Single bond
                elif o_charge == 0:  # Neutral O
                    return 2.0  # Double bond

        # C to O- bond: should be single (in carboxylate, the C-O- is single)
        if (z_i == 6 and z_j == 8) or (z_i == 8 and z_j == 6):
            o_idx = atom_i if z_i == 8 else atom_j
            o_charge = self.graph.formal_charges[o_idx]
            if o_charge == -1:
                return 1.0

        return order


def smiles_from_molecule(molecule, canonical: bool = True) -> str:
    """
    Generate SMILES from a Molecule object.

    Convenience function that creates a MolecularGraph and generates SMILES.

    Args:
        molecule: A Molecule object from chmpy.
        canonical: If True, generate canonical SMILES (default True).

    Returns:
        SMILES string.
    """
    from .adjacency import MolecularGraph

    graph = MolecularGraph.from_molecule(molecule)
    return to_smiles(graph, canonical=canonical)

"""
Atomic bond graph for crystal structures.

This module provides AtomicBondGraph for representing covalent bonds in a
crystal with full symmetry awareness. Editing the asymmetric unit bond list
automatically affects all symmetry-equivalent bonds.

Key features:
- Build bond graph from crystal connectivity
- Add/remove bonds with automatic symmetry propagation
- Query bonds at atoms, unique bonds with multiplicities
- Efficient storage: only asymmetric unit representatives are stored
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import numpy as np
from scipy.spatial import cKDTree as KDTree

from .symmetric_graph import (
    AlgebraicEdge,
    AlgebraicVertexRef,
    CosetTable,
    SymmetricGraph,
    apply_symop_to_vertex,
    canonical_edge_representative,
    compute_edge_orbit_size,
    normalize_edge,
)

if TYPE_CHECKING:
    from .crystal import Crystal
    from .space_group_table import SpaceGroupTable

LOG = logging.getLogger(__name__)


# Type aliases for clarity
AlgebraicAtomRef = AlgebraicVertexRef
AlgebraicBondIndex = AlgebraicEdge


@dataclass
class AtomicCosetTable(CosetTable):
    """
    Coset table specialized for atoms in a crystal.

    Maps symop indices to unit cell atom indices, handling atoms on
    special positions (Wyckoff sites with non-trivial site symmetry).

    Attributes:
        _n_symops: Number of space group operations
        _n_atoms: Number of atoms in unit cell
        symop_to_atom: (n_symops, n_asym) -> uc_atom_idx mapping
        atom_to_canonical_symop: (n_atoms,) canonical symop for each atom
        _site_symmetry_order: (n_atoms,) order of site symmetry
    """

    _n_symops: int
    _n_atoms: int
    symop_to_atom: np.ndarray  # (n_symops, n_asym_atoms) -> uc_atom_idx
    atom_to_canonical_symop: np.ndarray  # (n_uc_atoms,) -> symop_idx
    _site_symmetry_order: np.ndarray  # (n_uc_atoms,)

    @property
    def n_symops(self) -> int:
        return self._n_symops

    @property
    def n_vertices(self) -> int:
        return self._n_atoms

    def normalize_symop(self, symop_idx: int) -> tuple[int, int]:
        """
        For atoms, we do NOT normalize symops.

        Unlike molecules where all atoms move together (so one canonical symop
        per molecule makes sense), individual atoms can be on different Wyckoff
        positions with different site symmetries. Normalizing symops would
        incorrectly conflate different atoms.

        This method returns the symop unchanged. The symop_to_atom mapping
        should be used directly to find which UC atom a (symop, asym) pair
        generates.

        Returns:
            Tuple of (-1, symop_idx) - atom_idx is -1 since it depends on asym_idx
        """
        return -1, symop_idx

    def site_symmetry_order(self, vertex_idx: int) -> int:
        return int(self._site_symmetry_order[vertex_idx])

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        sg_table: SpaceGroupTable,
    ) -> AtomicCosetTable:
        """
        Build the atomic coset table from crystal structure.

        For each symop g, we determine which UC atom results when g is
        applied to each asymmetric unit atom. This is done by applying g
        to the atom position and finding the closest UC atom position.

        Args:
            crystal: The crystal structure
            sg_table: SpaceGroupTable for the space group

        Returns:
            AtomicCosetTable with symop-to-atom mappings
        """
        uc_dict = crystal.unit_cell_atoms()
        uc_frac = uc_dict["frac_pos"]
        uc_asym = uc_dict["asym_atom"]
        n_atoms = len(uc_frac)
        n_symops = sg_table.n_ops
        n_asym = crystal.nsites

        # Build symop_to_atom mapping
        # symop_to_atom[symop_idx, asym_idx] = uc_atom_idx
        symop_to_atom = np.full((n_symops, n_asym), -1, dtype=np.int32)

        # Get asymmetric unit positions
        asym_pos = crystal.site_positions

        for symop_idx in range(n_symops):
            R = sg_table.rotations[symop_idx]
            t = sg_table.translations[symop_idx]

            for asym_idx in range(n_asym):
                # Apply symop to asymmetric atom position
                pos = asym_pos[asym_idx]
                transformed = R @ pos + t

                # Wrap to [0, 1)
                transformed = np.mod(transformed + 10.0, 1.0)

                # Find closest UC atom position (handling PBC)
                best_atom = -1
                best_dist = float("inf")

                for atom_idx in range(n_atoms):
                    diff = transformed - uc_frac[atom_idx]
                    diff = diff - np.round(diff)  # Minimum image
                    dist = np.sum(diff**2)

                    if dist < best_dist:
                        best_dist = dist
                        best_atom = atom_idx

                if best_dist < 0.01:  # Tolerance
                    symop_to_atom[symop_idx, asym_idx] = best_atom
                else:
                    LOG.warning(
                        f"Symop {symop_idx} transforms atom {asym_idx} to "
                        f"{transformed}, no close match found (dist={best_dist:.4f})"
                    )

        # For each UC atom, find the canonical symop (smallest index)
        atom_to_canonical_symop = np.full(n_atoms, -1, dtype=np.int32)
        site_symmetry_order = np.zeros(n_atoms, dtype=np.int32)

        for symop_idx in range(n_symops):
            for asym_idx in range(n_asym):
                atom_idx = symop_to_atom[symop_idx, asym_idx]
                if atom_idx >= 0:
                    site_symmetry_order[atom_idx] += 1
                    if atom_to_canonical_symop[atom_idx] < 0:
                        atom_to_canonical_symop[atom_idx] = symop_idx

        return cls(
            _n_symops=n_symops,
            _n_atoms=n_atoms,
            symop_to_atom=symop_to_atom,
            atom_to_canonical_symop=atom_to_canonical_symop,
            _site_symmetry_order=site_symmetry_order,
        )


@dataclass
class AtomicBondGraph(SymmetricGraph):
    """
    Bond graph for a crystal structure with symmetry awareness.

    This class represents covalent bonds as a symmetric graph where only
    asymmetric unit representatives are stored. Adding or removing a bond
    automatically affects all symmetry-equivalent bonds.

    Attributes:
        crystal: The crystal structure
        n_asym_atoms: Number of atoms in asymmetric unit
        n_uc_atoms: Number of atoms in unit cell
        uc_atom_to_asym: Mapping from UC atom index to asym atom index
        asym_to_uc_atoms: Mapping from asym atom index to list of UC atom indices
    """

    crystal: Crystal
    n_asym_atoms: int
    n_uc_atoms: int
    uc_atom_to_asym: np.ndarray
    asym_to_canonical_symop: np.ndarray

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        tolerance: float = 0.4,
    ) -> AtomicBondGraph:
        """
        Build a bond graph from crystal connectivity.

        Uses covalent radii to determine bonding: atoms are bonded if
        their distance is less than sum of covalent radii plus tolerance.

        Args:
            crystal: The crystal structure
            tolerance: Bonding tolerance (default 0.4 Angstroms)

        Returns:
            AtomicBondGraph with bonds from crystal connectivity
        """
        from .space_group_table import SpaceGroupTable

        sg_table = SpaceGroupTable.from_space_group(crystal.space_group)
        coset_table = AtomicCosetTable.from_crystal(crystal, sg_table)

        # Get unit cell atoms and connectivity
        uc_dict = crystal.unit_cell_atoms()
        uc_asym = uc_dict["asym_atom"]
        uc_symop_codes = uc_dict["symop"]  # These are integer codes, not indices
        n_uc_atoms = len(uc_asym)
        n_asym_atoms = crystal.nsites

        # Convert symop integer codes to indices
        symops = crystal.space_group.symmetry_operations
        symop_code_to_idx = {s.integer_code: i for i, s in enumerate(symops)}
        uc_symop_idx = np.array(
            [symop_code_to_idx.get(code, 0) for code in uc_symop_codes],
            dtype=np.int32,
        )

        # Build asym_to_canonical_symop (as indices, not codes)
        asym_to_canonical_symop = np.zeros(n_asym_atoms, dtype=np.int32)
        for asym_idx in range(n_asym_atoms):
            # Find the first UC atom for this asym atom
            for uc_idx in range(n_uc_atoms):
                if uc_asym[uc_idx] == asym_idx:
                    asym_to_canonical_symop[asym_idx] = uc_symop_idx[uc_idx]
                    break

        # Get connectivity
        uc_graph, edge_cells = crystal.unit_cell_connectivity(tolerance=tolerance)

        # Convert bonds to algebraic form
        asym_edges = set()

        # Process each bond
        for (i, j), cell in edge_cells.items():
            # Get asym indices and symops for both atoms
            asym_i = uc_asym[i]
            asym_j = uc_asym[j]
            symop_i = uc_symop_idx[i]
            symop_j = uc_symop_idx[j]

            # Create algebraic atom references
            # Atom i is at cell (0,0,0) by convention
            atom_a = AlgebraicAtomRef(
                asym_idx=asym_i,
                symop_idx=symop_i,
                cell=(0, 0, 0),
            )

            # Atom j may be in a different cell
            atom_b = AlgebraicAtomRef(
                asym_idx=asym_j,
                symop_idx=symop_j,
                cell=cell,
            )

            # Create edge and find canonical representative
            edge = AlgebraicEdge.create(atom_a, atom_b)
            canonical = canonical_edge_representative(edge, sg_table, coset_table)
            asym_edges.add(canonical)

        return cls(
            sg_table=sg_table,
            coset_table=coset_table,
            n_asym_vertices=n_asym_atoms,
            asym_edges=asym_edges,
            crystal=crystal,
            n_asym_atoms=n_asym_atoms,
            n_uc_atoms=n_uc_atoms,
            uc_atom_to_asym=uc_asym,
            asym_to_canonical_symop=asym_to_canonical_symop,
        )

    def add_bond(
        self, atom_a: AlgebraicAtomRef, atom_b: AlgebraicAtomRef
    ) -> bool:
        """
        Add a bond between two atoms (and all equivalent bonds).

        Args:
            atom_a: First atom
            atom_b: Second atom

        Returns:
            True if the bond was newly added
        """
        return self.add_edge(atom_a, atom_b)

    def remove_bond(
        self, atom_a: AlgebraicAtomRef, atom_b: AlgebraicAtomRef
    ) -> bool:
        """
        Remove the bond between two atoms (and all equivalent bonds).

        Args:
            atom_a: First atom
            atom_b: Second atom

        Returns:
            True if the bond was removed
        """
        return self.remove_edge(atom_a, atom_b)

    def has_bond(
        self, atom_a: AlgebraicAtomRef, atom_b: AlgebraicAtomRef
    ) -> bool:
        """
        Check if a bond exists between two atoms.

        Args:
            atom_a: First atom
            atom_b: Second atom

        Returns:
            True if the bond exists
        """
        return self.has_edge(atom_a, atom_b)

    def bonds_at_atom(self, atom: AlgebraicAtomRef) -> list[AlgebraicBondIndex]:
        """
        Get all bonds incident to a specific atom.

        Args:
            atom: The atom to query

        Returns:
            List of bonds incident to this atom
        """
        return self.edges_at_vertex(atom)

    def unique_bonds(self) -> list[tuple[AlgebraicBondIndex, int]]:
        """
        Get unique bonds with their multiplicities.

        Returns:
            List of (bond, multiplicity) tuples
        """
        return self.unique_edges_with_multiplicities()

    def all_bonds(self) -> Iterator[AlgebraicBondIndex]:
        """
        Iterate over all bonds (full orbit expansion).

        Yields:
            All bonds in all orbits
        """
        return self.all_edges()

    def n_unique_bonds(self) -> int:
        """Return the number of unique bonds."""
        return self.n_unique_edges()

    def n_total_bonds(self) -> int:
        """Return the total number of bonds (sum of multiplicities)."""
        return self.n_total_edges()

    def make_atom_ref(
        self,
        asym_idx: int,
        symop_idx: int | None = None,
        cell: tuple[int, int, int] = (0, 0, 0),
    ) -> AlgebraicAtomRef:
        """
        Create an atom reference from asymmetric unit index.

        Args:
            asym_idx: Index in asymmetric unit
            symop_idx: Symop index (default: canonical symop for this atom)
            cell: Cell offset (default: origin)

        Returns:
            AlgebraicAtomRef for this atom
        """
        if symop_idx is None:
            symop_idx = int(self.asym_to_canonical_symop[asym_idx])

        return AlgebraicAtomRef(
            asym_idx=asym_idx,
            symop_idx=symop_idx,
            cell=cell,
        )

    def get_bond_atom_types(
        self, bond: AlgebraicBondIndex
    ) -> tuple[int, int]:
        """
        Get the atomic numbers for both atoms in a bond.

        Args:
            bond: The bond to query

        Returns:
            Tuple of (atomic_number_a, atomic_number_b)
        """
        asym_a = bond.src.asym_idx
        asym_b = bond.dst.asym_idx
        atoms = self.crystal.site_atoms
        return int(atoms[asym_a]), int(atoms[asym_b])

    def bonds_by_type(
        self, elem_a: int, elem_b: int
    ) -> list[tuple[AlgebraicBondIndex, int]]:
        """
        Get unique bonds between specific element types.

        Args:
            elem_a: Atomic number of first element
            elem_b: Atomic number of second element

        Returns:
            List of (bond, multiplicity) tuples for matching bonds
        """
        result = []
        for bond, mult in self.unique_bonds():
            type_a, type_b = self.get_bond_atom_types(bond)
            if (type_a == elem_a and type_b == elem_b) or (
                type_a == elem_b and type_b == elem_a
            ):
                result.append((bond, mult))
        return result

    def __repr__(self) -> str:
        return (
            f"AtomicBondGraph(n_asym_atoms={self.n_asym_atoms}, "
            f"n_unique_bonds={self.n_unique_bonds()}, "
            f"n_total_bonds={self.n_total_bonds()})"
        )

"""
Dimer indexing and symmetry equivalence for crystal structures.

This module provides data structures for tracking molecular dimers
(pairs of molecules) and determining their symmetry equivalence.

There are two approaches:
1. Distance-based: Group dimers by matching distances (with tolerance)
2. Algebraic: Use group theory - dimers equivalent if a space group operation
   maps one to the other (no tolerance needed, exact algebraic comparison)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree as KDTree

if TYPE_CHECKING:
    from .crystal import Crystal
    from .orbit import MoleculeOrbitTable
    from .space_group_table import SpaceGroupTable

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoleculeRef:
    """
    Reference to a molecule instance in the crystal.

    Attributes:
        asym_mol_idx: Index of the unique molecule type
        instance_idx: Index of this instance in MoleculeOrbitTable
        cell_offset: (h, k, l) periodic cell offset from reference cell
    """

    asym_mol_idx: int
    instance_idx: int
    cell_offset: tuple[int, int, int]

    def _key(self) -> tuple:
        """Return comparison key for ordering."""
        return (self.asym_mol_idx, self.instance_idx, self.cell_offset)

    def __lt__(self, other: MoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() < other._key()

    def __le__(self, other: MoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() <= other._key()

    def __gt__(self, other: MoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() > other._key()

    def __ge__(self, other: MoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() >= other._key()


@dataclass(frozen=True)
class AlgebraicMoleculeRef:
    """
    Algebraic reference to a molecule instance in the crystal.

    Unlike MoleculeRef which uses instance_idx (position in unit cell),
    this uses symop_idx which is the index of the symmetry operation that
    generated this molecule from the asymmetric unit. This enables purely
    algebraic symmetry operations using the group multiplication table.

    Attributes:
        asym_mol_idx: Index of the unique molecule type in asymmetric unit
        symop_idx: Index into space group's symmetry operation list
        cell: (h, k, l) periodic cell offset from reference cell
    """

    asym_mol_idx: int
    symop_idx: int
    cell: tuple[int, int, int]

    def _key(self) -> tuple:
        """Return comparison key for ordering."""
        return (self.asym_mol_idx, self.symop_idx, self.cell)

    def __lt__(self, other: AlgebraicMoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() < other._key()

    def __le__(self, other: AlgebraicMoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() <= other._key()

    def __gt__(self, other: AlgebraicMoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() > other._key()

    def __ge__(self, other: AlgebraicMoleculeRef) -> bool:
        """Lexicographic ordering for canonical form."""
        return self._key() >= other._key()


def apply_symop_to_algebraic_mol(
    mol: AlgebraicMoleculeRef,
    symop_idx: int,
    sg_table: SpaceGroupTable,
    coset_table: MolecularCosetTable | None = None,
) -> AlgebraicMoleculeRef:
    """
    Apply a space group operation to an algebraic molecule reference.

    This is a purely algebraic operation using the group multiplication table:
    - The new symop index is computed via the Cayley table
    - The cell offset is transformed by the rotation part of the symop
    - If coset_table is provided, normalizes the symop to the canonical one
      for the resulting molecule (important for Z' < 1 cases)

    Args:
        mol: The molecule reference to transform
        symop_idx: Index of the symop to apply
        sg_table: The precomputed space group table
        coset_table: Optional MolecularCosetTable for symop normalization

    Returns:
        Transformed AlgebraicMoleculeRef
    """
    # New generator: g_new = g_symop ∘ g_mol.symop
    # Using mult_table: mult[mol.symop_idx, symop_idx] = new_symop_idx
    new_symop_idx = sg_table.mult_table[mol.symop_idx, symop_idx]

    # Cell offset transforms by the rotation part of the applied symop
    # Note: The rotation acts on fractional coordinates, which are also
    # how cell offsets are represented
    R = sg_table.rotations[symop_idx]
    old_cell = np.array(mol.cell, dtype=np.float64)
    new_cell_float = R @ old_cell

    # Cell offsets should be integers (lattice translations)
    new_cell = tuple(np.round(new_cell_float).astype(int))

    # Normalize symop if coset table is provided (for Z' < 1)
    if coset_table is not None:
        _, canonical_symop = coset_table.normalize_symop(new_symop_idx)
        new_symop_idx = canonical_symop

    return AlgebraicMoleculeRef(
        asym_mol_idx=mol.asym_mol_idx,
        symop_idx=new_symop_idx,
        cell=new_cell,
    )


@dataclass(frozen=True)
class DimerIndex:
    """
    Canonical index for a molecular dimer (pair of molecules).

    The dimer is stored in canonical form where mol_a <= mol_b.
    This ensures each dimer has a unique representation.

    Attributes:
        mol_a: Reference to first molecule (always <= mol_b)
        mol_b: Reference to second molecule
        distance: Center-to-center distance between molecules
    """

    mol_a: MoleculeRef
    mol_b: MoleculeRef
    distance: float

    def __post_init__(self):
        # Verify canonical ordering
        if self.mol_a > self.mol_b:
            # Need to save values before swapping
            old_a, old_b = self.mol_a, self.mol_b
            object.__setattr__(self, "mol_a", old_b)
            object.__setattr__(self, "mol_b", old_a)

    @classmethod
    def create(
        cls,
        mol_a: MoleculeRef,
        mol_b: MoleculeRef,
        distance: float,
    ) -> DimerIndex:
        """Create a DimerIndex with automatic canonical ordering."""
        if mol_a <= mol_b:
            return cls(mol_a=mol_a, mol_b=mol_b, distance=distance)
        else:
            return cls(mol_a=mol_b, mol_b=mol_a, distance=distance)

    def is_homodimer(self) -> bool:
        """Check if both molecules are of the same type."""
        return self.mol_a.asym_mol_idx == self.mol_b.asym_mol_idx

    def involves_molecule_type(self, asym_mol_idx: int) -> bool:
        """Check if this dimer involves a specific molecule type."""
        return (
            self.mol_a.asym_mol_idx == asym_mol_idx
            or self.mol_b.asym_mol_idx == asym_mol_idx
        )


@dataclass(frozen=True)
class AlgebraicDimerIndex:
    """
    Algebraic index for a molecular dimer using symop indices.

    This class represents a dimer using algebraic molecule references
    (symop indices rather than instance indices). This enables exact
    equivalence checking via group theory without numerical tolerances.

    The dimer is stored in canonical form where mol_a <= mol_b.

    Attributes:
        mol_a: Algebraic reference to first molecule (always <= mol_b)
        mol_b: Algebraic reference to second molecule
    """

    mol_a: AlgebraicMoleculeRef
    mol_b: AlgebraicMoleculeRef

    def __post_init__(self):
        # Verify canonical ordering
        if self.mol_a > self.mol_b:
            old_a, old_b = self.mol_a, self.mol_b
            object.__setattr__(self, "mol_a", old_b)
            object.__setattr__(self, "mol_b", old_a)

    @classmethod
    def create(
        cls,
        mol_a: AlgebraicMoleculeRef,
        mol_b: AlgebraicMoleculeRef,
    ) -> AlgebraicDimerIndex:
        """Create an AlgebraicDimerIndex with automatic canonical ordering."""
        if mol_a <= mol_b:
            return cls(mol_a=mol_a, mol_b=mol_b)
        else:
            return cls(mol_a=mol_b, mol_b=mol_a)

    def is_homodimer(self) -> bool:
        """Check if both molecules are of the same type."""
        return self.mol_a.asym_mol_idx == self.mol_b.asym_mol_idx

    def involves_molecule_type(self, asym_mol_idx: int) -> bool:
        """Check if this dimer involves a specific molecule type."""
        return (
            self.mol_a.asym_mol_idx == asym_mol_idx
            or self.mol_b.asym_mol_idx == asym_mol_idx
        )

    def _key(self) -> tuple:
        """Return comparison key."""
        return (self.mol_a._key(), self.mol_b._key())

    def __hash__(self):
        return hash(self._key())


def apply_symop_to_algebraic_dimer(
    dimer: AlgebraicDimerIndex,
    symop_idx: int,
    sg_table: SpaceGroupTable,
    coset_table: MolecularCosetTable | None = None,
) -> AlgebraicDimerIndex:
    """
    Apply a space group operation to an algebraic dimer.

    This transforms both molecules in the dimer and returns the
    result in canonical form (mol_a <= mol_b).

    Args:
        dimer: The dimer to transform
        symop_idx: Index of the symop to apply
        sg_table: The precomputed space group table
        coset_table: Optional MolecularCosetTable for symop normalization

    Returns:
        Transformed AlgebraicDimerIndex in canonical form
    """
    new_a = apply_symop_to_algebraic_mol(dimer.mol_a, symop_idx, sg_table, coset_table)
    new_b = apply_symop_to_algebraic_mol(dimer.mol_b, symop_idx, sg_table, coset_table)
    return AlgebraicDimerIndex.create(new_a, new_b)


def normalize_algebraic_dimer(
    dimer: AlgebraicDimerIndex,
    sg_table: SpaceGroupTable,
) -> AlgebraicDimerIndex:
    """
    Normalize an algebraic dimer to a canonical representative.

    Normalization ensures:
    1. mol_a <= mol_b (lexicographic ordering)
    2. The "smaller" molecule is at cell (0,0,0)

    This is necessary because dimers that differ only by a lattice translation
    are equivalent.

    Args:
        dimer: The dimer to normalize
        sg_table: The precomputed space group table

    Returns:
        Normalized AlgebraicDimerIndex
    """
    mol_a = dimer.mol_a
    mol_b = dimer.mol_b

    # Determine which molecule should be mol_a (the smaller one)
    # Compare by (asym_mol_idx, symop_idx) first, ignoring cell
    key_a = (mol_a.asym_mol_idx, mol_a.symop_idx)
    key_b = (mol_b.asym_mol_idx, mol_b.symop_idx)

    if key_a > key_b:
        # Swap
        mol_a, mol_b = mol_b, mol_a

    # If same molecule type and symop, use cell to break tie
    if key_a == key_b:
        if mol_a.cell > mol_b.cell:
            mol_a, mol_b = mol_b, mol_a

    # Now translate so mol_a is at (0,0,0)
    shift = mol_a.cell

    new_a = AlgebraicMoleculeRef(
        asym_mol_idx=mol_a.asym_mol_idx,
        symop_idx=mol_a.symop_idx,
        cell=(0, 0, 0),
    )

    new_b_cell = tuple(c - s for c, s in zip(mol_b.cell, shift, strict=False))
    new_b = AlgebraicMoleculeRef(
        asym_mol_idx=mol_b.asym_mol_idx,
        symop_idx=mol_b.symop_idx,
        cell=new_b_cell,
    )

    # Use frozen dataclass directly instead of create() to avoid re-swapping
    return AlgebraicDimerIndex(mol_a=new_a, mol_b=new_b)


@dataclass
class DimerList:
    """
    List of all unique dimers within a cutoff distance.

    Attributes:
        dimers: List of DimerIndex objects
        n_dimers: Total number of unique dimers
        cutoff: Distance cutoff used
    """

    dimers: list[DimerIndex]
    n_dimers: int
    cutoff: float

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        mol_orbit: MoleculeOrbitTable,
        cutoff: float = 10.0,
    ) -> DimerList:
        """
        Build a list of all unique dimers within cutoff distance.

        Uses molecule centroids to determine neighbor pairs.

        Args:
            crystal: The crystal structure
            mol_orbit: MoleculeOrbitTable with molecule instances
            cutoff: Maximum center-to-center distance

        Returns:
            DimerList with all unique dimers
        """
        uc = crystal.unit_cell

        # Get molecule centroids
        centroids_frac = mol_orbit.instance_centroids_frac
        centroids_cart = mol_orbit.instance_centroids_cart
        n_mols = mol_orbit.n_uc_molecules

        # Determine how many cells to search
        cell_lengths = np.array(uc.lengths)
        n_cells = np.ceil(cutoff / cell_lengths).astype(int) + 1

        # Generate all cell translations
        h_range = np.arange(-n_cells[0], n_cells[0] + 1)
        k_range = np.arange(-n_cells[1], n_cells[1] + 1)
        l_range = np.arange(-n_cells[2], n_cells[2] + 1)

        # Build extended structure
        all_centroids = []
        all_cells = []
        all_mol_indices = []

        for h in h_range:
            for k in k_range:
                for l in l_range:
                    cell_offset = np.array([h, k, l])
                    shifted_frac = centroids_frac + cell_offset
                    shifted_cart = uc.to_cartesian(shifted_frac)
                    all_centroids.append(shifted_cart)
                    all_cells.extend([(h, k, l)] * n_mols)
                    all_mol_indices.extend(range(n_mols))

        all_centroids = np.vstack(all_centroids)
        all_cells = [tuple(c) for c in all_cells]
        all_mol_indices = np.array(all_mol_indices)

        # Build KDTree
        tree = KDTree(all_centroids)

        # Find all pairs within cutoff from reference cell (cell offset 0,0,0)
        dimers = []
        seen_pairs = set()

        for mol_i in range(n_mols):
            ref_centroid = centroids_cart[mol_i]
            neighbors = tree.query_ball_point(ref_centroid, cutoff)

            for j_extended in neighbors:
                mol_j = all_mol_indices[j_extended]
                cell_j = all_cells[j_extended]
                dist = np.linalg.norm(all_centroids[j_extended] - ref_centroid)

                # Skip self
                if mol_i == mol_j and cell_j == (0, 0, 0):
                    continue

                # Skip very small distances (shouldn't happen but safety check)
                if dist < 1e-10:
                    continue

                # Create molecule references
                asym_mol_i = mol_orbit.unit_cell_instances[mol_i].asym_mol_idx
                asym_mol_j = mol_orbit.unit_cell_instances[mol_j].asym_mol_idx

                ref_a = MoleculeRef(
                    asym_mol_idx=asym_mol_i,
                    instance_idx=mol_i,
                    cell_offset=(0, 0, 0),
                )
                ref_b = MoleculeRef(
                    asym_mol_idx=asym_mol_j,
                    instance_idx=mol_j,
                    cell_offset=cell_j,
                )

                # Create canonical dimer
                dimer = DimerIndex.create(ref_a, ref_b, dist)

                # Check for duplicates (considering symmetry)
                pair_key = (
                    dimer.mol_a.asym_mol_idx,
                    dimer.mol_a.instance_idx,
                    dimer.mol_a.cell_offset,
                    dimer.mol_b.asym_mol_idx,
                    dimer.mol_b.instance_idx,
                    dimer.mol_b.cell_offset,
                )

                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    dimers.append(dimer)

        # Sort by distance
        dimers.sort(key=lambda d: d.distance)

        return cls(
            dimers=dimers,
            n_dimers=len(dimers),
            cutoff=cutoff,
        )

    def get_dimers_involving(self, asym_mol_idx: int) -> list[DimerIndex]:
        """Get all dimers involving a specific molecule type."""
        return [d for d in self.dimers if d.involves_molecule_type(asym_mol_idx)]

    def get_homodimers(self) -> list[DimerIndex]:
        """Get all homodimers (same molecule type)."""
        return [d for d in self.dimers if d.is_homodimer()]

    def get_heterodimers(self) -> list[DimerIndex]:
        """Get all heterodimers (different molecule types)."""
        return [d for d in self.dimers if not d.is_homodimer()]

    def group_by_distance(
        self, tolerance: float = 0.01
    ) -> list[tuple[float, list[DimerIndex]]]:
        """
        Group dimers by distance within tolerance.

        Returns:
            List of (distance, dimers) tuples, sorted by distance
        """
        if not self.dimers:
            return []

        groups = []
        current_dist = self.dimers[0].distance
        current_group = [self.dimers[0]]

        for dimer in self.dimers[1:]:
            if abs(dimer.distance - current_dist) <= tolerance:
                current_group.append(dimer)
            else:
                groups.append((current_dist, current_group))
                current_dist = dimer.distance
                current_group = [dimer]

        groups.append((current_dist, current_group))
        return groups

    def unique_distances(self, tolerance: float = 0.01) -> list[float]:
        """Get list of unique dimer distances within tolerance."""
        groups = self.group_by_distance(tolerance)
        return [d for d, _ in groups]


@dataclass
class DimerEquivalenceClass:
    """
    A group of symmetry-equivalent dimers.

    All dimers in this class are related by space group symmetry operations.

    Attributes:
        representative: One representative dimer from the class
        members: All dimers in this equivalence class
        multiplicity: Number of equivalent dimers
        metadata: Optional metadata (e.g., central/neighbor types for non-permutative)
    """

    representative: DimerIndex
    members: list[DimerIndex]
    multiplicity: int
    metadata: dict = field(default_factory=dict)

    @property
    def distance(self) -> float:
        """Distance of dimers in this class."""
        return self.representative.distance

    @property
    def central_mol_type(self) -> int | None:
        """Central molecule type (for non-permutative enumeration)."""
        return self.metadata.get("central_type")

    @property
    def neighbor_mol_type(self) -> int | None:
        """Neighbor molecule type (for non-permutative enumeration)."""
        return self.metadata.get("neighbor_type")


def find_dimers_around_molecule(
    crystal: Crystal,
    mol_orbit: MoleculeOrbitTable,
    ref_mol_idx: int = 0,
    cutoff: float = 10.0,
) -> list[DimerIndex]:
    """
    Find all dimers surrounding a single reference molecule.

    This is the correct approach for dimer enumeration: pick ONE central
    molecule and find all its neighbors within cutoff.

    Args:
        crystal: The crystal structure
        mol_orbit: MoleculeOrbitTable with molecule instances
        ref_mol_idx: Index of reference molecule in unit cell (default: 0)
        cutoff: Maximum center-to-center distance

    Returns:
        List of DimerIndex objects for all dimers around the reference
    """
    uc = crystal.unit_cell

    # Get molecule centroids
    centroids_frac = mol_orbit.instance_centroids_frac
    centroids_cart = mol_orbit.instance_centroids_cart
    n_mols = mol_orbit.n_uc_molecules

    # Reference molecule centroid
    ref_centroid = centroids_cart[ref_mol_idx]

    # Determine how many cells to search
    cell_lengths = np.array(uc.lengths)
    n_cells = np.ceil(cutoff / cell_lengths).astype(int) + 1

    # Generate all cell translations
    h_range = np.arange(-n_cells[0], n_cells[0] + 1)
    k_range = np.arange(-n_cells[1], n_cells[1] + 1)
    l_range = np.arange(-n_cells[2], n_cells[2] + 1)

    # Build extended structure
    all_centroids = []
    all_cells = []
    all_mol_indices = []

    for h in h_range:
        for k in k_range:
            for l in l_range:
                cell_offset = np.array([h, k, l])
                shifted_frac = centroids_frac + cell_offset
                shifted_cart = uc.to_cartesian(shifted_frac)
                all_centroids.append(shifted_cart)
                all_cells.extend([(h, k, l)] * n_mols)
                all_mol_indices.extend(range(n_mols))

    all_centroids = np.vstack(all_centroids)
    all_mol_indices = np.array(all_mol_indices)

    # Build KDTree
    tree = KDTree(all_centroids)

    # Find all neighbors of the reference molecule
    neighbors = tree.query_ball_point(ref_centroid, cutoff)

    dimers = []
    asym_mol_ref = mol_orbit.unit_cell_instances[ref_mol_idx].asym_mol_idx

    ref_a = MoleculeRef(
        asym_mol_idx=asym_mol_ref,
        instance_idx=ref_mol_idx,
        cell_offset=(0, 0, 0),
    )

    for j_extended in neighbors:
        mol_j = all_mol_indices[j_extended]
        cell_j = all_cells[j_extended]
        dist = np.linalg.norm(all_centroids[j_extended] - ref_centroid)

        # Skip self
        if mol_j == ref_mol_idx and cell_j == (0, 0, 0):
            continue

        # Skip very small distances
        if dist < 1e-10:
            continue

        asym_mol_j = mol_orbit.unit_cell_instances[mol_j].asym_mol_idx

        ref_b = MoleculeRef(
            asym_mol_idx=asym_mol_j,
            instance_idx=mol_j,
            cell_offset=cell_j,
        )

        # Create dimer (NOT canonical form - keep reference as mol_a)
        dimer = DimerIndex(mol_a=ref_a, mol_b=ref_b, distance=dist)
        dimers.append(dimer)

    # Sort by distance
    dimers.sort(key=lambda d: d.distance)
    return dimers


def find_symmetry_unique_dimers(
    crystal: Crystal,
    mol_orbit: MoleculeOrbitTable,
    cutoff: float = 10.0,
    distance_tolerance: float = 0.01,
    permutative: bool = True,
) -> tuple[list[DimerEquivalenceClass], list[tuple[int, DimerIndex]]]:
    """
    Find symmetry-unique dimers within cutoff distance.

    For each symmetry-unique molecule, find all dimers surrounding it and
    group by equivalence. Two dimers are equivalent if they have the same
    separation (within tolerance) - this matches the existing chmpy approach
    using geometric comparison.

    Args:
        crystal: The crystal structure
        mol_orbit: MoleculeOrbitTable with molecule instances
        cutoff: Maximum center-to-center distance
        distance_tolerance: Tolerance for comparing distances
        permutative: If True (default), A-B and B-A dimers are equivalent.
            If False, distinguish dimers by which molecule type is the
            "central" molecule. This matters for heterodimers where you
            care about directionality (e.g., donor-acceptor interactions).

    Returns:
        Tuple of:
        - List of DimerEquivalenceClass objects (unique dimers)
        - List of (unique_idx, dimer) tuples (all dimers mapped to unique)
    """
    # For each unique molecule type, we need ONE reference molecule
    # Use the first instance of each unique molecule type
    n_unique = mol_orbit.n_unique_molecules

    # Map from asym_mol_idx to first instance index in unit cell
    first_instance_for_type = {}
    for i, inst in enumerate(mol_orbit.unit_cell_instances):
        if inst.asym_mol_idx not in first_instance_for_type:
            first_instance_for_type[inst.asym_mol_idx] = i

    unique_dimers = []
    all_dimers_mapped = []  # (unique_idx, dimer) for each dimer

    # Process each unique molecule type
    for asym_mol_idx in range(n_unique):
        ref_mol_idx = first_instance_for_type[asym_mol_idx]

        # Get all dimers around this reference molecule
        dimers = find_dimers_around_molecule(
            crystal, mol_orbit, ref_mol_idx, cutoff
        )

        # Group dimers by equivalence
        for dimer in dimers:
            # For non-permutative, we need to track ordered pairs
            # The "central" molecule type is asym_mol_idx (the one we're iterating from)
            # The "neighbor" molecule type is the other one
            central_type = asym_mol_idx
            neighbor_type = (
                dimer.mol_b.asym_mol_idx
                if dimer.mol_a.instance_idx == ref_mol_idx
                else dimer.mol_a.asym_mol_idx
            )

            # Check if this matches an existing unique dimer
            found_match = False
            for uid, unique_ec in enumerate(unique_dimers):
                # Distance must match
                if abs(dimer.distance - unique_ec.distance) > distance_tolerance:
                    continue

                # For non-permutative, also check ordered pair matches
                if not permutative:
                    unique_ec.metadata.get("ref_mol_idx")
                    rep_central = unique_ec.metadata.get("central_type")
                    rep_neighbor = unique_ec.metadata.get("neighbor_type")

                    if (central_type, neighbor_type) != (rep_central, rep_neighbor):
                        continue

                # Match found - add to this class
                unique_ec.members.append(dimer)
                all_dimers_mapped.append((uid, dimer))
                found_match = True
                break

            if not found_match:
                # New unique dimer
                uid = len(unique_dimers)
                ec = DimerEquivalenceClass(
                    representative=dimer,
                    members=[dimer],
                    multiplicity=1,  # Will be updated later
                )
                # Store metadata for non-permutative matching
                ec.metadata = {
                    "ref_mol_idx": ref_mol_idx,
                    "central_type": central_type,
                    "neighbor_type": neighbor_type,
                }
                unique_dimers.append(ec)
                all_dimers_mapped.append((uid, dimer))

    # Update multiplicities
    for ec in unique_dimers:
        ec.multiplicity = len(ec.members)

    # Sort unique dimers by distance
    # Need to also update the mapping indices
    sorted_indices = sorted(range(len(unique_dimers)), key=lambda i: unique_dimers[i].distance)
    index_mapping = {old: new for new, old in enumerate(sorted_indices)}
    unique_dimers = [unique_dimers[i] for i in sorted_indices]
    all_dimers_mapped = [(index_mapping[uid], d) for uid, d in all_dimers_mapped]

    return unique_dimers, all_dimers_mapped


# ============================================================================
# Algebraic dimer equivalence
#
# This section implements dimer equivalence using pure group theory:
# - MolecularCosetTable: tracks which symops generate which molecules
# - Canonical representatives: unique identifiers for equivalence classes
# - Orbit computation: determines multiplicity as orbit size under group action
# ============================================================================


@dataclass
class MolecularCosetTable:
    """
    Mapping between symop indices and unit cell molecules.

    When Z' < 1, multiple symops can generate the same molecule (they differ
    by the site symmetry). This table tracks which molecule each symop generates
    and provides a canonical symop for each molecule.

    Attributes:
        n_symops: Number of space group operations
        n_uc_molecules: Number of molecules in unit cell
        symop_to_mol: (n_symops,) symop_idx -> uc_mol_idx
        mol_to_canonical_symop: (n_uc_molecules,) uc_mol_idx -> canonical symop_idx
        site_symmetry_order: (n_uc_molecules,) order of site symmetry for each molecule
    """

    n_symops: int
    n_uc_molecules: int
    symop_to_mol: np.ndarray
    mol_to_canonical_symop: np.ndarray
    site_symmetry_order: np.ndarray

    @classmethod
    def from_crystal(
        cls,
        crystal: Crystal,
        mol_orbit: MoleculeOrbitTable,
        sg_table: SpaceGroupTable,
    ) -> MolecularCosetTable:
        """
        Build the molecular coset table from crystal structure.

        For each symop g, we determine which UC molecule results when g is
        applied to the asymmetric unit. This is done by applying g to a
        reference molecule's centroid and finding which UC molecule centroid
        it maps to.

        Args:
            crystal: The crystal structure
            mol_orbit: MoleculeOrbitTable with molecule instances
            sg_table: SpaceGroupTable for the space group

        Returns:
            MolecularCosetTable with symop-to-molecule mappings
        """
        n_symops = sg_table.n_ops
        n_mols = mol_orbit.n_uc_molecules

        # Get molecule centroids in fractional coordinates
        centroids_frac = mol_orbit.instance_centroids_frac

        # Use molecule 0 as reference
        ref_centroid = centroids_frac[0]

        # For each symop, apply it to the reference centroid and find which
        # molecule centroid it's closest to (modulo lattice)
        symop_to_mol = np.full(n_symops, -1, dtype=np.int32)

        for symop_idx in range(n_symops):
            # Apply symop to reference centroid
            R = sg_table.rotations[symop_idx]
            t = sg_table.translations[symop_idx]
            transformed = R @ ref_centroid + t

            # Wrap to [0, 1)
            transformed = np.mod(transformed + 10.0, 1.0)

            # Find closest molecule centroid (in fractional coords, handling PBC)
            best_mol = -1
            best_dist = float('inf')

            for mol_idx in range(n_mols):
                centroid = centroids_frac[mol_idx]
                # Distance considering PBC
                diff = transformed - centroid
                diff = diff - np.round(diff)  # Minimum image
                dist = np.sum(diff**2)

                if dist < best_dist:
                    best_dist = dist
                    best_mol = mol_idx

            if best_dist < 0.01:  # Tolerance for matching
                symop_to_mol[symop_idx] = best_mol
            else:
                LOG.warning(
                    f"Symop {symop_idx} transforms centroid to {transformed}, "
                    f"no close match found (best dist = {best_dist:.4f})"
                )

        # For each molecule, find the canonical symop (smallest index that generates it)
        mol_to_canonical_symop = np.full(n_mols, -1, dtype=np.int32)
        site_symmetry_order = np.zeros(n_mols, dtype=np.int32)

        for symop_idx in range(n_symops):
            mol_idx = symop_to_mol[symop_idx]
            if mol_idx >= 0:
                site_symmetry_order[mol_idx] += 1
                if mol_to_canonical_symop[mol_idx] < 0:
                    mol_to_canonical_symop[mol_idx] = symop_idx

        return cls(
            n_symops=n_symops,
            n_uc_molecules=n_mols,
            symop_to_mol=symop_to_mol,
            mol_to_canonical_symop=mol_to_canonical_symop,
            site_symmetry_order=site_symmetry_order,
        )

    def normalize_symop(self, symop_idx: int) -> tuple[int, int]:
        """
        Normalize a symop index to (uc_mol_idx, canonical_symop_idx).

        Args:
            symop_idx: The symop index to normalize

        Returns:
            Tuple of (uc_mol_idx, canonical_symop_idx)
        """
        mol_idx = self.symop_to_mol[symop_idx]
        if mol_idx < 0:
            return -1, symop_idx
        canonical = self.mol_to_canonical_symop[mol_idx]
        return mol_idx, canonical


@dataclass
class AlgebraicDimerEquivalenceClass:
    """
    A group of symmetry-equivalent dimers determined algebraically.

    All dimers in this class are related by space group symmetry operations.
    Unlike DimerEquivalenceClass, equivalence is determined by exact algebraic
    comparison (no distance tolerance).

    Attributes:
        representative: One representative dimer from the class
        member_indices: Indices of all dimers in this equivalence class
        multiplicity: Orbit size (number of distinct images under group action)
    """

    representative: AlgebraicDimerIndex
    member_indices: list[int]
    multiplicity: int


def canonical_dimer_representative(
    dimer: AlgebraicDimerIndex,
    sg_table: SpaceGroupTable,
    coset_table: MolecularCosetTable | None = None,
) -> AlgebraicDimerIndex:
    """
    Find the canonical representative of a dimer under group action.

    The canonical representative is the lexicographically smallest element
    of the orbit of the dimer under all space group operations.

    Args:
        dimer: The dimer to canonicalize
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional MolecularCosetTable for symop normalization

    Returns:
        The canonical representative (smallest in orbit)
    """
    best = normalize_algebraic_dimer(dimer, sg_table)

    for g in range(sg_table.n_ops):
        transformed = apply_symop_to_algebraic_dimer(dimer, g, sg_table, coset_table)
        transformed = normalize_algebraic_dimer(transformed, sg_table)

        if transformed._key() < best._key():
            best = transformed

    return best


def compute_dimer_orbit_size(
    dimer: AlgebraicDimerIndex,
    sg_table: SpaceGroupTable,
    coset_table: MolecularCosetTable | None = None,
) -> int:
    """
    Compute the orbit size of a dimer under the space group action.

    This is the number of distinct dimers obtained by applying all space
    group operations. This represents the multiplicity - how many
    symmetry-equivalent copies of this dimer exist in the crystal.

    Args:
        dimer: The dimer to compute orbit size for
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional MolecularCosetTable for symop normalization

    Returns:
        Size of the orbit (number of distinct images)
    """
    orbit = set()
    for g in range(sg_table.n_ops):
        transformed = apply_symop_to_algebraic_dimer(dimer, g, sg_table, coset_table)
        transformed = normalize_algebraic_dimer(transformed, sg_table)
        orbit.add(transformed._key())
    return len(orbit)


def build_algebraic_dimer_list(
    crystal: Crystal,
    mol_orbit: MoleculeOrbitTable,
    cutoff: float = 10.0,
) -> tuple[list[AlgebraicDimerIndex], list[float], SpaceGroupTable, MolecularCosetTable]:
    """
    Build a list of algebraic dimers from the crystal structure.

    This creates AlgebraicDimerIndex objects that use symop indices rather
    than instance indices, enabling algebraic symmetry operations.

    For proper dimer enumeration, we only consider dimers around ONE reference
    molecule per unique molecule type (the one generated by the identity symop).
    This ensures we don't count symmetry-equivalent dimers multiple times.

    Args:
        crystal: The crystal structure
        mol_orbit: MoleculeOrbitTable with molecule instances
        cutoff: Maximum center-to-center distance

    Returns:
        Tuple of:
        - List of AlgebraicDimerIndex objects (normalized, unique)
        - List of corresponding distances
        - SpaceGroupTable for the space group
        - MolecularCosetTable for symop/molecule mapping
    """
    from .space_group_table import SpaceGroupTable

    uc = crystal.unit_cell
    sg = crystal.space_group
    sg_table = SpaceGroupTable.from_space_group(sg)

    # Build coset table to get symop mapping
    coset_table = MolecularCosetTable.from_crystal(crystal, mol_orbit, sg_table)

    # Find the identity symop index
    identity_idx = sg_table.identity_idx()

    # For each unique molecule type, find ONE reference molecule
    # (preferably the one with identity symop)
    n_unique = mol_orbit.n_unique_molecules
    ref_mol_for_type = {}
    for mol_idx, inst in enumerate(mol_orbit.unit_cell_instances):
        asym_idx = inst.asym_mol_idx
        canonical_symop = coset_table.mol_to_canonical_symop[mol_idx]
        if asym_idx not in ref_mol_for_type:
            ref_mol_for_type[asym_idx] = mol_idx
        elif canonical_symop == identity_idx:
            # Prefer identity-generated molecule
            ref_mol_for_type[asym_idx] = mol_idx

    # Get molecule centroids
    centroids_frac = mol_orbit.instance_centroids_frac
    centroids_cart = mol_orbit.instance_centroids_cart
    n_mols = mol_orbit.n_uc_molecules

    # Determine how many cells to search
    cell_lengths = np.array(uc.lengths)
    n_cells = np.ceil(cutoff / cell_lengths).astype(int) + 1

    # Generate all cell translations
    h_range = np.arange(-n_cells[0], n_cells[0] + 1)
    k_range = np.arange(-n_cells[1], n_cells[1] + 1)
    l_range = np.arange(-n_cells[2], n_cells[2] + 1)

    # Build extended structure
    all_centroids = []
    all_cells = []
    all_mol_indices = []

    for h in h_range:
        for k in k_range:
            for l in l_range:
                cell_offset = np.array([h, k, l])
                shifted_frac = centroids_frac + cell_offset
                shifted_cart = uc.to_cartesian(shifted_frac)
                all_centroids.append(shifted_cart)
                all_cells.extend([(h, k, l)] * n_mols)
                all_mol_indices.extend(range(n_mols))

    all_centroids = np.vstack(all_centroids)
    all_mol_indices = np.array(all_mol_indices)

    # Build KDTree
    tree = KDTree(all_centroids)

    # Find all dimers from reference molecules only (one per unique type)
    dimers = []
    distances = []
    seen_dimers = set()

    for asym_mol_idx in range(n_unique):
        mol_i = ref_mol_for_type[asym_mol_idx]
        ref_centroid = centroids_cart[mol_i]
        neighbors = tree.query_ball_point(ref_centroid, cutoff)

        symop_idx_i = coset_table.mol_to_canonical_symop[mol_i]

        mol_a = AlgebraicMoleculeRef(
            asym_mol_idx=asym_mol_idx,
            symop_idx=symop_idx_i,
            cell=(0, 0, 0),
        )

        for j_extended in neighbors:
            mol_j = all_mol_indices[j_extended]
            cell_j = all_cells[j_extended]
            dist = np.linalg.norm(all_centroids[j_extended] - ref_centroid)

            # Skip self
            if mol_j == mol_i and cell_j == (0, 0, 0):
                continue

            # Skip very small distances
            if dist < 1e-10:
                continue

            asym_mol_j = mol_orbit.unit_cell_instances[mol_j].asym_mol_idx
            symop_idx_j = coset_table.mol_to_canonical_symop[mol_j]

            mol_b = AlgebraicMoleculeRef(
                asym_mol_idx=asym_mol_j,
                symop_idx=symop_idx_j,
                cell=cell_j,
            )

            # Create dimer and normalize (translate so one mol is at (0,0,0))
            dimer = AlgebraicDimerIndex.create(mol_a, mol_b)
            dimer = normalize_algebraic_dimer(dimer, sg_table)

            # Check for duplicates
            dimer_key = dimer._key()
            if dimer_key not in seen_dimers:
                seen_dimers.add(dimer_key)
                dimers.append(dimer)
                distances.append(dist)

    # Sort by distance
    sorted_indices = np.argsort(distances)
    dimers = [dimers[i] for i in sorted_indices]
    distances = [distances[i] for i in sorted_indices]

    return dimers, distances, sg_table, coset_table


def find_algebraic_equivalence_classes(
    dimers: list[AlgebraicDimerIndex],
    sg_table: SpaceGroupTable,
    coset_table: MolecularCosetTable | None = None,
) -> tuple[list[list[int]], list[int]]:
    """
    Group dimers by symmetry equivalence using pure algebra.

    Two dimers are equivalent if they have the same canonical representative
    under the space group action. This uses the group multiplication table
    for exact comparison - no numerical tolerances needed.

    Args:
        dimers: List of algebraic dimer indices
        sg_table: SpaceGroupTable for the space group
        coset_table: Optional MolecularCosetTable for symop normalization
            (important for Z' < 1 cases)

    Returns:
        Tuple of:
        - List of lists, where each inner list contains indices of equivalent dimers
        - List of orbit sizes for each equivalence class
    """
    n = len(dimers)
    if n == 0:
        return [], []

    # Compute canonical representative and orbit size for each dimer
    canonical_reps = []
    for dimer in dimers:
        canon = canonical_dimer_representative(dimer, sg_table, coset_table)
        canonical_reps.append(canon._key())

    # Group by canonical representative
    classes = defaultdict(list)
    for i, canon in enumerate(canonical_reps):
        classes[canon].append(i)

    class_list = list(classes.values())

    # Compute orbit size for each class (use first member as representative)
    orbit_sizes = []
    for member_indices in class_list:
        rep_dimer = dimers[member_indices[0]]
        orbit_size = compute_dimer_orbit_size(rep_dimer, sg_table, coset_table)
        orbit_sizes.append(orbit_size)

    return class_list, orbit_sizes


def find_symmetry_unique_dimers_algebraic(
    crystal: Crystal,
    mol_orbit: MoleculeOrbitTable,
    cutoff: float = 10.0,
) -> tuple[list[AlgebraicDimerEquivalenceClass], list[AlgebraicDimerIndex], list[float]]:
    """
    Find symmetry-unique dimers using algebraic group theory.

    This is the algebraic alternative to find_symmetry_unique_dimers().
    It determines equivalence by applying all space group operations and
    checking for exact algebraic equality - no distance tolerances needed.

    For Z' < 1 crystals (molecules on special positions), this properly
    handles the site symmetry by normalizing symop indices to canonical
    representatives for each molecule.

    Args:
        crystal: The crystal structure
        mol_orbit: MoleculeOrbitTable with molecule instances
        cutoff: Maximum center-to-center distance

    Returns:
        Tuple of:
        - List of AlgebraicDimerEquivalenceClass objects (unique dimers)
        - List of all AlgebraicDimerIndex objects
        - List of corresponding distances
    """
    # Build algebraic dimer list (also builds sg_table and coset_table internally)
    dimers, distances, sg_table, coset_table = build_algebraic_dimer_list(
        crystal, mol_orbit, cutoff
    )

    # Log site symmetry info
    for mol_idx in range(mol_orbit.n_uc_molecules):
        order = coset_table.site_symmetry_order[mol_idx]
        if order > 1:
            LOG.debug(
                f"Molecule {mol_idx} has site symmetry of order {order} "
                f"(Z' = {1/order:.2f})"
            )

    # Find equivalence classes using coset table for proper normalization
    equiv_classes, orbit_sizes = find_algebraic_equivalence_classes(
        dimers, sg_table, coset_table
    )

    # Build result
    result_classes = []
    for member_indices, orbit_size in zip(equiv_classes, orbit_sizes, strict=False):
        # Choose representative as the one with smallest index (first found)
        rep_idx = min(member_indices)
        rep = dimers[rep_idx]

        ec = AlgebraicDimerEquivalenceClass(
            representative=rep,
            member_indices=sorted(member_indices),
            multiplicity=orbit_size,  # Use orbit size as multiplicity
        )
        result_classes.append(ec)

    # Sort by distance of representative
    result_classes.sort(key=lambda ec: distances[ec.member_indices[0]])

    return result_classes, dimers, distances

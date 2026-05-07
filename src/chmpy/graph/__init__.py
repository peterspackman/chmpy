"""
Molecular graph module for ring detection, canonicalization, SMILES generation,
and substructure matching.

This module provides pure-Python implementations of common cheminformatics
algorithms without requiring external dependencies like graph_tool or RDKit
at runtime.
"""

from .adjacency import MolecularGraph
from .aromaticity import (
    count_pi_electrons,
    get_aromatic_rings,
    is_aromatic_atom,
    is_aromatic_bond,
    is_aromatic_ring,
    perceive_aromaticity,
)
from .bond_orders import (
    get_bond_order,
    hydrogen_count,
    implicit_hydrogen_count,
    is_double_bond,
    is_single_bond,
    is_triple_bond,
    perceive_bond_orders,
    total_bond_order,
)
from .canonicalization import (
    canonical_atom_map,
    canonical_ordering,
    equivalent_atoms,
    morgan_fingerprint,
    morgan_fingerprint_bits,
    reorder_graph,
    tanimoto_similarity,
)
from .formal_charges import assign_formal_charges, guess_formal_charges
from .rings import (
    find_all_rings,
    find_sssr,
    fused_ring_systems,
    is_in_ring,
    is_ring_bond,
    ring_membership,
    ring_sizes,
    smallest_ring_containing_atom,
)
from .smiles_writer import smiles_from_molecule, to_smiles
from .stereochemistry import (
    assign_stereochemistry,
    count_stereocenters,
    find_double_bond_stereo,
    find_stereocenters,
    get_double_bond_config,
    get_stereocenter_config,
    is_chiral,
)
from .substructure import (
    count_substructures,
    find_functional_groups,
    find_substructure,
    has_substructure,
    list_functional_groups,
)

__all__ = [
    # Core
    "MolecularGraph",
    # Rings
    "find_sssr",
    "find_all_rings",
    "is_in_ring",
    "ring_membership",
    "ring_sizes",
    "is_ring_bond",
    "smallest_ring_containing_atom",
    "fused_ring_systems",
    # Aromaticity
    "perceive_aromaticity",
    "is_aromatic_ring",
    "is_aromatic_atom",
    "is_aromatic_bond",
    "count_pi_electrons",
    "get_aromatic_rings",
    # Bond orders
    "perceive_bond_orders",
    "get_bond_order",
    "is_single_bond",
    "is_double_bond",
    "is_triple_bond",
    "total_bond_order",
    "hydrogen_count",
    "implicit_hydrogen_count",
    # Canonicalization
    "canonical_ordering",
    "canonical_atom_map",
    "equivalent_atoms",
    "morgan_fingerprint",
    "morgan_fingerprint_bits",
    "tanimoto_similarity",
    "reorder_graph",
    # Stereochemistry
    "assign_stereochemistry",
    "find_stereocenters",
    "find_double_bond_stereo",
    "get_stereocenter_config",
    "get_double_bond_config",
    "is_chiral",
    "count_stereocenters",
    # SMILES
    "to_smiles",
    "smiles_from_molecule",
    # Formal charges
    "guess_formal_charges",
    "assign_formal_charges",
    # Substructure
    "find_substructure",
    "has_substructure",
    "count_substructures",
    "find_functional_groups",
    "list_functional_groups",
]

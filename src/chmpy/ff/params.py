"""
Simple UFF parameter assignment for chmpy Crystal and Molecule objects using EEQ coordination numbers.
"""

import json
from pathlib import Path


def load_lj_params():
    """Load Lennard-Jones parameters from JSON file."""
    params_file = Path(__file__).parent / "lj_params.json"
    with open(params_file) as f:
        return json.load(f)


def assign_fit_lj_type_from_connectivity(
    atomic_num, neighbours_list, all_atomic_numbers
):
    """
    Assign fit_lj atom type based on connectivity, following neighcrys FIT potential rules.

    This follows the exact logic from neighcrys_axis.py for FIT potential labeling.

    Args:
        atomic_num: Atomic number of the atom to label
        neighbours_list: List of neighbor indices for each atom
        all_atomic_numbers: Array of atomic numbers for all atoms

    Returns:
        fit_lj atom type string
    """

    # Note: This requires full connectivity information, not just coordination number
    # For now, implementing a simplified version that can work with coordination numbers
    # A full implementation would need the molecule's bond connectivity

    if atomic_num == 1:  # Hydrogen
        # Without full connectivity, default to H_F1 for C-H and H_F2 for others
        # This is a simplification - full implementation would check what H is bonded to
        return "H_F1"  # Most common case

    elif atomic_num == 6:  # Carbon
        # All carbons map to C_F1 in FIT potential
        return "C_F1"

    elif atomic_num == 7:  # Nitrogen
        # All nitrogens map to N_F1 in FIT potential
        return "N_F1"

    elif atomic_num == 8:  # Oxygen
        # All oxygens map to O_F1 in FIT potential (except water which is O_Wa)
        return "O_F1"

    elif atomic_num == 9:  # Fluorine
        return "F_F1"

    elif atomic_num == 16:  # Sulfur
        return "S_F1"

    elif atomic_num == 17:  # Chlorine
        return "ClF1"

    else:
        # For unsupported elements, return None
        return None


def assign_fit_lj_type(atomic_num, coord_num, bonded_to=None):
    """
    Assign fit_lj atom type based on atomic number and coordination.

    Simplified version that uses coordination numbers when full connectivity
    is not available.

    Args:
        atomic_num: Atomic number
        coord_num: EEQ coordination number (float)
        bonded_to: Optional list of atomic numbers this atom is bonded to

    Returns:
        fit_lj atom type string
    """

    if atomic_num == 1:  # Hydrogen
        # H_F1 for C-H, H_F2 for O-H and N-H
        # Without bonding info, use coordination as proxy
        # Lower coordination typically means polar H (O-H, N-H)
        if coord_num < 0.8:
            return "H_F2"
        else:
            return "H_F1"

    elif atomic_num == 6:  # Carbon
        return "C_F1"

    elif atomic_num == 7:  # Nitrogen
        return "N_F1"

    elif atomic_num == 8:  # Oxygen
        return "O_F1"

    elif atomic_num == 9:  # Fluorine
        return "F_F1"

    elif atomic_num == 16:  # Sulfur
        return "S_F1"

    elif atomic_num == 17:  # Chlorine
        return "ClF1"

    else:
        # For unsupported elements, return None
        return None


def assign_uff_type_from_coordination(atomic_num, coord_num):
    """
    Assign UFF atom type based on atomic number and EEQ coordination number.

    Args:
        atomic_num: Atomic number
        coord_num: EEQ coordination number (float)

    Returns:
        UFF atom type string
    """

    if atomic_num == 1:  # Hydrogen
        return "H_"

    elif atomic_num == 6:  # Carbon
        if coord_num > 3.5:
            return "C_3"  # sp3 (tetrahedral)
        elif coord_num > 2.5:
            return "C_2"  # sp2 (trigonal)
        else:
            return "C_1"  # sp (linear)

    elif atomic_num == 7:  # Nitrogen
        if coord_num > 2.5:
            return "N_3"  # pyramidal
        elif coord_num > 1.5:
            return "N_2"  # trigonal
        else:
            return "N_1"  # linear

    elif atomic_num == 8:  # Oxygen
        if coord_num > 1.5:
            return "O_3"  # bent/tetrahedral
        else:
            return "O_2"  # linear (C=O)

    elif atomic_num == 16:  # Sulfur
        if coord_num > 3.5:
            return "S_3+6"
        elif coord_num > 2.5:
            return "S_3+4"
        else:
            return "S_3+2"

    elif atomic_num == 15:  # Phosphorus
        if coord_num > 3.5:
            return "P_3+5"
        else:
            return "P_3+3"

    # Common elements with fixed types
    elif atomic_num == 9:
        return "F_"
    elif atomic_num == 17:
        return "Cl"
    elif atomic_num == 35:
        return "Br"
    elif atomic_num == 53:
        return "I_"
    elif atomic_num == 14:
        return "Si3"
    elif atomic_num == 5:
        return "B_2"
    elif atomic_num == 13:
        return "Al3"

    # Common metals
    elif atomic_num == 12:
        return "Mg3+2"
    elif atomic_num == 20:
        return "Ca6+2"
    elif atomic_num == 30:
        return "Zn3+2"
    elif atomic_num == 26:
        return "Fe6+2" if coord_num > 4.5 else "Fe3+2"
    elif atomic_num == 29:
        return "Cu3+1"
    elif atomic_num == 28:
        return "Ni4+2"
    elif atomic_num == 27:
        return "Co6+3"
    elif atomic_num == 25:
        return "Mn6+2"
    elif atomic_num == 24:
        return "Cr6+3"
    elif atomic_num == 22:
        return "Ti6+4"
    elif atomic_num == 23:
        return "V_3+5"
    elif atomic_num == 42:
        return "Mo6+6"
    elif atomic_num == 74:
        return "W_6+6"

    else:
        # Generic fallback
        from chmpy.core.element import Element

        symbol = Element.from_atomic_number(atomic_num).symbol
        return f"{symbol}3+2"


def get_lj_parameters(obj, force_field="uff"):
    """
    Get Lennard-Jones atom types and parameters for Crystal or Molecule object.

    Args:
        obj: Crystal or Molecule object
        force_field: "uff", "uff4mof", or "fit_lj"

    Returns:
        tuple: (atom_types, parameters)
            atom_types: dict {atom_idx: atom_type}
            parameters: dict {atom_idx: {"sigma": float, "epsilon": float}}
    """

    # Load parameter database
    lj_params = load_lj_params()[force_field.lower()]

    # Get atomic numbers and coordination numbers
    if hasattr(obj, "unit_cell_atoms"):  # Crystal
        uc_atoms = obj.unit_cell_atoms()
        atomic_nums = uc_atoms["element"]
        coord_nums = obj.unit_cell_coordination_numbers()
    elif hasattr(obj, "atomic_numbers"):  # Molecule
        atomic_nums = obj.atomic_numbers
        coord_nums = obj.coordination_numbers  # Property, not method
    else:
        raise ValueError("Object must be Crystal or Molecule")

    atom_types = {}
    parameters = {}

    for i, (atomic_num, coord_num) in enumerate(
        zip(atomic_nums, coord_nums, strict=False)
    ):
        # Assign atom type based on force field and coordination
        if force_field.lower() == "fit_lj":
            atom_type = assign_fit_lj_type(atomic_num, coord_num)
        else:
            # UFF or UFF4MOF
            atom_type = assign_uff_type_from_coordination(atomic_num, coord_num)

        atom_types[i] = atom_type

        # Get parameters
        if atom_type and atom_type in lj_params:
            sigma, epsilon = lj_params[atom_type]
            parameters[i] = {"sigma": sigma, "epsilon": epsilon}
        else:
            # Fallback parameters or None for unsupported atoms
            if atom_type is None:
                print(f"Warning: No fit_lj type for atomic number {atomic_num}")
                parameters[i] = None
            else:
                parameters[i] = {"sigma": 3.0, "epsilon": 0.1}
                print(f"Warning: No parameters found for {atom_type}, using defaults")

    return atom_types, parameters


# Keep old function name for backwards compatibility
def get_uff_parameters(obj, force_field="uff"):
    """
    Deprecated: Use get_lj_parameters instead.

    Get UFF atom types and parameters for Crystal or Molecule object.

    Args:
        obj: Crystal or Molecule object
        force_field: "uff" or "uff4mof"

    Returns:
        tuple: (atom_types, parameters)
            atom_types: dict {atom_idx: uff_type}
            parameters: dict {atom_idx: {"sigma": float, "epsilon": float}}
    """
    return get_lj_parameters(obj, force_field)


def print_lj_summary(obj, force_field="uff"):
    """
    Print a summary of Lennard-Jones atom types and parameters.

    Args:
        obj: Crystal or Molecule object
        force_field: "uff", "uff4mof", or "fit_lj"
    """

    atom_types, parameters = get_lj_parameters(obj, force_field)

    # Get atomic info
    if hasattr(obj, "unit_cell_atoms"):  # Crystal
        uc_atoms = obj.unit_cell_atoms()
        atomic_nums = uc_atoms["element"]
        coord_nums = obj.unit_cell_coordination_numbers()  # Method for Crystal
        name = getattr(obj, "titl", "Crystal")
    elif hasattr(obj, "atomic_numbers"):  # Molecule
        atomic_nums = obj.atomic_numbers
        coord_nums = obj.coordination_numbers  # Property for Molecule
        name = getattr(obj, "molecular_formula", "Molecule")

    print(f"\nLJ Parameters for {name}")
    print(f"Force Field: {force_field.upper()}")
    print("=" * 70)
    print(
        f"{'Atom':>4} {'Element':>7} {'Coord':>6} {'Type':>10} {'σ (Å)':>8} {'ε (kcal/mol)':>12}"
    )
    print("-" * 70)

    for i, (atomic_num, coord_num) in enumerate(
        zip(atomic_nums, coord_nums, strict=False)
    ):
        atom_type = atom_types[i]
        params = parameters[i]

        if atom_type and params:
            print(
                f"{i + 1:4d} {atomic_num:7d} {coord_num:6.2f} {atom_type:>10s} {params['sigma']:8.3f} {params['epsilon']:12.6f}"
            )
        else:
            print(
                f"{i + 1:4d} {atomic_num:7d} {coord_num:6.2f} {'N/A':>10s} {'N/A':>8s} {'N/A':>12s}"
            )

    # Summary
    unique_types = {t for t in atom_types.values() if t is not None}
    print(f"\nUnique types: {len(unique_types)}")
    print(f"Types found: {sorted(unique_types)}")


# Keep old function name for backwards compatibility
def print_uff_summary(obj, force_field="uff"):
    """Deprecated: Use print_lj_summary instead."""
    return print_lj_summary(obj, force_field)


# Convenience functions
def crystal_lj_params(crystal, force_field="uff"):
    """Get Lennard-Jones parameters for Crystal object."""
    return get_lj_parameters(crystal, force_field)


def molecule_lj_params(molecule, force_field="uff"):
    """Get Lennard-Jones parameters for Molecule object."""
    return get_lj_parameters(molecule, force_field)


# Keep old function names for backwards compatibility
def crystal_uff_params(crystal, force_field="uff"):
    """Deprecated: Use crystal_lj_params instead."""
    return get_lj_parameters(crystal, force_field)


def molecule_uff_params(molecule, force_field="uff"):
    """Deprecated: Use molecule_lj_params instead."""
    return get_lj_parameters(molecule, force_field)

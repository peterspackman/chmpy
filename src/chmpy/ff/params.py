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


def get_uff_parameters(obj, force_field="uff"):
    """
    Get UFF atom types and parameters for Crystal or Molecule object.

    Args:
        obj: Crystal or Molecule object
        force_field: "uff" or "uff4mof"

    Returns:
        tuple: (atom_types, parameters)
            atom_types: dict {atom_idx: uff_type}
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

    for i, (atomic_num, coord_num) in enumerate(zip(atomic_nums, coord_nums, strict=False)):
        # Assign UFF type based on coordination
        uff_type = assign_uff_type_from_coordination(atomic_num, coord_num)
        atom_types[i] = uff_type

        # Get parameters
        if uff_type in lj_params:
            sigma, epsilon = lj_params[uff_type]
            parameters[i] = {"sigma": sigma, "epsilon": epsilon}
        else:
            # Fallback parameters
            parameters[i] = {"sigma": 3.0, "epsilon": 0.1}
            print(f"Warning: No parameters found for {uff_type}, using defaults")

    return atom_types, parameters


def print_uff_summary(obj, force_field="uff"):
    """
    Print a summary of UFF atom types and parameters.

    Args:
        obj: Crystal or Molecule object
        force_field: "uff" or "uff4mof"
    """

    atom_types, parameters = get_uff_parameters(obj, force_field)

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

    print(f"\nUFF Parameters for {name}")
    print(f"Force Field: {force_field.upper()}")
    print("=" * 70)
    print(
        f"{'Atom':>4} {'Element':>7} {'Coord':>6} {'UFF Type':>10} {'σ (Å)':>8} {'ε (kcal/mol)':>12}"
    )
    print("-" * 70)

    for i, (atomic_num, coord_num) in enumerate(zip(atomic_nums, coord_nums, strict=False)):
        uff_type = atom_types[i]
        params = parameters[i]

        print(
            f"{i + 1:4d} {atomic_num:7d} {coord_num:6.2f} {uff_type:>10s} {params['sigma']:8.3f} {params['epsilon']:12.6f}"
        )

    # Summary
    unique_types = set(atom_types.values())
    print(f"\nUnique types: {len(unique_types)}")
    print(f"Types found: {sorted(unique_types)}")


# Convenience functions
def crystal_uff_params(crystal, force_field="uff"):
    """Get UFF parameters for Crystal object."""
    return get_uff_parameters(crystal, force_field)


def molecule_uff_params(molecule, force_field="uff"):
    """Get UFF parameters for Molecule object."""
    return get_uff_parameters(molecule, force_field)

"""Force field atom typing and export functions for Crystal structures."""

import numpy as np

from chmpy.core.element import Element


def assign_atom_types(crystal, force_field="UFF", **kwargs):
    """
    Assign atom types and force field parameters to a crystal structure.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use for typing.
            Options: "UFF", "UFF4MOF", "DREIDING", "COMPASS"
        **kwargs: Additional arguments passed to the atom typing system

    Returns:
        dict: Dictionary with typing results containing:
            - atom_types: mapping of atom indices to (ff_type, descriptor) tuples
            - parameters: mapping of atom indices to ForceFieldParameters objects
            - force_field: name of the force field used
            - unique_types: set of unique atom types found

    Example:
        >>> crystal = Crystal.load("MOF.cif")
        >>> results = assign_atom_types(crystal, "UFF")
        >>> print(f"Found {len(results['unique_types'])} unique atom types")
        >>> for i, params in results["parameters"].items():
        ...     print(f"Atom {i}: {params.ff_type} (ε={params.epsilon:.3f})")
    """
    from chmpy.ff.params import ForceFieldType, type_crystal_structure

    # Convert string to enum if needed
    if isinstance(force_field, str):
        ff_enum = ForceFieldType(force_field.upper())
    else:
        ff_enum = force_field

    results = type_crystal_structure(crystal, ff_enum, **kwargs)

    # Cache results on the crystal object for future access
    crystal._atom_typing_results = results

    return results


def get_atom_types(crystal, force_field="UFF", use_cached=True, **kwargs):
    """
    Get atom type assignments for a crystal structure.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use
        use_cached (bool): Whether to use cached results if available
        **kwargs: Additional arguments for atom typing

    Returns:
        dict: Mapping of atom indices to (force_field_type, AtomTypeDescriptor) tuples
    """
    if use_cached and hasattr(crystal, "_atom_typing_results"):
        cached_ff = crystal._atom_typing_results.get("force_field", "").upper()
        if cached_ff == str(force_field).upper():
            return crystal._atom_typing_results["atom_types"]

    results = assign_atom_types(crystal, force_field, **kwargs)
    return results["atom_types"]


def get_ff_parameters(crystal, force_field="UFF", use_cached=True, **kwargs):
    """
    Get force field parameters for a crystal structure.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use
        use_cached (bool): Whether to use cached results if available
        **kwargs: Additional arguments for atom typing

    Returns:
        dict: Mapping of atom indices to ForceFieldParameters objects
    """
    if use_cached and hasattr(crystal, "_atom_typing_results"):
        cached_ff = crystal._atom_typing_results.get("force_field", "").upper()
        if cached_ff == str(force_field).upper():
            return crystal._atom_typing_results["parameters"]

    results = assign_atom_types(crystal, force_field, **kwargs)
    return results["parameters"]


def get_unique_atom_types(crystal, force_field="UFF", use_cached=True, **kwargs):
    """
    Get the set of unique atom types in a crystal structure.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use
        use_cached (bool): Whether to use cached results if available
        **kwargs: Additional arguments for atom typing

    Returns:
        set: Set of unique force field atom type strings
    """
    if use_cached and hasattr(crystal, "_atom_typing_results"):
        cached_ff = crystal._atom_typing_results.get("force_field", "").upper()
        if cached_ff == str(force_field).upper():
            return crystal._atom_typing_results["unique_types"]

    results = assign_atom_types(crystal, force_field, **kwargs)
    return results["unique_types"]


def get_lj_parameters_array(crystal, force_field="UFF", use_cached=True, **kwargs):
    """
    Get Lennard-Jones parameters as structured arrays for simulation input.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use
        use_cached (bool): Whether to use cached results if available
        **kwargs: Additional arguments for atom typing

    Returns:
        tuple: (atom_types_array, epsilon_array, sigma_array, mass_array, charges_array)
            where each array is ordered by atom index
    """
    parameters = get_ff_parameters(crystal, force_field, use_cached, **kwargs)
    uc_atoms = crystal.unit_cell_atoms()
    n_atoms = len(uc_atoms["element"])

    # Initialize arrays
    atom_types = []
    epsilons = np.zeros(n_atoms)
    sigmas = np.zeros(n_atoms)
    masses = np.zeros(n_atoms)
    charges = np.zeros(n_atoms)

    # Fill arrays in atom index order
    for i in range(n_atoms):
        if i in parameters:
            params = parameters[i]
            atom_types.append(params.ff_type)
            epsilons[i] = params.epsilon
            sigmas[i] = params.sigma
            masses[i] = params.mass if params.mass is not None else 0.0
            charges[i] = params.charge
        else:
            # Fallback for missing parameters
            element = Element[uc_atoms["element"][i]].symbol
            atom_types.append(f"{element}_generic")
            masses[i] = Element[uc_atoms["element"][i]].mass

    return atom_types, epsilons, sigmas, masses, charges


def export_lammps_data(crystal, filename, force_field="UFF", **kwargs):
    """
    Export crystal structure with atom types in LAMMPS data format.

    Args:
        crystal: Crystal instance
        filename (str): Output filename for LAMMPS data file
        force_field (str or ForceFieldType): Force field to use
        **kwargs: Additional arguments for atom typing and export

    Note:
        This is a placeholder - would need integration with LAMMPS export functionality
    """
    results = assign_atom_types(crystal, force_field, **kwargs)
    atom_types, epsilons, sigmas, masses, charges = get_lj_parameters_array(
        crystal, force_field, **kwargs
    )

    # This would integrate with existing LAMMPS export functionality
    # For now, just store the information
    lammps_data = {
        "atom_types": atom_types,
        "epsilons": epsilons,
        "sigmas": sigmas,
        "masses": masses,
        "charges": charges,
        "positions": crystal.unit_cell_atoms()["cart_pos"],
        "cell_parameters": crystal.unit_cell.parameters,
    }

    print(f"LAMMPS export functionality would write to {filename}")
    print(f"Found {len(results['unique_types'])} unique atom types:")
    for atom_type in sorted(results["unique_types"]):
        count = atom_types.count(atom_type)
        print(f"  {atom_type}: {count} atoms")

    return lammps_data


def export_raspa_files(crystal, force_field="UFF", output_dir=".", **kwargs):
    """
    Export force field parameters in RASPA format.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use
        output_dir (str): Directory to write RASPA files
        **kwargs: Additional arguments for atom typing

    Returns:
        dict: Paths to created RASPA files
    """
    from pathlib import Path

    _ = assign_atom_types(crystal, force_field, **kwargs)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get arrays for RASPA format
    atom_types, epsilons, sigmas, masses, charges = get_lj_parameters_array(
        crystal, force_field, **kwargs
    )

    # Create pseudo_atoms.def file
    pseudo_atoms_file = output_path / "pseudo_atoms.def"
    ff_mixing_file = output_path / "force_field_mixing_rules.def"

    # Write pseudo_atoms.def
    with open(pseudo_atoms_file, "w") as f:
        f.write("# Pseudo atoms definition file\n")
        f.write(f"# Generated by chmpy for crystal: {crystal.titl}\n")
        f.write(f"# Force field: {force_field}\n\n")

        unique_params = {}
        for i, ff_type in enumerate(atom_types):
            if ff_type not in unique_params:
                unique_params[ff_type] = {
                    "epsilon": epsilons[i],
                    "sigma": sigmas[i],
                    "mass": masses[i],
                    "charge": charges[i],
                    "element": Element[crystal.unit_cell_atoms()["element"][i]].symbol,
                }

        f.write(f"{len(unique_params)}\n")
        for ff_type, params in unique_params.items():
            f.write(
                f"{ff_type:12s} yes {params['element']:2s} {params['element']:2s} "
            )
            f.write(f"0 {params['mass']:8.3f} {params['charge']:8.3f} ")
            f.write(f"0.0 1.0 {params['sigma']:8.3f} 0 0 relative 0\n")

    # Write force_field_mixing_rules.def
    with open(ff_mixing_file, "w") as f:
        f.write("# Force field mixing rules\n")
        f.write(f"# Generated by chmpy for crystal: {crystal.titl}\n")
        f.write(f"# Force field: {force_field}\n\n")

        f.write("# general rule for Lorentz-Berthelot mixing\n")
        f.write("# LJ potential\n")
        f.write(f"{len(unique_params)}\n")

        for ff_type, params in unique_params.items():
            f.write(
                f"{ff_type:12s} lennard-jones {params['epsilon']:10.6f} {params['sigma']:10.6f}\n"
            )

        f.write("# general mixing rule\n")
        f.write("lorentz-berthelot\n")

    return {
        "pseudo_atoms": str(pseudo_atoms_file),
        "mixing_rules": str(ff_mixing_file),
    }


def atom_typing_summary(crystal, force_field="UFF", **kwargs):
    """
    Print a summary of atom typing results for a crystal.

    Args:
        crystal: Crystal instance
        force_field (str or ForceFieldType): Force field to use
        **kwargs: Additional arguments for atom typing
    """
    results = assign_atom_types(crystal, force_field, **kwargs)

    print(f"\nAtom Typing Summary for {crystal.titl}")
    print(f"Force Field: {results['force_field']}")
    print(f"Total atoms: {len(results['atom_types'])}")
    print(f"Unique types: {len(results['unique_types'])}")
    print("-" * 50)

    # Count atoms by type
    type_counts = {}
    for ff_type, _descriptor in results["atom_types"].values():
        type_counts[ff_type] = type_counts.get(ff_type, 0) + 1

    # Print sorted by count
    for ff_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        # Get example parameters
        example_params = None
        for params in results["parameters"].values():
            if params.ff_type == ff_type:
                example_params = params
                break

        if example_params:
            print(
                f"{ff_type:12s}: {count:3d} atoms  "
                f"(ε={example_params.epsilon:6.3f}, σ={example_params.sigma:6.3f})"
            )
        else:
            print(f"{ff_type:12s}: {count:3d} atoms")

    print("-" * 50)

    # Show any special environments
    special_envs = set()
    for _ff_type, descriptor in results["atom_types"].values():
        if descriptor.special_environment:
            special_envs.add(descriptor.special_environment)

    if special_envs:
        print(f"Special environments detected: {', '.join(special_envs)}")

    print()

from chmpy.core import Element, Molecule
from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell


def molecule_to_ase(mol, **kwargs):
    """
    Convert a Molecule object to an ASE Atoms object.

    Args:
        mol: a chmpy Molecule object

    Returns:
        ase.Atoms: ASE Atoms object representing the molecule
    """
    try:
        from ase import Atoms
    except ImportError as e:
        raise ImportError("ASE library is required.") from e

    atoms = Atoms(
        numbers=mol.atomic_numbers,
        positions=mol.positions,
        pbc=False,
    )

    if hasattr(mol, "properties") and mol.properties:
        for key, value in mol.properties.items():
            if isinstance(value, int | float | str | bool):
                atoms.info[f"molecule_{key}"] = value
    return atoms


def ase_to_molecule(atoms, **kwargs):
    """
    Convert an ASE Atoms object to a Molecule object.

    Args:
        atoms: ASE Atoms object representing the molecule

    Returns:
        Molecule: Molecule object atoms and position information
    """
    try:
        import ase  # noqa: F401
    except ImportError as e:
        raise ImportError("ASE library is required") from e

    if atoms.pbc.any():
        raise ValueError("Atoms object must not have periodic boundary conditions")

    return Molecule.from_arrays(atoms.get_atomic_numbers(), atoms.get_positions())


def crystal_to_ase(crystal, **kwargs):
    """
    Convert a Crystal object to an ASE Atoms object.

    Args:
        crystal: Crystal object with unit_cell, space_group, and asymmetric_unit

    Returns:
        ase.Atoms: ASE Atoms object representing the crystal structure
    """
    try:
        from ase import Atoms
    except ImportError as e:
        raise ImportError("ASE library is required.") from e

    uc_atoms = crystal.unit_cell_atoms()

    atomic_numbers = uc_atoms["element"]
    positions = uc_atoms["frac_pos"]

    cell = crystal.unit_cell.parameters

    atoms = Atoms(
        numbers=atomic_numbers,
        scaled_positions=positions,
        cell=cell,
        pbc=True,
    )

    if "occupation" in uc_atoms:
        atoms.arrays["occupancy"] = uc_atoms["occupation"]

    if "label" in uc_atoms:
        atoms.arrays["labels"] = uc_atoms["label"]

    atoms.arrays["asym_atom"] = uc_atoms["asym_atom"]
    atoms.arrays["symop"] = uc_atoms["symop"]

    atoms.info["space_group"] = crystal.space_group.symbol
    atoms.info["space_group_number"] = crystal.space_group.international_tables_number

    if hasattr(crystal, "properties") and crystal.properties:
        for key, value in crystal.properties.items():
            if isinstance(value, int | float | str | bool):
                atoms.info[f"crystal_{key}"] = value

    return atoms


def ase_to_crystal(atoms, **kwargs):
    """
    Convert an ASE Atoms object to a Crystal object. Assumes it is in P1

    Args:
        atoms: ASE Atoms object representing the crystal structure

    Returns:
        Crystal: Crystal object with unit_cell, space_group, and asymmetric_unit
    """
    try:
        import ase  # noqa: F401
    except ImportError as e:
        raise ImportError("ASE library is required") from e

    if not atoms.pbc.all():
        raise ValueError("Atoms object must have 3D periodic boundary conditions")

    cell_matrix = atoms.get_cell()
    unit_cell = UnitCell(cell_matrix)

    sg = SpaceGroup(1)

    atomic_numbers = atoms.get_atomic_numbers()
    positions_cart = atoms.get_positions()

    positions_frac = unit_cell.to_fractional(positions_cart)
    elements = [Element.from_atomic_number(num) for num in atomic_numbers]

    labels = atoms.arrays.get("labels", None)
    occupancy = atoms.arrays.get("occupancy", None)

    asymmetric_unit = AsymmetricUnit(
        elements=elements,
        positions=positions_frac,
        labels=labels,
        occupation=occupancy,
    )

    crystal = Crystal(
        unit_cell=unit_cell, space_group=sg, asymmetric_unit=asymmetric_unit
    )

    return crystal

"""FHI-aims format readers and writers.

This module handles FHI-aims geometry.in files for crystals and molecules.
"""

import logging

import numpy as np

from chmpy.core import Molecule
from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def crystal_to_geometry_string(crystal, use_cartesian=False):
    """Convert a Crystal object to an FHI-aims geometry.in string.

    Args:
        crystal: Crystal object to convert
        use_cartesian (bool, optional): If True, write atom positions in
            Cartesian coordinates. If False (default), use fractional coordinates.

    Returns:
        str: FHI-aims geometry.in format string

    Examples:
        >>> from chmpy import Crystal
        >>> crystal = Crystal.load("structure.cif")
        >>> geom_str = crystal_to_geometry_string(crystal)
    """
    lines = []

    # Write lattice vectors
    lattice = crystal.unit_cell.lattice
    for i in range(3):
        vec = lattice[i]
        lines.append(f"lattice_vector {vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")

    lines.append("")  # Blank line after lattice vectors

    # Get unit cell atoms
    uc_atoms = crystal.unit_cell_atoms()
    elements = [Element[x] for x in uc_atoms["element"]]

    if use_cartesian:
        # Convert fractional to Cartesian
        frac_pos = uc_atoms["frac_pos"]
        cart_pos = crystal.unit_cell.to_cartesian(frac_pos)
        for pos, el in zip(cart_pos, elements, strict=True):
            lines.append(f"atom {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {el.symbol}")
    else:
        # Use fractional coordinates
        frac_pos = uc_atoms["frac_pos"]
        for pos, el in zip(frac_pos, elements, strict=True):
            lines.append(f"atom_frac {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {el.symbol}")

    lines.append("")  # Trailing blank line
    return "\n".join(lines)


def molecule_to_geometry_string(molecule):
    """Convert a Molecule object to an FHI-aims geometry.in string.

    Args:
        molecule: Molecule object to convert

    Returns:
        str: FHI-aims geometry.in format string (no lattice vectors for molecules)

    Examples:
        >>> from chmpy import Molecule
        >>> mol = Molecule.load("water.xyz")
        >>> geom_str = molecule_to_geometry_string(mol)
    """
    lines = []

    # For molecules, just write atom positions in Cartesian coordinates
    for pos, el in zip(molecule.positions, molecule.elements, strict=True):
        lines.append(f"atom {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {el.symbol}")

    lines.append("")  # Trailing blank line
    return "\n".join(lines)


def to_geometry_string(obj, **kwargs):
    """Convert a Crystal or Molecule object to FHI-aims geometry.in format.

    Args:
        obj: Crystal or Molecule object
        **kwargs: Additional keyword arguments passed to the appropriate conversion function

    Returns:
        str: FHI-aims geometry.in format string
    """
    if isinstance(obj, Molecule):
        return molecule_to_geometry_string(obj)
    else:
        return crystal_to_geometry_string(obj, **kwargs)


def parse_geometry_string(contents):
    """Parse an FHI-aims geometry.in file.

    Args:
        contents (str): Contents of the geometry.in file

    Returns:
        dict: Dictionary containing:
            - lattice: (3, 3) array of lattice vectors (if periodic)
            - elements: List of Element objects
            - positions: (N, 3) array of atomic positions
            - fractional: bool indicating if positions are fractional

    Examples:
        >>> from pathlib import Path
        >>> geom_str = Path("geometry.in").read_text()
        >>> data = parse_geometry_string(geom_str)
    """
    lines = contents.splitlines()

    lattice_vectors = []
    elements = []
    positions = []
    fractional = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if not tokens:
            continue

        keyword = tokens[0]

        if keyword == "lattice_vector":
            # Parse lattice vector: lattice_vector x y z
            if len(tokens) < 4:
                LOG.warning("Incomplete lattice_vector line: %s", line)
                continue
            vec = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            lattice_vectors.append(vec)

        elif keyword == "atom":
            # Parse Cartesian atom: atom x y z element
            if len(tokens) < 5:
                LOG.warning("Incomplete atom line: %s", line)
                continue
            pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            el = Element.from_string(tokens[4])
            positions.append(pos)
            elements.append(el)

        elif keyword == "atom_frac":
            # Parse fractional atom: atom_frac x y z element
            if len(tokens) < 5:
                LOG.warning("Incomplete atom_frac line: %s", line)
                continue
            pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            el = Element.from_string(tokens[4])
            positions.append(pos)
            elements.append(el)
            fractional = True

    result = {
        "elements": elements,
        "positions": np.array(positions),
        "fractional": fractional,
    }

    if lattice_vectors:
        result["lattice"] = np.array(lattice_vectors)

    LOG.debug("Parsed %d atoms, %d lattice vectors", len(elements), len(lattice_vectors))

    return result

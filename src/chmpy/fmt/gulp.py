import logging
import re
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np

from chmpy.templates import load_template

LOG = logging.getLogger(__name__)
GULP_TEMPLATE = load_template("gulp")
NUMBER_REGEX = re.compile(
    r"([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s*([\w/\^\s]+)?"
)

Cell = namedtuple("Cell", "a b c alpha beta gamma")


def parse_value(string, with_units=False):
    """parse a value from a GULP output file to its appropriate type
    e.g. int, float, str etc. Will handle units with a space.

    Parameters
    ----------
    string: str
        the string containing the value to parse

    with_uncertainty: bool, optional
        return a tuple including uncertainty if a numeric type is expected

    Returns
    -------
    value
        the value coerced into the appropriate type

    >>> parse_value("2.3 kj/mol", with_units=True)
    (2.3, 'kj/mol')
    >>> parse_value("5 kgm^2", with_units=True)
    (5, 'kgm^2')
    >>> parse_value("string help")
    'string help'
    >>> parse_value("3.1415") * 4
    12.566
    """
    match = NUMBER_REGEX.match(string)
    try:
        if match and match:
            groups = match.groups()
            number = groups[0]
            number = float(number)
            if number.is_integer():
                number = int(number)
            if with_units and len(groups) > 1:
                return number, groups[1]
            return number
        else:
            s = string.strip()
            return s
    except Exception as e:
        print(e)
    return string


def crystal_to_gulp_input(
    crystal, keywords=None, additional_keywords=None, charges=None
):
    if additional_keywords is None:
        additional_keywords = {}
    if keywords is None:
        keywords = []
    pos = crystal.asymmetric_unit.positions
    el = crystal.asymmetric_unit.elements

    # Prepare atom data - always include charges (default to 0.0)
    if charges is not None:
        atoms = [
            (element, position, charge)
            for element, position, charge in zip(el, pos, charges, strict=False)
        ]
    else:
        atoms = [
            (element, position) for element, position in zip(el, pos, strict=False)
        ]

    return GULP_TEMPLATE.render(
        keywords=keywords,
        frac=True,
        cell=" ".join(f"{x:10.6f}" for x in crystal.unit_cell.parameters),
        atoms=atoms,
        spacegroup=crystal.space_group.crystal17_spacegroup_symbol(),
        origin_choice=getattr(crystal.space_group, "choice", 1),
        additional_keywords=additional_keywords,
    )


def molecule_to_gulp_input(molecule, keywords=None, additional_keywords=None):
    if additional_keywords is None:
        additional_keywords = {}
    if keywords is None:
        keywords = []
    pos = molecule.positions
    el = molecule.elements
    return GULP_TEMPLATE.render(
        keywords=keywords,
        frac=False,
        atoms=zip(el, pos, strict=False),
        additional_keywords=additional_keywords,
    )


def parse_single_line(line):
    toks = re.split(r"\s*[=:]\s*", line)
    if toks is not None and len(toks) == 2:
        return toks[0].strip(), toks[1].strip()
    return None


def parse_gulp_output(contents):
    lines = contents.splitlines()
    lines_iter = iter(lines)
    outputs = {}
    for line in lines_iter:
        result = parse_single_line(line)
        if result is not None:
            k, v = result
            outputs[k] = v
    return outputs


def parse_drv_file(drv_path: Path) -> dict[str, Any]:
    """
    Parse GULP .drv file to extract energy, gradients, and stress.

    The .drv file format:
    Line 1: energy <value> eV
    Line 2: coordinates cartesian Angstroms <natoms>
    Lines 3 to 2+natoms: atom_id element_id x y z
    Line 3+natoms: gradients cartesian eV/Ang <natoms>
    Lines 4+natoms to 3+2*natoms: atom_id gx gy gz
    Line 4+2*natoms: gradients strain eV
    Remaining lines: strain gradients

    Parameters
    ----------
    drv_path : Path
        Path to the .drv file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'energy', 'gradients', and 'stress_raw'
    """
    if not drv_path.exists():
        raise FileNotFoundError(f"GULP .drv file not found: {drv_path}")

    lines = drv_path.read_text().splitlines()
    if not lines:
        raise ValueError("Empty .drv file")

    line_idx = 0

    # Parse energy from first line: "energy                  -0.7783918216 eV"
    if line_idx >= len(lines):
        raise ValueError(
            f"Expected energy line at index {line_idx}, but file only has {len(lines)} lines"
        )

    energy_line = lines[line_idx].strip()
    if not energy_line:
        raise ValueError(f"Empty energy line at index {line_idx}")

    energy_parts = energy_line.split()
    if len(energy_parts) < 2:
        raise ValueError(
            f"Invalid energy line format: '{energy_line}' - expected at least 2 parts"
        )

    try:
        energy = float(energy_parts[1])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not parse energy from line '{energy_line}'") from e

    line_idx += 1

    # Parse coordinates header: "coordinates cartesian Angstroms     48"
    coord_header = lines[line_idx].strip()
    natoms = int(coord_header.split()[-1])
    line_idx += 1

    # Skip coordinate lines
    line_idx += natoms

    # Parse gradients header: "gradients cartesian eV/Ang     48"
    lines[line_idx].strip()
    line_idx += 1

    # Parse gradients (GULP outputs gradients directly)
    gradients = np.zeros((natoms, 3))
    for i in range(natoms):
        parts = lines[line_idx].split()
        # GULP gives gradients in eV/Ang
        # Handle potential numerical overflow (****** values)
        try:
            gradients[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            # Handle overflow case - set to large value
            gradients[i] = [
                1e10 if "*" in parts[1] else float(parts[1]),
                1e10 if "*" in parts[2] else float(parts[2]),
                1e10 if "*" in parts[3] else float(parts[3]),
            ]
        line_idx += 1

    # Parse strain gradients header: "gradients strain eV" (if present)
    stress_raw = np.zeros(6)  # Default to zero stress

    if line_idx < len(lines):
        strain_header = lines[line_idx].strip()
        if strain_header.startswith("gradients strain"):
            line_idx += 1

            # Parse strain gradients
            strain_gradients = []
            while line_idx < len(lines) and lines[line_idx].strip():
                line = lines[line_idx].strip()

                # Skip force constants section if present
                if "force_constants" in line:
                    line_idx += 1
                    # Skip the entire force constants block
                    while line_idx < len(lines) and lines[line_idx].strip():
                        line_idx += 1
                    break

                parts = lines[line_idx].split()
                try:
                    # Convert each part to float, handling overflow
                    for part in parts:
                        if "*" in part:
                            strain_gradients.append(1e10)
                        else:
                            strain_gradients.append(float(part))
                except ValueError:
                    # Skip lines that can't be parsed as numbers
                    pass
                line_idx += 1

            # Convert strain gradients to stress (requires volume from calling context)
            # For now, just return the raw strain gradients
            stress_raw = (
                np.array(strain_gradients[:6])
                if len(strain_gradients) >= 6
                else np.zeros(6)
            )

    # Look for force constants if we skipped them earlier
    force_constants = None
    # Reset to find force constants section
    for _i, line in enumerate(lines):
        if "force_constants" in line:
            # Parse force constants matrix if needed
            # For now, just flag that they exist
            force_constants = "present"
            break

    return {
        "energy": energy,
        "gradients": gradients,
        "stress_raw": stress_raw,  # Will need volume to convert to stress
        "natoms": natoms,
        "force_constants": force_constants,
    }

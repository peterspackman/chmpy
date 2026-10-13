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

    Args:
        string: the string containing the value to parse
        with_uncertainty: return a tuple including uncertainty if a numeric
            type is expected

    Returns:
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
        cell=" ".join(f"{x:.12f}" for x in crystal.unit_cell.parameters),
        atoms=atoms,
        spacegroup=crystal.space_group.crystal17_spacegroup_symbol(),
        origin_choice=_gulp_origin(crystal.space_group),
        additional_keywords=additional_keywords,
    )


def slab_to_gulp_input(
    slab,
    keywords=None,
    fixed=None,
    charges=None,
    additional_keywords=None,
    bulk_energy=None,
):
    """GULP input for a 2D periodic slab.

    The surface vectors are given by `svectors` and positions by
    `sfractional`, with the top of the slab at z = 0. Fixed atoms are
    written as region 2, which GULP holds fixed as the underlying bulk and
    excludes from the surface energy, so the energy reported is that of the
    top face; the rest are region 1.

    Args:
        slab: Slab
        keywords: GULP keywords
        fixed: (N,) boolean mask of region 2 atoms; defaults to `slab.fixed`
        charges: optional per atom (or, for cut slabs, per asymmetric unit
            atom) charges
        additional_keywords: dict of extra options written after the
            coordinates
        bulk_energy: energy (eV) of the bulk primitive cell; when given,
            `sbulkenergy` is written for the region 1 layers so GULP reports
            the surface energy (needs a cut slab and whole fixed layers)

    Returns:
        str: GULP input contents
    """
    from chmpy.core.element import Element

    keywords = keywords or []
    additional_keywords = additional_keywords or {}
    fixed = slab.fixed if fixed is None else np.asarray(fixed, dtype=bool)
    frac = slab.frac_positions
    z = slab.positions[:, 2] - slab.positions[:, 2].max()
    q = None
    if charges is not None:
        q = np.asarray(charges, dtype=float)
        if len(q) != len(slab):
            q = q[slab.asym_atom]
    lines = [" ".join(keywords), "title", slab.title, "end", "svectors"]
    for vec in slab.cell[:2]:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f}")

    regions = [~fixed, fixed] if fixed.any() else [np.ones(len(slab), dtype=bool)]
    for region, mask in enumerate(regions, start=1):
        lines.append(
            f"sfractional region {region}" if len(regions) > 1 else "sfractional"
        )
        for i in np.flatnonzero(mask):
            line = (
                f"{Element[slab.elements[i]].symbol} core "
                f"{frac[i, 0]:.10f} {frac[i, 1]:.10f} {z[i]:.10f}"
            )
            if q is not None:
                line += f" {q[i]:.6f}"
            lines.append(line)
    if bulk_energy is not None:
        lines.append(f"sbulkenergy {_region_1_layers(slab, fixed) * bulk_energy:.10f}")
    for keyword, argument in additional_keywords.items():
        if keyword == "species":
            lines.extend([keyword, str(argument)])
        else:
            lines.append(f"{keyword} {argument}")
    lines.append("")
    return "\n".join(lines)


def _region_1_layers(slab, fixed) -> int:
    "Number of whole layer periods in region 1, each one bulk primitive cell."
    if slab.layer is None or slab.layers is None:
        raise ValueError("sbulkenergy needs a slab with layer information")
    free_layers = np.unique(slab.layer[~fixed])
    if np.intersect1d(free_layers, slab.layer[fixed]).size:
        raise ValueError("Region 2 must consist of whole layers for sbulkenergy")
    return len(free_layers)


def gulp_string_to_slab(contents, vacuum: float = 15.0):
    """Read a Slab from a GULP surface input (svectors and sfractional).

    Region 2 atoms are marked fixed; shells are skipped. GULP surfaces have
    no third vector, so the slab is placed in a cell with `vacuum`
    Angstroms between images.
    """
    from chmpy.crystal.slab import Slab, parse_slab_title

    lines = contents.splitlines()
    svectors, elements, positions, fixed, metadata = [], [], [], [], {}
    region = None
    i = 0
    while i < len(lines):
        tokens = lines[i].split()
        i += 1
        if not tokens:
            continue
        word = tokens[0].lower()
        if word == "title":
            while i < len(lines) and lines[i].strip().lower() != "end":
                metadata = metadata or parse_slab_title(lines[i])
                i += 1
            i += 1
        elif word.startswith("svec"):
            svectors = [[float(x) for x in lines[i + j].split()[:2]] for j in range(2)]
            i += 2
        elif word.startswith("sfrac"):
            region = (
                int(tokens[2])
                if len(tokens) > 2 and tokens[1].lower().startswith("reg")
                else 1
            )
        elif region is not None:
            values = tokens[1:]
            kind = "core"
            if values and not _is_number(values[0]):
                kind = values[0].lower()
                values = values[1:]
            if len(values) < 3 or not all(_is_number(v) for v in values[:3]):
                region = None
                i -= 1
                continue
            if kind.startswith("s"):
                continue
            elements.append(re.sub(r"\d+$", "", tokens[0]))
            positions.append([float(x) for x in values[:3]])
            fixed.append(region == 2)
    if len(svectors) != 2 or not elements:
        raise ValueError("GULP input has no svectors or sfractional surface")
    a = np.array([*svectors[0], 0.0])
    b = np.array([*svectors[1], 0.0])
    frac = np.array(positions)
    z = frac[:, 2]
    height = z.max() - z.min() + vacuum
    cart = frac[:, :1] * a + frac[:, 1:2] * b
    cart[:, 2] = z - 0.5 * (z.max() + z.min()) + 0.5 * height
    return Slab(
        np.array([a, b, [0.0, 0.0, height]]), elements, cart, fixed=fixed, **metadata
    )


def _is_number(token) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _gulp_origin(space_group):
    """GULP's `origin` value, or None for the default (origin choice 1).

    Only origin choice 2 needs stating; "H"/"R" are axis settings, which GULP
    takes from the cell.
    """
    return "2" if str(getattr(space_group, "choice", "")) == "2" else None


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

    Args:
        drv_path: Path to the .drv file

    Returns:
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

    # Everything after the energy is optional. GULP writes no gradient section
    # at all when the run declared nothing variable -- an isolated molecule
    # with `noflag`, for instance -- and says so only in a warning, so a parser
    # that assumes the section is there fails with an index error a long way
    # from the cause.
    if line_idx >= len(lines):
        return {
            "energy": energy,
            "gradients": None,
            "stress_raw": None,
            "natoms": 0,
            "force_constants": None,
        }

    # Parse coordinates header: "coordinates cartesian Angstroms     48"
    coord_header = lines[line_idx].strip()
    natoms = int(coord_header.split()[-1])
    line_idx += 1

    # Skip coordinate lines
    line_idx += natoms

    # Parse gradients header: "gradients cartesian eV/Ang     48"
    if line_idx >= len(lines) or not lines[line_idx].strip().startswith("gradients"):
        return {
            "energy": energy,
            "gradients": None,
            "stress_raw": None,
            "natoms": natoms,
            "force_constants": None,
        }
    line_idx += 1

    # Parse gradients (GULP outputs gradients directly)
    if line_idx + natoms > len(lines):
        raise ValueError(
            f"the .drv file promises {natoms} gradient rows but holds "
            f"{len(lines) - line_idx}"
        )
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


def parse_elastic_constants(contents: str) -> np.ndarray | None:
    """The elastic constant matrix GULP prints under the `property` keyword.

    GULP computes these analytically, which makes them a reference for a
    finite-difference calculation rather than another estimate of the same
    kind.

    Args:
        contents: the text of a GULP output file

    Returns:
        (6, 6) elastic constants in GPa, or None if the output has none
    """
    marker = "Elastic Constant Matrix"
    start = contents.find(marker)
    if start < 0:
        return None

    rows = []
    for line in contents[start:].splitlines():
        fields = line.split()
        # a data row is an index 1-6 followed by six numbers
        if len(fields) == 7 and fields[0].isdigit() and 1 <= int(fields[0]) <= 6:
            try:
                rows.append([float(x) for x in fields[1:]])
            except ValueError:
                continue
        if len(rows) == 6:
            break
    if len(rows) != 6:
        return None
    return np.array(rows)

"""LAMMPS data files, input scripts, dumps, logs and `fix ave/time` output.

LAMMPS cells are lower triangular (a along x, b in the xy plane).
`lammps_cell` gives that form of a cell and the rotation to it; pass the
rotation to `stress_from_pressure` to get stresses in the original frame.

Type t is element `atom_types(...)[t - 1]`, which is also the order
`pair_coeff * *` expects. Written data files name each type's element in a
Masses comment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from jinja2 import Template

from chmpy.calc.system import System
from chmpy.core.element import Element
from chmpy.templates import load_template

LAMMPS_INPUT_TEMPLATE = load_template("lammps_input")

#: pressure unit of each LAMMPS unit style, in eV/A^3
PRESSURE_UNITS = {
    "metal": 1.0 / 1.602176634e6,  # bar
    "real": 101325.0 / 1.602176634e11,  # atm
}

#: Atoms section columns before x y z, by atom style
ATOM_STYLE_COLUMNS = {
    "atomic": ("id", "type"),
    "charge": ("id", "type", "q"),
    "molecular": ("id", "molecule", "type"),
    "full": ("id", "molecule", "type", "q"),
}

_STYLE_BY_WIDTH = {5: "atomic", 6: "charge", 7: "full"}

_SECTIONS = {
    "Atoms",
    "Velocities",
    "Masses",
    "Ellipsoids",
    "Lines",
    "Triangles",
    "Bodies",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
}


def lammps_cell(cell):
    """The LAMMPS form of a cell and the rotation that produces it.

    Args:
        cell: (3, 3) lattice vectors as rows, right-handed

    Returns:
        (lower, rotation): the lower-triangular cell and the rotation R with
        lower = cell @ R.T; vectors map as v @ R.T, tensors as R @ t @ R.T
    """
    cell = np.asarray(cell, dtype=float)
    if np.linalg.det(cell) <= 0:
        raise ValueError(
            "LAMMPS needs a right-handed cell; reorder or negate a lattice vector"
        )
    a, b, c = cell
    lx = np.linalg.norm(a)
    xy = np.dot(b, a) / lx
    ly = np.sqrt(np.dot(b, b) - xy**2)
    xz = np.dot(c, a) / lx
    yz = (np.dot(b, c) - xy * xz) / ly
    lz = np.sqrt(np.dot(c, c) - xz**2 - yz**2)
    lower = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    rotation = np.linalg.solve(cell, lower).T
    return lower, rotation


def atom_types(numbers, species=None) -> list[int]:
    """Atomic number of each LAMMPS type, in type order.

    Args:
        numbers: atomic numbers present
        species: elements in the order to use, may include extras; default
            is ascending atomic number

    Returns:
        list[int]: atomic number of each type
    """
    present = sorted({int(z) for z in numbers})
    if species is None:
        return present
    order = [Element[s].atomic_number for s in species]
    missing = set(present) - set(order)
    if missing:
        names = sorted(Element[z].symbol for z in missing)
        raise ValueError(f"species leaves out elements present: {names}")
    if len(set(order)) != len(order):
        raise ValueError(f"species repeats an element: {list(species)}")
    return order


def type_elements(numbers, species=None) -> list[str]:
    """Element symbols in type order, as `pair_coeff * *` takes them.

    Args:
        numbers: atomic numbers present
        species: type order, as for `atom_types`

    Returns:
        list[str]
    """
    return [Element[z].symbol for z in atom_types(numbers, species)]


def system_to_data_string(system, species=None, title="written by chmpy") -> str:
    """A LAMMPS data file (atom style atomic) for a periodic System, rotated
    into LAMMPS's frame with atoms wrapped into the box.

    Args:
        system: a periodic System
        species: type order, as for `atom_types`
        title: first line of the file

    Returns:
        str: data file contents
    """
    if not system.periodic:
        raise ValueError("system_to_data_string needs a periodic system")
    lower, _ = lammps_cell(system.cell)
    types = atom_types(system.numbers, species)
    type_of = {z: t for t, z in enumerate(types, start=1)}
    frac = system.scaled_positions % 1.0
    positions = frac @ lower

    (lx, _, _), (xy, ly, _), (xz, yz, lz) = lower
    lines = [
        title,
        "",
        f"{len(system)} atoms",
        f"{len(types)} atom types",
        "",
        f"0.0 {lx:.10f} xlo xhi",
        f"0.0 {ly:.10f} ylo yhi",
        f"0.0 {lz:.10f} zlo zhi",
        f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz",
        "",
        "Masses",
        "",
    ]
    for t, z in enumerate(types, start=1):
        element = Element[z]
        lines.append(f"{t} {element.mass:.6f} # {element.symbol}")
    lines += ["", "Atoms # atomic", ""]
    for i, (z, (x, y, w)) in enumerate(
        zip(system.numbers, positions, strict=True), start=1
    ):
        lines.append(f"{i} {type_of[int(z)]} {x:.10f} {y:.10f} {w:.10f}")
    lines.append("")
    return "\n".join(lines)


def to_lammps_input(
    data_file,
    pair,
    elements,
    sections=(),
    preamble="",
    units="metal",
    atom_style="atomic",
    boundary="p p p",
    title="written by chmpy",
    template=None,
    **context,
) -> str:
    """A LAMMPS input script: read a data file, set up a potential, then run
    the given sections.

    Args:
        data_file: data file path, as LAMMPS will see it
        pair: pair_style/pair_coeff commands; `{elements}` is replaced by the
            element symbols in type order
        elements: element of each type, in type order (see `type_elements`)
        sections: command blocks after the potential, as a list, or a dict
            whose keys become comment headings
        preamble: commands needed before `read_data`, e.g. `package`
        units, atom_style, boundary: LAMMPS settings
        title: comment on the first line
        template: jinja2 Template or template text replacing the built-in
            layout; it gets the arguments above, with `pair` filled in and
            `sections` as (name, text) pairs
        **context: extra variables for a custom template

    Returns:
        str
    """
    symbols = [Element[e].symbol for e in elements]
    if isinstance(sections, Mapping):
        sections = list(sections.items())
    else:
        sections = [(None, text) for text in sections]
    if isinstance(template, str):
        template = Template(template)
    template = template or LAMMPS_INPUT_TEMPLATE
    text = template.render(
        data_file=data_file,
        pair=pair.replace("{elements}", " ".join(symbols)).strip(),
        elements=symbols,
        sections=[(name, block.strip()) for name, block in sections],
        preamble=preamble.strip(),
        units=units,
        atom_style=atom_style,
        boundary=boundary,
        title=title,
        **context,
    )
    return text.rstrip() + "\n"


def _strip(line):
    content, _, comment = line.partition("#")
    return content.strip(), comment.strip()


def _section_name(content):
    if content in _SECTIONS or content.endswith("Coeffs"):
        return content
    return None


def _element_named(symbol):
    "Atomic number for an exact element symbol, else None"
    try:
        element = Element[symbol]
    except ValueError:
        return None
    return element.atomic_number if element.symbol == symbol else None


def _element_for_mass(mass):
    best = min((Element[z] for z in range(1, 87)), key=lambda e: abs(e.mass - mass))
    if abs(best.mass - mass) > 0.1:
        raise ValueError(
            f"no element has mass {mass}; pass species to name the atom types"
        )
    return best.atomic_number


def _style(width, atom_style, hint):
    "The Atoms section's atom style"
    style = atom_style or (hint.split()[0] if hint else None)
    if style is None:
        style = _STYLE_BY_WIDTH.get(width) or _STYLE_BY_WIDTH.get(width - 3)
    if style not in ATOM_STYLE_COLUMNS:
        raise ValueError(
            f"cannot read atom style {style!r}; pass one of "
            f"{sorted(ATOM_STYLE_COLUMNS)} as atom_style"
        )
    return style


def parse_data_string(contents, species=None, atom_style=None) -> System:
    """Read a LAMMPS data file.

    Elements are taken from `species`, else the Masses comments, else the
    nearest element by mass. Positions are relative to the box origin, with
    image flags applied.

    Args:
        contents: data file text
        species: element of each type, in type order
        atom_style: Atoms section style; defaults to the section's comment,
            else a guess from the column count

    Returns:
        System
    """
    header = {}
    bounds = {}
    tilt = (0.0, 0.0, 0.0)
    sections = {}
    hints = {}
    current = None
    for line in contents.splitlines()[1:]:
        content, comment = _strip(line)
        if not content:
            continue
        name = _section_name(content)
        if name is not None:
            current = name
            sections[name] = []
            hints[name] = comment
            continue
        if current is not None:
            sections[current].append((content.split(), comment))
            continue
        words = content.split()
        if content.endswith("xy xz yz"):
            tilt = tuple(float(w) for w in words[:3])
        elif len(words) == 4 and words[2][1:] == "lo":
            bounds[words[2][0]] = (float(words[0]), float(words[1]))
        elif words[0].isdigit():
            header[" ".join(words[1:])] = int(words[0])

    if "Atoms" not in sections:
        raise ValueError("data file has no Atoms section")
    origin = np.array([bounds[k][0] for k in "xyz"])
    lx, ly, lz = (bounds[k][1] - bounds[k][0] for k in "xyz")
    xy, xz, yz = tilt
    cell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])

    if species is not None:
        type_numbers = {
            t: Element[s].atomic_number for t, s in enumerate(species, start=1)
        }
    elif "Masses" in sections:
        type_numbers = {}
        for words, comment in sections["Masses"]:
            symbol = comment.split()[0] if comment else ""
            type_numbers[int(words[0])] = _element_named(symbol) or (
                _element_for_mass(float(words[1]))
            )
    else:
        raise ValueError("data file has no Masses section; pass species")

    rows = [words for words, _ in sections["Atoms"]]
    style = _style(len(rows[0]), atom_style, hints["Atoms"])
    columns = ATOM_STYLE_COLUMNS[style]
    width = len(columns) + 3

    table = sorted(rows, key=lambda r: int(r[0]))
    if "atoms" in header and len(table) != header["atoms"]:
        raise ValueError(
            f"header declares {header['atoms']} atoms, Atoms section has {len(table)}"
        )
    type_column = columns.index("type")
    numbers = [type_numbers[int(r[type_column])] for r in table]
    positions = np.array([[float(w) for w in r[width - 3 : width]] for r in table])
    positions -= origin
    if len(table[0]) >= width + 3:
        images = np.array([[int(w) for w in r[width : width + 3]] for r in table])
        positions += images @ cell
    return System(numbers, positions, cell=cell)


@dataclass
class DumpFrame:
    """One snapshot from a LAMMPS text dump.

    Attributes:
        timestep: the step it was written at
        cell: (3, 3) lower-triangular cell
        origin: (3,) box corner (xlo, ylo, zlo)
        columns: each per-atom quantity, ordered by atom id
    """

    timestep: int
    cell: np.ndarray
    origin: np.ndarray
    columns: dict

    @property
    def positions(self) -> np.ndarray:
        "(N, 3) positions relative to the box origin, unwrapped if available"
        for keys, scaled in (
            (("xu", "yu", "zu"), False),
            (("x", "y", "z"), False),
            (("xsu", "ysu", "zsu"), True),
            (("xs", "ys", "zs"), True),
        ):
            if all(k in self.columns for k in keys):
                values = np.column_stack([self.columns[k] for k in keys])
                return values @ self.cell if scaled else values - self.origin
        raise ValueError(f"dump has no coordinates: {sorted(self.columns)}")


def _dump_box(item, lines):
    "cell and origin from a BOX BOUNDS item and its three lines"
    if "abc" in item.split():
        raise ValueError("general triclinic dump boxes are not supported")
    values = np.array([[float(w) for w in line.split()] for line in lines])
    xy, xz, yz = values[:, 2] if values.shape[1] == 3 else (0.0, 0.0, 0.0)
    xlo = values[0, 0] - min(0.0, xy, xz, xy + xz)
    xhi = values[0, 1] - max(0.0, xy, xz, xy + xz)
    ylo = values[1, 0] - min(0.0, yz)
    yhi = values[1, 1] - max(0.0, yz)
    zlo, zhi = values[2, :2]
    cell = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
    return cell, np.array([xlo, ylo, zlo])


def parse_dump_string(contents) -> list[DumpFrame]:
    """Every frame of a LAMMPS text dump (`dump atom` or `dump custom`).

    Args:
        contents: dump file text

    Returns:
        list[DumpFrame]
    """
    lines = contents.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        timestep = int(lines[i + 1].split()[0])
        if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"expected NUMBER OF ATOMS at line {i + 3}")
        n = int(lines[i + 3])
        box_item = lines[i + 4]
        cell, origin = _dump_box(box_item, lines[i + 5 : i + 8])
        names = lines[i + 8].split()[2:]
        rows = [line.split() for line in lines[i + 9 : i + 9 + n]]
        if len(rows) < n:
            break
        table = np.array(rows, dtype=float)
        if "id" in names:
            table = table[np.argsort(table[:, names.index("id")], kind="stable")]
        columns = {name: table[:, k] for k, name in enumerate(names)}
        for name in ("id", "type"):
            if name in columns:
                columns[name] = columns[name].astype(int)
        frames.append(DumpFrame(timestep, cell, origin, columns))
        i += 9 + n
    return frames


def _numbers(words):
    try:
        return [float(w) for w in words]
    except ValueError:
        return None


def parse_log_string(contents) -> list[dict]:
    """Thermo output of each run in a LAMMPS log (one-line thermo style).

    Non-numeric lines such as warnings are skipped. An unfinished run keeps
    the rows written so far.

    Args:
        contents: log file text

    Returns:
        list[dict]: for each run, each thermo column as an array
    """
    runs = []
    names = None
    rows = []
    lines = iter(contents.splitlines())
    for line in lines:
        if line.startswith("Per MPI rank memory allocation"):
            names = next(lines, "").split()
            rows = []
        elif names is not None:
            if line.startswith("Loop time of"):
                runs.append(_columns(names, rows))
                names = None
                continue
            values = _numbers(line.split())
            if values is not None and len(values) == len(names):
                rows.append(values)
    if names is not None:
        runs.append(_columns(names, rows))
    return runs


def _columns(names, rows):
    table = np.array(rows, dtype=float).reshape(-1, len(names))
    return {name: table[:, k] for k, name in enumerate(names)}


def parse_ave_time_string(contents) -> dict:
    """Columns of a `fix ave/time` output file, keyed by header name.
    `mode vector` files are not supported.

    Args:
        contents: file text

    Returns:
        dict
    """
    names = None
    rows = []
    for line in contents.splitlines():
        if line.startswith("#"):
            words = line[1:].split()
            if words and not line.startswith("# Time-averaged"):
                names = words
            continue
        values = _numbers(line.split())
        if values:
            rows.append(values)
    if names is None:
        raise ValueError("no column header found in fix ave/time output")
    if rows and len(rows[0]) != len(names):
        raise ValueError(
            f"{len(names)} columns named but rows have {len(rows[0])} values; "
            "mode vector output is not supported"
        )
    return _columns(names, rows)


def stress_from_pressure(pressure, units="metal", rotation=None) -> np.ndarray:
    """A LAMMPS pressure tensor as a stress, in eV/A^3.

    Stress is positive in tension, the opposite sign to pressure.

    Args:
        pressure: (pxx, pyy, pzz, pxy, pxz, pyz), the order of `compute
            pressure`'s vector, or a (3, 3) tensor
        units: the LAMMPS unit style, "metal" (bar) or "real" (atm)
        rotation: rotation from `lammps_cell`; rotates the stress back to
            the original frame

    Returns:
        (3, 3) stress tensor
    """
    if units not in PRESSURE_UNITS:
        raise ValueError(f"units must be one of {sorted(PRESSURE_UNITS)}")
    pressure = np.asarray(pressure, dtype=float)
    if pressure.shape == (6,):
        xx, yy, zz, xy, xz, yz = pressure
        pressure = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    elif pressure.shape != (3, 3):
        raise ValueError(
            f"pressure must have 6 components or be 3x3, not {pressure.shape}"
        )
    stress = -pressure * PRESSURE_UNITS[units]
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        stress = rotation.T @ stress @ rotation
    return stress

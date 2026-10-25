import logging
from pathlib import Path

import numpy as np

from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def parse_xyz_string(contents, filename=None):
    """Convert provided xmol .xyz file contents into an array of
    atomic numbers and cartesian positions

    Args:
        contents: text contents of the .xyz file to read

    Returns:
        array of (N) atomic numbers and (N, 3) Cartesian positions read from
        the given file
    """
    lines = contents.splitlines()
    natom = int(lines[0].strip())
    LOG.debug("Expecting %d atoms %s", natom, "in " + filename if filename else "")
    elements = []
    positions = []
    for line in lines[2 : 2 + natom]:
        if not line.strip():
            break
        tokens = line.strip().split()
        xyz = tuple(float(x) for x in tokens[1:4])
        positions.append(xyz)
        elements.append(Element[tokens[0]])
    LOG.debug(
        "Found %d atoms lines in %s",
        len(elements),
        "in " + filename if filename else "",
    )
    return elements, np.asarray(positions)


def parse_traj_string(contents, filename=None):
    """Convert provided xmol .xyz file contents into list of arrays of
    atomic numbers and cartesian positions

    Args:
        contents: text contents of the .xyz file to read

    Returns:
        list of (N) :obj:`Element` and (N, 3) Cartesian positions read from
        the given file
    """
    lines = contents.splitlines()
    i = 0
    frames = []
    while i < len(lines):
        natom = int(lines[i].strip())
        elements = []
        positions = []
        i += 1
        comment = lines[i]
        i += 1
        for line in lines[i : i + natom]:
            if not line.strip():
                continue
            tokens = line.strip().split()
            xyz = tuple(float(x) for x in tokens[1:4])
            positions.append(xyz)
            elements.append(Element[tokens[0]])
        frames.append((elements, comment, np.asarray(positions)))
        i += natom
    return frames


def parse_xyz_file(filename):
    """Convert a provided xmol .xyz file into an array of
    atomic numbers and cartesian positions

    Args:
        filename: path to the .xyz file to read

    Returns:
        array of (N) atomic numbers and (N, 3) Cartesian positions read from
        the given file
    """
    path = Path(filename)
    return parse_xyz_string(path.read_text(), filename=str(path.absolute()))


def parse_traj_file(filename):
    """Convert a provided xmol .xyz file into an array of
    atomic numbers and cartesian positions

    Args:
        filename: path to the .xyz file to read

    Returns:
        array of (N) atomic numbers and (N, 3) Cartesian positions read from
        the given file
    """
    path = Path(filename)
    return parse_traj_string(path.read_text(), filename=str(path.absolute()))


def slab_to_extxyz_string(slab) -> str:
    """Extended XYZ for a slab, with occ-compatible surface metadata.

    Per-atom properties: species, pos, mol_idx (the molecule or atom of the
    slab each atom belongs to), fixed, and when known uc_idx (bulk unit cell
    molecule, -1 for atomic cuts) and layer.
    """
    lattice = " ".join(f"{x:.8f}" for x in slab.cell.flatten())
    header = []
    if slab.hkl is not None:
        h, k, l = slab.hkl
        header.append(f"MillerH={h} MillerK={k} MillerL={l}")
    if slab.offset is not None:
        header.append(f"Cut={slab.offset:.6f}")
    if slab.layers is not None:
        header.append(f"Layers={slab.layers}")
    if slab.symmetry is not None:
        header.append(f"PlaneGroup={slab.plane_group.symbol}")
    header.append(f'Lattice="{lattice}"')

    columns = [("mol_idx", "I", slab.fragment)]
    if slab.molecule is not None:
        columns.append(("uc_idx", "I", slab.molecule))
    if slab.layer is not None:
        columns.append(("layer", "I", slab.layer))
    columns.append(("fixed", "L", slab.fixed))
    spec = "".join(f":{name}:{kind}:1" for name, kind, _ in columns)
    header.append(f'Properties=species:S:1:pos:R:3{spec} pbc="T T T"')

    lines = [str(len(slab)), " ".join(header)]
    for i, (n, (x, y, z)) in enumerate(zip(slab.elements, slab.positions, strict=True)):
        values = []
        for _, kind, data in columns:
            values.append(("T" if data[i] else "F") if kind == "L" else str(data[i]))
        lines.append(
            f"{Element[n].symbol} {x:14.8f} {y:14.8f} {z:14.8f} " + " ".join(values)
        )
    lines.append("")
    return "\n".join(lines)


def parse_extxyz_string(contents) -> tuple[dict, dict]:
    """Parse a single-frame extended XYZ.

    Returns:
        the header key=value pairs, and per-atom columns keyed by property
        name (species as symbols, pos as an (N, 3) array)
    """
    import shlex

    lines = contents.splitlines()
    natom = int(lines[0].split()[0])
    info = {}
    for token in shlex.split(lines[1]):
        key, _, value = token.partition("=")
        info[key] = value
    spec = info.get("Properties", "species:S:1:pos:R:3").split(":")
    rows = [line.split() for line in lines[2 : 2 + natom]]
    columns, start = {}, 0
    for name, kind, width in zip(spec[::3], spec[1::3], spec[2::3], strict=True):
        width = int(width)
        raw = [r[start : start + width] for r in rows]
        start += width
        if kind == "R":
            values = np.array(raw, dtype=float)
        elif kind == "I":
            values = np.array(raw, dtype=int)
        elif kind == "L":
            values = np.array([[x.upper() in ("T", "TRUE") for x in r] for r in raw])
        else:
            values = np.array(raw, dtype=object)
        columns[name] = values[:, 0] if width == 1 else values
    return info, columns


def extxyz_string_to_slab(contents):
    "Read a Slab from extended XYZ written by `slab_to_extxyz_string`."
    from chmpy.crystal.slab import Slab

    info, columns = parse_extxyz_string(contents)
    if "Lattice" not in info:
        raise ValueError("A slab extxyz needs a Lattice")
    cell = np.array(info["Lattice"].split(), dtype=float).reshape(3, 3)
    kwargs = {}
    if "MillerH" in info:
        kwargs["hkl"] = tuple(int(info[f"Miller{x}"]) for x in "HKL")
    if "Cut" in info:
        kwargs["offset"] = float(info["Cut"])
    if "Layers" in info:
        kwargs["layers"] = int(info["Layers"])
    for name, key in (
        ("fixed", "fixed"),
        ("layer", "layer"),
        ("uc_idx", "molecule"),
        ("mol_idx", "fragment"),
    ):
        if name in columns:
            kwargs[key] = columns[name]
    return Slab(cell, list(columns["species"]), columns["pos"], **kwargs)

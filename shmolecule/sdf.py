import numpy as np
import logging
from collections import defaultdict
from pathlib import Path

LOG = logging.getLogger(__name__)

_COUNTS_FIELDS = (
    ("atoms", int, 3),
    ("bonds", int, 3),
    ("atom_list", int, 3),
    ("obselete", None, 3),
    ("chiral", bool, 3),
    ("stext", int, 3),
    ("obselete", None, 3),
    ("obselete", None, 3),
    ("obselete", None, 3),
    ("obselete", None, 3),
    ("additional", int, 3),
    ("version", str, None),
)

_ATOM_FIELDS = (
    ("x", float, 10),
    ("y", float, 10),
    ("z", float, 10),
    ("space", None, 1),
    ("symbol", lambda x: x.strip(), 3),
    ("mass_difference", int, 2),
    ("charge", int, 3),
    ("stereo", int, 3),
    ("hydrogen_count", int, 3),
    ("stereo_care_box", bool, 3),
    ("valence", int, 3),
    ("h0_designator", bool, 3),
    ("not_used", None, 3),
    ("not_used", None, 3),
    ("mapping", int, 3),
    ("inversion", int, 3),
    ("exact_exchange", int, 3),
)

_BOND_FIELDS = (
    ("left", int, 3),
    ("right", int, 3),
    ("type", int, 3),
    ("stereo", int, 3),
    ("not_used", None, 3),
    ("topology", int, 3),
    ("center_status", int, 3),
)


def parse_atom_lines(lines):
    atom_data = defaultdict(list)
    for line in lines:
        n = 0
        for (name, parser, length) in _ATOM_FIELDS:
            if parser is not None:
                atom_data[name].append(parser(line[n : n + length]))
            n += length
    return {x: np.array(y) for x, y in atom_data.items()}


def parse_bond_lines(lines):
    bond_data = defaultdict(list)
    for line in lines:
        n = 0
        for (name, parser, length) in _BOND_FIELDS:
            if parser is not None:
                bond_data[name].append(parser(line[n : n + length]))
            n += length
    return {x: np.array(y) for x, y in bond_data.items()}


def parse_counts_line(line):
    n = 0
    result = {}
    for (name, parser, length) in _COUNTS_FIELDS:
        if length is None:
            result[name] = parser(line[n:].strip())
        else:
            if parser is not None:
                result[name] = parser(line[n : n + length])
            n += length
    return result


def parse_data_lines(lines):
    sections = "".join(lines).strip().split("> <")
    result = {}
    for section in sections:
        if not section:
            continue
        i = section.find(">")
        k = section[:i]
        v = section[i + 1 :]
        result[k] = v
    return result


def parse_property_line(line):
    n = int(line[:4])
    l = 4
    props = []
    for i in range(n):
        props.append((int(line[l : l + 4]), int(line[l + 4 : l + 8])))
    return props


def parse_property_lines(lines):
    charges = []
    isotopes = []

    for line in lines:
        if "CHG" in line:
            charges += parse_property_line(line[6:])
        elif "ISO" in line:
            isotopes += parse_property_line(line[6:])
    return charges, isotopes


def parse_sdf_file(filename, limit=None, progress=False, keep_sdf_text=False):
    contents = Path(filename).read_text()
    compounds = contents.split("$$$$\n")
    results = []
    if limit is None:
        limit = len(compounds)
    update = lambda x: None

    if progress:
        from tqdm import tqdm

        pbar = tqdm(desc="Loading SDF V2000 file", total=len(compounds), leave=False)
        update = pbar.update

    for compound in compounds[:limit]:
        lines = compound.splitlines()
        if len(lines) == 0:
            continue
        header = lines[:3]
        counts = parse_counts_line(lines[3])
        if counts["version"] != "V2000":
            raise ValueError("Only V2000 files are supported")
        l, u = 4, 4 + counts["atoms"]
        atom_lines = lines[l:u]
        atoms = parse_atom_lines(atom_lines)
        l, u = u, u + counts["bonds"]
        bond_lines = lines[l:u]
        bonds = parse_bond_lines(bond_lines)
        l = u
        while lines[u].startswith("M "):
            u += 1
        property_lines = lines[l : u - 1]
        charges, isotopes = parse_property_lines(property_lines)
        LOG.debug("Overriding charges with values from M  CHG block")
        for idx, charge in charges:
            atoms["charge"][idx - 1] = charge
        LOG.debug("Ignoring isotopes set with M  ISO")
        data_lines = lines[u:]
        additional_data = parse_data_lines(data_lines)
        result = {
            "header": header,
            "atoms": atoms,
            "bonds": bonds,
            "data": additional_data,
        }
        if keep_sdf_text:
            result["sdf"] = compound
        results.append(result)
        update(1)

    if progress:
        pbar.close()

    return results

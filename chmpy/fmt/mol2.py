from collections import defaultdict
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)

# Might be useful one day, currently mostly ignored
_ATOM_TYPES = {
    "C.3": "carbon sp3",
    "C.2": "carbon sp2",
    "C.1": "carbon sp",
    "C.ar": "carbon aromatic",
    "C.cat": "cabocation (C+) used only in a guadinium group",
    "N.3": "nitrogen sp3",
    "N.2": "nitrogen sp2",
    "N.1": "nitrogen sp",
    "N.ar": "nitrogen aromatic",
    "N.am": "nitrogen amide",
    "N.pl3": "nitrogen trigonal planar",
    "N.4": "nitrogen sp3 positively charged",
    "O.3": "oxygen sp3",
    "O.2": "oxygen sp2",
    "O.co2": "oxygen in carboxylate and phosphate groups",
    "O.spc": "oxygen in Single Point Charge (SPC) water model",
    "O.t3p": "oxygen in Transferable Intermolecular Potential (TIP3P) water model",
    "S.3": "sulfur sp3",
    "S.2": "sulfur sp2",
    "S.O": "sulfoxide sulfur",
    "S.O2/S.o2": "sulfone sulfur",
    "P.3": "phosphorous sp3",
    "F": "fluorine",
    "H": "hydrogen",
    "H.spc": "hydrogen in Single Point Charge (SPC) water model",
    "H.t3p": "hydrogen in Transferable Intermolecular Potential (TIP3P) water model",
    "LP": "lone pair",
    "Du": "dummy atom",
    "Du.C": "dummy carbon",
    "Any": "any atom",
    "Hal": "halogen",
    "Het": "heteroatom = N, O, S, P",
    "Hev": "heavy atom (non hydrogen)",
    "Li": "lithium",
    "Na": "sodium",
    "Mg": "magnesium",
    "Al": "aluminum",
    "Si": "silicon",
    "K": "potassium",
    "Ca": "calcium",
    "Cr.thm": "chromium (tetrahedral)",
    "Cr.oh": "chromium (octahedral)",
    "Mn": "manganese",
    "Fe": "iron",
    "Co.oh": "cobalt (octahedral)",
    "Cu": "copper",
}

_ATOM_FIELDS = (
    ("id", int),
    ("name", str),
    ("x", float),
    ("y", float),
    ("z", float),
    ("type", str),
    ("mol_id", int),
    ("mol_name", str),
    ("charge", float),
    ("status_bits", str),
)

_BOND_FIELDS = (
    ("bond_id", int),
    ("origin", int),
    ("target", int),
    ("type", str),
    ("status_bits", str),
)


def parse_atom_lines(lines):
    atom_data = defaultdict(list)
    for line in lines[1:]:
        for (n, f), tok in zip(_ATOM_FIELDS, line.split()):
            atom_data[n].append(f(tok))
    return atom_data


def parse_bond_lines(lines):
    bond_data = defaultdict(list)
    for line in lines[1:]:
        for (n, f), tok in zip(_BOND_FIELDS, line.split()):
            bond_data[n].append(f(tok))
    return bond_data


def parse_mol2_string(string):
    # only parse bonds and atoms for now
    atom_section = "@<TRIPOS>ATOM"
    bond_section = "@<TRIPOS>BOND"
    atom_lines = []
    bond_lines = []
    lines = string.splitlines()
    unknown_lines = []
    app = unknown_lines
    for i in range(len(lines)):
        line = lines[i].strip()
        if "@" in line:
            app = unknown_lines
            if atom_section in line:
                app = atom_lines
            elif bond_section in line:
                app = bond_lines
        if line:
            app.append(line)
    return parse_atom_lines(atom_lines), parse_bond_lines(bond_lines)


def parse_mol2_file(filename):
    return parse_mol2_string(Path(filename).read_text())

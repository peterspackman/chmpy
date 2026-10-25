import logging
import re
from pathlib import Path

from chmpy.crystal import SymmetryOperation

LOG = logging.getLogger(__name__)


def _parse_titl(line):
    return line.split()[-1]


def _parse_cell(line):
    tokens = [float(x) for x in line.split()[1:]]
    return {
        "wavelength": tokens[0],
        "lengths": tuple(tokens[1:4]),
        "angles": tuple(tokens[4:7]),
    }


def _parse_int(line):
    return int(line.split()[1])


def _parse_sfac(line):
    return tuple(line.split()[1:])


def _parse_symm(line):
    return SymmetryOperation.from_string_code(line[4:])


def _parse_fvar(line):
    return tuple(float(x) for x in line.split()[1:])


def _occupation(sof, fvar):
    # SHELXL encodes site occupation factors as 10 * m + p:
    # |m| = 1 is fixed at p, otherwise p * fv(m), or p * (fv(-m) - 1) if m < 0
    if abs(sof) <= 5:
        return sof
    m = int(sof / 10)
    p = sof - 10 * m
    if m == 1:
        return p
    if m == -1:
        return -p
    fv = fvar[abs(m) - 1] if abs(m) <= len(fvar) else 1.0
    return p * fv if m > 0 else p * (fv - 1)


def _parse_atom_line(sfac, line, fvar=()):
    tokens = line.split()
    label = tokens[0]
    LOG.debug("Parsing atom line: %s", line)
    sfac_idx = int(tokens[1])
    a, b, c = (float(x) for x in tokens[2:5])
    occupation = 1.0
    if len(tokens) > 5:
        occupation = _occupation(float(tokens[5]), fvar)
    return {
        "label": label,
        "element": sfac[sfac_idx - 1],
        "position": (a, b, c),
        "occupation": occupation,
    }


SHELX_LINE_KEYS = {
    "TITL": _parse_titl,
    "CELL": _parse_cell,
    "ZERR": _parse_int,
    "LATT": _parse_int,
    "SFAC": _parse_sfac,
    "SYMM": _parse_symm,
    "FVAR": _parse_fvar,
}

# other SHELXL instructions, which are ignored
_IGNORED_INSTRUCTIONS = {
    "ABIN", "ACTA", "AFIX", "ANIS", "ANSC", "ANSR", "BASF", "BEDE", "BIND",
    "BLOC", "BOND", "BUMP", "CGLS", "CHIV", "CONF", "CONN", "DAMP", "DANG",
    "DEFS", "DELU", "DFIX", "DISP", "EADP", "EGEN", "EQIV", "ESEL", "EXTI",
    "EXYZ", "FEND", "FLAT", "FMAP", "FRAG", "FREE", "GRID", "HFIX", "HKLF",
    "HTAB", "ISOR", "LAUE", "LIST", "L.S.", "LONE", "MERG", "MOLE", "MORE",
    "MOVE", "MPLA", "NCSY", "NEUT", "OMIT", "PART", "PLAN", "PRIG", "REM",
    "RESI", "RIGU", "RTAB", "SADI", "SAME", "SHEL", "SIMU", "SIZE", "SPEC",
    "STIR", "SUMP", "SWAT", "TEMP", "TIME", "TWIN", "TWST", "UNIT", "WGHT",
    "WIGL", "WPDB", "XNPD",
}  # fmt: skip


def _logical_lines(file_content):
    # join lines continued with a trailing '=' (e.g. anisotropic atoms)
    line_number, buffer = 0, ""
    for i, line in enumerate(file_content.split("\n"), start=1):
        if not buffer:
            line_number = i
        line = line.rstrip()
        if line.endswith("=") and not line.upper().startswith("REM"):
            buffer += line[:-1] + " "
            continue
        yield line_number, buffer + line
        buffer = ""
    if buffer:
        yield line_number, buffer


def parse_shelx_file_content(file_content):
    """Read a SHELX formatted crystal structure from
    a string

    Args:
        file_content: text contents of the SHELX .res file to read

    Returns:
        dictionary of parsed shelx data
    """
    shelx_dict = {"SYMM": [SymmetryOperation.from_string_code("x,y,z")], "ATOM": []}
    for line_number, line in _logical_lines(file_content):
        try:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            key = line.split()[0][:4].upper()
            if key == "END":
                break
            elif key == "SYMM":
                shelx_dict[key].append(SHELX_LINE_KEYS[key](line))
            elif key in SHELX_LINE_KEYS:
                shelx_dict[key] = SHELX_LINE_KEYS[key](line)
            elif key in _IGNORED_INSTRUCTIONS:
                continue
            elif re.match(r"Q\d+$", line.split()[0], re.IGNORECASE):
                # residual electron density peaks
                continue
            else:
                shelx_dict["ATOM"].append(
                    _parse_atom_line(
                        shelx_dict["SFAC"], line, shelx_dict.get("FVAR", ())
                    )
                )
        except Exception as e:
            raise ValueError(f"Error parsing shelx string: line {line_number}") from e
    return shelx_dict


def parse_shelx_file(filename):
    """Read a SHELX formatted .res file.

    Args:
        filename: path to the shelx .res file to read

    Returns:
        dictionary of parsed shelx data
    """
    return parse_shelx_file_content(Path(filename).read_text())


def _cell_string(value):
    rvalues = [int(x) if x.is_integer() else round(x, 6) for x in value]
    return "CELL 0.7 {} {} {} {} {} {}".format(*rvalues)


def _atom_lines(atoms):
    return "\n".join(atoms)


def to_res_contents(shelx_data):
    """
    Args:
        shelx_data: dictionary of data to write into a SHELX .res format

    Returns:
        the string encoded contents of this shelx_data
    """
    SHELX_FORMATTERS = {
        "TITL": lambda x: f"TITL {x}",
        "CELL": _cell_string,
        "LATT": lambda x: f"LATT {x}",
        "SYMM": lambda symm: "\n".join(f"SYMM {x}" for x in symm),
        "SFAC": lambda x: "SFAC " + " ".join(x),
        "ATOM": _atom_lines,
    }
    sections = []
    for key in SHELX_FORMATTERS:
        sections.append(SHELX_FORMATTERS[key](shelx_data[key]))
    return "\n".join(sections)

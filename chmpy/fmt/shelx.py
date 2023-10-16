import logging
from pathlib import Path
from chmpy.crystal import SymmetryOperation, SpaceGroup

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


def _parse_atom_line(sfac, line):
    tokens = line.split()
    ntokens = len(tokens)
    label = tokens[0]
    LOG.debug("Parsing atom line: %s", line)
    sfac_idx = int(tokens[1])
    a, b, c = (float(x) for x in tokens[2:5])
    occupation = 1.0
    if len(tokens) > 5:
        occupation = float(tokens[5])
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
    "FVAR": None,
    "UNIT": None,
    "REM ": None,
    "MORE": None,
    "TIME": None,
    "OMIT": None,
    "ESEL": None,
    "EGEN": None,
    "LIST": None,
    "FMAP": None,
    "PLAN": None,
    "MOLE": None,
    "HKLF": None,
}


def parse_shelx_file_content(file_content):
    """Read a SHELX formatted crystal structure from
    a string
    Parameters
    ----------
    file_content: str
        text contents of the SHELX .res file to read

    Returns
    -------
    dict
        dictionary of parsed shelx data
    """
    contents = file_content.split("\n")
    shelx_dict = {"SYMM": [SymmetryOperation.from_string_code("x,y,z")], "ATOM": []}
    for line_number, line in enumerate(contents, start=1):
        try:
            line = line.strip()
            if not line:
                continue
            key = line[:4].upper()
            if key == "END":
                break
            elif key == "SYMM":
                shelx_dict[key].append(SHELX_LINE_KEYS[key](line))
            elif key not in SHELX_LINE_KEYS:
                shelx_dict["ATOM"].append(_parse_atom_line(shelx_dict["SFAC"], line))
            else:
                f = SHELX_LINE_KEYS[key]
                if f is None:
                    continue
                shelx_dict[key] = f(line)
        except Exception as e:
            raise ValueError(f"Error parsing shelx string: line {line_number}") from e
    return shelx_dict


def parse_shelx_file(filename):
    """Read a SHELX formatted .res file.
    Parameters
    ----------
    filename: str
        path to the shelx .res file to read

    Returns
    -------
    dict
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
    Parameters
    ----------
    shelx_data: dict
        dictionary of data to write into a SHELX .res format

    Returns
    -------
    str
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

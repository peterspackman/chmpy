import logging
from pathlib import Path

import numpy as np

from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def parse_gen_string(contents, filename=None):
    """Convert provided xmol .xyz file contents into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    contents: str
        text contents of the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        List[Element], (N, 3) positions, (4, 3) lattice vectors, bool (if fractional)
        read from the given file
    """
    lines = contents.splitlines()
    natom_str, kind = lines[0].split()
    natom = int(natom_str)
    kind = kind.strip()
    LOG.debug("Expecting %d atoms %s", natom, "in " + filename if filename else "")
    elements_map = [Element[x.strip()] for x in lines[1].split()]

    arr = [[float(x) for x in line.split()] for line in lines[natom + 2 : natom + 6]]
    elements = []
    positions = []
    for line in lines[2 : natom + 2]:
        if not line.strip():
            break
        tokens = line.strip().split()
        xyz = tuple(float(x) for x in tokens[2:5])
        positions.append(xyz)
        el = elements_map[int(tokens[1]) - 1]
        elements.append(el)
    LOG.debug(
        "Found %d atoms lines in %s",
        len(elements),
        "in " + filename if filename else "",
    )
    return elements, np.asarray(positions), np.asarray(arr), kind == "F"


def parse_gen_file(filename):
    """Convert a provided DFTB+ .gen file into an array of
    atomic numbers, positions and

    Parameters
    ----------
    filename: str
        path to the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        List[Element], (N, 3) positions, (4, 3) lattice vectors, bool (if fractional)
        read from the given file
    """
    path = Path(filename)
    return parse_gen_string(path.read_text(), filename=str(path.absolute()))

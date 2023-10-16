import logging
from pathlib import Path
import numpy as np
from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def parse_tmol_string(contents, filename=None):
    """Convert provided turbomole coord file contents into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    contents: str
        text contents of the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        array of (N) atomic numbers and (N, 3) Cartesian positions
        read from the given file
    """
    lines = contents.splitlines()
    angstroms = "angs" in lines[0]
    elements = []
    positions = []
    for line in lines[1:]:
        stripped = line.strip()
        if (not stripped) or "$end" in line:
            break
        if "$" in line:
            continue
        tokens = line.strip().split()
        xyz = tuple(float(x) for x in tokens[:3])
        positions.append(xyz)
        elements.append(Element[tokens[3]])
    LOG.debug(
        "Found %d atoms lines in %s",
        len(elements),
        "in " + filename if filename else "",
    )
    positions = np.array(positions)
    if not angstroms:
        from chmpy.util.unit import units

        positions = units.angstrom(positions)
    return elements, positions


def parse_tmol_file(filename):
    """Convert a provided turbomole coord file into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    filename: str
        path to the turbomole file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        array of (N) atomic numbers and (N, 3) Cartesian positions
        read from the given file
    """
    path = Path(filename)
    return parse_tmol_string(path.read_text(), filename=str(path.absolute()))

import logging
from pathlib import Path
import numpy as np
from .element_data import atomic_number

LOG = logging.getLogger(__name__)


def parse_xyz_file(filename):
    """Convert a provided xmol .xyz file into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    filename: str
        path to the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        array of (N) atomic numbers and (N, 3) Cartesian positions
        read from the given file
    """
    lines = Path(filename).read_text().splitlines()
    natom = int(lines[0].strip())
    LOG.debug("Expecting %d atoms in %s", natom, filename)
    elements = []
    positions = []
    for line in lines[2:]:
        if not line.strip():
            break
        tokens = line.strip().split()
        el = atomic_number(tokens[0])
        xyz = tuple(float(x) for x in tokens[1:4])
        positions.append(xyz)
        elements.append(el)
    LOG.debug("Found %d atoms lines in %s", len(elements), filename)
    return np.asarray(elements), np.asarray(positions)

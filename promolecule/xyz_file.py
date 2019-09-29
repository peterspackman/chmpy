import logging
from pathlib import Path
import numpy as np
from .element_data import atomic_number

LOG = logging.getLogger(__name__)


def parse_xyz_file(filename):
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

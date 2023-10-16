import logging
from pathlib import Path
import numpy as np
from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def parse_poscar(poscar_string):
    "Read in a VASP POSCAR or CONTCAR file"
    result = {
        "name": None,
        "direct": None,
        "elements": None,
        "positions": None,
    }
    lines = poscar_string.splitlines()
    result["name"] = lines[0].strip()
    scale_factor = float(lines[1])
    result["direct"] = scale_factor * np.fromstring(
        " ".join(lines[2:5]), sep=" "
    ).reshape((3, 3))
    elements = []
    for el, x in zip(lines[5].split(), lines[6].split()):
        elements += [Element[el]] * int(x)
    result["elements"] = elements
    result["coord_type"] = lines[7].strip().lower()
    N = len(elements)
    result["positions"] = np.fromstring(" ".join(lines[8 : 8 + N]), sep=" ").reshape(
        (-1, 3)
    )
    return result

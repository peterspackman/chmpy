from . import atomic_form_factors
from . import _sfac
import numpy as np


UNIQUE_REFLECTION_COMBINATIONS = { ("triclinic", "-1", "*"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 0, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 1, 0), ((-1, 0, 0), (0, 1, 0), (0, 0, -1))),
        ((0, 1, -1), ((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    ),
    ("monoclinic", "2/m", "A"): (
        ((0, 0, 0), ((0, 1, 0), (1, 0, 0), (0, 0, 1))),
        ((0, -1, 1), ((0, -1, 0), (1, 0, 0), (0, 0, 1))),
    ),
    ("monoclinic", "2/m", "B"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 0, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("monoclinic", "2/m", "C"): (
        ((0, 0, 0), ((1, 0, 0), (0, 0, 1), (0, 1, 0))),
        ((-1, 1, 0), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)))
    ),
    ("orthorhombic", "mmm", "*"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("tetragonal", "4/mmm", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("tetragonal", "4/m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 2, 0), ((1, 1, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("hexagonal", "6/mmm", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("hexagonal", "6/m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 2, 0), ((0, 1, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("hexagonal", "-3m1", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((0, 1, 1), ((0, 1, 0), (1, 1, 0), (0, 0, 1)))
    ),
    ("hexagonal", "-31m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 1, -1), ((1, 0, 0), (1, 1, 0), (0, 0, -1))),
    ),
    ("hexagonal", "-3m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 0, -1), (1, 1, 1))),
        ((1, 1, 0), ((1, 0, -1), (0, 0, -1), (1, 1, 1)))
    ),
    ("hexagonal", "-3", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 0, -1), (1, 1, 1))),
        ((1, 1, 0), ((1, 0, -1), (0, 0, -1), (1, 1, 1))),
        ((0, -1, -2), ((1, 0, 0), (1, 0,-1), (-1, -1, -1))),
        ((1, 0, -2),  ((1, 0, -1), (0, 0,-1), (-1, -1, -1))),
    ),
    ("cubic", "m3m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),
    ),
    ("cubic", "m3", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),
        ((1, 2, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1)))
    )
}

def pruned_reflections(crystal, wavelength):
    a = crystal.unit_cell.a
    b = crystal.unit_cell.b
    c = crystal.unit_cell.c
    recip = crystal.unit_cell.reciprocal_lattice.copy()
    apexes, mats = [], []
    for apex, mat in UNIQUE_REFLECTION_COMBINATIONS[("orthorhombic", "mmm", "*")]:
        apexes.append(np.array(apex, dtype=np.int32))
        mats.append(np.array(mat, dtype=np.int32))
    apexes = np.vstack(apexes)
    mats = np.stack(mats)
    return _sfac.pruned_reflections(a, b, c, recip, apexes, mats, wavelength)

def calculate_structure_factors(crystal):
    indices = [
        _sfac.get_form_factor_index(el.symbol)
        for el in crystal.elements
    ]

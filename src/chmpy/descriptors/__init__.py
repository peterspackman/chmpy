import numpy as np

from .cna import StructureType, adaptive_cna, fixed_cna, interval_cna
from .symmetry_function_ani1 import SymmetryFunctionsANI1

__all__ = [
    "selling_scalars",
    "StructureType",
    "SymmetryFunctionsANI1",
    "adaptive_cna",
    "fixed_cna",
    "interval_cna",
]


def selling_scalars(unit_cell):
    a = unit_cell.direct[0, :]
    b = unit_cell.direct[1, :]
    c = unit_cell.direct[2, :]
    d = -np.sum(unit_cell.direct, axis=0)
    return np.asarray(
        (
            np.vdot(b, c),
            np.vdot(a, c),
            np.vdot(a, b),
            np.vdot(a, d),
            np.vdot(b, d),
            np.vdot(c, d),
        )
    )

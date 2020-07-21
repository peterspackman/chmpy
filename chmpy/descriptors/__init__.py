import numpy as np
from .symmetry_function_ani1 import SymmetryFunctionsANI1

__all__ = [
    "selling_scalars",
    "SymmetryFunctionsANI1",
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

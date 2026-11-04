import numpy as np

from .cna import StructureType, adaptive_cna, fixed_cna, interval_cna
from .pdd import PDD, amd, pdd, pdd_emd, pdd_rms
from .shape_measure import (
    IDEAL_SHAPES,
    ShapeMatch,
    continuous_shape_measure,
    coordination_shapes,
    shape_match,
    shape_measures,
)
from .symmetry_function_ani1 import SymmetryFunctionsANI1

__all__ = [
    "IDEAL_SHAPES",
    "PDD",
    "ShapeMatch",
    "selling_scalars",
    "StructureType",
    "SymmetryFunctionsANI1",
    "adaptive_cna",
    "amd",
    "continuous_shape_measure",
    "coordination_shapes",
    "fixed_cna",
    "interval_cna",
    "pdd",
    "pdd_emd",
    "pdd_rms",
    "shape_match",
    "shape_measures",
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

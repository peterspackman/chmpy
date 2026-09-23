"""Geometry optimisation: what to vary, and how to vary it.

`relax` is the entry point for the usual case. Underneath, a `Coordinates`
object says what the degrees of freedom are -- Cartesian atoms, a
symmetry-adapted cell strain, a crystal's asymmetric unit -- and `TrustRegion`
drives them. `elastic_tensor` uses the same machinery to strain a cell and
watch the stress come back.

`XtbOptimizer` and the GULP drivers are a separate thing: they hand the whole
problem to an external program.
"""

from .coordinates import Atomic, AtomicStrain, Coordinates, Strain
from .curvature import LimitedMemoryBFGS
from .elastic import ElasticResult, elastic_tensor
from .hessian import stretch_hessian
from .relax import Stage, coordinates_for, relax, two_stage
from .strain import (
    cartesian_rotations,
    invariant_elastic_basis,
    invariant_strain_basis,
)
from .symmetry import SymmetryAdapted
from .trust_region import Relaxation, Step, TrustRegion, TrustRegionOptions
from .xtb import XtbEnergyEvaluator, XtbOptimizer

__all__ = [
    "Atomic",
    "AtomicStrain",
    "Coordinates",
    "ElasticResult",
    "LimitedMemoryBFGS",
    "Relaxation",
    "Stage",
    "Step",
    "Strain",
    "SymmetryAdapted",
    "TrustRegion",
    "TrustRegionOptions",
    "XtbEnergyEvaluator",
    "XtbOptimizer",
    "cartesian_rotations",
    "coordinates_for",
    "elastic_tensor",
    "invariant_elastic_basis",
    "invariant_strain_basis",
    "relax",
    "stretch_hessian",
    "two_stage",
]

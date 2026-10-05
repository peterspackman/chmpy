"""Energies, forces and stresses of atomic systems.

    from chmpy import Crystal
    from chmpy.calc import Calculator, System

    system = System.from_crystal(Crystal.load("structure.cif"))
    calc = Calculator.from_ase(published_mlip_calculator)
    result = calc(system, ("energy", "forces", "stress"))

Adapters live in `chmpy.calc.adapters`: `ase` in both directions,
`metatomic` for models driven directly, and `gulp` for the external
program, which is the one that can supply analytic second derivatives.

Write one by subclassing `Calculator` and implementing `compute`; see
`chmpy.calc.base` for the contract and `chmpy.calc.potentials` for a worked
example.
"""

from .base import Calculator, PropertyNotAvailable, ShiftedCalculator, SumCalculator
from .neighbors import NeighborList, Neighbors
from .potentials import LennardJones, PairPotential
from .result import (
    ALL_PROPERTIES,
    ENERGIES,
    ENERGY,
    FORCES,
    STRESS,
    CalculatorStats,
    GradientCheck,
    Result,
)
from .system import System

__all__ = [
    "ALL_PROPERTIES",
    "ENERGIES",
    "ENERGY",
    "FORCES",
    "STRESS",
    "Calculator",
    "CalculatorStats",
    "GradientCheck",
    "LennardJones",
    "NeighborList",
    "Neighbors",
    "PairPotential",
    "PropertyNotAvailable",
    "Result",
    "ShiftedCalculator",
    "SumCalculator",
    "System",
]

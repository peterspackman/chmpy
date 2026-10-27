"""Harmonic lattice dynamics: force constants, phonons, and what follows.

    from chmpy.vib import force_constants

    constants = force_constants(crystal, calculator)
    constants.frequencies([0.0, 0.0, 0.0], units="cm-1")

Force constants are indexed `(i, j, shift)`, the same way a neighbour list is,
so the dynamical matrix is a phase sum over exactly the pairs that exist and no
supercell index appears in it. See `chmpy.vib.force_constants`.

`chmpy.vib.einstein_debye` estimates the vibrational free energy from the unit
cell alone -- Gamma-point optical modes plus a Debye model of the acoustic ones
fitted to the elastic tensor -- which is what polymorph ranking usually needs
when a converged supercell is out of reach.

`chmpy.vib.christoffel` checks a set of force constants against an elastic
tensor for the same structure: the two describe the same physics in the
long-wavelength limit by completely different routes, so the acoustic slopes
have to agree.
"""

from .christoffel import (
    christoffel_velocities,
    compare_with_elastic,
    density,
    phonon_velocities,
)
from .einstein_debye import (
    DebyeModel,
    EinsteinDebye,
    debye_model,
    einstein_debye,
    einstein_debye_properties,
    gamma_einstein_debye,
)
from .force_constants import ForceConstants, force_constants, symmetrise
from .thermo import ThermalProperties, q_mesh, thermal_properties

__all__ = [
    "DebyeModel",
    "EinsteinDebye",
    "ForceConstants",
    "ThermalProperties",
    "christoffel_velocities",
    "compare_with_elastic",
    "debye_model",
    "density",
    "einstein_debye",
    "einstein_debye_properties",
    "force_constants",
    "gamma_einstein_debye",
    "phonon_velocities",
    "q_mesh",
    "symmetrise",
    "thermal_properties",
]

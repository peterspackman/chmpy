"""Harmonic lattice dynamics: force constants, phonons, and what follows.

    from chmpy.vib import force_constants

    constants = force_constants(crystal, calculator)
    constants.frequencies([0.0, 0.0, 0.0], units="cm-1")

Force constants are indexed `(i, j, shift)`, the same way a neighbour list is,
so the dynamical matrix is a phase sum over exactly the pairs that exist and no
supercell index appears in it. See `chmpy.vib.force_constants`.

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
from .force_constants import ForceConstants, force_constants, symmetrise

__all__ = [
    "ForceConstants",
    "christoffel_velocities",
    "compare_with_elastic",
    "density",
    "force_constants",
    "phonon_velocities",
    "symmetrise",
]

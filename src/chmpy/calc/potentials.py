"""Pair potentials.

`PairPotential` turns a distance dependence into a complete calculator. A
subclass sets `cutoff` and implements `pair`, returning the pair energy and its
derivative; analytic forces, stress, per-atom energies and the neighbour list
follow from those.

    class Morse(PairPotential):
        cutoff = 8.0

        def pair(self, r, zi, zj):
            x = np.exp(-self.a * (r - self.r0))
            return self.d * (x**2 - 2 * x), -2 * self.a * self.d * (x**2 - x)

The virial is written once here rather than in each subclass, since it is the
easiest part to get wrong: `dE/deps_ab = sum_pairs phi'(r) v_a v_b / r`, summed
over the pair vectors, which carry their lattice translations.
"""

from __future__ import annotations

import numpy as np

from .base import Calculator
from .neighbors import NeighborList
from .result import ENERGIES, ENERGY, FORCES, STRESS, Result


class PairPotential(Calculator):
    """A calculator for an energy that is a sum over pairs of a function of distance.

    Subclasses set `cutoff` and implement `pair`. Everything else -- forces,
    stress, per-atom energies, neighbour lists -- follows from those.

    Attributes:
        cutoff: interaction range in Angstroms
        skin: Verlet skin for the neighbour list, as a fraction of the cutoff
        shift_energy: subtract `pair(cutoff)` so the energy is continuous at
            the cutoff. Forces are unaffected; the energy is not, and a
            discontinuity there upsets an optimiser far more than the shift.
    """

    provides = {"energy", "forces", "stress", "energies"}
    cutoff: float = 6.0
    skin: float = 0.1
    shift_energy: bool = True

    def __init__(self, cutoff=None, skin=None, shift_energy=None, **kwargs):
        super().__init__(**kwargs)
        if cutoff is not None:
            self.cutoff = float(cutoff)
        if skin is not None:
            self.skin = float(skin)
        if shift_energy is not None:
            self.shift_energy = bool(shift_energy)
        self.neighbors = NeighborList(
            self.cutoff, full=False, skin=self.skin * self.cutoff, self_pairs=True
        )

    def pair(self, r: np.ndarray, zi: np.ndarray, zj: np.ndarray):
        """The pair energy and its derivative with respect to distance.

        Args:
            r: (P,) pair distances in Angstroms
            zi: (P,) atomic number of the first atom of each pair
            zj: (P,) atomic number of the second atom

        Returns:
            (energy, denergy_dr), both (P,), in eV and eV/A
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement pair(r, zi, zj)"
        )

    def compute(self, system, want):
        pairs = self.neighbors.compute(system)
        numbers = np.asarray(system.numbers)
        zi, zj = numbers[pairs.i], numbers[pairs.j]
        r = pairs.distances

        energy, denergy_dr = self.pair(r, zi, zj)
        if self.shift_energy and len(r):
            # each pair is shifted by its own elements' value at the cutoff
            at_cutoff, _ = self.pair(np.full(len(r), self.cutoff), zi, zj)
            energy = energy - at_cutoff

        n = len(system)
        forces = stress = energies = None

        if FORCES in want or STRESS in want:
            # v runs from i to j, so dr/dr_i = -v/r and the force on i is
            # +phi'(r) v / r, pulling i towards j when phi' > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                unit = np.where(r[:, None] > 0, pairs.vectors / r[:, None], 0.0)
            pair_force = denergy_dr[:, None] * unit

        if FORCES in want:
            forces = np.zeros((n, 3))
            np.add.at(forces, pairs.i, pair_force)
            np.add.at(forces, pairs.j, -pair_force)

        if STRESS in want:
            volume = system.volume
            stress = (pair_force.T @ pairs.vectors) / volume
            stress = 0.5 * (stress + stress.T)

        if ENERGIES in want:
            energies = np.zeros(n)
            np.add.at(energies, pairs.i, 0.5 * energy)
            np.add.at(energies, pairs.j, 0.5 * energy)

        return Result(
            energy=float(energy.sum()),
            forces=forces,
            stress=stress,
            energies=energies,
            volume=system.volume,
        )


class LennardJones(PairPotential):
    """The 12-6 Lennard-Jones potential, one set of parameters for all elements.

    Cheap, differentiable and with a known minimum, which makes it the test
    fixture for the optimiser as well as a usable potential for rare gases.

    Args:
        epsilon: well depth in eV
        sigma: distance at which the potential crosses zero, Angstroms
        cutoff: interaction range, Angstroms
    """

    def __init__(self, epsilon: float = 0.0103, sigma: float = 3.40, **kwargs):
        kwargs.setdefault("cutoff", 3.0 * sigma)
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)

    def pair(self, r, zi, zj):
        x = (self.sigma / r) ** 6
        energy = 4.0 * self.epsilon * (x * x - x)
        denergy_dr = 4.0 * self.epsilon * (-12.0 * x * x + 6.0 * x) / r
        return energy, denergy_dr

    def __repr__(self) -> str:
        return (
            f"<LennardJones epsilon={self.epsilon} eV sigma={self.sigma} A "
            f"cutoff={self.cutoff} A>"
        )


class ExternalStress(Calculator):
    """A constant applied stress, to add to another calculator.

    The energy is `V0 sigma : eps`, with `eps = sym(F) - I` the small strain of
    the cell `h = h0 F` from a reference cell `h0`, and `V0` its volume. A
    structure that minimises `E + V0 sigma : eps` has its own stress equal to
    `-sigma`. With `sigma = P I` this is a hydrostatic pressure, to first order
    in the strain.

        relax(crystal, calc + ExternalStress(sigma_gpa, reference_cell))

    Args:
        stress: (3, 3) applied stress in GPa, in the ASE sign convention
            (positive is tensile on the structure's own stress)
        reference_cell: (3, 3) lattice vectors as rows that the strain is
            measured from
    """

    provides = frozenset({ENERGY, FORCES, STRESS})

    def __init__(self, stress, reference_cell, **kwargs):
        super().__init__(**kwargs)
        from .result import EV_PER_ANGSTROM3_TO_GPA

        stress = np.asarray(stress, dtype=float)
        self.stress = 0.5 * (stress + stress.T) / EV_PER_ANGSTROM3_TO_GPA
        self.reference_cell = np.array(reference_cell, dtype=float)
        self.reference_volume = abs(float(np.linalg.det(self.reference_cell)))

    def __repr__(self) -> str:
        from .result import EV_PER_ANGSTROM3_TO_GPA

        return (
            f"<ExternalStress max {np.abs(self.stress).max() * EV_PER_ANGSTROM3_TO_GPA:.4g}"
            " GPa>"
        )

    def compute(self, system, want):
        # h = h0 F with lattice vectors as rows
        deformation = np.linalg.solve(self.reference_cell, np.asarray(system.cell))
        strain = 0.5 * (deformation + deformation.T) - np.eye(3)
        energy = self.reference_volume * float(np.sum(self.stress * strain))
        # a further strain d about the current cell takes F to F (I + d), so
        # dE = V0 sigma : sym(F d), i.e. dE/dd = V0 F^T sigma, symmetrised
        derivative = deformation.T @ self.stress * self.reference_volume
        stress = 0.5 * (derivative + derivative.T) / system.volume
        return Result(
            energy=energy,
            forces=np.zeros((len(system), 3)) if FORCES in want else None,
            stress=stress if STRESS in want else None,
            volume=system.volume,
        )

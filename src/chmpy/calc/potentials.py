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
from .result import ENERGIES, FORCES, STRESS, Result


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
            at_cutoff, _ = self.pair(
                np.full(1, self.cutoff), zi[:1] * 0 + zi[:1], zj[:1]
            )
            energy = energy - at_cutoff[0]

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

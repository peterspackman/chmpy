"""Einstein-Debye vibrational free energies, from a single unit cell.

The vibrational free energy of a molecular crystal needs the whole phonon
dispersion, which needs a supercell large enough for the force constants to
die away, which for a flexible drug-like molecule can mean thousands of atoms
per displacement. The Einstein-Debye model (C. J. Nickerson and E. R. Johnson,
Phys. Chem. Chem. Phys., 2026, DOI 10.1039/D6CP00089D) gets within about
1 kJ/mol of that from the unit cell alone:

* **Optical modes, Einstein.** Molecular crystals have optical branches that
  barely disperse, so the 3N - 3 optical modes at Gamma stand in for every
  wavevector. Gamma-point force constants are exact from a 1x1x1 "supercell":
  every periodic image of a displaced atom moves with it, which is what a
  q = 0 mode is.
* **Acoustic modes, Debye.** The three acoustic branches go to zero at Gamma
  and dominate the low-temperature free energy, so they cannot be left out.
  They are treated as a Debye solid with one cutoff frequency, found from the
  elastic tensor: the Christoffel sound velocities `v` along the 13 directions
  `(a, b, c)`, `a, b, c in {0, +-1}`, of the reciprocal lattice, each extended
  to the zone boundary with a sinusoidal dispersion,

      omega_max = 2 v k_cb / pi,       k_cb = |(a, b, c)/2 . B|

  where `B` is the reciprocal lattice *including* the 2 pi, and averaged over
  all 39 velocities. The `2 / pi` is the sine curve that matches the slope
  `v` at Gamma and is flat at the boundary; a straight line to the boundary
  overestimates it.

For one Debye "atom" per cell, with `x = hbar omega_D / kT`,

    F = 9/8 hbar omega_D + kT [3 ln(1 - exp(-x)) - D(x)]
    U = 9/8 hbar omega_D + 3 kT D(x)
    S = k [4 D(x) - 3 ln(1 - exp(-x))]
    C = 3k [4 D(x) - 3x / (exp(x) - 1)]

with `D(x) = 3/x^3 int_0^x t^3 / (exp(t) - 1) dt` the Debye function. The
total is the sum of the two parts, per unit cell.

    from chmpy.vib import einstein_debye

    result = einstein_debye(relaxed, calculator, temperatures=[100, 298.15])
    result.free_energy                 # eV per cell
    result.per_molecule("free_energy") # kJ/mol per molecule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from chmpy.util.unit import EV_TO_KJ_PER_MOL

from .christoffel import christoffel_velocities, density
from .force_constants import ForceConstants, force_constants
from .thermo import BOLTZMANN_EV, HBAR_EV_S, ZERO_FREQUENCY, harmonic_properties

LOG = logging.getLogger(__name__)

#: the 13 directions (a, b, c), a, b, c in {0, +-1}, one of each +-pair, in
#: units of the reciprocal lattice vectors
DEBYE_DIRECTIONS = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)


@dataclass
class DebyeModel:
    """The acoustic part: a Debye solid fitted to the elastic tensor.

    Attributes:
        frequency: Debye angular frequency, rad/s
        velocities: (13, 3) Christoffel velocities along `DEBYE_DIRECTIONS`,
            m/s, ascending in each row
        density: kg/m^3
    """

    frequency: float
    velocities: np.ndarray
    density: float

    @property
    def temperature(self) -> float:
        "Debye temperature, K"
        return HBAR_EV_S * self.frequency / BOLTZMANN_EV

    @property
    def mean_velocity(self) -> float:
        "Average of the 39 sound velocities, m/s"
        return float(self.velocities.mean())

    def properties(self, temperature: float):
        """Free energy, entropy, heat capacity and internal energy per cell.

        Returns:
            (F eV, S eV/K, C eV/K, U eV) of the three acoustic modes
        """
        energy = HBAR_EV_S * self.frequency
        zero_point = 9.0 / 8.0 * energy
        if temperature <= 0 or energy <= 0:
            return zero_point, 0.0, 0.0, zero_point
        kT = BOLTZMANN_EV * temperature
        x = energy / kT
        debye = debye_function(x)
        log_term = float(np.log(-np.expm1(-x)))
        free_energy = zero_point + kT * (3.0 * log_term - debye)
        internal = zero_point + 3.0 * kT * debye
        entropy = BOLTZMANN_EV * (4.0 * debye - 3.0 * log_term)
        heat_capacity = 3.0 * BOLTZMANN_EV * (4.0 * debye - 3.0 * x / np.expm1(x))
        return free_energy, entropy, heat_capacity, internal


def debye_function(x: float) -> float:
    """The third-order Debye function, `3/x^3 int_0^x t^3 / (exp(t) - 1) dt`."""
    if x <= 0:
        return 1.0
    if x > 150.0:
        # the integral has reached pi^4 / 15 to double precision
        return np.pi**4 / (5.0 * x**3)
    integral, _ = quad(lambda t: t**3 / np.expm1(t) if t > 0 else 0.0, 0.0, x)
    return 3.0 * integral / x**3


def debye_model(c_voigt, cell, mass) -> DebyeModel:
    """The Debye frequency of a crystal from its elastic constants.

    Args:
        c_voigt: (6, 6) elastic constants in GPa, or an `ElasticTensor`
        cell: (3, 3) lattice vectors as rows, Angstroms
        mass: total mass of the cell in amu, or (N,) atomic masses

    Returns:
        DebyeModel
    """
    c_voigt = np.asarray(getattr(c_voigt, "c_voigt", c_voigt), dtype=float)
    cell = np.asarray(cell, dtype=float)
    rho = density(np.atleast_1d(mass), cell)
    # rad per Angstrom, as rows
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    boundary = 0.5 * DEBYE_DIRECTIONS @ reciprocal
    velocities = np.array([christoffel_velocities(c_voigt, rho, k) for k in boundary])
    # |k_cb| in rad / m
    k_cb = np.linalg.norm(boundary, axis=1) * 1e10
    omega_max = 2.0 * velocities * k_cb[:, None] / np.pi
    return DebyeModel(
        frequency=float(omega_max.mean()), velocities=velocities, density=rho
    )


@dataclass
class EinsteinDebye:
    """Einstein-Debye vibrational thermodynamics, per unit cell.

    Attributes:
        temperatures: (T,) K
        free_energy: (T,) vibrational Helmholtz free energy, eV
        entropy: (T,) eV/K
        heat_capacity: (T,) at constant volume, eV/K
        internal_energy: (T,) vibrational internal energy, eV
        zero_point: zero-point energy, eV
        optical: (T, 4) the optical part's F, S, C, U
        acoustic: (T, 4) the acoustic part's F, S, C, U
        optical_frequencies: (3N - 3,) Gamma-point optical frequencies,
            rad/s, imaginary ones negative
        debye: the acoustic `DebyeModel`
        n_imaginary: optical modes left out as imaginary
        z: molecules per cell, for `per_molecule`, or None if unknown
    """

    temperatures: np.ndarray
    free_energy: np.ndarray
    entropy: np.ndarray
    heat_capacity: np.ndarray
    internal_energy: np.ndarray
    zero_point: float
    optical: np.ndarray
    acoustic: np.ndarray
    optical_frequencies: np.ndarray
    debye: DebyeModel
    n_imaginary: int
    z: int | None = None

    def per_molecule(self, name: str = "free_energy", units: str = "kj/mol"):
        """A quantity per molecule rather than per cell.

        Args:
            name: "free_energy", "internal_energy", "entropy",
                "heat_capacity" or "zero_point"
            units: "kj/mol" (per K for entropy and heat capacity) or "ev"

        Returns:
            the quantity divided by `z`, in those units
        """
        if not self.z:
            raise ValueError("the number of molecules per cell, z, is not known")
        factor = {"kj/mol": EV_TO_KJ_PER_MOL, "ev": 1.0}
        if units.lower() not in factor:
            raise ValueError(f"unknown units {units!r}, expected one of {list(factor)}")
        return np.asarray(getattr(self, name)) * factor[units.lower()] / self.z


def einstein_debye_properties(
    optical_frequencies, debye: DebyeModel, temperatures, z=None
) -> EinsteinDebye:
    """Combine Gamma-point optical modes with a Debye model of the acoustic ones.

    This is the whole model once its two inputs exist, so it can be used with
    frequencies and an elastic tensor from anywhere.

    Args:
        optical_frequencies: (3N - 3,) Gamma-point optical angular frequencies,
            rad/s, with imaginary modes negative. Imaginary modes and any
            within `ZERO_FREQUENCY` of zero are left out, and counted.
        debye: a `DebyeModel`, from `debye_model`
        temperatures: K
        z: molecules per cell, if known

    Returns:
        EinsteinDebye
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    omega = np.asarray(optical_frequencies, dtype=float)
    imaginary = int(np.sum(omega < -ZERO_FREQUENCY))
    if imaginary:
        LOG.warning(
            "%d imaginary optical modes at Gamma, down to %.1f cm-1, are left "
            "out of the free energy; the structure is not at a minimum",
            imaginary,
            omega.min() / (2 * np.pi * 2.99792458e10),
        )
    kept = omega[omega > ZERO_FREQUENCY]

    optical = []
    for T in temperatures:
        F, S, C = harmonic_properties(kept, T)
        optical.append((F, S, C, F + T * S))
    optical = np.array(optical).reshape(-1, 4)
    acoustic = np.array([debye.properties(T) for T in temperatures]).reshape(-1, 4)
    total = optical + acoustic
    return EinsteinDebye(
        temperatures=temperatures,
        free_energy=total[:, 0],
        entropy=total[:, 1],
        heat_capacity=total[:, 2],
        internal_energy=total[:, 3],
        zero_point=harmonic_properties(kept, 0.0)[0] + debye.properties(0.0)[0],
        optical=optical,
        acoustic=acoustic,
        optical_frequencies=omega,
        debye=debye,
        n_imaginary=imaginary,
        z=z,
    )


def einstein_debye(
    structure,
    calculator,
    temperatures,
    *,
    elastic=None,
    z=None,
    displacement: float | None = None,
    symmetry: bool = True,
    info=None,
    progress=None,
    elastic_options=None,
) -> EinsteinDebye:
    """Einstein-Debye vibrational free energy of a relaxed crystal.

    Costs one set of Gamma-point displacements in the unit cell, plus an
    elastic tensor unless one is given.

    Args:
        structure: a relaxed `Crystal`, or a periodic `System`
        calculator: a `chmpy.calc.Calculator`
        temperatures: K
        elastic: the elastic constants, as a (6, 6) array in GPa, an
            `ElasticTensor`, or the `ElasticResult` from
            `chmpy.opt.elastic_tensor`. Computed if not given.
        z: molecules per cell. Found from a `Crystal`'s unit-cell molecules
            if not given.
        displacement: finite-difference step in Angstroms
        symmetry: use a `Crystal`'s space group to reduce the displacements,
            and the elastic strains if the tensor is computed here
        info: model inputs that are not geometry, carried on every `System`
        progress: as for `force_constants`
        elastic_options: keyword arguments for `chmpy.opt.elastic_tensor`

    Returns:
        EinsteinDebye, per unit cell
    """
    options = {} if displacement is None else {"displacement": displacement}
    constants = force_constants(
        structure,
        calculator,
        supercell=(1, 1, 1),
        info=info,
        progress=progress,
        symmetry=symmetry,
        **options,
    )
    if elastic is None:
        from chmpy.opt import elastic_tensor

        elastic = elastic_tensor(
            structure,
            calculator,
            symmetry=symmetry,
            info=info,
            progress=progress,
            **(elastic_options or {}),
        )
    elastic = getattr(elastic, "tensor", elastic)
    if hasattr(elastic, "is_stable") and not elastic.is_stable():
        LOG.warning(
            "the elastic tensor is not positive definite, so some sound "
            "velocities are not real and the acoustic free energy is unreliable"
        )
    if z is None and hasattr(structure, "unit_cell_molecules"):
        z = len(structure.unit_cell_molecules())
    return gamma_einstein_debye(constants, elastic, temperatures, z=z)


def gamma_einstein_debye(
    constants: ForceConstants, elastic, temperatures, z=None
) -> EinsteinDebye:
    """Einstein-Debye thermodynamics from force constants already measured.

    Only the Gamma-point modes of `constants` are used, so they may come from
    a 1x1x1 cell or from a larger supercell.

    Args:
        constants: a `ForceConstants`
        elastic: (6, 6) elastic constants in GPa, or an `ElasticTensor`
        temperatures: K
        z: molecules per cell, if known
    """
    debye = debye_model(elastic, constants.cell, constants.masses)
    return einstein_debye_properties(
        constants.optical_frequencies(units="rad/s"), debye, temperatures, z=z
    )

"""Sound velocities, from either end.

An elastic tensor and a set of force constants describe the same physics in
the long-wavelength limit, and they are computed by completely different
routes -- one from stresses under finite strain, the other from forces under
finite displacement in a supercell. Where they must agree is the slope of the
acoustic branches:

* From elasticity, the Christoffel equation
  `det|C_ijkl n_j n_l - rho v^2 delta_ik| = 0` gives three velocities along a
  direction `n`.
* From lattice dynamics, `omega(q) / |k|` as `q -> 0` along the same direction
  gives the same three.

That makes this the sharpest available test of a force-constant calculation,
and in particular of the two parts of it that are easy to get subtly wrong: the
supercell bookkeeping and the multiplicity weights in the phase sum. Both show
up as acoustic branches with the wrong slope, and nothing else about the
calculation looks amiss when they do.
"""

from __future__ import annotations

import numpy as np

#: amu -> kg, Angstrom^3 -> m^3
AMU_TO_KG = 1.66053906660e-27
ANGSTROM3_TO_M3 = 1e-30

#: Voigt index of each (i, j) pair
VOIGT_INDEX = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def full_tensor(c_voigt) -> np.ndarray:
    """The (3, 3, 3, 3) elastic tensor from a Voigt 6x6 matrix."""
    c_voigt = np.asarray(c_voigt, dtype=float)
    return c_voigt[VOIGT_INDEX[:, :, None, None], VOIGT_INDEX[None, None, :, :]]


def christoffel_velocities(c_voigt, density, direction) -> np.ndarray:
    """The three acoustic velocities along a direction, from elastic constants.

    Args:
        c_voigt: (6, 6) elastic constants in GPa
        density: kg/m^3
        direction: (3,) propagation direction, normalised internally

    Returns:
        (3,) velocities in m/s, ascending -- two quasi-transverse and one
        quasi-longitudinal for a general direction
    """
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    tensor = full_tensor(c_voigt) * 1e9  # GPa -> Pa
    acoustic = np.einsum("ijkl,j,l->ik", tensor, n, n)
    eigenvalues = np.linalg.eigvalsh(0.5 * (acoustic + acoustic.T))
    return np.sqrt(np.maximum(eigenvalues, 0.0) / density)


def density(masses, cell) -> float:
    """Mass density in kg/m^3 from amu and a cell in Angstroms."""
    volume = abs(np.linalg.det(np.asarray(cell, dtype=float)))
    return float(np.sum(masses) * AMU_TO_KG / (volume * ANGSTROM3_TO_M3))


def phonon_velocities(constants, direction, magnitude: float = 1e-4) -> np.ndarray:
    """The three acoustic velocities along a direction, from force constants.

    Evaluated at a wavevector short enough to be in the linear regime, which
    is what makes it comparable with `christoffel_velocities`.

    Args:
        constants: a `ForceConstants`
        direction: (3,) Cartesian propagation direction
        magnitude: length of the probe wavevector in inverse Angstroms

    Returns:
        (3,) velocities in m/s, ascending
    """
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)

    # a plane wave exp(i k.R) with R = S h has phase 2 pi q.S when
    # k = 2 pi q h^-T, so the fractional wavevector for a Cartesian k is
    # q = k h^T / (2 pi)
    wavevector = magnitude * n
    q = wavevector @ np.asarray(constants.cell).T / (2 * np.pi)

    omega = constants.frequencies(q, units="rad/s")[:3]
    # |k| in inverse metres
    return np.abs(omega) / (magnitude * 1e10)


def compare_with_elastic(constants, c_voigt, directions=None) -> dict:
    """Acoustic velocities from force constants against those from elasticity.

    Args:
        constants: a `ForceConstants`
        c_voigt: (6, 6) elastic constants in GPa for the same structure
        directions: (M, 3) propagation directions, or None for a default set
            spanning axes, face diagonals and a body diagonal

    Returns:
        a dict with `directions`, `phonon`, `elastic` (both (M, 3) in m/s),
        and `max_relative_error`
    """
    if directions is None:
        directions = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    rho = density(constants.masses, constants.cell)

    from_phonons = np.array([phonon_velocities(constants, n) for n in directions])
    from_elastic = np.array(
        [christoffel_velocities(c_voigt, rho, n) for n in directions]
    )
    scale = max(float(np.abs(from_elastic).max()), 1e-12)
    return {
        "directions": directions,
        "phonon": from_phonons,
        "elastic": from_elastic,
        "density": rho,
        "max_relative_error": float(np.abs(from_phonons - from_elastic).max() / scale),
    }

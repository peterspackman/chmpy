"""Harmonic thermodynamics: free energy, entropy and heat capacity.

Per mode of angular frequency `w`, with `x = hbar w / kT`:

    F = hbar w / 2 + kT ln(1 - exp(-x))
    S = k [x / (exp(x) - 1) - ln(1 - exp(-x))]
    C = k x^2 exp(x) / (exp(x) - 1)^2

summed over modes and averaged over a q-mesh, giving values per unit cell.
The three acoustic modes at Gamma (w = 0) are excluded. Imaginary modes are
excluded and counted: a structure with any is not at a minimum, and the
numbers describe only its stable modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: reduced Planck constant, eV s
HBAR_EV_S = 6.582119569e-16

#: Boltzmann constant, eV / K
BOLTZMANN_EV = 8.617333262e-5

#: modes below this, in rad/s (about 0.03 cm-1), count as zero
ZERO_FREQUENCY = 2 * np.pi * 1e9

#: dynamical matrices held at once while diagonalising, bytes
FREQUENCY_BATCH_BYTES = 256 * 2**20


@dataclass
class ThermalProperties:
    """Harmonic thermodynamics of a crystal, per unit cell.

    Attributes:
        temperatures: (T,) in K
        free_energy: (T,) vibrational Helmholtz free energy, eV
        entropy: (T,) eV / K
        heat_capacity: (T,) at constant volume, eV / K
        zero_point: zero-point energy, eV
        mesh: q-mesh used
        n_modes: modes included over the whole mesh
        n_imaginary: modes excluded as imaginary
        gamma: (T, 3) the Gamma correction included in F, S and Cv, or None
        shift: offset of the mesh from Gamma in mesh steps
    """

    temperatures: np.ndarray
    free_energy: np.ndarray
    entropy: np.ndarray
    heat_capacity: np.ndarray
    zero_point: float
    mesh: tuple
    n_modes: int
    n_imaginary: int
    gamma: np.ndarray | None = None
    shift: tuple = (0.0, 0.0, 0.0)


def mesh_shift(mesh, shift=0.0) -> np.ndarray:
    """The offset of a mesh from Gamma along each axis, in mesh steps.

    Args:
        mesh: (3,) points along each reciprocal axis
        shift: 0 for a Gamma-centred mesh, 0.5 for one centred half a step
            away along every axis, or (3,) of either per axis

    Returns:
        (3,) each 0 or 0.5
    """
    shift = np.broadcast_to(np.asarray(shift, dtype=float), (3,)) % 1.0
    if not np.all(np.isclose(shift, 0.0) | np.isclose(shift, 0.5)):
        raise ValueError(
            f"a mesh shift must be 0 or half a step along each axis, got {shift}; "
            "other shifts break the q, -q pairing the mesh sums rely on"
        )
    return np.where(np.isclose(shift, 0.5), 0.5, 0.0)


def q_mesh(mesh, shift=0.0) -> np.ndarray:
    """A mesh of fractional wavevectors in [-0.5, 0.5).

    Args:
        mesh: (3,) points along each reciprocal axis
        shift: offset from Gamma in mesh steps; see `mesh_shift`

    Returns:
        (n1 n2 n3, 3) fractional q-points
    """
    offset = mesh_shift(mesh, shift)
    axes = [(np.arange(n) + o) / n for n, o in zip(mesh, offset, strict=True)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    return grid - np.round(grid + 1e-12)


def harmonic_free_energy(frequencies, temperature: float) -> float:
    "Free energy of a set of modes (angular frequencies in rad/s), eV"
    return harmonic_properties(frequencies, temperature)[0]


def harmonic_properties(frequencies, temperature: float):
    """Free energy, entropy and heat capacity of a set of modes.

    Args:
        frequencies: angular frequencies in rad/s, all positive
        temperature: K

    Returns:
        (free energy eV, entropy eV/K, heat capacity eV/K)
    """
    energies = HBAR_EV_S * np.asarray(frequencies, dtype=float)
    zero_point = 0.5 * float(energies.sum())
    if temperature <= 0:
        return zero_point, 0.0, 0.0
    kT = BOLTZMANN_EV * temperature
    x = energies / kT
    one_minus = -np.expm1(-x)  # 1 - exp(-x), accurate for small x
    free_energy = zero_point + kT * float(np.log(one_minus).sum())
    occupation = np.exp(-x) / one_minus  # 1 / (exp(x) - 1)
    entropy = BOLTZMANN_EV * float((x * occupation - np.log(one_minus)).sum())
    heat_capacity = BOLTZMANN_EV * float(
        (x * x * np.exp(-x) / (one_minus * one_minus)).sum()
    )
    return free_energy, entropy, heat_capacity


def mesh_frequencies(constants, mesh, shift=0.0):
    """Mode frequencies over a q-mesh, with the zero and imaginary ones removed.

    Frequencies at `q` and `-q` are the same, so only one of each such pair
    is diagonalised and its modes counted twice. At Gamma the three uniform
    translations are projected out rather than recognised by being small, so
    an acoustic-sum-rule residue is never counted as an imaginary mode.

    Returns:
        (positive angular frequencies in rad/s as a flat array, number of
        imaginary modes removed). Each q-point's modes appear once per point
        of the full mesh they stand for.
    """
    points, multiplicity = _irreducible_by_inversion(q_mesh(mesh, shift))
    gamma = np.all(points == 0, axis=1)
    kept, imaginary = [], 0

    def keep(omega, count):
        nonlocal imaginary
        imaginary += count * int(np.sum(omega < -ZERO_FREQUENCY))
        kept.append(np.repeat(omega[omega > ZERO_FREQUENCY], count))

    if gamma.any():
        keep(constants.optical_frequencies(units="rad/s"), 1)
    rest, counts = points[~gamma], multiplicity[~gamma]
    batch = max(1, FREQUENCY_BATCH_BYTES // (16 * (3 * constants.n_atoms) ** 2))
    for start in range(0, len(rest), batch):
        omega = constants.frequencies(rest[start : start + batch], units="rad/s")
        for row, count in zip(omega, counts[start : start + batch], strict=True):
            keep(row, int(count))
    return np.concatenate(kept), imaginary


def _irreducible_by_inversion(points):
    """One of each `q`, `-q` pair of a mesh, and how many points it stands for.

    `points` are in [-0.5, 0.5), so `-q` of a point on the -0.5 face is folded
    back onto the same face before matching.
    """
    folded = -points
    folded = folded - np.round(folded + 1e-12)
    keys = {tuple(np.round(p * 1e8).astype(np.int64)): k for k, p in enumerate(points)}
    chosen, multiplicity = [], []
    seen = np.zeros(len(points), dtype=bool)
    for k, p in enumerate(folded):
        if seen[k]:
            continue
        partner = keys[tuple(np.round(p * 1e8).astype(np.int64))]
        seen[k] = seen[partner] = True
        chosen.append(k)
        multiplicity.append(1 if partner == k else 2)
    return points[chosen], np.array(multiplicity)


def thermal_properties(
    constants, temperatures, mesh=(8, 8, 8), gamma_correction=None, shift=0.0
) -> ThermalProperties:
    """Harmonic thermodynamics of a crystal from its force constants.

    Args:
        constants: a `chmpy.vib.ForceConstants`
        temperatures: K
        mesh: q-mesh; converge it for the property you need
        gamma_correction: put back the three acoustic modes at Gamma that the
            mesh drops (see `chmpy.vib.gamma`). True takes the long-wave
            acoustic tensor from `constants`; an elastic tensor (a (6, 6)
            array in GPa, `ElasticTensor` or `ElasticResult`) takes it from
            that instead. None or False leaves them out. On a shifted mesh
            no mode is dropped, and the correction is instead for the
            acoustic branches' singular behaviour near Gamma.
        shift: offset of the mesh from Gamma in mesh steps: 0.5 centres it
            half a step away along every axis; see `mesh_shift`

    Returns:
        ThermalProperties, per unit cell
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    offset = mesh_shift(mesh, shift)
    frequencies, imaginary = mesh_frequencies(constants, mesh, offset)
    n_q = int(np.prod(mesh))
    values = np.array([harmonic_properties(frequencies, T) for T in temperatures])
    values = values.reshape(-1, 3) / n_q
    zero_point = harmonic_properties(frequencies, 0.0)[0] / n_q

    gamma = None
    if gamma_correction is not None and gamma_correction is not False:
        from .gamma import gamma_weight, long_wave_tensor

        elastic = None if gamma_correction is True else gamma_correction
        tensor = long_wave_tensor(constants, elastic)
        weights = gamma_weight(tensor, mesh, np.append(temperatures, 0.0), shift=offset)
        gamma = np.array(weights).T / n_q
        zero_point += gamma[-1, 0]
        gamma = gamma[:-1]
        values = values + gamma

    return ThermalProperties(
        temperatures=temperatures,
        free_energy=values[:, 0],
        entropy=values[:, 1],
        heat_capacity=values[:, 2],
        zero_point=zero_point,
        mesh=tuple(mesh),
        n_modes=len(frequencies),
        n_imaginary=imaginary,
        gamma=gamma,
        shift=tuple(offset.tolist()),
    )

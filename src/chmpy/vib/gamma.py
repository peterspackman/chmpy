"""The Gamma-point correction for harmonic thermodynamics on a q-mesh.

Summing over a Gamma-centred mesh drops the three acoustic modes at Gamma,
whose frequency is zero, and that missing piece is most of the error of a
coarse mesh. Near Gamma those modes are sound waves, `omega_j(x)^2 =
eig D2(x)` with `D2(x)_ik = K_ikab x_a x_b` quadratic in the fractional
wavevector `x`, so what the mesh misses can be computed: the finite part of
the lattice sum of their free energy over the mesh-index lattice `m = n x`,

    FP[s] = int phi(m) s(m) d^3m - sum_(m != 0) phi(m) s(m)

with a smooth cutoff `phi(m) = erfc((|A m| - R) / w) / 2` that the result
does not depend on: it is 1 to round-off at the origin, and its Fourier
transform is a Gaussian, so the lattice sum aliases at ~exp(-(pi w)^2). This is the order-zero term of the singular Euler-Maclaurin
expansion (Buchheit and Kessler, Nonlinearity, 2022,
doi:10.1088/1361-6544/ac73d0). Adding `W / N` to the mesh average replaces
the dropped modes; for F the error then falls as h^4-h^5 rather than
h^3 ln h.

A mesh shifted half a step from Gamma is the lattice `m + 1/2` instead. It
contains no zero-frequency mode to put back, but the sum still misses the
singular part of the acoustic free energy near Gamma, and the same finite
part, now over every point of the shifted lattice, recovers it.

Everything here is in angular frequency: `tau = kT / hbar` and `omega` in
rad/s, `K` in (rad/s)^2 per (fractional unit)^2.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from .force_constants import EV_PER_ANGSTROM2_AMU_TO_RAD2_S2
from .thermo import BOLTZMANN_EV, HBAR_EV_S

#: smooth cutoff radius and width, in mesh units
RADIUS = 12.0
WIDTH = 2.0

#: above this `4 pi^2 tau / c_max` the classical weight is exact to ~exp(-20)
CLASSICAL = 20.0

#: thermal lattice sums stop where `hbar omega / kT` reaches this
THERMAL_RANGE = 25.0

#: GPa -> Pa, and 1/A^2 -> 1/m^2
_GPA_PER_A2 = 1e9 * 1e20
_AMU_KG = 1.66053906660e-27


@cache
def _sphere():
    "Lebedev directions (n, 3) and weights summing to 4 pi"
    from scipy.integrate import lebedev_rule

    points, weights = lebedev_rule(131)
    return points.T, weights


def _cutoff(r):
    return 0.5 * erfc((r - RADIUS) / WIDTH)


@cache
def _radial(power: int, log: bool = False) -> float:
    "int_0^inf phi(r) r^power [ln r] dr"
    if log:
        integrand = lambda r: _cutoff(r) * r**power * np.log(r)  # noqa: E731
    else:
        integrand = lambda r: _cutoff(r) * r**power  # noqa: E731
    return quad(integrand, 0, RADIUS + 8 * WIDTH, limit=400)[0]


# -- the long-wave tensor ------------------------------------------------------


def long_wave_tensor(constants, elastic=None, step: float = 5e-3) -> np.ndarray:
    """K_ikab, with `omega^2` of the acoustic modes `eig K_ikab x_a x_b`.

    Args:
        constants: the `ForceConstants` the mesh sum uses
        elastic: (6, 6) elastic constants in GPa, an `ElasticTensor` or an
            `ElasticResult`, to take D2 from. By default it is read off the
            force constants themselves, which is consistent with the mesh
            sum by construction.
        step: fractional wavevector for reading D2 off the force constants

    Returns:
        (3, 3, 3, 3) in (rad/s)^2 per (fractional unit)^2
    """
    if elastic is not None:
        return _from_elastic(elastic, constants.cell, constants.masses)
    return _from_constants(constants, step)


def _from_elastic(elastic, cell, masses) -> np.ndarray:
    from .christoffel import full_tensor

    elastic = getattr(elastic, "tensor", elastic)
    c_voigt = np.asarray(getattr(elastic, "c_voigt", elastic), dtype=float)
    stiffness = full_tensor(c_voigt)  # C_ijkl
    cell = np.asarray(cell, dtype=float)
    density = np.sum(masses) * _AMU_KG / (abs(np.linalg.det(cell)) * 1e-30)
    # Cartesian angular wavevector per fractional unit, rad/A
    reciprocal = 2 * np.pi * np.linalg.inv(cell)
    tensor = np.einsum("ijkl,ja,lb->ikab", stiffness, reciprocal, reciprocal)
    tensor = 0.5 * (tensor + tensor.transpose(0, 1, 3, 2))
    return tensor * _GPA_PER_A2 / density


def _from_constants(constants, step) -> np.ndarray:
    """D2 from the acoustic Schur complement of the dynamical matrix at small q.

    The optical modes are relaxed out rather than separated by eigenvalue, so
    no mode assignment is needed and internal relaxation is included. The
    bilinear form is recovered by polarisation from the three axes and three
    face diagonals, each at two step lengths for a Richardson estimate.
    """
    directions = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float
    )
    qs = np.concatenate([step * directions, 2 * step * directions])
    matrices = constants.dynamical_matrices(qs)

    root_mass = np.sqrt(constants.masses)
    translations = np.kron(root_mass[:, None], np.eye(3)) / np.linalg.norm(root_mass)
    complement = constants._translation_complement()
    acoustic = []
    for matrix in matrices:
        tt = translations.T @ matrix @ translations
        tq = translations.T @ matrix @ complement
        qq = complement.T @ matrix @ complement
        schur = tt - tq @ np.linalg.solve(qq, tq.conj().T)
        acoustic.append((0.5 * (schur + schur.conj().T)).real)
    acoustic = np.array(acoustic) * EV_PER_ANGSTROM2_AMU_TO_RAD2_S2
    near, far = acoustic[:6] / step**2, acoustic[6:] / (2 * step) ** 2
    quadratic = (4 * near - far) / 3

    tensor = np.zeros((3, 3, 3, 3))
    for a in range(3):
        tensor[:, :, a, a] = quadratic[a]
    for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        mixed = 0.5 * (quadratic[3 + k] - quadratic[a] - quadratic[b])
        tensor[:, :, a, b] = tensor[:, :, b, a] = mixed
    return tensor


# -- the Gamma weight ----------------------------------------------------------


def gamma_weight(tensor, mesh, temperatures, shift=(0.0, 0.0, 0.0)):
    """What the three dropped acoustic modes at Gamma contribute, per Gamma cell.

    Divide by the number of mesh points and add to the mesh average.

    Args:
        tensor: (3, 3, 3, 3) long-wave tensor from `long_wave_tensor`
        mesh: (3,) mesh
        temperatures: K
        shift: (3,) offset of the mesh from Gamma, 0 or 0.5 mesh steps

    Returns:
        (free energy eV, entropy eV/K, heat capacity eV/K), each (T,)
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    lattice = _GammaLattice(tensor, mesh, shift)
    values = np.array([lattice.weight(T) for T in temperatures]).reshape(-1, 3)
    return values[:, 0], values[:, 1], values[:, 2]


class _GammaLattice:
    """The mesh-index lattice, its cutoff, and the sums that do not depend on T."""

    def __init__(self, tensor, mesh, shift=(0.0, 0.0, 0.0)):
        h = 1.0 / np.asarray(mesh, dtype=float)
        self.shift = np.broadcast_to(np.asarray(shift, dtype=float), (3,)) % 1.0
        # whether the mesh contains Gamma, and so drops its three modes
        self.centred = not np.any(self.shift)
        # D2 at mesh-index vectors m, x = m / n
        self.tensor = tensor * h[None, None, :, None] * h[None, None, None, :]
        directions, self.weights = _sphere()
        self.speeds = self._frequencies(directions)  # per mesh unit, (n, 3)
        if self.speeds.min() <= 0:
            raise ValueError(
                "the long-wave acoustic tensor is not positive definite, so the "
                "crystal is mechanically unstable and the Gamma weight is undefined"
            )

        # cutoff in the sound-speed metric |A m|^2 ~ tr D2(m), scaled so the
        # ellipsoid contains the ball of radius RADIUS in mesh units
        metric = np.linalg.cholesky(np.einsum("iiab->ab", self.tensor)).T
        self.metric = metric / np.linalg.norm(metric, 2)
        inverse = np.linalg.inv(self.metric)
        reach = RADIUS + 6 * WIDTH
        half = np.ceil(reach * np.linalg.norm(inverse, axis=1)).astype(int) + 1
        points = _half_box(half, self.shift)
        radii = np.linalg.norm(points @ self.metric.T, axis=1)
        inside = radii < reach
        self.points, self.phi = points[inside], _cutoff(radii[inside])
        # continuum integrals of homogeneous functions in the metric of A
        self.metric_directions = directions @ inverse.T
        self.metric_weights = self.weights / np.linalg.det(self.metric)
        self._classical = None
        self._zero_point = None

    def _d2(self, m):
        return np.einsum("ikab,...a,...b->...ik", self.tensor, m, m)

    def _frequencies(self, m):
        return np.sqrt(np.clip(np.linalg.eigvalsh(self._d2(m)), 0, None))

    def classical_constant(self) -> float:
        "C = FP[1/2 ln det D2(m)]"
        if self._classical is None:
            half_log = lambda m: 0.5 * np.linalg.slogdet(self._d2(m))[1]  # noqa: E731
            integral = 3 * self.metric_weights.sum() * _radial(2, log=True) + _radial(
                2
            ) * np.sum(self.metric_weights * half_log(self.metric_directions))
            self._classical = integral - 2 * np.sum(self.phi * half_log(self.points))
        return self._classical

    def zero_point(self) -> float:
        "FP[sum_j omega_j / 2], rad/s"
        if self._zero_point is None:
            integral = _radial(3) * np.sum(
                self.metric_weights[:, None]
                * self._frequencies(self.metric_directions)
                / 2
            )
            lattice = np.sum(self.phi[:, None] * self._frequencies(self.points))
            self._zero_point = integral - lattice
        return self._zero_point

    def weight(self, temperature):
        "(F eV, S eV/K, Cv eV/K) of the Gamma cell at one temperature"
        if temperature <= 0:
            return HBAR_EV_S * self.zero_point(), 0.0, 0.0
        tau = BOLTZMANN_EV * temperature / HBAR_EV_S
        k = BOLTZMANN_EV
        if 4 * np.pi**2 * tau / self.speeds.max() >= CLASSICAL:
            constant = self.classical_constant()
            if not self.centred:
                # no point at the origin, so a constant has no finite part
                return HBAR_EV_S * tau * constant, -k * constant, 0.0
            log_tau = np.log(tau)
            free = HBAR_EV_S * tau * (constant - 3 * log_tau)
            return free, k * (3 * log_tau + 3 - constant), 3 * k

        # Debye continuum: F = -a kT, S = 4a k, Cv = 12a k
        a = np.pi**4 / 45 * tau**3 * np.sum(self.weights[:, None] * self.speeds**-3.0)
        reach = THERMAL_RANGE * tau / self.speeds.min()
        points = _half_box(np.full(3, int(np.ceil(reach)) + 1), self.shift)
        points = points[np.linalg.norm(points, axis=1) < reach]
        x = self._frequencies(points).ravel() / tau
        x = x[(x > 0) & (x < THERMAL_RANGE)]
        one_minus = -np.expm1(-x)
        log_term = np.log(one_minus)
        occupation = np.exp(-x) / one_minus
        free = HBAR_EV_S * (self.zero_point() - a * tau - 2 * tau * log_term.sum())
        entropy = k * (4 * a - 2 * np.sum(x * occupation - log_term))
        heat_capacity = k * (12 * a - 2 * np.sum(x * x * occupation / one_minus))
        return free, entropy, heat_capacity


def _half_box(half_widths, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """(M, 3) one of each `p`, `-p` pair of the points `p = m + shift` in a box
    about the origin, without the origin. Every sum here is of an even
    function, so it is twice the sum over these. A shift of half a step
    keeps the lattice closed under `p -> -p`."""
    axes = [np.arange(-k, k + 1) + s for k, s in zip(half_widths, shift, strict=True)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    first = np.argmax(points != 0, axis=1)
    positive = points[np.arange(len(points)), first] > 0
    return points[positive]

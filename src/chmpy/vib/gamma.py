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
transform is a Gaussian, so the lattice sum aliases at ~exp(-(pi w)^2).
`|A m|` measures m in the sound-speed metric, which a spherical cutoff in
mesh units does not: on anisotropic meshes that costs percents. This is the
order-zero term of the singular Euler-Maclaurin expansion (Buchheit and
Kessler, Nonlinearity, 2022, doi:10.1088/1361-6544/ac73d0). Adding `W / N` to
the mesh average replaces the dropped modes; for F the error then falls as
h^4-h^5 rather than h^3 ln h.

The free energy of a mode is `hbar w / 2 + kT ln(1 - exp(-x))`. The
zero-point part is homogeneous, so its continuum integral is a radial
constant times an angular sum. The thermal part is smooth away from Gamma,
so `(1 - phi)` times it sums to its own integral to well below tolerance, and
its finite part needs only the same cutoff lattice, at any temperature, with
a one-dimensional radial quadrature per direction for the integral. The
entropy and heat capacity are the same finite parts of their own per-mode
functions. All the eigenvalues are computed once, so each temperature costs
one pass over them.

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

#: smooth cutoff radius and width, in mesh units: 1 - phi(0) ~ 1e-9, and
#: the result agrees with R = 16, w = 2.5 to a few parts in 10^6
RADIUS = 6.0
WIDTH = 1.5

#: the cutoff is taken as zero beyond RADIUS + TAIL * WIDTH, where
#: erfc(TAIL) / 2 ~ 1e-17
TAIL = 6.0

#: thermal integrands are cut off where `hbar omega / kT` reaches this
THERMAL_RANGE = 40.0

#: Lebedev order for the thermal continuum integrals, which are smooth in
#: direction
THERMAL_ORDER = 35

#: GPa -> Pa, and 1/A^2 -> 1/m^2
_GPA_PER_A2 = 1e9 * 1e20
_AMU_KG = 1.66053906660e-27


@cache
def _sphere(order: int = 131):
    "Lebedev directions (n, 3) and weights summing to 4 pi"
    from scipy.integrate import lebedev_rule

    points, weights = lebedev_rule(order)
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
        reach = RADIUS + TAIL * WIDTH
        half = np.ceil(reach * np.linalg.norm(inverse, axis=1)).astype(int) + 1
        points = _half_box(half, self.shift)
        radii = np.linalg.norm(points @ self.metric.T, axis=1)
        inside = radii < reach
        self.phi = _cutoff(radii[inside])
        # D2 eigenvalues at the lattice points, shared by every finite part
        self.frequencies = np.sqrt(
            np.clip(_eigenvalues(self._d2(points[inside])), 0, None)
        )

        # continuum integrals of homogeneous functions in the metric of A
        self.metric_directions = directions @ inverse.T
        self.metric_weights = self.weights / np.linalg.det(self.metric)
        self._zero_point = None
        self._continuum = None

    def _d2(self, m):
        "(..., 3, 3) D2 at mesh-index vectors, as one matrix product"
        m = np.asarray(m, dtype=float)
        outer = (m[..., :, None] * m[..., None, :]).reshape(*m.shape[:-1], 9)
        return (outer @ self.tensor.reshape(9, 9).T).reshape(*m.shape[:-1], 3, 3)

    def _frequencies(self, m):
        return np.sqrt(np.clip(_eigenvalues(self._d2(m)), 0, None))

    def zero_point(self) -> float:
        "FP[sum_j omega_j / 2], rad/s"
        if self._zero_point is None:
            integral = _radial(3) * np.sum(
                self.metric_weights[:, None]
                * self._frequencies(self.metric_directions)
                / 2
            )
            lattice = np.sum(self.phi[:, None] * self.frequencies)
            self._zero_point = integral - lattice
        return self._zero_point

    def weight(self, temperature):
        """(F eV, S eV/K, Cv eV/K) of the Gamma cell at one temperature.

        The thermal part `kT ln(1 - exp(-x))` is smooth except near the
        origin, so its finite part needs only the same cutoff lattice as the
        zero point: `(1 - phi)` times it is smooth and decays, and its sum and
        integral agree to far below the tolerance at any temperature. No
        separate classical regime, and no temperature-dependent box.
        """
        if temperature <= 0:
            return HBAR_EV_S * self.zero_point(), 0.0, 0.0
        tau = BOLTZMANN_EV * temperature / HBAR_EV_S
        k = BOLTZMANN_EV

        # only modes below THERMAL_RANGE kT contribute to the thermal parts
        warm = self.frequencies[:, 0] < THERMAL_RANGE * tau
        x = self.frequencies[warm] / tau
        phi = self.phi[warm, None]
        lattice = [2 * np.sum(phi * g) for g in _thermal(x)]
        continuum = self._thermal_continuum(tau)
        free, entropy, heat = (c - s for c, s in zip(continuum, lattice, strict=True))
        return (
            HBAR_EV_S * (self.zero_point() + tau * free),
            k * entropy,
            k * heat,
        )

    def _thermal_continuum(self, tau):
        """int phi(r) g(omega / tau) d^3m for the three thermal functions g.

        Per direction and branch the frequency is `c r` along the radius in
        the metric of A, so with `x = c r / tau` each is
        `(tau / c)^3 int x^2 phi(x tau / c) g(x) dx`, cut off where g has
        decayed or phi has.
        """
        if self._continuum is None:
            directions, weights = _sphere(THERMAL_ORDER)
            inverse = np.linalg.inv(self.metric)
            speeds = self._frequencies(directions @ inverse.T)
            self._continuum = (speeds, weights / np.linalg.det(self.metric))
        speeds, weights = self._continuum
        scale = tau / speeds  # (n, 3): radius per unit x
        top = np.minimum(THERMAL_RANGE, (RADIUS + TAIL * WIDTH) / scale)
        nodes, node_weights = _gauss_legendre()
        x = top[..., None] * nodes  # (n, 3, q)
        measure = (
            (top * scale**3)[..., None]
            * node_weights
            * x**2
            * _cutoff(x * scale[..., None])
        )
        return [
            float(np.sum(weights[:, None, None] * measure * g)) for g in _thermal(x)
        ]


def _thermal(x):
    """Per mode, in units of kT and k: ln(1 - e^-x), the entropy and the heat
    capacity. Zero where x is not positive."""
    x = np.asarray(x, dtype=float)
    positive = x > 0
    safe = np.where(positive, x, 1.0)
    one_minus = -np.expm1(-safe)
    log_term = np.log(one_minus)
    occupation = np.exp(-safe) / one_minus
    free = log_term
    entropy = safe * occupation - log_term
    heat = safe * safe * occupation / one_minus
    return [np.where(positive, value, 0.0) for value in (free, entropy, heat)]


@cache
def _gauss_legendre(order: int = 64):
    "Gauss-Legendre nodes and weights on [0, 1]"
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1), 0.5 * weights


def _eigenvalues(matrices) -> np.ndarray:
    """Ascending eigenvalues of symmetric 3x3 matrices, in closed form.

    The trigonometric solution of the characteristic cubic: an order of
    magnitude faster than LAPACK on many small matrices, and accurate to
    round-off relative to the largest eigenvalue, which is all a sum of them
    needs.
    """
    a = np.asarray(matrices, dtype=float)
    q = np.trace(a, axis1=-2, axis2=-1) / 3
    off = a[..., 0, 1] ** 2 + a[..., 0, 2] ** 2 + a[..., 1, 2] ** 2
    shifted = a - q[..., None, None] * np.eye(3)
    p = np.sqrt((np.einsum("...ii->...i", shifted) ** 2).sum(axis=-1) / 6 + off / 3)
    safe = np.where(p > 0, p, 1.0)
    half_det = _determinant(shifted / safe[..., None, None]) / 2
    angle = np.arccos(np.clip(half_det, -1.0, 1.0)) / 3
    largest = q + 2 * p * np.cos(angle)
    smallest = q + 2 * p * np.cos(angle + 2 * np.pi / 3)
    middle = 3 * q - largest - smallest
    return np.stack([smallest, middle, largest], axis=-1)


def _determinant(a) -> np.ndarray:
    "Determinants of 3x3 matrices, written out"
    return (
        a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
        - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
        + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
    )


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

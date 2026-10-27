"""Quasi-harmonic geometry: the cell minimising G = E_static + F_vib at T.

The cell is `h = h0 (I + sum_k q_k B_k)` over the strain directions `B_k` the
space group allows, so an anisotropic crystal can expand differently along
each axis. At every cell the atoms are relaxed on the static energy (the zero
static internal stress approximation, ZSISA), and

    G(q) = E_static(q) + F_vib(q, T)

is minimised on a model, not by searching:

* E_static is quadratic about the current cell, with the analytic static
  stress as its gradient and, as its Hessian, the static elastic tensor
  projected onto the strain basis as `V0 B_k : C : B_l`, computed once and
  updated by BFGS from the static stresses, which are free of noise;
* F_vib is a parabola along each strain direction through the current cell
  and the cells strained by +/-h, which gives its slope and curvature;
* the step is Newton on the sum, bounded, and the iteration stops once it is
  small.

F_vib is never differentiated over less than `h`, 1% by default. Its strain
derivative is in effect a third derivative of the energy, which
machine-learned potentials reproduce far less smoothly than the energy itself:
over a few tenths of a percent of strain, their force constants can wiggle
enough to change a central difference by tenths of a GPa. A parabola through
points 1% apart averages that out, the elastic tensor carries the coupling
between directions, and no cell is ever evaluated just to be rejected, so the
cost is fixed at 2n + 1 force-constant sets per iteration for n strain
directions.

Imaginary modes mean the harmonic free energy is not defined. Dropping them
would make the finite differences measure which modes were dropped, so a
step that reaches a cell with imaginary modes is halved, and imaginary modes
at the starting cell stop the calculation.

    result = qha_structure(crystal, calculator, temperature=293.0)
    tensor = elastic_tensor(result.structure, calculator)  # quasi-static C
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

from .force_constants import force_constants
from .gamma import MechanicallyUnstable
from .thermo import thermal_properties

LOG = logging.getLogger(__name__)

#: finite-difference step in the strain amplitudes for dF_vib/dq
DEFAULT_STRAIN = 0.005

#: strain amplitude the F_vib parabola is fitted over in `qha_structure`
FIT_STRAIN = 0.01

#: largest change in any strain amplitude in one Newton step
MAX_STEP = 0.02


class ImaginaryModes(RuntimeError):
    "A cell had imaginary phonon modes, so its free energy is not defined"


@dataclass
class ThermalStress:
    """The vibrational stress of a structure at a temperature.

    Attributes:
        stress: (3, 3) eV/A^3, `(1/V) dF_vib/d eps`, ASE sign convention
        free_energy: F_vib of the unstrained structure, eV per cell
        temperature: K
    """

    stress: np.ndarray
    free_energy: float
    temperature: float

    @property
    def stress_gpa(self) -> np.ndarray:
        return self.stress * EV_PER_ANGSTROM3_TO_GPA


@dataclass
class QHAResult:
    """A quasi-harmonic structure and how it was reached.

    Attributes:
        structure: the crystal minimising G at `temperature`, atoms relaxed
        temperature: K
        free_energy: G = E_static + F_vib at the minimum, eV per cell
        strain: (3, 3) strain from the starting cell
        converged: whether the step fell below tolerance
        history: per iteration, the amplitudes, G, the largest gradient and
            the step taken
    """

    structure: object
    temperature: float
    free_energy: float
    strain: np.ndarray
    converged: bool
    history: list = field(default_factory=list)

    @property
    def volume_change(self) -> float:
        "V / V_start - 1"
        return float(np.linalg.det(np.eye(3) + self.strain) - 1)


class _FreeEnergySurface:
    """G(q) and its gradient for one crystal, temperature and model."""

    def __init__(
        self,
        crystal,
        calculator,
        temperature,
        *,
        mesh,
        supercell,
        cutoff,
        strain,
        info,
        fmax,
        gamma_correction=False,
    ):
        from chmpy.opt.coordinates import strain_basis_for

        self.reference = crystal
        self.cell = np.asarray(crystal.unit_cell.direct)
        self.volume = crystal.unit_cell.volume()
        self.basis = strain_basis_for(crystal)
        self.calculator = calculator
        self.temperature = float(temperature)
        self.mesh, self.supercell, self.cutoff = mesh, supercell, cutoff
        self.strain, self.info, self.fmax = strain, info, fmax
        self.gamma_correction = gamma_correction
        self._crystals = {}
        self._vibrational = {}
        self._elastic = None

    def deformation(self, q):
        return np.eye(3) + np.einsum("k,kab->ab", q, self.basis)

    def _key(self, q):
        return tuple(np.round(q, 12))

    def crystal(self, q):
        "The crystal at amplitudes q with its atoms relaxed on the static energy"
        key = self._key(q)
        if key not in self._crystals:
            from chmpy.crystal import Crystal, UnitCell
            from chmpy.opt import relax
            from chmpy.opt.relax import coordinates_for

            crystal = Crystal(
                UnitCell(self.cell @ self.deformation(q)),
                self.reference.space_group,
                self.reference.asymmetric_unit,
            )
            if coordinates_for(crystal, cell=False).n_dof:
                outcome = relax(
                    crystal,
                    self.calculator,
                    cell=False,
                    info=self.info,
                    fmax=self.fmax,
                    steps=500,
                )
                if not outcome.converged:
                    LOG.warning("atomic relaxation at q=%s did not converge", key)
                crystal = outcome.structure
            self._crystals[key] = crystal
        return self._crystals[key]

    def static(self, q):
        "(E_static, dE_static/dq) at q"
        from chmpy.calc.system import System

        crystal = self.crystal(q)
        result = self.calculator(
            System.from_crystal(crystal, **(self.info or {})), ("energy", "stress")
        )
        # dE/dq_k = V <F^-T sigma, B_k>, as for the strain coordinates
        stress = np.linalg.solve(self.deformation(q).T, result.stress * result.volume)
        return result.energy, np.einsum("kab,ab->k", self.basis, stress)

    def vibrational(self, q):
        "F_vib at q, raising ImaginaryModes if any are found"
        key = self._key(q)
        if key not in self._vibrational:
            constants = force_constants(
                self.crystal(q),
                self.calculator,
                supercell=self.supercell,
                cutoff=self.cutoff,
                info=self.info,
            )
            correction = self.gamma_correction
            if isinstance(correction, str) and correction == "elastic":
                correction = self.elastic()
            try:
                properties = thermal_properties(
                    constants,
                    [self.temperature],
                    mesh=self.mesh,
                    gamma_correction=correction,
                )
            except MechanicallyUnstable as unstable:
                raise ImaginaryModes(
                    f"acoustic modes near Gamma are imaginary at strain amplitudes "
                    f"{np.round(q, 5).tolist()}: {unstable}"
                ) from unstable
            if properties.n_imaginary:
                raise ImaginaryModes(
                    f"{properties.n_imaginary} imaginary modes at strain amplitudes "
                    f"{np.round(q, 5).tolist()}: the harmonic free energy is not "
                    "defined there"
                )
            self._vibrational[key] = float(properties.free_energy[0])
        return self._vibrational[key]

    def value(self, q):
        return self.static(q)[0] + self.vibrational(q)

    def vibrational_gradient(self, q):
        gradient = np.zeros(len(q))
        for k in range(len(q)):
            step = np.zeros(len(q))
            step[k] = self.strain
            gradient[k] = (self.vibrational(q + step) - self.vibrational(q - step)) / (
                2 * self.strain
            )
        return gradient

    def vibrational_parabolas(self, q):
        """F_vib at q, and its slope and curvature along each strain direction,
        from the parabola through q and q +/- strain."""
        h = self.strain
        centre = self.vibrational(q)
        slope, curvature = np.zeros(len(q)), np.zeros(len(q))
        for k in range(len(q)):
            step = np.zeros(len(q))
            step[k] = h
            plus, minus = self.vibrational(q + step), self.vibrational(q - step)
            slope[k] = (plus - minus) / (2 * h)
            curvature[k] = (plus - 2 * centre + minus) / h**2
        return centre, slope, curvature

    def elastic(self):
        "The static elastic tensor of the reference cell"
        if self._elastic is None:
            from chmpy.opt import elastic_tensor

            self._elastic = elastic_tensor(
                self.reference, self.calculator, info=self.info, fmax=self.fmax
            )
        return self._elastic

    def hessian(self):
        "V0 B_k : C : B_l from the static elastic tensor at the start"
        from chmpy.opt.strain import elastic_from_voigt, to_vector

        stiffness = elastic_from_voigt(self.elastic().c_voigt) / EV_PER_ANGSTROM3_TO_GPA
        vectors = to_vector(self.basis)
        return self.volume * vectors @ stiffness @ vectors.T


def thermal_stress(
    crystal,
    calculator,
    temperature: float,
    *,
    mesh=(4, 4, 4),
    supercell=None,
    cutoff: float | None = None,
    strain: float = DEFAULT_STRAIN,
    info=None,
    fmax: float = 1e-3,
    gamma_correction=False,
) -> ThermalStress:
    """`(1/V) dF_vib/d eps` over the strains a crystal's symmetry allows.

    Raises:
        ImaginaryModes: if any strained cell has imaginary modes
    """
    surface = _FreeEnergySurface(
        crystal,
        calculator,
        temperature,
        mesh=mesh,
        supercell=supercell,
        cutoff=cutoff,
        strain=strain,
        info=info,
        fmax=fmax,
        gamma_correction=gamma_correction,
    )
    q = np.zeros(len(surface.basis))
    gradient = surface.vibrational_gradient(q)
    return ThermalStress(
        stress=np.einsum("k,kab->ab", gradient, surface.basis) / surface.volume,
        free_energy=surface.vibrational(q),
        temperature=float(temperature),
    )


def qha_structure(
    crystal,
    calculator,
    temperature: float,
    *,
    mesh=(4, 4, 4),
    supercell=None,
    cutoff: float | None = None,
    strain: float = FIT_STRAIN,
    info=None,
    fmax: float = 1e-3,
    xtol: float = 2e-4,
    max_iterations: int = 10,
    max_step: float = MAX_STEP,
    gamma_correction=False,
) -> QHAResult:
    """Minimise G = E_static + F_vib over the symmetry-allowed strains.

    Args:
        crystal: a statically relaxed `Crystal`; strains are measured from its
            cell
        calculator: a `chmpy.calc.Calculator`
        temperature: K
        mesh: q-mesh for F_vib
        gamma_correction: add the Gamma correction to F_vib, so a coarse mesh
            can stand in for a dense one. True reads the long-wave tensor off
            each cell's own force constants; "elastic" takes it from the
            reference cell's elastic tensor, which is computed anyway and is
            more robust when the force constants are noisy near Gamma
        supercell, cutoff: passed to `force_constants`; size the supercell
            from the model's receptive field, not its cutoff radius
        strain: half-width of the F_vib parabola along each strain direction
        info: model inputs that are not geometry
        fmax: force tolerance for the atomic relaxations, eV/A
        xtol: convergence on the largest strain amplitude of the next step;
            2e-4 is a volume change of about 0.06%
        max_iterations: Newton steps before giving up
        max_step: largest change in any strain amplitude per step

    Returns:
        QHAResult

    Raises:
        ImaginaryModes: if any cell visited has imaginary modes
    """
    surface = _FreeEnergySurface(
        crystal,
        calculator,
        temperature,
        mesh=mesh,
        supercell=supercell,
        cutoff=cutoff,
        strain=strain,
        info=info,
        fmax=fmax,
        gamma_correction=gamma_correction,
    )
    static_hessian = surface.hessian()
    q = np.zeros(len(surface.basis))
    history, converged = [], False
    previous, previous_static, step = None, None, None
    for _ in range(max_iterations):
        try:
            F, slope, curvature = surface.vibrational_parabolas(q)
        except ImaginaryModes as unstable:
            if step is None:
                raise
            # the step reached a dynamically unstable cell: go half as far
            LOG.info("halving a step that reached an unstable cell: %s", unstable)
            step *= 0.5
            q = previous + step
            continue
        E, static_gradient = surface.static(q)
        if previous is not None:
            # BFGS on the static part alone: its gradient is analytic and free
            # of noise, and the lattice softens as it expands
            s, y = q - previous, static_gradient - previous_static
            if s @ y > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
                hs = static_hessian @ s
                static_hessian = (
                    static_hessian
                    + np.outer(y, y) / (y @ s)
                    - np.outer(hs, hs) / (s @ hs)
                )
        previous_static = static_gradient
        gradient = static_gradient + slope
        hessian = static_hessian + np.diag(curvature)
        if np.linalg.eigvalsh(hessian).min() <= 0:
            # F_vib curving down faster than the lattice stiffens: trust the
            # static curvature alone for this step
            hessian = static_hessian
        step = -np.linalg.solve(hessian, gradient)
        step *= min(1.0, max_step / max(float(np.abs(step).max()), 1e-300))
        largest = (
            float(np.abs(gradient).max()) / surface.volume * EV_PER_ANGSTROM3_TO_GPA
        )
        history.append(
            {
                "q": q.tolist(),
                "G": E + F,
                "gradient_gpa": largest,
                "step": step.tolist(),
            }
        )
        if float(np.abs(step).max()) < xtol:
            converged = True
            break
        previous = q.copy()
        q = q + step
    if not converged:
        LOG.warning(
            "the quasi-harmonic cell did not converge to a step of %g in %d steps",
            xtol,
            max_iterations,
        )
    G = surface.value(q)
    return QHAResult(
        structure=surface.crystal(q),
        temperature=float(temperature),
        free_energy=G,
        strain=surface.deformation(q) - np.eye(3),
        converged=converged,
        history=history,
    )


# -- isotropic: a volume scan --------------------------------------------------


@dataclass
class VolumeQHA:
    """Quasi-harmonic equilibrium from an isotropic volume scan.

    Attributes:
        temperatures: (T,) K
        volumes: (V,) cell volumes sampled, A^3, those with imaginary modes
            left out
        static_energies: (V,) E_static at each, atoms relaxed, eV per cell
        free_energies: (T, V) F_vib at each, eV per cell
        volume: (T,) equilibrium volume, A^3
        free_energy: (T,) G at the equilibrium volume, eV per cell
        bulk_modulus: (T,) isothermal bulk modulus at that volume, GPa
        in_range: (T,) whether the minimum lies inside the sampled volumes;
            where it does not, volume, free_energy and bulk_modulus are NaN
            rather than an extrapolation
        reference_volume: volume of the starting (static) cell, A^3
        n_imaginary: (V,) imaginary modes found at every volume sampled
        sampled: (V,) every volume sampled, including any left out
    """

    temperatures: np.ndarray
    volumes: np.ndarray
    static_energies: np.ndarray
    free_energies: np.ndarray
    volume: np.ndarray
    free_energy: np.ndarray
    bulk_modulus: np.ndarray
    in_range: np.ndarray
    reference_volume: float
    n_imaginary: np.ndarray
    sampled: np.ndarray

    @property
    def volume_change(self) -> np.ndarray:
        "(T,) V(T) / V_static - 1"
        return self.volume / self.reference_volume - 1

    @property
    def thermal_expansion(self) -> np.ndarray:
        "(T,) volumetric expansion coefficient (1/V) dV/dT, 1/K"
        if len(self.temperatures) < 2:
            return np.full(len(self.temperatures), np.nan)
        return np.gradient(self.volume, self.temperatures) / self.volume


def vinet(volume, energy, minimum, bulk, derivative):
    """The Vinet equation of state, E(V) in the units of `energy` and `bulk`*V."""
    eta = (np.asarray(volume) / minimum) ** (1.0 / 3.0)
    x = 1.5 * (derivative - 1.0) * (1.0 - eta)
    return energy + 2.0 * bulk * minimum / (derivative - 1.0) ** 2 * (
        2.0 - (2.0 - 2.0 * x) * np.exp(x)
    )


def _fit_vinet(volumes, values):
    """(E0, V0, B0 in eV/A^3, B0') of a Vinet curve through the points.

    Fitted relative to the lowest point and in units of the mean volume: the
    energies are thousands of eV with differences of meV, which leaves the
    raw least-squares problem too badly scaled to converge.
    """
    from scipy.optimize import least_squares

    volumes, values = np.asarray(volumes, float), np.asarray(values, float)
    unit, offset = float(volumes.mean()), float(values.min())
    v, e = volumes / unit, values - offset
    a, b, _ = np.polyfit(v, e, 2)
    v0 = -b / (2 * a) if a > 0 else float(v[np.argmin(e)])
    v0 = float(np.clip(v0, 0.5, 1.5))
    guess = [float(e.min()), v0, max(2 * a * v0, 1e-6), 5.0]
    fit = least_squares(
        lambda p: vinet(v, *p) - e,
        guess,
        bounds=([-np.inf, 0.3, 1e-9, 1.01], [np.inf, 3.0, np.inf, 30.0]),
        x_scale="jac",
    )
    e0, v0, b0, b1 = fit.x
    # B0 came out in eV per (unit volume); the energy scale is unchanged
    return np.array([e0 + offset, v0 * unit, b0 / unit, b1])


def _descend(values, start) -> int:
    "The local minimum of a sequence reached by walking downhill from `start`"
    index = int(start)
    while True:
        if index > 0 and values[index - 1] < values[index]:
            index -= 1
        elif index < len(values) - 1 and values[index + 1] < values[index]:
            index += 1
        else:
            return index


def qha_volume(
    crystal,
    calculator,
    temperatures,
    *,
    scales=(0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10),
    mesh=(4, 4, 4),
    supercell=None,
    cutoff: float | None = None,
    gamma_correction=False,
    info=None,
    fmax: float = 1e-3,
    shift=0.0,
    window: int = 2,
) -> VolumeQHA:
    """Quasi-harmonic V(T) and G(T) by scaling the cell isotropically.

    Each volume costs one atomic relaxation and one set of force constants,
    whatever the number of temperatures: F_vib at every temperature is a sum
    over the same modes. G(V) = E_static(V) + F_vib(V, T) is then fitted with
    the Vinet equation of state at each temperature, about its lowest sampled
    point, which smooths over noise in F_vib that a derivative would amplify.

    The cell shape is kept, so this is exact for cubic crystals and an
    approximation otherwise; `qha_structure` lets the shape change too, at a
    cost that grows with the number of strain directions and a sensitivity to
    noise in the model's third derivatives.

    Args:
        crystal: a statically relaxed `Crystal`
        calculator: a `chmpy.calc.Calculator`
        temperatures: K
        scales: volumes to sample, relative to the static cell. The fitted
            minimum at every temperature should lie inside them; `in_range`
            on the result says whether it does.
        mesh, shift: q-mesh for F_vib
        supercell, cutoff: passed to `force_constants`
        gamma_correction: as for `thermal_properties`; True reads the
            long-wave tensor off each volume's own force constants
        info: model inputs that are not geometry
        fmax: force tolerance for the atomic relaxations, eV/A
        window: the equation of state is fitted to this many volumes either
            side of the minimum, not to all of them. The minimum is followed
            downhill from the static cell, upward in temperature: far from
            it the relaxed atoms can switch to another branch of the energy
            surface, which is not the same crystal expanding

    Returns:
        VolumeQHA, per unit cell
    """
    from chmpy.calc.system import System
    from chmpy.crystal import Crystal, UnitCell
    from chmpy.opt import relax
    from chmpy.opt.relax import coordinates_for

    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    reference = np.asarray(crystal.unit_cell.direct)
    reference_volume = crystal.unit_cell.volume()
    sampled, energies, free, imaginary = [], [], [], []
    for scale in scales:
        scaled = Crystal(
            UnitCell(reference * float(scale) ** (1.0 / 3.0)),
            crystal.space_group,
            crystal.asymmetric_unit,
        )
        if coordinates_for(scaled, cell=False).n_dof:
            outcome = relax(
                scaled, calculator, cell=False, info=info, fmax=fmax, steps=500
            )
            if not outcome.converged:
                LOG.warning("atomic relaxation at volume x%g did not converge", scale)
            scaled = outcome.structure
        energy = calculator.energy(System.from_crystal(scaled, **(info or {})))
        constants = force_constants(
            scaled, calculator, supercell=supercell, cutoff=cutoff, info=info
        )
        try:
            properties = thermal_properties(
                constants,
                temperatures,
                mesh=mesh,
                gamma_correction=gamma_correction,
                shift=shift,
            )
            n_imaginary, values = properties.n_imaginary, properties.free_energy
        except MechanicallyUnstable:
            n_imaginary, values = -1, np.full(len(temperatures), np.nan)
        sampled.append(scaled.unit_cell.volume())
        energies.append(energy)
        free.append(values)
        imaginary.append(n_imaginary)

    sampled, energies = np.array(sampled), np.array(energies)
    free, imaginary = np.array(free).T, np.array(imaginary)
    keep = imaginary == 0
    if not keep.all():
        LOG.warning(
            "left out %d of %d volumes with imaginary modes or an unstable "
            "long-wave tensor: x%s",
            int((~keep).sum()),
            len(keep),
            np.round(np.asarray(scales)[~keep], 4).tolist(),
        )
    if keep.sum() < 2 * window + 1:
        raise ImaginaryModes(
            f"only {int(keep.sum())} of {len(keep)} volumes are dynamically "
            "stable, too few for an equation of state"
        )
    volumes = sampled[keep]

    minima, values, bulk, in_range = [], [], [], []
    lowest = int(np.argmin(np.abs(volumes - reference_volume)))
    order = np.argsort(temperatures)
    fitted = {}
    for t in order:
        G = energies[keep] + free[t, keep]
        # follow the minimum continuously from the static cell, upward in
        # temperature, rather than taking the lowest point anywhere: far from
        # the minimum a relaxation can land on another branch of the energy
        # surface, which may even be lower, but is not this crystal expanding
        lowest = _descend(G, lowest)
        start = int(np.clip(lowest - window, 0, max(len(G) - 2 * window - 1, 0)))
        part = slice(start, start + 2 * window + 1)
        _, v0, b0, _ = parameters = _fit_vinet(volumes[part], G[part])
        inside = volumes[part]
        if 0 < lowest < len(G) - 1 and inside.min() <= v0 <= inside.max():
            fitted[t] = (v0, vinet(v0, *parameters), b0 * EV_PER_ANGSTROM3_TO_GPA, True)
        else:
            # never report an extrapolated minimum
            fitted[t] = (np.nan, np.nan, np.nan, False)
    for t in range(len(temperatures)):
        minima.append(fitted[t][0])
        values.append(fitted[t][1])
        bulk.append(fitted[t][2])
        in_range.append(fitted[t][3])
    minima, in_range = np.array(minima), np.array(in_range)
    if not in_range.all():
        LOG.warning(
            "the G minimum lies outside the sampled volumes at %s K; widen `scales`",
            temperatures[~in_range].tolist(),
        )
    return VolumeQHA(
        temperatures=temperatures,
        volumes=volumes,
        static_energies=energies[keep],
        free_energies=free[:, keep],
        volume=minima,
        free_energy=np.array(values),
        bulk_modulus=np.array(bulk),
        in_range=in_range,
        reference_volume=reference_volume,
        n_imaginary=imaginary,
        sampled=sampled,
    )

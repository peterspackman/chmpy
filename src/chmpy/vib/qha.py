"""Quasi-harmonic geometry: the cell minimising G = E_static + F_vib at T.

The cell is `h = h0 (I + sum_k q_k B_k)` over the strain directions `B_k` the
space group allows, so an anisotropic crystal can expand differently along
each axis. At every cell the atoms are relaxed on the static energy (the zero
static internal stress approximation, ZSISA), and

    G(q) = E_static(q) + F_vib(q, T)

is minimised by damped Newton steps:

* the gradient is the static stress, analytically, plus dF_vib/dq_k by
  central differences, each needing force constants at a strained cell;
* the Hessian is the static elastic tensor projected onto the strain basis,
  `V0 B_k : C : B_l`, computed once at the start;
* each step is bounded and accepted only if G actually decreases, so a large
  thermal stress cannot carry the cell to a region where the crystal has
  come apart, which a fixed applied stress does.

Imaginary modes mean the harmonic free energy is not defined. Dropping them
would make the finite differences measure which modes were dropped, so a
trial cell with imaginary modes is a rejected step, and imaginary modes at
the current cell or the cells its gradient needs stop the calculation.

    result = qha_structure(crystal, calculator, temperature=293.0)
    tensor = elastic_tensor(result.structure, calculator)  # quasi-static C
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

from .force_constants import force_constants
from .thermo import thermal_properties

LOG = logging.getLogger(__name__)

#: finite-difference step in the strain amplitudes for dF_vib/dq
DEFAULT_STRAIN = 0.005

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
        converged: whether the gradient fell below tolerance
        history: per iteration, the amplitudes, G and the largest gradient
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
        self._crystals = {}
        self._vibrational = {}

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
            properties = thermal_properties(
                constants, [self.temperature], mesh=self.mesh
            )
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

    def hessian(self):
        "V0 B_k : C : B_l from the static elastic tensor at the start"
        from chmpy.opt import elastic_tensor
        from chmpy.opt.strain import elastic_from_voigt, to_vector

        result = elastic_tensor(
            self.reference, self.calculator, info=self.info, fmax=self.fmax
        )
        stiffness = elastic_from_voigt(result.c_voigt) / EV_PER_ANGSTROM3_TO_GPA
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
    strain: float = DEFAULT_STRAIN,
    info=None,
    fmax: float = 1e-3,
    gtol: float = 1e-3,
    max_iterations: int = 20,
    max_step: float = MAX_STEP,
) -> QHAResult:
    """Minimise G = E_static + F_vib over the symmetry-allowed strains.

    Args:
        crystal: a statically relaxed `Crystal`; strains are measured from its
            cell
        calculator: a `chmpy.calc.Calculator`
        temperature: K
        mesh: q-mesh for F_vib
        supercell, cutoff: passed to `force_constants`; size the supercell
            from the model's receptive field, not its cutoff radius
        strain: finite-difference step for dF_vib/dq
        info: model inputs that are not geometry
        fmax: force tolerance for the atomic relaxations, eV/A
        gtol: convergence on the largest gradient component, as a stress in GPa
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
    )
    hessian = surface.hessian()
    q = np.zeros(len(surface.basis))
    G = surface.value(q)
    history, converged = [], False
    previous = None
    for _ in range(max_iterations):
        _, static_gradient = surface.static(q)
        gradient = static_gradient + surface.vibrational_gradient(q)
        largest = (
            float(np.abs(gradient).max()) / surface.volume * EV_PER_ANGSTROM3_TO_GPA
        )
        history.append({"q": q.tolist(), "G": G, "gradient_gpa": largest})
        if largest < gtol:
            converged = True
            break
        if previous is not None:
            # BFGS: the static C misses the vibrational softening of G, which
            # leaves plain Newton converging only linearly
            s, y = q - previous[0], gradient - previous[1]
            if s @ y > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
                hs = hessian @ s
                hessian = (
                    hessian + np.outer(y, y) / (y @ s) - np.outer(hs, hs) / (s @ hs)
                )
        previous = (q.copy(), gradient.copy())
        step = -np.linalg.solve(hessian, gradient)
        step *= min(1.0, max_step / max(float(np.abs(step).max()), 1e-300))
        for _ in range(6):
            trial = q + step
            try:
                trial_G = surface.value(trial)
                if trial_G < G:
                    # the next gradient needs the neighbouring cells too; they
                    # are cached, so this costs nothing for an accepted step
                    surface.vibrational_gradient(trial)
            except ImaginaryModes as unstable:
                # the step left the region where the crystal is dynamically
                # stable; that is a step to shorten, not a failure
                LOG.info("rejected a trial cell: %s", unstable)
                trial_G = np.inf
            if trial_G < G:
                q, G = trial, trial_G
                break
            step *= 0.5
        else:
            LOG.warning("no step along the Newton direction lowered G; stopping")
            break
    if not converged:
        LOG.warning(
            "the quasi-harmonic cell did not converge to %g GPa in %d steps",
            gtol,
            max_iterations,
        )
    return QHAResult(
        structure=surface.crystal(q),
        temperature=float(temperature),
        free_energy=G,
        strain=surface.deformation(q) - np.eye(3),
        converged=converged,
        history=history,
    )

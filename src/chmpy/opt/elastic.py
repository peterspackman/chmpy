"""Elastic constants, by straining and re-relaxing.

The elastic tensor is the second derivative of the energy with respect to
strain, `C_ij = (1/V) d2E / de_i de_j`, and it is computed here the way it is
measured: deform the cell by a small strain, let the atoms settle, and read the
stress that comes back.

Two things are easy to get wrong and are handled here.

**The ions have to relax.** The clamped-ion tensor -- strain the cell and read
the stress without letting the atoms move -- is a different and much stiffer
quantity, sometimes by a factor of two for a molecular crystal, where a strain
is taken up mostly by molecules rearranging rather than by bonds stretching.
`relax_ions=True` (the default) runs a fixed-cell relaxation at every strained
geometry, which is what makes this expensive and what makes it right.

**The result has to respect the point group.** Finite differences and a finite
ionic relaxation leave a bit of the tensor outside the subspace the symmetry
allows -- a monoclinic crystal comes back with small non-zero constants where
its group forbids any. Those components are noise, and leaving them in puts a
spurious anisotropy into every modulus derived from the tensor. The tensor is
projected onto the invariant subspace of the crystal's point group, computed in
`chmpy.opt.strain` rather than looked up from a table of crystal systems, and
the size of that projection is reported as `symmetry_residual` -- a free and
rather sensitive check on whether the strain step and the relaxation tolerance
were tight enough.

The ionic relaxation itself runs in P1. That is deliberate: a general strain
breaks the crystal's symmetry, so constraining the atoms to the parent group's
asymmetric unit would hold them in places the strained structure does not
require them to be, and the tensor would come out too stiff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA
from chmpy.calc.system import System

from .coordinates import Atomic
from .strain import (
    cartesian_rotations,
    elastic_from_voigt,
    elastic_to_voigt,
    invariant_elastic_basis,
    project_elastic,
)
from .trust_region import TrustRegion

LOG = logging.getLogger(__name__)

#: Default engineering strain. Large enough for the stress response to clear
#: the noise in a float32 model, small enough to stay in the linear regime.
#: Measured on benzene with PET-MAD: at 0.004 the tensor came out with a
#: negative bulk modulus and a fifth of it outside the symmetry-allowed
#: subspace; at 0.02 it agreed with an independent calculation to 10%.
DEFAULT_STRAIN = 0.01

#: Relative uncertainty above which a tensor is worth complaining about.
#: Measured against the elastic-tensors2025 set, PET-MAD tensors land between
#: 0.4% and 13%, and the ones at the top of that range are soft crystals whose
#: *absolute* error is small -- so this is set where a tensor is likely to be
#: genuinely broken (the benzene case that came back with a negative bulk
#: modulus measured 20%) rather than merely soft.
NOISE_WARNING = 0.15

#: Strains tried in turn by `strain="auto"`, smallest first.
AUTO_STRAINS = (0.005, 0.01, 0.02, 0.04)

#: above this, the reference structure is not at equilibrium and "the elastic
#: constants" is no longer a well-defined thing to ask for, GPa
RESIDUAL_STRESS_WARNING = 0.5


@dataclass
class ElasticResult:
    """An elastic tensor and how much to believe it.

    Attributes:
        tensor: the `chmpy.ext.elastic_tensor.ElasticTensor`, in GPa
        evaluations: calculator evaluations used
        relaxed_ions: whether the atoms were relaxed at each strain
        residual_stress: largest stress component of the reference structure in
            GPa. Elastic constants are defined at equilibrium; a large value
            here means the answer is not the quantity it is called.
        asymmetry: largest `|C_ij - C_ji|` before the tensor was symmetrised,
            in GPa. The two are computed from different strain columns and must
            agree for a conservative model, so their difference is noise. This
            estimate exists for every crystal system.
        symmetry_residual: how much the symmetry projection changed the tensor,
            in GPa. Components symmetry forbids are noise too, and this is the
            more sensitive of the two measures -- but a triclinic crystal
            forbids nothing, so it is identically zero there.
        n_independent: independent elastic constants the point group allows
        strain: the engineering strain used
    """

    tensor: object
    evaluations: int
    relaxed_ions: bool
    residual_stress: float
    asymmetry: float
    symmetry_residual: float
    n_independent: int
    strain: float
    relaxations: list = field(default_factory=list)

    @property
    def c_voigt(self) -> np.ndarray:
        "(6, 6) elastic constants in GPa"
        return self.tensor.c_voigt

    @property
    def noise(self) -> float:
        "The larger of the two free error estimates, in GPa"
        return max(self.asymmetry, self.symmetry_residual)

    @property
    def noise_fraction(self) -> float:
        """Estimated noise as a fraction of the largest constant.

        Two measurements come free with the calculation, and the larger is
        taken. Above a few percent the tensor is not converged: the strain was
        too small for the stress response to clear the calculator's noise, or
        the ionic relaxation was not run hard enough, or both.
        """
        largest = float(np.abs(self.c_voigt).max())
        return self.noise / largest if largest > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"<ElasticResult {self.n_independent} independent constants, "
            f"{self.evaluations} evaluations, "
            f"symmetry residual {self.symmetry_residual:.3g} GPa>"
        )

    def __str__(self) -> str:
        averages = self.tensor.averages()
        return "\n".join(
            [
                repr(self),
                repr(self.tensor),
                f"bulk modulus  (Hill) {averages['bulk_modulus_avg']['hill']:8.3f} GPa",
                f"shear modulus (Hill) {averages['shear_modulus_avg']['hill']:8.3f} GPa",
                f"stable: {self.tensor.is_stable()}, "
                f"noise {100 * self.noise_fraction:.1f}% of the largest constant",
            ]
        )


def voigt_strain(index: int, magnitude: float) -> np.ndarray:
    """The strain tensor for one engineering Voigt component.

    Voigt order is xx, yy, zz, yz, xz, xy, and the shear components carry the
    engineering factor of two: `e_3 = 2 * eps_yz`, so a unit of `e_3` is half a
    unit in each of the two off-diagonal entries.

    Args:
        index: 0-5
        magnitude: the engineering strain

    Returns:
        (3, 3) symmetric strain tensor
    """
    strain = np.zeros((3, 3))
    if index < 3:
        strain[index, index] = magnitude
    else:
        a, b = ((1, 2), (0, 2), (0, 1))[index - 3]
        strain[a, b] = strain[b, a] = 0.5 * magnitude
    return strain


def elastic_tensor(
    structure,
    calculator,
    *,
    strain: float = DEFAULT_STRAIN,
    relax_ions: bool = True,
    symmetry: bool = True,
    rotations=None,
    info=None,
    fmax: float = 0.01,
    steps: int = 200,
    logger=None,
) -> ElasticResult:
    """Compute the elastic tensor of a relaxed structure.

    The structure should already be relaxed: elastic constants are second
    derivatives *at* a minimum, and a residual stress makes them ambiguous.
    `residual_stress` on the result says how well that held.

    Args:
        structure: a relaxed `Crystal` or `System`
        calculator: a `chmpy.calc.Calculator`
        strain: engineering strain magnitude for the finite difference, or
            "auto" to try successively larger ones until the tensor's own
            noise estimate falls below a few percent. A strain too small for
            the stress response to clear the calculator's noise is the usual
            reason an elastic tensor comes out wrong, and how small is too
            small depends on the calculator, not on the structure.
        relax_ions: relax the atoms at each strained cell. Leave this on unless
            you specifically want clamped-ion constants.
        symmetry: project the result onto the tensors the point group allows
        rotations: (M, 3, 3) Cartesian point operations to symmetrise with.
            Taken from a `Crystal`'s space group when not given.
        info: model inputs that are not geometry, carried on every `System`
            the calculator sees
        fmax: force convergence for the ionic relaxations, eV/A
        steps: iteration cap for each ionic relaxation
        logger: called with a line per strain point, e.g. `print`

    Returns:
        ElasticResult
    """
    system = _as_system(structure, info)
    if not system.periodic:
        raise ValueError("elastic constants need a periodic structure")

    rotations = _rotations_for(structure, rotations, symmetry)
    basis = invariant_elastic_basis(rotations)

    reference_cell = np.array(system.cell)
    reference_scaled = system.scaled_positions
    started = calculator.stats.calls

    reference = calculator(system, ("energy", "stress"))
    residual = reference.smax
    if residual > RESIDUAL_STRESS_WARNING:
        LOG.warning(
            "the reference structure carries %.3f GPa of stress; elastic "
            "constants are defined at equilibrium, so relax it first",
            residual,
        )

    attempts = AUTO_STRAINS if strain == "auto" else (float(strain),)
    for attempt, size in enumerate(attempts):
        raw, relaxations, evaluations = _strain_sweep(
            system,
            calculator,
            reference_cell,
            reference_scaled,
            size,
            relax_ions,
            fmax,
            steps,
            logger,
        )
        asymmetry = float(np.abs(raw - raw.T).max())
        voigt = 0.5 * (raw + raw.T)

        symmetry_residual = 0.0
        if symmetry:
            projected = elastic_to_voigt(
                project_elastic(elastic_from_voigt(voigt), basis)
            )
            symmetry_residual = float(np.abs(projected - voigt).max())
            voigt = projected

        largest = float(np.abs(voigt).max())
        noise = max(asymmetry, symmetry_residual)
        converged = largest > 0 and noise <= NOISE_WARNING * largest
        if converged or attempt == len(attempts) - 1:
            break
        LOG.info(
            "strain %.3f left %.0f%% noise in the elastic tensor; trying %.3f",
            size,
            100 * noise / largest,
            attempts[attempt + 1],
        )

    if not converged:
        _warn_about_noise(noise, largest, size, fmax)

    from chmpy.ext.elastic_tensor import ElasticTensor

    return ElasticResult(
        tensor=ElasticTensor(voigt),
        evaluations=calculator.stats.calls - started,
        relaxed_ions=relax_ions,
        residual_stress=residual,
        asymmetry=asymmetry,
        symmetry_residual=symmetry_residual,
        n_independent=len(basis),
        strain=size,
        relaxations=relaxations,
    )


def _warn_about_noise(noise, largest, strain, fmax) -> None:
    """Say plainly that a tensor is not converged, and which knob to turn.

    Two estimates come free. `C_ij` and `C_ji` are computed from different
    strain columns and must agree for a conservative model, so the asymmetry of
    the raw tensor is pure noise -- and that one exists for every crystal
    system. The components symmetry forbids are noise too and are the more
    sensitive measure, but a triclinic crystal forbids nothing, so relying on
    them alone leaves the lowest-symmetry structures unchecked, which is where
    a silent wrong answer is least likely to be noticed.

    The usual cause is a strain too small for the stress response to clear the
    calculator's noise. The signal grows as the strain and the noise does not,
    while the error from anharmonicity grows as its square, so there is an
    optimum -- and for a float32 model it sits well above the value a
    double-precision code would use.
    """
    LOG.warning(
        "the elastic tensor carries %.2f GPa of noise, %.0f%% of its largest "
        "constant: it has not converged and should not be used. The strain "
        "(now %.3f) is the usual cause -- the stress response has to clear the "
        "calculator's own noise -- followed by the ionic relaxation (now "
        "fmax=%.3f eV/A, which for a noisy model works as a step budget rather "
        'than a tolerance). `strain="auto"` searches for a workable value.',
        noise,
        100 * noise / largest if largest > 0 else float("nan"),
        strain,
        fmax,
    )


def _strain_sweep(
    system,
    calculator,
    reference_cell,
    reference_scaled,
    strain,
    relax_ions,
    fmax,
    steps,
    logger,
):
    """Stress response to a strain in each Voigt direction.

    Returns the raw `dsigma_i / de_j` in GPa, before symmetrisation, so the
    caller can read the asymmetry as a noise estimate.
    """
    started = calculator.stats.calls
    columns, relaxations, carried = [], [], None
    for index in range(6):
        responses = []
        for sign in (1, -1):
            deformation = np.eye(3) + voigt_strain(index, sign * strain)
            strained = System(
                system.numbers,
                reference_scaled @ (reference_cell @ deformation),
                reference_cell @ deformation,
                True,
                dict(system.info),
            )
            outcome = None
            if relax_ions:
                outcome, carried = _settle(strained, calculator, fmax, steps, carried)
                if outcome is not None:
                    relaxations.append(outcome)
            responses.append(calculator(strained, ("energy", "stress")).stress_voigt)
            if logger is not None:
                settled = (
                    f"{outcome.steps} ion steps"
                    if outcome is not None
                    else "no internal freedoms"
                    if relax_ions
                    else "clamped ions"
                )
                logger(
                    f"  voigt {index} {'+' if sign > 0 else '-'}{strain:g}  {settled}"
                )
        columns.append((responses[0] - responses[1]) / (2.0 * strain))
    raw = np.array(columns).T * EV_PER_ANGSTROM3_TO_GPA
    return raw, relaxations, calculator.stats.calls - started


def _settle(system, calculator, fmax, steps, carried):
    """Relax the atoms of a strained cell, reusing the last force constants.

    The curvature of the internal coordinates barely changes between one strain
    point and the next, so the model learned at the first is a good starting
    point for all twelve. The model object itself is handed on, rather than a
    copy of its matrix, so it keeps accumulating across the whole calculation.
    """
    coordinates = Atomic(system)
    if coordinates.n_dof == 0:
        return None, carried
    reuse = carried if carried is not None and carried.n == coordinates.n_dof else None
    optimiser = TrustRegion(coordinates, calculator, model=reuse)
    outcome = optimiser.run(fmax=fmax, steps=steps)
    if not outcome.converged:
        LOG.warning(
            "the ionic relaxation at one strain point did not converge "
            "(fmax %.4g eV/A after %d steps); the elastic constants will be "
            "too stiff",
            outcome.measures.get("fmax", float("nan")),
            outcome.steps,
        )
    return outcome, outcome.model


def _as_system(structure, info=None) -> System:
    kwargs = dict(info) if info else {}
    if isinstance(structure, System):
        system = structure.copy()
        system.info.update(kwargs)
        return system
    if hasattr(structure, "space_group"):
        return System.from_crystal(structure, **kwargs)
    return System.from_molecule(structure, **kwargs)


def _rotations_for(structure, rotations, symmetry):
    if not symmetry:
        return np.zeros((0, 3, 3))
    if rotations is not None:
        return np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    if hasattr(structure, "space_group"):
        return cartesian_rotations(structure)
    LOG.info(
        "no space group to symmetrise with, so the tensor is left triclinic; "
        "pass `rotations=` if the structure has symmetry"
    )
    return np.zeros((0, 3, 3))

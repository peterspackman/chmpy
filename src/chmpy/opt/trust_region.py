"""A scaled trust-region quasi-Newton optimiser.

At each iteration a quadratic model `m(p) = g.p + p.B.p / 2` is minimised
inside a ball of radius `delta`; the step is taken, and the ratio of the actual
reduction to the predicted one decides whether to keep it and what to do with
the radius. The implementation follows klasp's `trust_region.hpp`, generalised
from rigid bodies to any `Coordinates`.

Four details carry most of the robustness, each of them a failure mode that had
to be fixed rather than a refinement:

* **Scaled coordinates.** Degrees of freedom are measured in Angstroms of
  atomic displacement (`Coordinates.scale`), so one radius and one gradient
  tolerance mean the same thing for an atom, a cell strain and anything added
  later. Unscaled, the radius is set by whichever freedom carries the largest
  units.

* **Powell damping.** Plain BFGS must skip the curvature update whenever
  `y.s <= 0`, which is where the model is worst -- and in a trust region those
  are also the steps that get rejected, so a model that causes a rejection
  never learns from it. Damping blends `y` towards `Bs` by just enough to keep
  the update positive definite, so every step contributes.

* **The radius shrinks only on rejection.** Shrinking whenever `rho < 1/4`
  regardless of acceptance is the textbook rule; measured against this
  objective it over-reacts to a noisy energy and collapses the radius.

* **A floor on the radius, and restarts.** Repeated rejection drives the radius
  geometrically to zero; once a step is too small to change the energy, every
  later trial is rejected and the structure is stuck. On reaching the floor the
  model is reset from the accepted point, and after a few restarts it pins at
  the floor and keeps inching.

A fifth is about the calculator rather than the algorithm: below the smallest
energy difference a calculator can resolve, the reduction ratio carries no
information and is not computed. See `_judge`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .curvature import for_size

LOG = logging.getLogger(__name__)

#: a step clipped below this fraction means the parameterisation is at a bound,
#: and the reference is moved rather than the radius shrunk
REANCHOR_FRACTION = 0.5


@dataclass
class TrustRegionOptions:
    """Tuning for `TrustRegion`. The defaults are klasp's measured ones.

    Attributes:
        eta: smallest reduction ratio that still accepts a step
        rho_max: above this the model is so wrong the step is treated as failed
        delta0: initial and post-restart radius, in Angstroms of displacement
        delta_max: largest radius
        delta_min: floor below which the radius has collapsed
        energy_noise: smallest energy difference worth believing, in eV. None
            asks the calculator.
        max_restarts: how many times to reset the model before pinning at the
            floor
        grow: factor to grow the radius by on a good step at the boundary
        shrink: factor to shrink by on a rejected step
        shrink_hard: factor for a step that was not merely bad but unusable
    """

    eta: float = 0.10
    rho_max: float = 10.0
    #: smallest believable energy difference, eV. None asks the calculator
    #: (`Calculator.energy_noise`), which is almost always what you want.
    energy_noise: float | None = None
    delta0: float = 0.20
    delta_max: float = 1.0
    delta_min: float = 1e-4
    max_restarts: int = 3
    grow: float = 2.0
    shrink: float = 0.5
    shrink_hard: float = 0.25


@dataclass
class Step:
    """One iteration, for the trajectory."""

    step: int
    energy: float
    measures: dict
    radius: float
    rho: float
    accepted: bool
    restarted: bool = False
    reanchored: bool = False


@dataclass
class Relaxation:
    """The outcome of a relaxation.

    Attributes:
        converged: whether every convergence criterion was met
        steps: iterations taken
        energy: final energy in eV
        measures: final convergence measures, e.g. fmax in eV/A and smax in GPa
        evaluations: calculator evaluations used
        history: one `Step` per iteration
        structure: the relaxed structure
        result: the calculator result at the final geometry
        model: the `CurvatureModel` the run ended with, so a later stage can
            start from what this one learned
        stages: for a staged relaxation, the `Relaxation` of each stage
    """

    converged: bool
    steps: int
    energy: float
    measures: dict
    evaluations: int
    history: list = field(default_factory=list)
    structure: object = None
    result: object = None
    stalled: bool = False
    model: object = None
    stages: list = field(default_factory=list)

    @property
    def accepted_steps(self) -> int:
        "How many trial steps were kept"
        return sum(1 for step in self.history if step.accepted)

    def __repr__(self) -> str:
        state = "converged" if self.converged else "NOT converged"
        measures = ", ".join(f"{k}={v:.4g}" for k, v in self.measures.items())
        return (
            f"<Relaxation {state} in {self.steps} steps "
            f"({self.evaluations} evaluations), E={self.energy:.6f} eV, {measures}>"
        )

    def __str__(self) -> str:
        lines = [repr(self), f"  {self.accepted_steps} of {self.steps} steps accepted"]
        if len(self.stages) > 1:
            lines.extend(f"    {stage!r}" for stage in self.stages)
        return "\n".join(lines)


class TrustRegion:
    """Relax a structure by a scaled dogleg trust region with damped BFGS.

    Args:
        coordinates: the degrees of freedom to vary
        calculator: what to evaluate energies with
        options: tuning, or None for the defaults
        pressure: external hydrostatic pressure in GPa, applied when the
            coordinates include a cell
        hessian: (n_dof, n_dof) starting model of the curvature, in scaled
            coordinates. The default is the identity, which is a reasonable
            model precisely because the coordinates are scaled to Angstroms.
            Passing a better one -- an analytic or previously converged
            Hessian -- is the single biggest saving available on a hard system.
        model: the curvature model, or None to build one. Anything with
            `matvec`, `solve`, `update`, `reset` and `rescale` will do; see
            `chmpy.opt.curvature`.
    """

    def __init__(
        self,
        coordinates,
        calculator,
        options=None,
        pressure: float = 0.0,
        hessian=None,
        model=None,
    ):
        self.coordinates = coordinates
        self.calculator = calculator
        self.options = options or TrustRegionOptions()
        self.pressure = float(pressure)
        self.hessian = None if hessian is None else np.array(hessian, dtype=float)
        self.model = model

    def run(
        self,
        fmax: float = 0.01,
        smax: float = 0.05,
        steps: int = 200,
        callback=None,
        logger=None,
    ) -> Relaxation:
        """Relax until converged or out of steps.

        Args:
            fmax: force convergence in eV/A, on the largest component
            smax: stress convergence in GPa, on the largest component. Ignored
                when the coordinates hold no cell degrees of freedom.
            steps: maximum iterations
            callback: called with each `Step`
            logger: a callable given a one-line summary of each step, e.g.
                `print`

        Returns:
            Relaxation
        """
        options = self.options
        coordinates = self.coordinates
        tolerances = {"fmax": fmax, "smax": smax}
        started = self.calculator.stats.calls

        x = coordinates.get()
        scale = coordinates.scale()
        result = self._evaluate(x)
        energy = self._energy(result)
        gradient = self._gradient(result, scale)

        n = coordinates.n_dof
        model = self.model
        if model is None:
            model = for_size(n, initial=self.hessian)
        if model.n != n:
            raise ValueError(
                f"the curvature model has {model.n} degrees of freedom, but "
                f"the coordinates have {n}"
            )
        LOG.debug("relaxing %d degrees of freedom with %r", n, model)
        radius = options.delta0
        restarts = 0
        stalled = False
        history = []
        converged = self._converged(result, tolerances)

        for iteration in range(1, steps + 1):
            if converged:
                break

            step, predicted = _dogleg(model, gradient, radius)

            reanchored = False
            fraction = coordinates.step_fraction(x, step / scale)
            if fraction < REANCHOR_FRACTION:
                # The step has run into a bound on the parameterisation. Move
                # the reference here -- the geometry does not change, so the
                # result and its gradient still stand -- and ask again. Waiting
                # for an accepted step to do this would wait for ever: at a
                # bound the step is clipped to nothing, so nothing is ever
                # accepted and the radius shrinks to the floor instead.
                anchored = coordinates.reanchor(x, force=True)
                if anchored is not None:
                    reanchored = True
                    x = anchored
                    scale, previous = coordinates.scale(), scale
                    model.rescale(previous, scale)
                    gradient = self._gradient(result, scale)
                    step, predicted = _dogleg(model, gradient, radius)
                    fraction = coordinates.step_fraction(x, step / scale)
            if fraction < 1.0:
                step = step * fraction
                predicted = model.quadratic(gradient, step)

            if predicted <= 0.0 and np.linalg.norm(step) <= 1e-14:
                # nothing left to try: the model proposes no move at all
                stalled = True
                break

            trial_x = x + step / scale
            trial_result = self._evaluate(trial_x)
            trial_energy = self._energy(trial_result)
            trial_gradient = self._gradient(trial_result, scale)

            noise = (
                self.calculator.energy_noise(energy)
                if options.energy_noise is None
                else options.energy_noise
            )
            rho, accept, unusable = _judge(
                trial_energy, energy, predicted, options.eta, options.rho_max, noise
            )

            # Both branches learn from the step: y = g(x + p) - g(x) is valid
            # curvature whether or not the point is kept, and throwing away the
            # rejected ones is what makes a bad model self-perpetuating.
            if np.isfinite(trial_energy):
                model.update(step, trial_gradient - gradient)

            restarted = False
            if accept:
                x, energy, gradient, result = (
                    trial_x,
                    trial_energy,
                    trial_gradient,
                    trial_result,
                )
                if rho > 0.75 and np.linalg.norm(step) > 0.8 * radius:
                    radius = min(options.grow * radius, options.delta_max)
                anchored = coordinates.reanchor(x)
                if anchored is not None:
                    # the geometry has not changed, only its parameterisation,
                    # so the result still stands and the gradient is re-derived
                    reanchored = True
                    x = anchored
                    scale, previous = coordinates.scale(), scale
                    model.rescale(previous, scale)
                    gradient = self._gradient(result, scale)
                converged = self._converged(result, tolerances)
            else:
                coordinates.set(x)
                radius, restart = _shrink(radius, unusable, restarts, options)
                if restart:
                    restarts += 1
                    restarted = True
                    model.reset()

            entry = Step(
                iteration,
                energy,
                coordinates.measures(result),
                radius,
                rho,
                accept,
                restarted,
                reanchored,
            )
            history.append(entry)
            if logger is not None:
                logger(_format(entry))
            if callback is not None:
                callback(entry)

        coordinates.set(x)
        if stalled:
            LOG.warning(
                "the optimiser stalled: the model proposes no step at all, which "
                "usually means the parameterisation has run into a bound it "
                "cannot be re-anchored out of"
            )
        return Relaxation(
            converged=converged,
            steps=len(history),
            energy=energy,
            measures=coordinates.measures(result),
            evaluations=self.calculator.stats.calls - started,
            history=history,
            structure=coordinates.system,
            result=result,
            stalled=stalled,
            model=model,
        )

    # -- internals -----------------------------------------------------------

    def _evaluate(self, x):
        self.coordinates.set(x)
        return self.calculator(self.coordinates.system, self.coordinates.wanted)

    def _energy(self, result) -> float:
        """The objective: the enthalpy when a pressure is applied."""
        if self.pressure:
            from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

            return (
                result.energy
                + (self.pressure / EV_PER_ANGSTROM3_TO_GPA)
                * self.coordinates.system.volume
            )
        return result.energy

    def _gradient(self, result, scale) -> np.ndarray:
        gradient = self.coordinates.gradient(result)
        if self.pressure:
            from .coordinates import pressure_term

            gradient = gradient + pressure_term(self.coordinates, self.pressure)
        return gradient / scale

    def _converged(self, result, tolerances) -> bool:
        measures = self.coordinates.measures(result)
        return all(
            value <= tolerances[name]
            for name, value in measures.items()
            if name in tolerances
        )


def _format(step: Step) -> str:
    measures = " ".join(f"{k}={v:9.5f}" for k, v in step.measures.items())
    flags = "".join(
        (
            "+" if step.accepted else "-",
            "R" if step.restarted else " ",
            "A" if step.reanchored else " ",
        )
    )
    return (
        f"{step.step:4d} {flags}  E={step.energy:16.8f}  {measures}  "
        f"delta={step.radius:8.2e}  rho={step.rho:7.3f}"
    )


# -- the numerics -------------------------------------------------------------


def _dogleg(model, gradient, radius):
    """Minimise the quadratic model inside `||p|| <= radius`, approximately.

    Powell's dogleg: take the Newton step if it fits, otherwise interpolate
    between the steepest-descent minimiser and the Newton step out to the
    boundary. Falls back to steepest descent to the boundary when the model
    cannot be inverted, which is where the Newton step would be nonsense.

    Written against `CurvatureModel`, so the same step works for the dense
    model and the limited-memory one -- the latter supplies `B^-1 g` from the
    two-loop recursion and `B v` from the compact representation, both in
    `O(n m)`.

    Returns:
        (step, predicted reduction)
    """
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= 1e-30:
        return np.zeros_like(gradient), 0.0

    newton = None
    try:
        newton = model.solve(-gradient)
        if np.linalg.norm(newton) <= radius:
            return newton, model.quadratic(gradient, newton)
    except np.linalg.LinAlgError:
        pass

    curvature = float(gradient @ model.matvec(gradient))
    to_boundary = -(radius / gradient_norm) * gradient
    if curvature <= 0 or newton is None:
        return to_boundary, model.quadratic(gradient, to_boundary)

    # the steepest-descent minimiser along -g
    cauchy = -(gradient @ gradient) / curvature * gradient
    if np.linalg.norm(cauchy) >= radius:
        return to_boundary, model.quadratic(gradient, to_boundary)

    direction = newton - cauchy
    a = float(direction @ direction)
    b = 2.0 * float(cauchy @ direction)
    c = float(cauchy @ cauchy) - radius * radius
    discriminant = b * b - 4 * a * c
    tau = (-b + np.sqrt(discriminant)) / (2 * a) if a > 0 and discriminant > 0 else 1.0
    step = cauchy + np.clip(tau, 0.0, 1.0) * direction
    return step, model.quadratic(gradient, step)


def _judge(trial_energy, energy, predicted, eta, rho_max, noise=0.0):
    """Accept or reject a trial step from the model-versus-actual ratio.

    The measured reduction is credited with one unit of `noise`, which is the
    smallest energy difference the calculator can resolve. Without it, a step
    whose true reduction is below that resolution measures as exactly zero, so
    `rho` is zero, the step is rejected, and the radius shrinks -- repeatedly,
    until the optimiser gives up with the structure still moving. This is not
    a hypothetical: a model running in float32 cannot resolve 1.5e-5 eV at
    -190 eV, and the steps near a minimum are smaller than that long before the
    forces are converged.

    Crediting the noise makes the test read "did the energy fall, as far as we
    can tell?" rather than "did it fall by a number we cannot measure?". When
    the predicted reduction is far above the noise the correction is
    negligible, so nothing changes where the ratio test was working.

    Below the noise the ratio is abandoned rather than credited. Dividing a
    quantity that is all noise by a prediction smaller still gives a huge
    number, which the `rho_max` test reads as a catastrophically wrong model
    and answers with a hard shrink -- measured stalling a quadratic at
    `fmax = 9e-9` for a thousand iterations, at a minimum it was a few steps
    from.
    """
    finite = np.isfinite(trial_energy)
    has_prediction = predicted > 1e-30
    if not (finite and has_prediction):
        return -1.0, False, not finite
    if predicted <= noise:
        # The model predicts a change smaller than the calculator can measure.
        # The ratio is then meaningless in both directions, and computing it
        # anyway produces an enormous number that reads as "the model is wildly
        # wrong" -- which shrinks the radius hard and stalls the optimiser at a
        # point it could have walked away from. Take the step if the energy did
        # not visibly rise; there is nothing else to go on.
        accepted = trial_energy <= energy + noise
        return (1.0 if accepted else -1.0), accepted, False
    rho = (energy - trial_energy + noise) / predicted
    unusable = rho > rho_max
    return rho, bool(not unusable and rho > eta), unusable


def _shrink(radius, unusable, restarts, options):
    """What to do with the radius after a rejected step."""
    shrunk = radius * (options.shrink_hard if unusable else options.shrink)
    if shrunk >= options.delta_min:
        return shrunk, False
    if restarts < options.max_restarts:
        return options.delta0, True
    return options.delta_min, False

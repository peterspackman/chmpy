"""Relaxing a structure to a local minimum.

    from chmpy import Crystal
    from chmpy.calc import Calculator
    from chmpy.opt import relax

    crystal = Crystal.load("structure.cif")
    result = relax(crystal, calculator, fmax=0.01, smax=0.05)
    result.structure.save("relaxed.cif")

`relax` picks the parameterisation: a `Crystal` varies its asymmetric unit and
the strains its space group allows, a `Molecule` varies its atoms, and a
`System` varies atoms and cell if it has one. Anything more particular is a
`Coordinates` object and a `TrustRegion` built by hand.

It runs a sequence of `Stage`s, and by default that sequence is one. The
familiar protocol of settling the internal coordinates at fixed cell before
relaxing everything together is available as `stages="two-stage"` but is not
the default, because neither argument for it holds here. The first is that the
stress is expensive and should be deferred: for a model that differentiates its
energy it comes out of the same backward pass as the forces, measured at under
two percent of the call. The second is that the cell should not chase a
geometry that is still moving, which describes an optimiser that cannot tell
the two kinds of freedom apart; these are scaled to a common unit and the trust
region handles them together. Measured on a molecular crystal, splitting the
relaxation costs more calls than it saves.

A split still earns its keep when the stages differ in something other than
which freedoms move -- a cheap potential for the first pass and an expensive
one for the second -- and the `Stage` list is the right shape for that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from chmpy.calc.system import System

from .coordinates import Atomic, AtomicStrain, strain_basis_for
from .symmetry import SymmetryAdapted
from .trust_region import Relaxation, TrustRegion

#: how much looser the fixed-cell stage is than the final one. Tight enough to
#: be worth doing, loose enough not to converge internal coordinates against a
#: cell that is about to change.
PRERELAX_FACTOR = 5.0


@dataclass
class Stage:
    """One step of a relaxation protocol.

    Attributes:
        cell: vary the cell during this stage
        fmax: force convergence for this stage, eV/A
        smax: stress convergence for this stage, GPa; ignored at fixed cell
        steps: maximum iterations for this stage
        hessian: "model" builds a stretch model from the geometry, "identity"
            starts blind, "carry" continues from the previous stage's learned
            curvature (falling back to "model" for the first stage), or pass an
            array
        name: what to call it in the log
    """

    cell: bool = True
    fmax: float = 0.01
    smax: float = 0.05
    steps: int = 200
    hessian: object = "model"
    name: str = ""


def two_stage(fmax: float = 0.01, smax: float = 0.05, steps: int = 200) -> list[Stage]:
    """Settle the atoms at fixed cell, then relax the cell with them.

    Args:
        fmax: final force convergence, eV/A
        smax: final stress convergence, GPa
        steps: maximum iterations per stage

    Returns:
        two `Stage`s
    """
    return [
        Stage(
            cell=False,
            fmax=PRERELAX_FACTOR * fmax,
            steps=steps,
            hessian="model",
            name="fixed cell",
        ),
        Stage(
            cell=True,
            fmax=fmax,
            smax=smax,
            steps=steps,
            hessian="carry",
            name="variable cell",
        ),
    ]


def coordinates_for(
    structure, cell: bool = True, symmetry: bool = True, fixed=None, info=None
):
    """The degrees of freedom to relax a structure with.

    Args:
        structure: a `Crystal`, `Molecule` or `System`
        cell: vary the cell. Ignored for anything not periodic.
        symmetry: for a `Crystal`, vary the asymmetric unit and the strains its
            space group allows. With this off the crystal is relaxed in P1 and
            its symmetry is free to break.
        fixed: boolean mask of atoms to hold still -- over the asymmetric unit
            when symmetry is on, over the unit cell otherwise
        info: model inputs that are not geometry, e.g. a charge or a spin,
            carried on the `System` the calculator sees

    Returns:
        a `Coordinates`
    """
    if symmetry and hasattr(structure, "space_group"):
        return SymmetryAdapted(structure, cell=cell, fixed=fixed, info=info)

    system = _as_system(structure, info)
    if not system.periodic or not cell:
        return Atomic(system, fixed=fixed)
    basis = strain_basis_for(structure, symmetry=symmetry)
    return AtomicStrain(system, basis=basis, fixed=fixed)


def relax(
    structure,
    calculator,
    *,
    cell: bool = True,
    symmetry: bool = True,
    fixed=None,
    info=None,
    fmax: float = 0.01,
    smax: float = 0.05,
    steps: int = 200,
    pressure: float = 0.0,
    options=None,
    hessian="model",
    stages=None,
    logger=None,
    callback=None,
) -> Relaxation:
    """Relax a structure to a local minimum.

    Args:
        structure: a `Crystal`, `Molecule` or `System`
        calculator: a `chmpy.calc.Calculator`
        cell: vary the cell as well as the atoms
        symmetry: keep a crystal's space group, exactly, by making the
            asymmetric unit the degrees of freedom
        fixed: boolean mask of atoms to hold still
        info: model inputs that are not geometry -- a charge, a spin, an
            external field -- carried on the `System` the calculator sees.
            Models that need them fail without them.
        fmax: force convergence in eV/A
        smax: stress convergence in GPa
        steps: maximum iterations per stage
        pressure: external hydrostatic pressure in GPa, so the objective is the
            enthalpy `E + PV` rather than the energy
        options: `TrustRegionOptions`, or None for the defaults
        hessian: the starting curvature model for a single-stage run: "model",
            "identity", or an array
        stages: a list of `Stage`, or "two-stage" for the fixed-cell-then-
            variable-cell protocol. The default is a single stage, which was
            measured to be as good or better -- see the module docstring.
        logger: called with a one-line summary of each step, e.g. `print`
        callback: called with each `Step`

    Returns:
        a `Relaxation`, whose `structure` is of the same kind as the input and
        whose `stages` holds one `Relaxation` per stage that ran -- one for the
        default protocol, and none for stages skipped as having no freedoms
    """
    _check_hessian(hessian)
    plan = _plan(stages, structure, cell, fmax, smax, steps, hessian)

    current = structure
    carried = None
    finished = []
    for stage in plan:
        coordinates = coordinates_for(
            current, cell=stage.cell, symmetry=symmetry, fixed=fixed, info=info
        )
        if coordinates.n_dof == 0:
            # nothing to vary -- a fixed-cell stage for a structure whose atoms
            # are all on special positions, say. Evaluating it would cost a
            # call to say so.
            continue
        if logger is not None and len(plan) > 1:
            logger(f"--- {stage.name or 'stage'}: {coordinates!r}")
        optimiser = TrustRegion(
            coordinates,
            calculator,
            options=options,
            pressure=pressure,
            hessian=_starting_hessian(stage.hessian, coordinates, carried),
        )
        outcome = optimiser.run(
            fmax=stage.fmax,
            smax=stage.smax,
            steps=stage.steps,
            logger=logger,
            callback=callback,
        )
        outcome.structure = _rebuild(current, coordinates)
        current = outcome.structure
        carried = outcome.model
        finished.append(outcome)

    if not finished:
        raise ValueError(
            "this structure has no degrees of freedom to relax: every atom is "
            "fixed or on a special position, and the cell is not being varied"
        )
    return _combine(finished)


# -- the protocol -------------------------------------------------------------


#: the ways of asking for a starting curvature model
HESSIAN_CHOICES = ("model", "identity", "carry")


def _check_hessian(hessian) -> None:
    """Refuse an unknown `hessian` up front.

    Worth doing eagerly: the default plan is two stages with their own hessian
    settings, so a misspelled top-level one would otherwise be ignored in
    silence rather than corrected.
    """
    if isinstance(hessian, str) and hessian not in HESSIAN_CHOICES:
        raise ValueError(
            f"hessian must be one of {HESSIAN_CHOICES} or an array, not {hessian!r}"
        )


def _plan(stages, structure, cell, fmax, smax, steps, hessian) -> list[Stage]:
    """Turn the `stages` argument into a list of `Stage`."""
    single = [
        Stage(cell=cell, fmax=fmax, smax=smax, steps=steps, hessian=hessian, name="")
    ]
    if isinstance(stages, str):
        if stages == "single":
            return single
        if stages == "two-stage":
            return two_stage(fmax=fmax, smax=smax, steps=steps)
        raise ValueError(
            f"stages must be 'two-stage', 'single' or a list of Stage, not {stages!r}"
        )
    if stages is not None:
        return [replace(stage) for stage in stages]
    return single


def _starting_hessian(hessian, coordinates, carried):
    """Resolve a stage's `hessian` into an array or None."""
    if isinstance(hessian, str) and hessian == "carry":
        if carried is None:
            hessian = "model"
        else:
            return _extend_hessian(carried, coordinates)
    if hessian is None or (isinstance(hessian, str) and hessian == "identity"):
        return None
    if isinstance(hessian, str):
        _check_hessian(hessian)
        from .hessian import stretch_hessian

        return stretch_hessian(coordinates)
    return np.asarray(hessian, dtype=float)


def _extend_hessian(previous, coordinates) -> np.ndarray:
    """Carry a stage's learned curvature into a stage with more freedoms.

    Every parameterisation here puts its atomic degrees of freedom first and
    its cell ones last, and the atomic ones mean the same thing in both stages
    -- the cell has not moved yet, so their scaling is unchanged. So the block
    the previous stage learned is dropped in as it stands, and the model fills
    the cell block it knows nothing about.

    A limited-memory model is materialised for this, which costs `O(n^2 m)` --
    affordable precisely because a stage small enough to be worth carrying is
    small.
    """
    from .curvature import LimitedMemoryBFGS
    from .hessian import stretch_hessian

    hessian = stretch_hessian(coordinates)
    if not isinstance(previous, LimitedMemoryBFGS):
        return hessian
    learned = previous.to_dense()
    if learned.shape[0] > coordinates.n_dof:
        # fewer freedoms than before: nothing sensible to carry
        return hessian
    shared = learned.shape[0]
    hessian[:shared, :shared] = learned
    return hessian


def _combine(stages: list[Relaxation]) -> Relaxation:
    """One `Relaxation` describing a whole protocol."""
    last = stages[-1]
    return Relaxation(
        converged=last.converged,
        steps=sum(stage.steps for stage in stages),
        energy=last.energy,
        measures=last.measures,
        evaluations=sum(stage.evaluations for stage in stages),
        history=[step for stage in stages for step in stage.history],
        structure=last.structure,
        result=last.result,
        stalled=any(stage.stalled for stage in stages),
        model=last.model,
        stages=stages,
    )


# -- structures ---------------------------------------------------------------


def _as_system(structure, info=None) -> System:
    if isinstance(structure, System):
        if info:
            structure.info.update(info)
        return structure
    kwargs = dict(info) if info else {}
    if hasattr(structure, "space_group"):
        return System.from_crystal(structure, **kwargs)
    return System.from_molecule(structure, **kwargs)


def _rebuild(original, coordinates):
    """Give back the same kind of object that was passed in."""
    if isinstance(coordinates, SymmetryAdapted):
        return coordinates.to_crystal()
    system = coordinates.system
    if isinstance(original, System):
        return system
    if hasattr(original, "space_group"):
        # relaxed without symmetry: every atom of the cell moved independently,
        # so P1 is the only description certainly still true of the result
        return system.to_crystal(1)
    return system.to_molecule()

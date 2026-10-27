"""Elastic constants by finite strain.

`C_ij = (1/V) d2E / de_i de_j` is computed by straining the cell in each
direction, relaxing the atoms at fixed cell, and differencing the stress.

* **Relaxed atoms.** `relax_atoms=True` (the default) relaxes the atoms at
  every strained geometry. Clamped-atom constants can be much stiffer,
  particularly for molecular crystals.
* **Strain selection.** Since `C (R e R^T) = R (C e) R^T`, one strain gives
  the response for its whole point-group orbit. Only strains not already
  covered are applied (2 for cubic, 3 for hexagonal, 6 for triclinic), and the
  tensor is a least-squares fit over all rotated responses.
* **Subgroup relaxation.** A strain keeps the operations with
  `R eps R^T = eps`, and the atoms relax within that subgroup
  (`chmpy.opt.symmetry.InvariantAtomic`). This cannot find a lower-symmetry
  minimum if the reference is a saddle point in P1; `tensor.is_stable()` will
  usually flag that, and `symmetry=False` relaxes in P1.
* **Projection.** The result is projected onto the tensors the point group
  allows. `symmetry_residual` reports the larger of the fit misfit and the
  size of that projection, and `asymmetry` the largest `|C_ij - C_ji|`;
  both are noise estimates.

`method="hessian"` gets the same relaxed-ion tensor without relaxing anything:
`C = C0 - L^T Phi^+ L / V` from the clamped-ion constants, the internal-strain
tensor and the Gamma-point Hessian, every piece one central difference and the
lot one batch of evaluations. See `hessian_elastic_tensor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA
from chmpy.calc.system import System

from .coordinates import Atomic
from .progress import reporter
from .strain import (
    cartesian_rotations,
    elastic_from_voigt,
    elastic_to_voigt,
    invariant_elastic_basis,
    project_elastic,
    to_vector,
)
from .symmetry import InvariantAtomic, crystal_atom_map
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

#: strain for the clamped-ion part of method="hessian"; nothing relaxes, so
#: the response is linear and a small strain is safe
HESSIAN_STRAIN = 0.005

#: atomic displacement for the Gamma-point Hessian of method="hessian", A
DEFAULT_DISPLACEMENT = 0.01

#: above this, the reference structure is not at equilibrium and "the elastic
#: constants" is no longer a well-defined thing to ask for, GPa
RESIDUAL_STRESS_WARNING = 0.5


@dataclass
class ElasticResult:
    """An elastic tensor and how much to believe it.

    Attributes:
        tensor: the `chmpy.ext.elastic_tensor.ElasticTensor`, in GPa
        evaluations: calculator evaluations used
        relaxed_atoms: whether the atoms were relaxed at each strain
        residual_stress: largest stress component of the reference structure in
            GPa. Elastic constants are defined at equilibrium; a large value
            here means the answer is not the quantity it is called.
        asymmetry: largest `|C_ij - C_ji|` before the tensor was symmetrised,
            in GPa. The two are computed from different strain columns and must
            agree for a conservative model, so their difference is noise. This
            estimate exists for every crystal system.
        symmetry_residual: how far the measured stresses are from what the
            point group allows, in GPa (fit misfit or projection size, whichever
            is larger). Always zero for triclinic.
        n_independent: independent elastic constants the point group allows
        strain: the engineering strain used
        strains: Voigt indices actually applied
        method: "strain" (finite strain, atoms relaxed at each) or "hessian"
            (clamped-ion constants corrected through the internal-strain
            tensor and the Gamma-point Hessian)
        internal_strain_noise: for "hessian", the disagreement between the
            internal-strain tensor measured from strained cells and from
            displaced ones, propagated into the constants, in GPa
        clamped: for "hessian", the clamped-ion tensor C0 in GPa
    """

    tensor: object
    evaluations: int
    relaxed_atoms: bool
    residual_stress: float
    asymmetry: float
    symmetry_residual: float
    n_independent: int
    strain: float
    relaxations: list = field(default_factory=list)
    strains: tuple = (0, 1, 2, 3, 4, 5)
    method: str = "strain"
    internal_strain_noise: float | None = None
    clamped: np.ndarray | None = None

    @property
    def c_voigt(self) -> np.ndarray:
        "(6, 6) elastic constants in GPa"
        return self.tensor.c_voigt

    @property
    def noise(self) -> float:
        "The largest of the free error estimates, in GPa"
        return max(
            self.asymmetry, self.symmetry_residual, self.internal_strain_noise or 0.0
        )

    @property
    def noise_fraction(self) -> float:
        """Estimated noise as a fraction of the largest constant.

        Two measurements come free with the calculation, and the larger is
        taken. Above a few percent the tensor is not converged: the strain was
        too small for the stress response to clear the calculator's noise, or
        the atomic relaxation was not run hard enough, or both.
        """
        largest = float(np.abs(self.c_voigt).max())
        return self.noise / largest if largest > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"<ElasticResult {self.method}, {self.n_independent} independent "
            f"constants, {len(self.strains)} strains, {self.evaluations} "
            f"evaluations, symmetry residual {self.symmetry_residual:.3g} GPa>"
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
    relax_atoms: bool = True,
    symmetry: bool = True,
    rotations=None,
    info=None,
    fmax: float = 0.01,
    steps: int = 200,
    progress=None,
    method: str = "strain",
    displacement: float = DEFAULT_DISPLACEMENT,
    force_constants=None,
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
        relax_atoms: relax the atoms at each strained cell. Leave this on unless
            you specifically want clamped-atom constants.
        symmetry: use the point group to choose the strains, relax the atoms
            within the subgroup each strain preserves, and project the result
            onto the tensors the point group allows. When False, all six
            strains are applied and the atoms relax in P1.
        rotations: (M, 3, 3) Cartesian point operations. Taken from a
            `Crystal`'s space group by default. With a `System` they are used
            for strain selection and projection only; the atoms relax in P1.
        info: model inputs that are not geometry, carried on every `System`
            the calculator sees
        fmax: force convergence for the atomic relaxations, eV/A
        steps: iteration cap for each atomic relaxation
        progress: True to print progress, or a callable given a
            `chmpy.opt.progress.Progress` per event. Each strain point is a
            stage; the atomic relaxation at it is nested inside.
        method: "strain" relaxes the atoms at each strained cell. "hessian"
            relaxes none: it takes the clamped-ion constants from strained
            cells and corrects them with the internal-strain tensor and the
            Gamma-point Hessian, `C = C0 - L^T Phi^+ L / V`, all from one batch
            of finite differences. Much cheaper, and exact to second order;
            see `hessian_elastic_tensor`.
        displacement: atomic finite-difference step for method="hessian", A
        force_constants: for method="hessian", `ForceConstants` already
            measured for this structure (any supercell); only the strained
            cells are then computed

    Returns:
        ElasticResult
    """
    if method == "hessian":
        return hessian_elastic_tensor(
            structure,
            calculator,
            strain=HESSIAN_STRAIN if strain == "auto" else strain,
            displacement=displacement,
            symmetry=symmetry,
            rotations=rotations,
            info=info,
            force_constants=force_constants,
            progress=progress,
        )
    if method != "strain":
        raise ValueError(f"unknown method {method!r}, expected 'strain' or 'hessian'")
    system = _as_system(structure, info)
    if not system.periodic:
        raise ValueError("elastic constants need a periodic structure")

    rotations = _rotations_for(structure, rotations, symmetry)
    basis = invariant_elastic_basis(rotations)
    atom_map = (
        crystal_atom_map(structure, system)
        if symmetry and hasattr(structure, "space_group")
        else None
    )
    strains = independent_strains(rotations)

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

    report = reporter(progress)
    report(
        "elastic",
        "setup",
        f"elastic tensor: {len(strains)} of 6 strains needed "
        f"(Voigt {', '.join(str(index) for index in strains)}), "
        f"{len(basis)} independent constants",
    )

    attempts = AUTO_STRAINS if strain == "auto" else (float(strain),)
    for attempt, size in enumerate(attempts):
        responses, relaxations = _strain_sweep(
            system,
            calculator,
            reference_cell,
            reference_scaled,
            strains,
            size,
            relax_atoms,
            fmax,
            steps,
            report,
            atom_map,
        )
        raw, misfit = fit_elastic(responses, rotations)
        asymmetry = float(np.abs(raw - raw.T).max())
        voigt = 0.5 * (raw + raw.T)

        symmetry_residual = misfit
        if symmetry:
            projected = elastic_to_voigt(
                project_elastic(elastic_from_voigt(voigt), basis)
            )
            symmetry_residual = max(misfit, float(np.abs(projected - voigt).max()))
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
        report(
            "elastic",
            "auto strain",
            f"strain {size:g} left {100 * noise / largest:.0f}% noise; "
            f"trying {attempts[attempt + 1]:g}",
        )

    if not converged:
        _warn_about_noise(noise, largest, size, fmax)

    from chmpy.ext.elastic_tensor import ElasticTensor

    return ElasticResult(
        tensor=ElasticTensor(voigt),
        evaluations=calculator.stats.calls - started,
        relaxed_atoms=relax_atoms,
        residual_stress=residual,
        asymmetry=asymmetry,
        symmetry_residual=symmetry_residual,
        n_independent=len(basis),
        strain=size,
        relaxations=relaxations,
        strains=tuple(strains),
    )


def hessian_elastic_tensor(
    structure,
    calculator,
    *,
    strain: float = HESSIAN_STRAIN,
    displacement: float = DEFAULT_DISPLACEMENT,
    symmetry: bool = True,
    rotations=None,
    info=None,
    force_constants=None,
    progress=None,
) -> ElasticResult:
    """Relaxed-ion elastic constants from second derivatives, without relaxing.

    Expanding the energy to second order in the strain `e` and the internal
    displacements `u` and minimising over `u` gives

        C = C0 - (1/V) L^T Phi^+ L

    with `C0` the clamped-ion constants (stress response at fixed fractional
    coordinates), `Phi` the Gamma-point Hessian of the cell, and
    `L = d2E/du de` the internal-strain tensor. Every piece is a central
    difference, so the whole calculation is one batch of independent
    evaluations: the symmetry-unique displacements of the unit cell for
    `Phi`, and twelve strained cells for `C0` and `L`.

    `L` is measured twice -- as `-dF/de` on every atom from the strained
    cells, and as `d(virial)/du` from the displaced ones, completed by
    symmetry -- and the two have to agree. Their average is used, and half
    the difference between the tensors each gives is `internal_strain_noise`
    on the result. The three uniform translations are projected out of `Phi`
    before it is inverted, which is where the acoustic sum rule comes in.

    Args:
        structure: a relaxed `Crystal` or `System`
        calculator: a `chmpy.calc.Calculator`; every evaluation goes through
            one `calculator.batch` call, which falls back to evaluating one
            at a time for calculators without a native batch path
        strain: engineering strain for `C0` and `L`
        displacement: atomic step for `Phi`, Angstroms
        symmetry: displace only symmetry-unique atoms, fold `L` onto the
            space group, and project `C` onto the point group
        rotations: (M, 3, 3) Cartesian point operations, as for
            `elastic_tensor`
        info: model inputs that are not geometry
        force_constants: `ForceConstants` already measured for this
            structure; their Gamma-point Hessian is used and only the strained
            cells are computed
        progress: as for `elastic_tensor`

    Returns:
        ElasticResult with method="hessian"
    """
    from chmpy.ext.elastic_tensor import ElasticTensor
    from chmpy.vib.force_constants import DisplacementPlan

    system = _as_system(structure, info)
    if not system.periodic:
        raise ValueError("elastic constants need a periodic structure")
    n = len(system)
    volume = system.volume
    rotations = _rotations_for(structure, rotations, symmetry)
    basis = invariant_elastic_basis(rotations)
    has_group = symmetry and hasattr(structure, "space_group")
    started = calculator.stats.calls

    reference = calculator(system, ("energy", "stress"))
    residual = reference.smax
    if residual > RESIDUAL_STRESS_WARNING:
        LOG.warning(
            "the reference structure carries %.3f GPa of stress; elastic "
            "constants are defined at equilibrium, so relax it first",
            residual,
        )

    cell, scaled = np.array(system.cell), system.scaled_positions
    strained = []
    for index in range(6):
        for sign in (1, -1):
            deformation = np.eye(3) + voigt_strain(index, sign * strain)
            strained.append(
                System(
                    system.numbers,
                    scaled @ (cell @ deformation),
                    cell @ deformation,
                    True,
                    dict(system.info),
                )
            )
    plan = None
    displaced = []
    if force_constants is None:
        plan = DisplacementPlan(structure, system, (1, 1, 1), symmetry, displacement)
        displaced = plan.systems()

    report = reporter(progress)
    report(
        "elastic",
        "setup",
        f"elastic tensor from the Hessian: {len(displaced)} displaced and "
        f"{len(strained)} strained cells in one batch, "
        f"{len(basis)} independent constants",
    )
    results = calculator.batch(displaced + strained, ("energy", "forces", "stress"))
    moved, deformed = results[: len(displaced)], results[len(displaced) :]

    if plan is not None:
        force_constants = plan.solve([result.forces for result in moved])
    hessian = force_constants.gamma_hessian()
    if hessian.shape != (3 * n, 3 * n):
        raise ValueError(
            f"the force constants are for {hessian.shape[0] // 3} atoms, the "
            f"structure has {n}"
        )

    # clamped-ion constants and the internal-strain tensor, one strain column
    # at a time
    clamped = np.zeros((6, 6))
    internal = np.zeros((3 * n, 6))
    for index in range(6):
        plus, minus = deformed[2 * index], deformed[2 * index + 1]
        clamped[:, index] = (plus.stress_voigt - minus.stress_voigt) / (2 * strain)
        internal[:, index] = -(plus.forces - minus.forces).ravel() / (2 * strain)
    if has_group:
        from .symmetry import crystal_atom_map

        internal = _symmetrise_internal_strain(
            internal, crystal_atom_map(structure, system), cell
        )

    # the same tensor from the other side: d(virial)/du along each
    # displacement made, completed by symmetry. The two estimates of C differ
    # only through L, so half their difference is what L's noise costs.
    correction = _relaxation_correction(hessian, internal, n) / volume
    internal_noise = None
    if plan is not None:
        from_displacements = _internal_strain_from_displacements(plan, moved, volume)
        if has_group:
            from_displacements = _symmetrise_internal_strain(
                from_displacements, plan.atom_map, cell
            )
        other = _relaxation_correction(hessian, from_displacements, n) / volume
        internal_noise = (
            0.5 * float(np.abs(other - correction).max()) * EV_PER_ANGSTROM3_TO_GPA
        )
        correction = (
            _relaxation_correction(hessian, 0.5 * (internal + from_displacements), n)
            / volume
        )

    raw = (clamped - correction) * EV_PER_ANGSTROM3_TO_GPA
    asymmetry = float(np.abs(raw - raw.T).max())
    voigt = 0.5 * (raw + raw.T)
    symmetry_residual = 0.0
    if symmetry:
        projected = elastic_to_voigt(project_elastic(elastic_from_voigt(voigt), basis))
        symmetry_residual = float(np.abs(projected - voigt).max())
        voigt = projected

    result = ElasticResult(
        tensor=ElasticTensor(voigt),
        evaluations=calculator.stats.calls - started,
        relaxed_atoms=True,
        residual_stress=residual,
        asymmetry=asymmetry,
        symmetry_residual=symmetry_residual,
        n_independent=len(basis),
        strain=strain,
        method="hessian",
        internal_strain_noise=internal_noise,
        clamped=0.5 * (clamped + clamped.T) * EV_PER_ANGSTROM3_TO_GPA,
    )
    largest = float(np.abs(voigt).max())
    if largest > 0 and result.noise > NOISE_WARNING * largest:
        _warn_about_noise(result.noise, largest, strain, float("nan"))
    return result


def _relaxation_correction(hessian, internal, n) -> np.ndarray:
    """`L^T Phi^+ L` with the three uniform translations projected out.

    A uniform translation costs nothing, so `Phi` is singular along it and `L`
    has no component there; both are expressed in an orthonormal basis of the
    complement before solving. A Hessian that is not positive definite there
    means the reference is not a minimum, and the relaxation it implies is
    not one either.
    """
    translations = np.kron(np.ones((n, 1)), np.eye(3)) / np.sqrt(n)
    basis, _ = np.linalg.qr(translations, mode="complete")
    complement = basis[:, 3:]
    reduced = complement.T @ hessian @ complement
    reduced = 0.5 * (reduced + reduced.T)
    projected = complement.T @ internal
    eigenvalues, vectors = np.linalg.eigh(reduced)
    if eigenvalues.min() <= 0:
        LOG.warning(
            "the Gamma-point Hessian has %d non-positive eigenvalues (lowest "
            "%.3g eV/A^2): the reference is not a minimum, and the relaxed-ion "
            "constants from it are not meaningful",
            int(np.sum(eigenvalues <= 0)),
            eigenvalues.min(),
        )
    keep = np.abs(eigenvalues) > 1e-10 * np.abs(eigenvalues).max()
    inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    return projected.T @ inverse @ projected


def _internal_strain_from_displacements(plan, results, volume) -> np.ndarray:
    """`L = d(virial)/du` for every atom, from the displaced cells alone.

    Each displacement `d` of a representative atom measures `d . L_r`; each
    operation `s` of its site group turns that into another equation,
    `(R^T d) . L_r = R^T (d . L_r) R` on the strain indices, and the rest of
    the orbit follows from `L_(g(r)) = R (x) R (x) R . L_r`.
    """
    n = len(plan.system)
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    cartesian = plan.cartesian
    tensor = np.zeros((n, 3, 3, 3))
    results = iter(results)
    for representative, stabiliser, directions in plan.plan:
        rows, responses = [], []
        for direction in directions:
            plus, minus = next(results), next(results)
            voigt = volume * (plus.stress_voigt - minus.stress_voigt)
            voigt /= 2 * plan.displacement
            measured = np.zeros((3, 3))
            for index, (a, b) in enumerate(pairs):
                measured[a, b] = measured[b, a] = voigt[index]
            for operation in stabiliser:
                rotation = cartesian[operation]
                rows.append(rotation.T @ direction)
                responses.append((rotation.T @ measured @ rotation).ravel())
        solved = (np.linalg.pinv(np.array(rows)) @ np.array(responses)).reshape(3, 3, 3)
        images = plan.atom_map.permutation[:, representative]
        _, first = np.unique(images, return_index=True)
        for operation in first:
            rotation = cartesian[operation]
            tensor[images[operation]] = np.einsum(
                "xa,yb,zc,abc->xyz", rotation, rotation, rotation, solved
            )
    return np.stack([tensor[:, :, a, b] for a, b in pairs], axis=-1).reshape(3 * n, 6)


def _symmetrise_internal_strain(internal, atom_map, cell) -> np.ndarray:
    """Fold `L` onto the space group: the group average of
    `L_(g(i)) = R (x) R (x) R . L_i` over the operations."""
    n = internal.shape[0] // 3
    tensor = np.zeros((n, 3, 3, 3))
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    columns = internal.reshape(n, 3, 6)
    for index, (a, b) in enumerate(pairs):
        tensor[:, :, a, b] = tensor[:, :, b, a] = columns[:, :, index]
    rotations = atom_map.cartesian(cell)
    folded = np.zeros_like(tensor)
    for rotation, permutation in zip(rotations, atom_map.permutation, strict=True):
        rotated = np.einsum("xa,yb,zc,nabc->nxyz", rotation, rotation, rotation, tensor)
        np.add.at(folded, permutation, rotated)
    folded /= len(rotations)
    return np.stack([folded[:, :, a, b] for a, b in pairs], axis=-1).reshape(3 * n, 6)


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
        "calculator's own noise -- followed by the atomic relaxation (now "
        "fmax=%.3f eV/A, which for a noisy model works as a step budget rather "
        'than a tolerance). `strain="auto"` searches for a workable value.',
        noise,
        100 * noise / largest if largest > 0 else float("nan"),
        strain,
        fmax,
    )


def independent_strains(rotations) -> list[int]:
    """The Voigt strains to apply, skipping any already spanned by the
    point-group orbits of earlier ones. Cubic gives `[0, 3]`, hexagonal
    `[0, 2, 3]`, triclinic all six.

    Args:
        rotations: (M, 3, 3) Cartesian point operations, or empty for none

    Returns:
        Voigt indices, in increasing order
    """
    rotations = _with_identity(rotations)
    chosen, span, rank = [], np.zeros((0, 6)), 0
    for index in range(6):
        strain = voigt_strain(index, 1.0)
        orbit = to_vector(np.einsum("rij,jk,rlk->ril", rotations, strain, rotations))
        extended = np.vstack([span, orbit])
        extended_rank = np.linalg.matrix_rank(extended, tol=1e-8)
        if extended_rank == rank:
            continue
        chosen.append(index)
        span, rank = extended, extended_rank
        if rank == 6:
            break
    return chosen


def fit_elastic(responses, rotations):
    """Least-squares Voigt tensor from strain responses and their rotations.

    Each `(index, response)` is the stress per unit engineering strain along
    Voigt direction `index`; each rotation adds the pair `(R e R^T, R s R^T)`.

    Returns:
        (raw (6, 6) tensor, largest residual of the fit). The residual is
        nonzero only when responses disagree with the symmetry.
    """
    rotations = _with_identity(rotations)
    strains, stresses = [], []
    for index, response in responses:
        strain = voigt_strain(index, 1.0)
        stress = _tensor_from_voigt(response)
        for rotation in rotations:
            strains.append(_engineering_voigt(rotation @ strain @ rotation.T))
            stresses.append(_stress_voigt(rotation @ stress @ rotation.T))
    strains, stresses = np.array(strains).T, np.array(stresses).T
    tensor = stresses @ np.linalg.pinv(strains)
    misfit = float(np.abs(tensor @ strains - stresses).max(initial=0.0))
    return tensor, misfit


def _with_identity(rotations) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    return rotations if len(rotations) else np.eye(3)[None]


def _tensor_from_voigt(voigt) -> np.ndarray:
    "A stress tensor from (xx, yy, zz, yz, xz, xy)"
    xx, yy, zz, yz, xz, xy = voigt
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _stress_voigt(tensor) -> np.ndarray:
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[1, 2],
            tensor[0, 2],
            tensor[0, 1],
        ]
    )


def _engineering_voigt(tensor) -> np.ndarray:
    "A strain tensor as engineering Voigt, with the factor of two on shears"
    return _stress_voigt(tensor) * np.array([1, 1, 1, 2, 2, 2])


def _strain_sweep(
    system,
    calculator,
    reference_cell,
    reference_scaled,
    strains,
    strain,
    relax_atoms,
    fmax,
    steps,
    report,
    atom_map,
):
    """Central-difference stress response along each chosen Voigt direction.

    Returns `(index, dsigma/de)` pairs in GPa, unsymmetrised.
    """
    responses, relaxations, carried = [], [], None
    total = 2 * len(strains)
    for position, index in enumerate(strains):
        kept = None
        if atom_map is not None:
            kept = atom_map.preserving_strain(voigt_strain(index, 1.0), reference_cell)
            # +/- of one strain share a parameterisation; different strains
            # generally don't (in P1 they all do)
            carried = None
        stresses = []
        for half, sign in enumerate((1, -1)):
            point = 2 * position + half
            stage = f"voigt {index} {'+' if sign > 0 else '-'}{strain:g}"
            symmetry = f", {len(kept)} operations kept" if kept is not None else ""
            report("elastic", stage, f"{stage}{symmetry}", index=point, total=total)
            deformation = np.eye(3) + voigt_strain(index, sign * strain)
            strained = System(
                system.numbers,
                reference_scaled @ (reference_cell @ deformation),
                reference_cell @ deformation,
                True,
                dict(system.info),
            )
            outcome = None
            if relax_atoms:
                outcome, carried = _settle(
                    strained,
                    calculator,
                    fmax,
                    steps,
                    carried,
                    kept,
                    report.nested("elastic", stage),
                )
                if outcome is not None:
                    relaxations.append(outcome)
            stresses.append(calculator(strained, ("energy", "stress")).stress_voigt)
            settled = (
                f"{outcome.steps} relaxation steps"
                if outcome is not None
                else "no internal freedoms"
                if relax_atoms
                else "clamped atoms"
            )
            report(
                "elastic",
                stage,
                f"{stage}: {settled}",
                index=point,
                total=total,
                done=True,
            )
        response = (stresses[0] - stresses[1]) / (2.0 * strain)
        responses.append((index, response * EV_PER_ANGSTROM3_TO_GPA))
    return responses, relaxations


def _settle(system, calculator, fmax, steps, carried, atom_map=None, report=None):
    """Relax the atoms of a strained cell at fixed cell.

    `carried` is the curvature model from the previous strain point, reused
    when the parameterisation matches. With an `atom_map` the atoms relax
    within those operations.
    """
    coordinates = (
        InvariantAtomic(system, atom_map) if atom_map is not None else Atomic(system)
    )
    if coordinates.n_dof == 0:
        return None, carried
    reuse = carried if carried is not None and carried.n == coordinates.n_dof else None
    optimiser = TrustRegion(coordinates, calculator, model=reuse)
    outcome = optimiser.run(fmax=fmax, steps=steps, progress=report)
    if not outcome.converged:
        LOG.warning(
            "the atomic relaxation at one strain point did not converge "
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

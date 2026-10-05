"""Quasi-harmonic geometry: the structure minimising E_static + F_vib at T.

The free energy is minimised over the strains the space group allows, not
only the volume, so an anisotropic crystal expands differently along each
axis.

`thermal_stress` gives `sigma_th = (1/V) dF_vib/d eps` by central differences
along each allowed strain direction; each strained cell keeps the space group,
its atoms are relaxed at fixed cell, and its force constants give F_vib on a
q-mesh. `qha_structure` relaxes the static energy against that stress (through
`chmpy.calc.ExternalStress`), recomputes it at the new cell, and repeats until
the cell stops changing. At the fixed point the static stress cancels the
thermal stress, which is the condition d(E_static + F_vib)/d eps = 0.

    result = qha_structure(crystal, calculator, temperature=295.0)
    tensor = elastic_tensor(result.structure, calculator)  # quasi-static C
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.calc.potentials import ExternalStress
from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

from .force_constants import force_constants
from .thermo import thermal_properties

LOG = logging.getLogger(__name__)

#: strain amplitude for the free-energy finite difference
DEFAULT_STRAIN = 0.005


@dataclass
class ThermalStress:
    """The vibrational stress of a structure at a temperature.

    Attributes:
        stress: (3, 3) eV/A^3, `(1/V) dF_vib/d eps`, ASE sign convention
        free_energy: F_vib of the unstrained structure, eV per cell
        temperature: K
        n_imaginary: imaginary modes found over all the strained cells
    """

    stress: np.ndarray
    free_energy: float
    temperature: float
    n_imaginary: int

    @property
    def stress_gpa(self) -> np.ndarray:
        return self.stress * EV_PER_ANGSTROM3_TO_GPA


@dataclass
class QHAResult:
    """A quasi-harmonic structure and how it was reached.

    Attributes:
        structure: the structure minimising E_static + F_vib at `temperature`
        temperature: K
        thermal_stress: the last `ThermalStress`
        converged: whether the cell stopped changing within `max_iterations`
        history: per iteration, the cell lengths and largest thermal stress
    """

    structure: object
    temperature: float
    thermal_stress: ThermalStress
    converged: bool
    history: list = field(default_factory=list)


def _strained(crystal, deformation):
    "The crystal with its cell deformed and its fractional coordinates kept"
    from chmpy.crystal import Crystal, UnitCell

    return Crystal(
        UnitCell(np.asarray(crystal.unit_cell.direct) @ deformation),
        crystal.space_group,
        crystal.asymmetric_unit,
    )


def _settle_atoms(crystal, calculator, info, fmax):
    "Relax the atoms at fixed cell, if they have any freedom"
    from chmpy.opt import relax
    from chmpy.opt.relax import coordinates_for

    if coordinates_for(crystal, cell=False).n_dof == 0:
        return crystal
    result = relax(crystal, calculator, cell=False, info=info, fmax=fmax, steps=500)
    if not result.converged:
        LOG.warning("atomic relaxation of a strained cell did not converge")
    return result.structure


def _vibrational_free_energy(
    crystal, calculator, temperature, mesh, supercell, cutoff, info
):
    constants = force_constants(
        crystal, calculator, supercell=supercell, cutoff=cutoff, info=info
    )
    properties = thermal_properties(constants, [temperature], mesh=mesh)
    return properties.free_energy[0], properties.n_imaginary


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

    Args:
        crystal: a relaxed `Crystal`
        calculator: a `chmpy.calc.Calculator`
        temperature: K
        mesh: q-mesh for the free energy
        supercell, cutoff: passed to `force_constants`
        strain: finite-difference amplitude along each strain direction
        info: model inputs that are not geometry
        fmax: force tolerance for relaxing the atoms of each strained cell

    Returns:
        ThermalStress
    """
    from chmpy.opt.coordinates import strain_basis_for

    def free_energy(structure):
        return _vibrational_free_energy(
            structure, calculator, temperature, mesh, supercell, cutoff, info
        )

    centre, imaginary = free_energy(crystal)
    derivative = np.zeros((3, 3))
    for direction in strain_basis_for(crystal):
        values = []
        for sign in (1, -1):
            strained = _settle_atoms(
                _strained(crystal, np.eye(3) + sign * strain * direction),
                calculator,
                info,
                fmax,
            )
            value, count = free_energy(strained)
            values.append(value)
            imaginary += count
        # the basis is orthonormal, so the gradient's components rebuild it
        derivative += (values[0] - values[1]) / (2 * strain) * direction
    if imaginary:
        LOG.warning(
            "%d imaginary modes across the strained cells; the free energy "
            "describes only the stable modes",
            imaginary,
        )
    return ThermalStress(
        stress=derivative / crystal.unit_cell.volume(),
        free_energy=centre,
        temperature=float(temperature),
        n_imaginary=imaginary,
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
    smax: float = 1e-3,
    max_iterations: int = 10,
    tolerance: float = 1e-5,
) -> QHAResult:
    """Relax a crystal to the minimum of E_static + F_vib at a temperature.

    Args:
        crystal: a `Crystal`, ideally already relaxed statically
        calculator: a `chmpy.calc.Calculator`
        temperature: K
        mesh, supercell, cutoff, strain: see `thermal_stress`
        info: model inputs that are not geometry
        fmax, smax: tolerances for each relaxation, eV/A and GPa
        max_iterations: thermal-stress updates before giving up
        tolerance: largest relative change in a cell vector that counts as
            converged

    Returns:
        QHAResult
    """
    from chmpy.opt import relax

    current = crystal
    history = []
    converged = False
    sigma = None
    for _ in range(max_iterations):
        sigma = thermal_stress(
            current,
            calculator,
            temperature,
            mesh=mesh,
            supercell=supercell,
            cutoff=cutoff,
            strain=strain,
            info=info,
            fmax=fmax,
        )
        reference = np.asarray(current.unit_cell.direct)
        applied = calculator + ExternalStress(sigma.stress_gpa, reference)
        updated = relax(current, applied, info=info, fmax=fmax, smax=smax).structure
        change = float(
            (
                np.linalg.norm(np.asarray(updated.unit_cell.direct) - reference, axis=1)
                / np.linalg.norm(reference, axis=1)
            ).max()
        )
        history.append(
            {
                "lengths": [float(x) for x in updated.unit_cell.lengths],
                "thermal_stress_gpa": float(np.abs(sigma.stress_gpa).max()),
                "cell_change": change,
            }
        )
        current = updated
        if change < tolerance:
            converged = True
            break
    if not converged:
        LOG.warning(
            "the quasi-harmonic cell was still changing after %d iterations",
            max_iterations,
        )
    return QHAResult(
        structure=current,
        temperature=float(temperature),
        thermal_stress=sigma,
        converged=converged,
        history=history,
    )

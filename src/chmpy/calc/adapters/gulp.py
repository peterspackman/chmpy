"""GULP as a calculator.

`chmpy.exe.Gulp` already runs GULP and parses its `.drv` file, which carries
the energy, the Cartesian gradients and the strain gradients -- everything this
interface needs. This wraps that as a `Calculator`, so GULP can be used
wherever any other one can: relaxation, elastic constants, force constants.

    calc = GulpCalculator(potentials='''
        species
        C core  0.0
        buck
        C core C core  1000.0 0.3 0.0 0.0 12.0
    ''')

The potential is the caller's: GULP will not guess one, and a run with no
potential returns zero for everything. Pass the lines you would put in a GULP
input -- a `library` directive, explicit `buck`/`lennard` blocks, whatever.

Being an external program run per evaluation, this is slow in a way none of the
in-process calculators are: each call writes an input, starts a process, and
reads files back. It earns that where GULP has something the alternatives do
not -- an established force field, or second derivatives computed analytically
rather than by finite differences, which makes it a reference for checking
`chmpy.vib`.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from chmpy.util.exe import which

from ..base import Calculator, PropertyNotAvailable
from ..result import FORCES, STRESS, Result

LOG = logging.getLogger(__name__)

#: keywords that make GULP report derivatives for a fixed geometry. `conp`
#: asks for the strain derivatives as well as the Cartesian ones.
DERIVATIVE_KEYWORDS = ("single", "gradient", "conp")


class GulpCalculator(Calculator):
    """Energies, forces and stresses from GULP.

    Args:
        potentials: the potential specification, as GULP input lines. Required:
            without it GULP has nothing to compute.
        keywords: GULP keywords. The default asks for a single-point energy
            with gradients at constant pressure, which is what yields forces
            and stress.
        charges: (n_asym,) charges to write into the input, or None
        working_directory: where to run. A temporary directory per call by
            default, which keeps concurrent calculators from colliding.
        timeout: seconds before a run is abandoned

    Examples:
        Checking a set of force constants against GULP's analytic ones::

            calc = GulpCalculator(potentials=library)
            constants = force_constants(crystal, calc)
    """

    provides = {"energy", "forces", "stress"}
    #: GULP prints ten significant figures to the .drv file
    energy_precision = 1e-10

    def __init__(
        self,
        potentials: str,
        keywords=DERIVATIVE_KEYWORDS,
        charges=None,
        working_directory=None,
        timeout: float = 600.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not which("gulp"):
            raise ImportError(
                "GULP was not found on PATH; this calculator runs the `gulp` "
                "executable, which chmpy does not ship"
            )
        if not potentials or not potentials.strip():
            raise ValueError(
                "GULP needs a potential: without one it reports zero for every "
                "property, which is indistinguishable from a working run"
            )
        self.potentials = potentials
        self.keywords = list(keywords)
        self.charges = charges
        self.working_directory = working_directory
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"<GulpCalculator keywords={self.keywords}>"

    def compute(self, system, want):
        from chmpy.exe.gulp import Gulp

        contents = self._input(system)
        directory = self.working_directory
        with tempfile.TemporaryDirectory() as scratch:
            job = Gulp(
                contents,
                working_directory=str(directory or scratch),
                timeout=self.timeout,
            )
            job.run()
            energy = job.energy
            gradients = job.gradients
            strain = job.stress_raw

        if energy is None:
            raise PropertyNotAvailable(
                "GULP produced no energy; its output is in the job's working "
                "directory. A missing or misspelled potential is the usual cause."
            )
        if energy == 0.0 and len(system) > 1:
            raise PropertyNotAvailable(
                "GULP returned exactly zero for a system of "
                f"{len(system)} atoms, which means no potential matched any pair. "
                "The species names in the potential have to match the ones "
                "written to the input, which are bare element symbols; a typed "
                "force field such as DREIDING expects C_3 or O_2 instead and "
                "will silently match nothing."
            )

        forces = None
        if FORCES in want:
            if gradients is None:
                raise PropertyNotAvailable(
                    "GULP returned no gradients; `gradient` must be among the "
                    f"keywords, which are {self.keywords}"
                )
            forces = -np.asarray(gradients, dtype=float)

        stress = None
        if STRESS in want and system.periodic:
            if strain is None:
                raise PropertyNotAvailable(
                    "GULP returned no strain derivatives; `conp` must be among "
                    f"the keywords, which are {self.keywords}"
                )
            stress = _to_matrix(np.asarray(strain, dtype=float)) / system.volume

        return Result(
            energy=float(energy),
            forces=forces,
            stress=stress,
            volume=system.volume,
        )

    def _input(self, system) -> str:
        """A GULP input for this geometry, with the potential appended.

        A periodic system is written in P1: a `System` carries no symmetry, it
        is the unit cell as the calculator sees it, and asserting a space group
        here would be inventing one. Symmetry-constrained relaxation happens a
        level up, in `chmpy.opt`, which hands this the expanded cell.

        An isolated system is written as a molecule, with `conv` in place of
        `conp`: there is no cell to hold at constant pressure. The substitution
        is not cosmetic. GULP reads per-atom optimisation flags after the
        coordinates unless a keyword says otherwise, so `conp` cannot simply be
        dropped; and the obvious replacement, `noflag`, silences that but
        leaves no atom marked as a variable, so GULP returns the energy and no
        gradients at all -- with a warning easily lost in its output. This path
        exists because a lattice energy needs the gas-phase molecules as well
        as the crystal.
        """
        from chmpy.fmt.gulp import crystal_to_gulp_input, molecule_to_gulp_input

        if system.periodic:
            body = crystal_to_gulp_input(
                system.to_crystal(1), keywords=self.keywords, charges=self.charges
            )
        else:
            keywords = [k for k in self.keywords if k != "conp"]
            if "conv" not in keywords and "noflag" not in keywords:
                keywords.append("conv")
            body = molecule_to_gulp_input(system.to_molecule(), keywords=keywords)
        return f"{body}\n{self.potentials}\n"


def _to_matrix(strain_gradients) -> np.ndarray:
    """GULP's six strain derivatives as a symmetric (3, 3).

    They come in Voigt order -- xx, yy, zz, yz, xz, xy -- as `dE/de` for the
    engineering strain, which is already what the shear components of a stress
    tensor are, so the off-diagonals are placed rather than halved.
    """
    values = np.asarray(strain_gradients, dtype=float).ravel()
    if values.size != 6:
        raise ValueError(
            f"expected six strain derivatives from GULP, got {values.size}"
        )
    xx, yy, zz, yz, xz, xy = values
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def library_calculator(name: str, **kwargs) -> GulpCalculator:
    """A calculator using one of GULP's bundled potential libraries.

    Args:
        name: the library to load, e.g. "dreiding" or "streitz"
        **kwargs: passed to `GulpCalculator`

    Returns:
        GulpCalculator
    """
    return GulpCalculator(potentials=f"library {name}", **kwargs)


def available() -> bool:
    "Whether the `gulp` executable can be found"
    return bool(which("gulp"))


def _find_library(name: str) -> Path | None:
    """Where GULP keeps its libraries, if the environment says."""
    import os

    root = os.environ.get("GULP_LIB")
    if not root:
        return None
    candidate = Path(root) / f"{name}.lib"
    return candidate if candidate.exists() else None

"""Conversion to and from `ase.calculators`.

`Calculator.from_ase(calculator)` brings one in, which is how most
machine-learned potentials are distributed. `calc.as_ase()` sends one the other
way, for workflows that expect an ASE calculator.

Coming in, the adapter keeps a single `Atoms` object for the whole run and
writes the geometry into its arrays, rather than constructing one per
evaluation, and calls `calculate` directly with the properties wanted and the
list of what changed. It cannot avoid the `self.atoms = atoms.copy()` inside
`ase.calculators.calculator.Calculator.calculate`, which most implementations
reach through `super()`.

Going out, `_want` decides what to ask the wrapped calculator for. ASE requests
one property at a time and re-enters `calculate` for each one it does not find
in `results`, so answering a cell-filter step narrowly would evaluate the same
geometry twice; see that method for the rule.
"""

from __future__ import annotations

import numpy as np

from chmpy.util.optional import require

from ..base import Calculator, PropertyNotAvailable
from ..result import ENERGIES, ENERGY, FORCES, STRESS, Result
from ..system import System

#: ASE property names for the ones this interface knows about
_ASE_NAMES = {
    ENERGY: "energy",
    FORCES: "forces",
    STRESS: "stress",
    ENERGIES: "energies",
}


def voigt_to_matrix(stress) -> np.ndarray:
    """A (6,) Voigt stress as a symmetric (3, 3) matrix.

    Args:
        stress: (6,) as xx, yy, zz, yz, xz, xy, or an already-square (3, 3)

    Returns:
        (3, 3) symmetric stress
    """
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape == (3, 3):
        return stress
    xx, yy, zz, yz, xz, xy = stress
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def matrix_to_voigt(stress) -> np.ndarray:
    """A symmetric (3, 3) stress as a (6,) Voigt vector."""
    s = np.asarray(stress, dtype=np.float64)
    return np.array(
        [
            s[0, 0],
            s[1, 1],
            s[2, 2],
            0.5 * (s[1, 2] + s[2, 1]),
            0.5 * (s[0, 2] + s[2, 0]),
            0.5 * (s[0, 1] + s[1, 0]),
        ]
    )


class AseCalculator(Calculator):
    """A published ASE calculator, wrapped.

    Args:
        calculator: any `ase.calculators.calculator.Calculator`
        provides: override the properties to take from it. By default this is
            read from its `implemented_properties`, which is occasionally
            optimistic -- a model that lists `stress` but raises for an
            isolated system, say.
        info: entries to put in `atoms.info` on every call, for models that
            want a charge or a spin there

    Examples:
        Wrapping a model published as an ASE calculator::

            from mace.calculators import mace_mp

            calc = Calculator.from_ase(mace_mp())
            calc(System.from_crystal(crystal), ("energy", "forces", "stress"))
    """

    def __init__(self, calculator, provides=None, info=None, **kwargs):
        require("ase", "wrapping an ASE calculator")
        self.ase_calculator = calculator
        self.info = dict(info) if info else {}
        if provides is None:
            implemented = set(getattr(calculator, "implemented_properties", ["energy"]))
            provides = {
                name for name, ase_name in _ASE_NAMES.items() if ase_name in implemented
            }
            provides.add(ENERGY)
        self.provides = frozenset(provides)
        super().__init__(**kwargs)
        self._atoms = None
        self._direct = True
        self._direct_error = None

    def __repr__(self) -> str:
        return f"<AseCalculator {type(self.ase_calculator).__name__}>"

    def compute(self, system, want):
        atoms = self._sync(system)
        properties = sorted(_ASE_NAMES[name] for name in want)

        if self._direct:
            try:
                self.ase_calculator.calculate(atoms, properties, ["positions", "cell"])
                return self._unpack(self.ase_calculator.results, system, want)
            except Exception as exc:
                # Not every calculator can be driven this way -- some need the
                # properties negotiated by `get_property`. Take the slower route
                # that ASE itself takes, and keep taking it.
                self._direct = False
                self._direct_error = exc

        atoms.calc = self.ase_calculator
        try:
            atoms.get_potential_energy()
        except Exception as exc:
            if self._direct_error is not None:
                raise RuntimeError(
                    f"{type(self.ase_calculator).__name__} failed both when its "
                    f"calculate() was called directly ({self._direct_error!r}) and "
                    f"through ase.Atoms.get_potential_energy()"
                ) from exc
            raise
        return self._unpack(self.ase_calculator.results, system, want)

    def _sync(self, system) -> ase.Atoms:  # noqa: F821
        """Write `system` into the persistent Atoms object, building it once."""
        atoms = self._atoms
        numbers = np.asarray(system.numbers)
        if (
            atoms is None
            or len(atoms) != len(system)
            or not np.array_equal(atoms.numbers, numbers)
        ):
            atoms = self._atoms = system.to_ase()
        else:
            atoms.positions[:] = system.positions
            if not np.array_equal(np.asarray(atoms.cell), system.cell):
                atoms.set_cell(np.asarray(system.cell))
            if not np.array_equal(atoms.pbc, system.pbc):
                atoms.set_pbc(np.asarray(system.pbc))
        atoms.info.update(self.info)
        atoms.info.update(system.info)
        return atoms

    def _unpack(self, results, system, want) -> Result:
        if "energy" not in results:
            raise PropertyNotAvailable(
                f"{type(self.ase_calculator).__name__} returned no energy; it "
                f"produced {sorted(results)}"
            )
        stress = None
        if STRESS in want and "stress" in results:
            stress = voigt_to_matrix(results["stress"])
        return Result(
            energy=float(results["energy"]),
            forces=np.asarray(results["forces"]) if FORCES in want else None,
            stress=stress,
            energies=np.asarray(results["energies"]) if ENERGIES in want else None,
            volume=system.volume,
        )


def as_ase_calculator(calculator):
    """Wrap a chmpy `Calculator` as an ASE one.

    Args:
        calculator: a `chmpy.calc.Calculator`

    Returns:
        an `ase.calculators.calculator.Calculator` delegating to it
    """
    ase_calculator_module = require(
        "ase.calculators.calculator", "exposing a calculator to ASE"
    )
    base = ase_calculator_module.Calculator

    class ChmpyCalculator(base):
        """An `ase.Atoms`-facing view of a chmpy calculator."""

        implemented_properties = sorted(
            _ASE_NAMES[name] for name in calculator.provides
        )

        def __init__(self):
            super().__init__()
            self.calculator = calculator
            self._system = None

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            super().calculate(atoms, list(properties), system_changes or [])
            system = self._reuse(atoms)
            result = self.calculator(system, self._want(properties, system))

            self.results = {"energy": result.energy, "free_energy": result.energy}
            if result.forces is not None:
                self.results["forces"] = result.forces
            if result.stress is not None:
                self.results["stress"] = matrix_to_voigt(result.stress)
            if result.energies is not None:
                self.results["energies"] = result.energies

        def _want(self, properties, system):
            """What to ask the calculator for, given what ASE asked for.

            ASE requests one property at a time and re-enters `calculate` for
            each one it does not find in `self.results`, so a cell-filter step
            asks for the forces and then the stress at the same geometry. For
            a model that differentiates its energy, both come out of the same
            backward pass, so answering the first request narrowly would double
            the cost of every step.

            The rule: a request for any derivative asks for every derivative
            the calculator computes *natively*. A request for the energy alone
            stays an energy alone, and a derivative the calculator does not
            provide is never volunteered -- finite-differencing a stress nobody
            asked for costs twelve energy evaluations.
            """
            asked = {
                name for name, ase_name in _ASE_NAMES.items() if ase_name in properties
            }
            want = asked | {ENERGY}
            if asked & {FORCES, STRESS}:
                want |= frozenset(self.calculator.provides) & {FORCES, STRESS}
            if not system.periodic:
                want -= {STRESS}
            return want

        def _reuse(self, atoms):
            """Keep one System alive so the calculator's cache stays warm.

            The geometry is written only when it has actually changed. ASE asks
            for one property at a time -- a cell filter's `get_forces` wants the
            forces and then the stress -- and writing the same positions back
            would bump the version and turn each of those requests into a fresh
            evaluation of an unchanged structure.
            """
            system = self._system
            if (
                system is None
                or len(system) != len(atoms)
                or not np.array_equal(system.numbers, atoms.numbers)
                or not np.array_equal(system.pbc, atoms.pbc)
            ):
                return self._replace(atoms)
            if not np.array_equal(system.positions, atoms.positions):
                system.set_positions(atoms.positions)
            if np.any(atoms.pbc) and not np.array_equal(
                system.cell, np.asarray(atoms.cell)
            ):
                system.set_cell(np.asarray(atoms.cell))
            system.info.update(atoms.info)
            return system

        def _replace(self, atoms):
            self._system = System.from_ase(atoms)
            return self._system

        def __repr__(self):
            return f"<ase view of {self.calculator!r}>"

    return ChmpyCalculator()

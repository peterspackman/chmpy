"""Energies, forces and stresses of a `System`.

A calculator subclasses `Calculator`, declares which properties it computes
natively, and implements `compute`:

    class Harmonic(Calculator):
        provides = {"energy", "forces"}

        def compute(self, system, want):
            displacement = system.positions - self.reference
            return Result(
                energy=0.5 * self.k * float((displacement**2).sum()),
                forces=-self.k * displacement,
            )

`compute` is called only with properties the subclass declared, and may return
more than it was asked for -- the extras are cached rather than discarded.
Around it the base class provides:

* results cached on `(system, version)`, so repeated queries of an unchanged
  geometry cost one evaluation;
* finite-difference forces and stress for derivatives a subclass does not
  declare, so an energy-only calculator still works under variable cell;
* `check_gradients`, comparing analytic derivatives against finite differences;
* `batch`, a loop by default and one model call for backends that override
  `compute_batch`;
* `energy_noise`, the smallest energy difference the calculator can resolve,
  which the optimiser needs in order not to read rounding as a result;
* conversion to and from `ase.calculators`;
* composition: `a + b` sums two calculators, and `calc.shifted(references)`
  subtracts per-element reference energies;
* counters on `calc.stats`.

Units are eV and Angstroms throughout; see `chmpy.calc.result`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from .result import (
    ALL_PROPERTIES,
    ENERGIES,
    ENERGY,
    FORCES,
    STRESS,
    CalculatorStats,
    GradientCheck,
    Result,
    normalise_properties,
)
from .system import System

LOG = logging.getLogger(__name__)

#: default step for finite differences, Angstroms (and dimensionless for strain)
DEFAULT_FD_STEP = 1e-4


class PropertyNotAvailable(RuntimeError):
    """A calculator was asked for something it cannot produce."""


def _check_stress_defined(want, system) -> None:
    """Stress is a derivative with respect to a strain, so it needs a cell."""
    if STRESS in want and not system.periodic:
        raise PropertyNotAvailable(
            "stress is only defined for a system periodic in all three "
            f"directions; this one has pbc={tuple(bool(p) for p in system.pbc)}"
        )


class _CacheEntry:
    __slots__ = ("system", "version", "result", "available")

    def __init__(self, system, result, available):
        self.system = system
        self.version = system.version
        self.result = result
        self.available = available


class Calculator(ABC):
    """Base class for anything that computes energies of a `System`.

    Attributes:
        provides: the properties `compute` produces without help. Anything else
            that is asked for is filled in by finite differences.
        cache_size: how many geometries to remember results for.
        stats: call counts and wall time.
    """

    #: what `compute` returns natively; everything else is finite-differenced
    provides: frozenset = frozenset({ENERGY})
    #: how many geometries to remember results for
    cache_size: int = 8
    #: displacement for finite differences, Angstroms
    fd_step: float = DEFAULT_FD_STEP
    #: relative precision of the energies this calculator returns. None means
    #: "work it out from the first energy"; see `energy_noise`.
    energy_precision: float | None = None

    def __init_subclass__(cls, **kwargs):
        """Let a subclass write `provides = {"energy", "forces"}` and mean it."""
        super().__init_subclass__(**kwargs)
        if "provides" in cls.__dict__:
            cls.provides = frozenset(cls.provides)

    def __init__(self, cache_size: int | None = None, fd_step: float | None = None):
        # Calling this is optional: everything it sets has a working default, so
        # a subclass that forgets `super().__init__()` still gets caching and
        # statistics rather than an AttributeError three layers down.
        if cache_size is not None:
            self.cache_size = cache_size
        if fd_step is not None:
            self.fd_step = fd_step

    @property
    def stats(self) -> CalculatorStats:
        "Evaluation counts and wall time"
        stats = self.__dict__.get("_stats")
        if stats is None:
            stats = self.__dict__["_stats"] = CalculatorStats()
        return stats

    @property
    def _results(self) -> OrderedDict:
        cache = self.__dict__.get("_result_cache")
        if cache is None:
            cache = self.__dict__["_result_cache"] = OrderedDict()
        return cache

    # -- the one thing a subclass must write ---------------------------------

    @abstractmethod
    def compute(self, system: System, want: frozenset) -> Result:
        """Compute `want` for `system`. Implemented by subclasses.

        `want` is always a subset of `self.provides` and always contains
        `"energy"`. Returning more than was asked for is welcome -- the extra
        properties are cached rather than discarded.

        Args:
            system: the geometry to evaluate
            want: property names, a subset of `self.provides`

        Returns:
            Result holding at least the requested properties
        """

    def compute_batch(self, systems: list[System], want: frozenset) -> list[Result]:
        """Compute `want` for several systems.

        The default runs `compute` on each in turn. Override it when the
        underlying model can evaluate a list in one pass, which is most of what
        a GPU offers when the cells are small.
        """
        return [self.compute(system, want) for system in systems]

    # -- what callers use ----------------------------------------------------

    def __call__(self, system: System, want=(ENERGY,)) -> Result:
        """Evaluate `system`, returning at least the properties in `want`.

        Args:
            system: the geometry to evaluate
            want: a property name or an iterable of them

        Returns:
            Result, from cache when the geometry has not changed
        """
        want = normalise_properties(want)
        _check_stress_defined(want, system)
        cached = self._lookup(system, want)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached

        native = want & frozenset(self.provides)
        started = time.perf_counter()
        result = self.compute(system, native)
        self.stats.record(native, time.perf_counter() - started)
        result = self._validate(result, system, native)

        missing = want - self._available(result)
        if missing:
            result = self._finite_difference(system, result, missing)

        self._store(system, result)
        return result

    #: `calc.evaluate(system, ...)` reads better in a pipeline than `calc(...)`
    evaluate = __call__

    def batch(self, systems, want=(ENERGY,)) -> list[Result]:
        """Evaluate several systems, using the native batch path where there is one.

        Cached systems are dropped from the batch and only the rest go to the
        model, so re-evaluating a partly unchanged set is cheap.
        """
        want = normalise_properties(want)
        systems = list(systems)
        for system in systems:
            _check_stress_defined(want, system)
        results: list[Result | None] = [None] * len(systems)

        todo = []
        for index, system in enumerate(systems):
            cached = self._lookup(system, want)
            if cached is not None:
                self.stats.cache_hits += 1
                results[index] = cached
            else:
                todo.append(index)

        if todo:
            native = want & frozenset(self.provides)
            started = time.perf_counter()
            computed = self.compute_batch([systems[i] for i in todo], native)
            elapsed = time.perf_counter() - started
            self.stats.batched_calls += 1
            self.stats.seconds += elapsed
            self.stats.calls += len(todo)
            for name in native:
                self.stats.per_property[name] = self.stats.per_property.get(
                    name, 0
                ) + len(todo)
            if len(computed) != len(todo):
                raise PropertyNotAvailable(
                    f"{type(self).__name__}.compute_batch returned "
                    f"{len(computed)} results for {len(todo)} systems"
                )
            for index, result in zip(todo, computed, strict=True):
                system = systems[index]
                result = self._validate(result, system, native)
                missing = want - self._available(result)
                if missing:
                    result = self._finite_difference(system, result, missing)
                self._store(system, result)
                results[index] = result

        return results

    def energy(self, system: System) -> float:
        "Total energy in eV"
        return self(system, (ENERGY,)).energy

    def energies(self, system: System) -> np.ndarray:
        "(N,) per-atom energies in eV"
        return self(system, (ENERGIES,)).energies

    def forces(self, system: System) -> np.ndarray:
        "(N, 3) forces in eV/A"
        return self(system, (FORCES,)).forces

    def stress(self, system: System) -> np.ndarray:
        "(3, 3) stress in eV/A^3"
        return self(system, (STRESS,)).stress

    def virial(self, system: System) -> np.ndarray:
        "(3, 3) `-dE/deps` in eV"
        return self(system, (STRESS,)).virial

    def energy_noise(self, energy: float) -> float:
        """How much of an energy difference this calculator cannot resolve, in eV.

        This matters more than it sounds. A model running in float32 returns
        energies with about seven significant figures, so for a cell at -190 eV
        one unit in the last place is 1.5e-5 eV -- and near a minimum the real
        energy change over an optimiser step is smaller than that. An optimiser
        that compares energies then sees a reduction of exactly zero, decides
        its model is wrong, and shrinks its way to a halt while the structure is
        still moving. `chmpy.opt.TrustRegion` asks for this number and stops
        drawing conclusions from differences below it.

        `energy_precision` sets the relative precision, and a calculator that
        knows it should say so -- `MetatomicCalculator` reads it off the model's
        dtype. Left as None it is inferred from the energies that come back: a
        float32 model's energies are all exactly representable in float32.

        The inference waits for several *distinct* energies before concluding
        anything, because single round numbers are exactly representable in
        float32 too: deciding on one sample called -1.0, -0.5 and -100.0
        float32, which would hand an analytic potential a noise floor of 1e-5
        eV and let the optimiser accept steps that genuinely go uphill. What
        the inference still cannot do is tell a float32 model from a calculator
        whose energies are *all* round numbers, since those are float32 values
        by definition. Real potentials do not behave that way, but a test
        double might -- set `energy_precision` explicitly if yours does.

        Args:
            energy: the energy in question, eV

        Returns:
            the smallest energy difference worth believing, in eV
        """
        precision = self.energy_precision
        if precision is None:
            precision = self.__dict__.get("_detected_precision")
        if precision is None:
            return float(np.spacing(abs(energy)))
        return float(precision * max(abs(energy), 1.0))

    #: distinct energies that must all be float32-representable before the
    #: precision is inferred to be float32
    precision_samples: int = 3

    def _detect_precision(self, energy: float) -> None:
        """Infer the precision of the energies, once there is enough evidence."""
        if self.energy_precision is not None:
            return
        if "_detected_precision" in self.__dict__:
            return
        if energy == 0.0 or not np.isfinite(energy):
            return
        if float(np.float32(energy)) != energy:
            # one energy that float32 cannot hold settles it
            self.__dict__["_detected_precision"] = float(np.finfo(np.float64).eps)
            return
        seen = self.__dict__.setdefault("_precision_samples", set())
        seen.add(energy)
        if len(seen) >= self.precision_samples:
            self.__dict__["_detected_precision"] = float(np.finfo(np.float32).eps)

    # -- composition ---------------------------------------------------------

    def __add__(self, other: Calculator) -> Calculator:
        if not isinstance(other, Calculator):
            return NotImplemented
        return SumCalculator([self, other])

    def shifted(self, reference_energies: dict) -> Calculator:
        """This calculator with a per-element reference energy subtracted.

        Args:
            reference_energies: atomic number (or element symbol) -> eV

        Returns:
            a calculator whose energies are measured from those references
        """
        return ShiftedCalculator(self, reference_energies)

    # -- checking ------------------------------------------------------------

    def check_gradients(
        self, system: System, step: float | None = None, tolerance: float = 1e-5
    ) -> GradientCheck:
        """Verify analytic forces and stress against finite differences.

        Only derivatives the calculator declares in `provides` are checked --
        anything finite-differenced is trivially self-consistent. Print the
        result: it names the worst component.

        A failing check is not always a wrong derivative. The numeric side has
        its own error, largest at small steps where a difference of two nearly
        equal energies loses precision, and at large steps where the quadratic
        term stops being negligible. On a stiff crystal the two have their
        minima at different steps -- forces prefer the larger, stress the
        smaller -- so a check that fails at one step is worth repeating at
        another before concluding anything.

        Args:
            system: geometry to check at, ideally one away from a stationary
                point and with no exact symmetry hiding a sign error
            step: displacement in Angstroms and strain increment; defaults to
                `self.fd_step`
            tolerance: relative to the largest analytic component

        Returns:
            GradientCheck, whose `ok` is the verdict
        """
        step = self.fd_step if step is None else step
        analytic = self(system, frozenset(self.provides))

        forces_error = forces_scale = forces_worst = None
        if FORCES in self.provides:
            numeric = self._numeric_forces(system, step)
            difference = np.abs(analytic.forces - numeric)
            forces_error = float(difference.max())
            forces_scale = float(np.abs(analytic.forces).max())
            forces_worst = tuple(
                int(i) for i in np.unravel_index(difference.argmax(), difference.shape)
            )

        stress_error = stress_scale = stress_worst = None
        if STRESS in self.provides and system.periodic:
            numeric = self._numeric_stress(system, step)
            difference = np.abs(analytic.stress - numeric)
            stress_error = float(difference.max())
            stress_scale = float(np.abs(analytic.stress).max())
            stress_worst = tuple(
                int(i) for i in np.unravel_index(difference.argmax(), difference.shape)
            )

        return GradientCheck(
            forces_error,
            forces_scale,
            forces_worst,
            stress_error,
            stress_scale,
            stress_worst,
            tolerance,
            step,
        )

    # -- interoperability ----------------------------------------------------

    @staticmethod
    def from_ase(calculator, **kwargs) -> Calculator:
        """Wrap an ASE calculator, which is how most models are published.

        The wrapper keeps one `ase.Atoms` alive and writes the geometry into
        it, rather than building a new one per evaluation.

        Args:
            calculator: any `ase.calculators.calculator.Calculator`

        Returns:
            a `Calculator` wrapping it
        """
        from .adapters.ase import AseCalculator

        return AseCalculator(calculator, **kwargs)

    def as_ase(self):
        """Expose this calculator as an `ase.calculators.calculator.Calculator`,
        for workflows that expect one.
        """
        from .adapters.ase import as_ase_calculator

        return as_ase_calculator(self)

    # -- relaxation ----------------------------------------------------------

    def relax(self, structure, **kwargs):
        """Relax a `Crystal`, `Molecule` or `System` with this calculator.

        Keyword arguments go to `chmpy.opt.relax`.
        """
        from chmpy.opt import relax

        return relax(structure, self, **kwargs)

    def lattice_energy(self, crystal, **kwargs):
        """The lattice energy of a molecular crystal with this calculator.

        Keyword arguments go to `chmpy.opt.lattice_energy`.
        """
        from chmpy.opt import lattice_energy

        return lattice_energy(crystal, self, **kwargs)

    def elastic_tensor(self, structure, **kwargs):
        """The elastic tensor of an already-relaxed structure.

        Keyword arguments go to `chmpy.opt.elastic_tensor`.
        """
        from chmpy.opt import elastic_tensor

        return elastic_tensor(structure, self, **kwargs)

    # -- internals -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{type(self).__name__} provides={sorted(self.provides)}>"

    def clear_cache(self) -> None:
        "Forget every remembered result"
        self._results.clear()

    def _lookup(self, system, want):
        entry = self._results.get(id(system))
        if entry is None or entry.system is not system:
            return None
        if entry.version != system.version or not want <= entry.available:
            return None
        self._results.move_to_end(id(system))
        return entry.result

    def _store(self, system, result):
        cache = self._results
        # the entry holds the system, so its id cannot be reused while cached
        cache[id(system)] = _CacheEntry(system, result, self._available(result))
        cache.move_to_end(id(system))
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    @staticmethod
    def _available(result) -> frozenset:
        return frozenset(name for name in ALL_PROPERTIES if result.has(name))

    def _validate(self, result, system, want) -> Result:
        """Check shapes and finiteness, and attach the volume the stress needs."""
        who = type(self).__name__
        if not isinstance(result, Result):
            raise TypeError(f"{who}.compute must return a Result, got {type(result)}")
        if not np.isfinite(result.energy):
            raise PropertyNotAvailable(f"{who} returned a non-finite energy")
        self._detect_precision(result.energy)

        n = len(system)
        for name, shape in ((FORCES, (n, 3)), (STRESS, (3, 3)), (ENERGIES, (n,))):
            value = getattr(result, name)
            if value is None:
                if name in want:
                    raise PropertyNotAvailable(
                        f"{who} declares {name!r} in `provides` but returned None"
                    )
                continue
            if value.shape != shape:
                raise ValueError(
                    f"{who} returned {name} with shape {value.shape}, expected {shape}"
                )
            if not np.all(np.isfinite(value)):
                raise PropertyNotAvailable(f"{who} returned non-finite {name}")

        undeclared = self._available(result) - frozenset(self.provides)
        if undeclared and not self.__dict__.get("_warned_undeclared"):
            self.__dict__["_warned_undeclared"] = True
            LOG.warning(
                "%s returned %s but does not list %s in `provides`; they will be "
                "used, but declaring them avoids a needless finite-difference "
                "fallback when they are requested on their own",
                who,
                sorted(undeclared),
                "them" if len(undeclared) > 1 else "it",
            )

        if result.stress is not None and result.volume == 0.0:
            result = _replace_volume(result, system.volume)
        return result

    def _finite_difference(self, system, result, missing) -> Result:
        """Fill in derivatives the calculator does not provide."""
        forces = result.forces
        stress = result.stress
        if FORCES in missing:
            forces = self._numeric_forces(system, self.fd_step)
        if STRESS in missing:
            if not system.periodic:
                raise PropertyNotAvailable(
                    "stress is only defined for a periodic system"
                )
            stress = self._numeric_stress(system, self.fd_step)
        if ENERGIES in missing:
            raise PropertyNotAvailable(
                f"{type(self).__name__} does not provide per-atom energies, and "
                "they cannot be finite-differenced"
            )
        return Result(
            energy=result.energy,
            forces=forces,
            stress=stress,
            energies=result.energies,
            volume=system.volume,
            extra=result.extra,
        )

    def _energy_at(self, scratch, positions, cell=None) -> float:
        """Energy of a scratch system at a displaced geometry."""
        if cell is not None:
            scratch.set_cell(cell)
        scratch.set_positions(positions)
        self.stats.finite_difference_calls += 1
        return float(self.compute(scratch, frozenset({ENERGY})).energy)

    def _numeric_forces(self, system, step) -> np.ndarray:
        """Central-difference forces: 6N energy evaluations."""
        scratch = system.copy()
        positions = np.array(system.positions)
        forces = np.zeros_like(positions)
        for i in range(len(system)):
            for d in range(3):
                shifted = positions.copy()
                shifted[i, d] += step
                plus = self._energy_at(scratch, shifted)
                shifted[i, d] -= 2 * step
                minus = self._energy_at(scratch, shifted)
                forces[i, d] = -(plus - minus) / (2 * step)
        return forces

    def _numeric_stress(self, system, step) -> np.ndarray:
        """Central-difference stress: 12 energy evaluations.

        Deforms by `x = I + eps` applied as `cell @ x` with the atoms carried
        along in fractional coordinates, which is the strain the stress is the
        derivative with respect to. The shear components put `step / 2` in both
        off-diagonal entries so that `eps` stays symmetric and the derivative is
        with respect to the tensor component, not the engineering shear.
        """
        scratch = system.copy()
        cell = np.array(system.cell)
        scaled = system.scaled_positions
        volume = system.volume
        stress = np.zeros((3, 3))

        def energy_for(deformation):
            deformed = cell @ deformation
            return self._energy_at(scratch, scaled @ deformed, deformed)

        for i in range(3):
            x = np.eye(3)
            x[i, i] += step
            plus = energy_for(x)
            x[i, i] -= 2 * step
            minus = energy_for(x)
            stress[i, i] = (plus - minus) / (2 * step * volume)

        for i, j in ((0, 1), (0, 2), (1, 2)):
            x = np.eye(3)
            x[i, j] = x[j, i] = 0.5 * step
            plus = energy_for(x)
            x[i, j] = x[j, i] = -0.5 * step
            minus = energy_for(x)
            stress[i, j] = stress[j, i] = (plus - minus) / (2 * step * volume)

        return stress


def _replace_volume(result: Result, volume: float) -> Result:
    return Result(
        energy=result.energy,
        forces=result.forces,
        stress=result.stress,
        energies=result.energies,
        volume=volume,
        extra=result.extra,
    )


class SumCalculator(Calculator):
    """Several calculators added together, e.g. an MLIP plus a dispersion term.

    Built by `calc_a + calc_b`. Provides the intersection of what its parts
    provide, since a property only one of them has cannot be summed.
    """

    def __init__(self, calculators, **kwargs):
        flattened = []
        for calculator in calculators:
            if isinstance(calculator, SumCalculator):
                flattened.extend(calculator.calculators)
            else:
                flattened.append(calculator)
        self.calculators = flattened
        self.provides = frozenset.intersection(
            *(frozenset(c.provides) for c in flattened)
        )
        super().__init__(**kwargs)

    def compute(self, system, want):
        results = [calculator(system, want) for calculator in self.calculators]
        energy = sum(r.energy for r in results)
        forces = stress = energies = None
        if FORCES in want:
            forces = sum(r.forces for r in results)
        if STRESS in want:
            stress = sum(r.stress for r in results)
        if ENERGIES in want:
            energies = sum(r.energies for r in results)
        return Result(energy, forces, stress, energies, volume=system.volume)

    def __repr__(self) -> str:
        return " + ".join(type(c).__name__ for c in self.calculators)


class ShiftedCalculator(Calculator):
    """A calculator with a per-element reference energy removed.

    Lattice and formation energies want energies measured from isolated atoms
    or from some other reference, and the shift is constant for a fixed
    composition, so it changes no derivative.
    """

    def __init__(self, calculator, reference_energies, **kwargs):
        from chmpy.core import Element

        self.calculator = calculator
        self.references = {}
        for key, value in reference_energies.items():
            number = key if isinstance(key, int) else Element[key].atomic_number
            self.references[int(number)] = float(value)
        self.provides = frozenset(calculator.provides)
        super().__init__(**kwargs)

    def _shift(self, system) -> float:
        return sum(self.references.get(int(z), 0.0) for z in system.numbers)

    def compute(self, system, want):
        result = self.calculator(system, want)
        shift = self._shift(system)
        energies = result.energies
        if energies is not None:
            energies = energies - np.array(
                [self.references.get(int(z), 0.0) for z in system.numbers]
            )
        return Result(
            energy=result.energy - shift,
            forces=result.forces,
            stress=result.stress,
            energies=energies,
            volume=result.volume,
            extra=result.extra,
        )

    def __repr__(self) -> str:
        return f"<Shifted {self.calculator!r}>"

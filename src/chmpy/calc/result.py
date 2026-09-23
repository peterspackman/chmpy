"""What a calculator returns, and the names of the things it can be asked for.

Units throughout are eV and Angstroms:

    energy      eV
    energies    eV, per atom, summing to `energy`
    forces      eV/A, `-dE/dr`
    stress      eV/A^3, `(1/V) dE/deps`

`stress` follows the ASE sign convention, where `eps` is the symmetric strain
applied as `r -> (I + eps) r` (so, for row-stored positions and lattice
vectors, `P @ (I + eps)` and `A @ (I + eps)`). A crystal under compression has
a negative `pressure`. `virial = -V * stress = -dE/deps` is the same quantity
without the volume, which is what a machine-learned model usually differentiates
and what the strain degrees of freedom in `chmpy.opt` want.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: total energy, eV
ENERGY = "energy"
#: per-atom energies, eV
ENERGIES = "energies"
#: forces, eV/A
FORCES = "forces"
#: stress, eV/A^3
STRESS = "stress"

#: every property this interface knows about
ALL_PROPERTIES = frozenset({ENERGY, ENERGIES, FORCES, STRESS})

#: eV/A^3 -> GPa
EV_PER_ANGSTROM3_TO_GPA = 160.21766208


def normalise_properties(want) -> frozenset:
    """Validate a requested property set, naming anything unrecognised.

    Args:
        want: a property name, or an iterable of them

    Returns:
        frozenset of property names, always including `energy`
    """
    if isinstance(want, str):
        want = (want,)
    want = frozenset(want) | {ENERGY}
    unknown = want - ALL_PROPERTIES
    if unknown:
        raise ValueError(
            f"unknown propert{'y' if len(unknown) == 1 else 'ies'} "
            f"{sorted(unknown)}; known properties are {sorted(ALL_PROPERTIES)}"
        )
    return want


@dataclass(frozen=True, slots=True)
class Result:
    """The properties a calculator computed for one geometry.

    Anything not asked for is None rather than zero, so a missing property is a
    `TypeError` at the point of use instead of a plausible-looking answer.
    """

    energy: float
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    energies: np.ndarray | None = None
    volume: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("forces", "stress", "energies"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, np.asarray(value, dtype=np.float64))

    @property
    def virial(self) -> np.ndarray:
        "(3, 3) `-dE/deps` in eV, i.e. the stress without the volume"
        if self.stress is None:
            raise ValueError("this result has no stress, so it has no virial")
        return -self.volume * self.stress

    @property
    def stress_voigt(self) -> np.ndarray:
        "(6,) stress as xx, yy, zz, yz, xz, xy"
        s = self.stress
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

    @property
    def stress_gpa(self) -> np.ndarray:
        "(3, 3) stress in GPa"
        return self.stress * EV_PER_ANGSTROM3_TO_GPA

    @property
    def pressure(self) -> float:
        "Hydrostatic pressure in GPa, positive under compression"
        return float(-np.trace(self.stress) / 3.0 * EV_PER_ANGSTROM3_TO_GPA)

    @property
    def fmax(self) -> float:
        "Largest force on any atom, eV/A"
        return float(np.abs(self.forces).max()) if self.forces is not None else 0.0

    @property
    def smax(self) -> float:
        "Largest stress component, GPa"
        return float(np.abs(self.stress_gpa).max()) if self.stress is not None else 0.0

    def has(self, name: str) -> bool:
        "Whether this result carries the named property"
        return getattr(self, name, None) is not None

    def __repr__(self) -> str:
        parts = [f"energy={self.energy:.6f} eV"]
        if self.forces is not None:
            parts.append(f"fmax={self.fmax:.4g} eV/A")
        if self.stress is not None:
            parts.append(f"smax={self.smax:.4g} GPa")
        return f"<Result {', '.join(parts)}>"


@dataclass(frozen=True)
class GradientCheck:
    """How well a calculator's analytic derivatives match finite differences.

    Produced by `Calculator.check_gradients`. `ok` is the verdict; the rest is
    there so a failure says which component and by how much.
    """

    forces_error: float | None
    forces_scale: float | None
    forces_worst: tuple | None
    stress_error: float | None
    stress_scale: float | None
    stress_worst: tuple | None
    tolerance: float
    step: float

    @property
    def ok(self) -> bool:
        """True when every checked derivative is within `tolerance` of numeric.

        The comparison is relative to the largest analytic component, with only
        enough of a floor to survive a geometry where everything is zero. An
        earlier version floored it at 1.0 eV/A, which quietly made the check
        pass for anything near a stationary point -- the very place a gradient
        bug is easiest to hide.
        """
        for error, scale in (
            (self.forces_error, self.forces_scale),
            (self.stress_error, self.stress_scale),
        ):
            if error is None:
                continue
            if error > self.tolerance * max(scale, 1e-9):
                return False
        return True

    def __str__(self) -> str:
        lines = [
            f"gradient check ({'PASS' if self.ok else 'FAIL'}), "
            f"step={self.step:g}, tolerance={self.tolerance:g}"
        ]
        if self.forces_error is not None:
            lines.append(
                f"  forces  max |analytic - numeric| = {self.forces_error:.3e} eV/A"
                f"   (largest force {self.forces_scale:.3e} eV/A,"
                f" worst atom {self.forces_worst[0]} xyz[{self.forces_worst[1]}])"
            )
        if self.stress_error is not None:
            lines.append(
                f"  stress  max |analytic - numeric| = {self.stress_error:.3e} eV/A^3"
                f" (largest stress {self.stress_scale:.3e} eV/A^3,"
                f" worst component {self.stress_worst})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<GradientCheck {'ok' if self.ok else 'FAILED'}>"


@dataclass
class CalculatorStats:
    """Call counts and wall time, so "why is this slow" has an answer.

    Read it off any calculator as `calc.stats`.
    """

    calls: int = 0
    cache_hits: int = 0
    batched_calls: int = 0
    finite_difference_calls: int = 0
    seconds: float = 0.0
    per_property: dict = field(default_factory=dict)

    def record(self, want, seconds: float) -> None:
        self.calls += 1
        self.seconds += seconds
        for name in want:
            self.per_property[name] = self.per_property.get(name, 0) + 1

    def reset(self) -> None:
        self.__init__()

    @property
    def seconds_per_call(self) -> float:
        "Mean wall time of an actual evaluation, excluding cache hits"
        return self.seconds / self.calls if self.calls else 0.0

    def __str__(self) -> str:
        rows = [
            f"{self.calls} evaluations in {self.seconds:.3f} s "
            f"({self.seconds_per_call * 1e3:.2f} ms each)"
        ]
        if self.cache_hits:
            rows.append(f"{self.cache_hits} cache hits")
        if self.batched_calls:
            rows.append(f"{self.batched_calls} batched calls")
        if self.finite_difference_calls:
            rows.append(f"{self.finite_difference_calls} finite-difference evaluations")
        for name in sorted(self.per_property):
            rows.append(f"  {name}: {self.per_property[name]}")
        return "\n".join(rows)

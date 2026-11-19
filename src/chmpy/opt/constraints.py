"""Geometric constraints, shared by chmpy's optimiser and input writers.

`Bond`, `Angle` and `Dihedral` hold an internal coordinate at `value`, or at
its starting value when `value` is None; `Fixed` holds atoms (or some of their
Cartesian components) where they start. Atom indices are 0-based; distances
are in Angstrom and angles in degrees. The same objects go to
`relax(..., constraints=...)` and to `chmpy.fmt.orca.GeomSettings`.

`Constrained` is the optimiser's view: it varies only the directions that
leave every constraint unchanged, from one basis fixed at the start, and
corrects each trial geometry back onto the constraints, so the optimiser sees
a smooth energy of its variables alone. Internal coordinates are measured
between positions as given, without periodic images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chmpy.calc.result import ENERGY, FORCES

from .coordinates import Coordinates

#: residual below which a geometry satisfies its constraints, A or radians
TOLERANCE = 1e-9

#: Gauss-Newton iterations allowed to reach the constraints
MAX_CORRECTIONS = 50

#: finite-difference step for constraint derivatives, A
DERIVATIVE_STEP = 1e-5

#: within this of 0 or 180 degrees an angle has no usable derivative
LINEAR_ANGLE = 1.0


@dataclass(frozen=True)
class Bond:
    """The distance between two atoms, held at `value` or its starting value."""

    i: int
    j: int
    value: float | None = None
    code = "B"

    @property
    def atoms(self):
        return (self.i, self.j)

    def measure(self, positions):
        "Angstrom"
        a, b = (np.asarray(positions[n], dtype=float) for n in self.atoms)
        return float(np.linalg.norm(a - b))

    def _internal(self, positions):
        return self.measure(positions)


@dataclass(frozen=True)
class Angle:
    """The angle i-j-k at atom j, in degrees."""

    i: int
    j: int
    k: int
    value: float | None = None
    code = "A"

    @property
    def atoms(self):
        return (self.i, self.j, self.k)

    def measure(self, positions):
        "degrees"
        a, b, c = (np.asarray(positions[n], dtype=float) for n in self.atoms)
        u, v = a - b, c - b
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def _internal(self, positions):
        return np.radians(self.measure(positions))


@dataclass(frozen=True)
class Dihedral:
    """The torsion i-j-k-l about the j-k bond, in degrees."""

    i: int
    j: int
    k: int
    l: int  # noqa: E741
    value: float | None = None
    code = "D"

    @property
    def atoms(self):
        return (self.i, self.j, self.k, self.l)

    def measure(self, positions):
        "degrees, in (-180, 180], IUPAC sign"
        a, b, c, d = (np.asarray(positions[n], dtype=float) for n in self.atoms)
        b1, b2, b3 = b - a, c - b, d - c
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        y = np.linalg.norm(b2) * np.dot(b1, n2)
        return float(np.degrees(np.arctan2(y, np.dot(n1, n2))))

    def _internal(self, positions):
        return np.radians(self.measure(positions))


@dataclass(frozen=True)
class Fixed:
    """Atoms held at their starting Cartesian positions.

    `axes` restricts this to some components, e.g. "z" to keep atoms in a
    plane while they move within it.
    """

    atoms: tuple
    axes: str = "xyz"

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(sorted({int(a) for a in self.atoms})))
        axes = self.axes.lower()
        if not axes or set(axes) - set("xyz"):
            raise ValueError(f"axes must be drawn from 'xyz', not {self.axes!r}")
        object.__setattr__(self, "axes", "".join(a for a in "xyz" if a in axes))


INTERNAL = (Bond, Angle, Dihedral)


def check_atoms(constraints, n_atoms) -> None:
    """Refuse constraints naming atoms that do not exist, or one atom twice."""
    for entry in constraints:
        bad = [a for a in entry.atoms if not 0 <= a < n_atoms]
        if bad:
            raise ValueError(f"{entry} names atoms {bad}, but there are {n_atoms}")
        if isinstance(entry, INTERNAL) and len(set(entry.atoms)) != len(entry.atoms):
            raise ValueError(f"{entry} repeats an atom")


def _wrapped(entry, difference):
    "A dihedral difference brought into (-pi, pi]; others unchanged"
    if isinstance(entry, Dihedral):
        return -((-difference + np.pi) % (2 * np.pi) - np.pi)
    return difference


class Constrained(Coordinates):
    """Cartesian atomic positions at a fixed cell, subject to constraints.

    The structure is first moved onto the constraints, which only changes it
    when a target `value` differs from the starting one. The variables are
    then the amplitudes along an orthonormal basis of the directions the
    constraints allow there; each trial point is corrected back onto the
    constraints, and the gradient is the force projected onto what the
    constraints allow at the current geometry, so a converged result is a
    constrained minimum.

    Args:
        system: the structure to vary
        constraints: `Bond`, `Angle`, `Dihedral` and `Fixed` entries
        fixed: (N,) boolean mask of atoms to hold still, or None
    """

    wanted = frozenset({ENERGY, FORCES})

    def __init__(self, system, constraints, fixed=None):
        super().__init__(system)
        n = len(system)
        check_atoms(constraints, n)

        free = np.ones((n, 3), dtype=bool)
        if fixed is not None:
            free[np.asarray(fixed, dtype=bool)] = False
        for entry in constraints:
            if isinstance(entry, Fixed):
                for axis in entry.axes:
                    free[list(entry.atoms), "xyz".index(axis)] = False
        self.free = free
        self._components = np.flatnonzero(free.ravel())
        self._column = np.full(3 * n, -1)
        self._column[self._components] = np.arange(len(self._components))

        self.internals = [c for c in constraints if isinstance(c, INTERNAL)]
        positions = np.array(system.positions)
        self._targets = np.array(
            [
                c._internal(positions)
                if c.value is None
                else (c.value if isinstance(c, Bond) else np.radians(c.value))
                for c in self.internals
            ]
        )
        self._check_usable(positions)

        self.anchor = self._correct(positions.ravel()[self._components])
        self._write(self.anchor)
        self.basis = _null_space(self._jacobian(self.anchor), len(self.anchor))
        self._x = np.zeros(self.basis.shape[1])

    @property
    def n_dof(self) -> int:
        return self.basis.shape[1]

    def get(self) -> np.ndarray:
        return self._x.copy()

    def set(self, x) -> None:
        x = np.asarray(x, dtype=float)
        self._write(self._correct(self.anchor + self.basis @ x))
        self._x = x.copy()

    def gradient(self, result) -> np.ndarray:
        return self.basis.T @ self._projected(-result.forces.ravel()[self._components])

    def scale(self) -> np.ndarray:
        return np.ones(self.n_dof)

    def measures(self, result) -> dict:
        """fmax is the largest force component the constraints do not hold"""
        free_forces = self._projected(result.forces.ravel()[self._components])
        return {"fmax": float(np.abs(free_forces).max(initial=0.0))}

    def violation(self) -> np.ndarray:
        "How far each internal constraint is from its target, A or degrees"
        residual = self._residual(self._free_positions())
        return np.array(
            [
                r if isinstance(c, Bond) else np.degrees(r)
                for c, r in zip(self.internals, residual, strict=True)
            ]
        )

    # -- internals -----------------------------------------------------------

    def _free_positions(self) -> np.ndarray:
        return np.array(self.system.positions).ravel()[self._components]

    def _full(self, p) -> np.ndarray:
        positions = np.array(self.system.positions).ravel()
        positions[self._components] = p
        return positions.reshape(-1, 3)

    def _write(self, p) -> None:
        self.system.set_positions(self._full(p))

    def _residual(self, p) -> np.ndarray:
        positions = self._full(p)
        return np.array(
            [
                _wrapped(c, c._internal(positions) - target)
                for c, target in zip(self.internals, self._targets, strict=True)
            ]
        )

    def _jacobian(self, p) -> np.ndarray:
        """d(constraint)/d(free component), by central differences over the
        components of each constraint's own atoms"""
        jacobian = np.zeros((len(self.internals), len(p)))
        for row, entry in enumerate(self.internals):
            for atom in entry.atoms:
                for axis in range(3):
                    column = self._column[3 * atom + axis]
                    if column < 0:
                        continue
                    step = np.zeros_like(p)
                    step[column] = DERIVATIVE_STEP
                    plus = entry._internal(self._full(p + step))
                    minus = entry._internal(self._full(p - step))
                    jacobian[row, column] = _wrapped(entry, plus - minus) / (
                        2 * DERIVATIVE_STEP
                    )
            if not jacobian[row].any():
                raise ValueError(
                    f"{entry} involves only fixed atoms, so it cannot move"
                )
        return jacobian

    def _correct(self, p) -> np.ndarray:
        "The nearby geometry satisfying every internal constraint"
        if not self.internals:
            return p
        for _ in range(MAX_CORRECTIONS):
            residual = self._residual(p)
            if np.abs(residual).max() < TOLERANCE:
                return p
            jacobian = self._jacobian(p)
            p = p - np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        raise ValueError(
            "the constraints could not be satisfied together (largest residual "
            f"{np.abs(self._residual(p)).max():.3g} A or rad); check for "
            "conflicting targets"
        )

    def _projected(self, vector) -> np.ndarray:
        "The part of a free-component vector the constraints allow"
        if not self.internals:
            return vector
        jacobian = self._jacobian(self._free_positions())
        held = np.linalg.lstsq(jacobian.T, vector, rcond=None)[0]
        return vector - jacobian.T @ held

    def _check_usable(self, positions) -> None:
        for entry in self.internals:
            if isinstance(entry, Angle):
                angles = [entry.measure(positions)]
                if entry.value is not None:
                    angles.append(entry.value)
                if any(min(a, 180.0 - a) < LINEAR_ANGLE for a in angles):
                    raise ValueError(
                        f"{entry} is linear, where an angle has no direction "
                        "to hold; fix the atoms or constrain the bonds instead"
                    )


def _null_space(jacobian, n) -> np.ndarray:
    "Orthonormal basis of the directions leaving the constraints unchanged"
    if len(jacobian) == 0:
        return np.eye(n)
    _, singular, vt = np.linalg.svd(jacobian)
    rank = int(np.sum(singular > 1e-8 * singular.max()))
    return vt[rank:].T

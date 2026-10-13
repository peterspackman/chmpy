"""Symmetry of a crystal surface (hkl), derived exactly from the space group.

Operations are re-expressed in a primitive surface basis (a2, b2, w), with
a2, b2 spanning the plane and w one lattice plane up the normal, so that a
fractional coordinate z counts lattice planes. In that basis an operation
keeping the plane orientation has rotation [[A, b], [0, 0, s]] with
s = +1 (normal kept) or s = -1 (normal reversed), and:

- s = +1, t_z integer: maps the semi-infinite crystal onto itself; its
  in-plane part (A, t_xy) is an element of the surface's plane group.
- s = +1, t_z fractional: maps a termination at z = c to one at c + t_z.
- s = -1: exchanges top and bottom, z -> t_z - z.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

import numpy as np

from .integer_lattice import lattice_basis
from .miller_plane import MillerPlane, reduce_hkl
from .plane_group import PlaneGroup, identify_plane_group
from .space_group import SpaceGroup
from .symmetry_operation import SymmetryOperation
from .unit_cell import UnitCell


@dataclass(frozen=True)
class SurfaceOperation:
    """A space group operation expressed in the surface basis.

    Attributes:
        operation: the operation in the bulk (conventional) setting
        rotation: (3, 3) integer rotation in the surface basis
        translation: three `Fraction`s in the surface basis, reduced mod 1
    """

    operation: SymmetryOperation
    rotation: np.ndarray
    translation: tuple[Fraction, Fraction, Fraction]

    @property
    def normal_sign(self) -> int:
        "+1 if the operation keeps the surface normal, -1 if it reverses it."
        return int(self.rotation[2, 2])

    @property
    def plane_operation(self):
        "The in-plane part (A, t_xy) acting on 2D surface coordinates."
        return self.rotation[:2, :2], self.translation[:2]

    def apply(self, coordinates: np.ndarray) -> np.ndarray:
        "Apply to (N, 3) surface fractional coordinates."
        t = np.array([float(x) for x in self.translation])
        return coordinates @ self.rotation.T + t


def _fraction(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(int(x))


def _fraction_matrix(m):
    return np.array([[_fraction(x) for x in row] for row in m], dtype=object)


def _fraction_inverse(m):
    "Inverse of a 3x3 matrix of `Fraction`s by the adjugate."
    m = [[_fraction(x) for x in row] for row in m]
    cof = [
        [
            m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]
            - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3]
            for j in range(3)
        ]
        for i in range(3)
    ]
    det = sum(m[0][j] * cof[0][j] for j in range(3))
    return np.array(
        [[cof[j][i] / det for j in range(3)] for i in range(3)], dtype=object
    )


def primitive_lattice(space_group: SpaceGroup) -> np.ndarray:
    """Rows: a primitive basis of the lattice, in conventional fractional
    coordinates, as `Fraction`s (identity for primitive groups)."""
    centring = [
        op.translation
        for op in space_group.symmetry_operations
        if np.array_equal(np.round(op.rotation), np.eye(3))
    ]
    generators = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]
    for t in centring:
        generators.append([Fraction(round(float(x) * 12), 12) for x in t])
    basis = np.array(lattice_basis(generators), dtype=object)
    if _det(basis) < 0:
        basis[0] *= -1
    return basis


def _det(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _primitive_hkl(hkl, primitive_rows) -> tuple[int, int, int]:
    "Miller indices of the same plane orientation in the primitive basis."
    values = [
        sum(Fraction(h) * q for h, q in zip(hkl, row, strict=True))
        for row in primitive_rows
    ]
    scale = lcm(*(v.denominator for v in values))
    ints = [int(v * scale) for v in values]
    g = gcd(*ints)
    return tuple(x // g for x in ints)


@dataclass
class SurfaceSymmetry:
    """Exact symmetry of the surface (hkl) of a crystal.

    Attributes:
        hkl: reduced Miller indices in the conventional setting
        plane: the plane in the primitive lattice (its `basis` is in
            primitive fractional coordinates)
        surface_vectors: (3, 3) `Fraction` rows (a2, b2, w) in conventional
            fractional coordinates
        operations: all operations keeping the plane orientation, one per
            coset of the lattice, in the surface basis
        plane_group: plane group of the semi-infinite surface
    """

    hkl: tuple[int, int, int]
    plane: MillerPlane
    surface_vectors: np.ndarray
    operations: list[SurfaceOperation]
    plane_group: PlaneGroup

    @classmethod
    def from_space_group(cls, space_group: SpaceGroup, unit_cell: UnitCell, hkl):
        hkl = reduce_hkl(hkl)
        primitive_rows = primitive_lattice(space_group)
        primitive_cell = UnitCell(
            np.array(primitive_rows, dtype=float) @ unit_cell.direct
        )
        plane = MillerPlane(_primitive_hkl(hkl, primitive_rows), primitive_cell)
        surface_vectors = _fraction_matrix(plane.basis).dot(primitive_rows)
        m = surface_vectors.T
        m_inv = _fraction_inverse(m)

        seen = {}
        for op in space_group.symmetry_operations:
            r = _fraction_matrix(np.round(op.rotation).astype(int))
            t = np.array(
                [Fraction(round(float(x) * 12), 12) for x in op.translation],
                dtype=object,
            )
            r_s = m_inv.dot(r).dot(m)
            if r_s[2][0] != 0 or r_s[2][1] != 0:
                continue
            if any(x.denominator != 1 for x in r_s.flat):
                raise ValueError(f"Operation {op} is not integral in the surface basis")
            t_s = tuple(x % 1 for x in m_inv.dot(t))
            rotation = np.array(r_s, dtype=int)
            key = (rotation.tobytes(), t_s)
            if key not in seen:
                seen[key] = SurfaceOperation(op, rotation, t_s)
        operations = list(seen.values())

        in_plane = [
            s.plane_operation
            for s in operations
            if s.normal_sign == 1 and s.translation[2] == 0
        ]
        plane_group = identify_plane_group(in_plane, plane.metric_2d)
        return cls(hkl, plane, surface_vectors, operations, plane_group)

    @classmethod
    def from_crystal(cls, crystal, hkl):
        return cls.from_space_group(crystal.space_group, crystal.unit_cell, hkl)

    @property
    def surface_operations(self) -> list[SurfaceOperation]:
        "Operations of the semi-infinite crystal (normal kept, t_z = 0)."
        return [
            s for s in self.operations if s.normal_sign == 1 and s.translation[2] == 0
        ]

    @property
    def shift_operations(self) -> list[SurfaceOperation]:
        "Operations relating terminations at z = c and z = c + t_z."
        return [
            s for s in self.operations if s.normal_sign == 1 and s.translation[2] != 0
        ]

    @property
    def flip_operations(self) -> list[SurfaceOperation]:
        "Operations exchanging the two sides of the plane, z -> t_z - z."
        return [s for s in self.operations if s.normal_sign == -1]

    @property
    def is_polar_direction(self) -> bool:
        "True when no operation reverses the normal, so (hkl) and (-h-k-l) differ."
        return not self.flip_operations

    def equivalent_offsets(self, offset: float) -> list[float]:
        """Cut offsets (fractions of the layer period) equivalent to `offset`
        on this face, reduced to [0, 1)."""
        return sorted(
            {
                (offset + float(s.translation[2])) % 1.0
                for s in self.operations
                if s.normal_sign == 1
            }
        )

    @property
    def layer_spacing(self) -> float:
        "Spacing between successive lattice planes (Angstroms)."
        return self.plane.d_spacing

    def __repr__(self):
        h, k, l = self.hkl
        return f"<SurfaceSymmetry ({h} {k} {l}) {self.plane_group.symbol}>"

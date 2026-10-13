"""Lattice planes (hkl): exact surface bases and symmetry-equivalent faces.

A face (hkl) is described by a unimodular integer basis (a2, b2, w) of the
lattice in which a2 and b2 span the plane (h.a2 = h.b2 = 0) and w steps one
plane along the normal (h.w = 1). All symmetry arithmetic is on integers.
"""

from __future__ import annotations

from functools import reduce
from math import gcd

import numpy as np

from .space_group import SpaceGroup
from .symmetry_operation import SymmetryOperation
from .unit_cell import UnitCell


def reduce_hkl(hkl) -> tuple[int, int, int]:
    "Miller indices divided by their common factor."
    h = tuple(int(x) for x in hkl)
    if len(h) != 3 or h == (0, 0, 0):
        raise ValueError(f"Miller indices must be three integers, not all zero: {hkl}")
    g = reduce(gcd, h)
    return tuple(x // g for x in h)


def _column_reduce(hkl: tuple[int, int, int]) -> np.ndarray:
    """Unimodular integer U with hkl @ U = (0, 0, 1), hkl primitive."""
    v = np.array(hkl, dtype=int)
    u = np.eye(3, dtype=int)
    while np.count_nonzero(v) > 1:
        nonzero = np.flatnonzero(v)
        p = nonzero[np.argmin(np.abs(v[nonzero]))]
        for j in nonzero:
            if j == p:
                continue
            q = v[j] // v[p]
            v[j] -= q * v[p]
            u[:, j] -= q * u[:, p]
    p = int(np.flatnonzero(v)[0])
    if v[p] < 0:
        u[:, p] *= -1
    order = [i for i in range(3) if i != p] + [p]
    return u[:, order]


def _lagrange_reduce(a, b, metric):
    "Gauss-Lagrange reduction of a 2D integer lattice basis under `metric`."

    def dot(x, y):
        return x @ metric @ y

    while True:
        if dot(b, b) < dot(a, a):
            a, b = b, a
        mu = int(round(dot(a, b) / dot(a, a)))
        if mu == 0:
            return a, b
        b = b - mu * a


def surface_basis(hkl, unit_cell: UnitCell) -> np.ndarray:
    """Integer basis for the lattice plane (hkl), rows (a2, b2, w).

    a2, b2 span the plane and are Lagrange reduced (|a2| <= |b2|,
    |a2.b2| <= |a2|^2 / 2); w satisfies h.w = 1 with the smallest in-plane
    component; the basis is right-handed (determinant +1).
    """
    h = reduce_hkl(hkl)
    metric = unit_cell.metric_tensor
    u = _column_reduce(h)
    a, b = _lagrange_reduce(u[:, 0], u[:, 1], metric)
    w = u[:, 2]

    gram = np.array(
        [[a @ metric @ a, a @ metric @ b], [b @ metric @ a, b @ metric @ b]]
    )
    x, y = np.linalg.solve(gram, [a @ metric @ w, b @ metric @ w])
    candidates = [
        w - i * a - j * b
        for i in (int(np.floor(x)), int(np.ceil(x)))
        for j in (int(np.floor(y)), int(np.ceil(y)))
    ]
    w = min(candidates, key=lambda c: (c @ metric @ c, tuple(-c)))

    basis = np.array([a, b, w], dtype=int)
    if round(np.linalg.det(basis)) < 0:
        basis[1] *= -1
    return basis


class MillerPlane:
    """The lattice plane (hkl) of a unit cell, with its surface basis.

    Attributes:
        hkl: reduced Miller indices
        unit_cell: the bulk unit cell
        basis: (3, 3) integer rows (a2, b2, w) in bulk fractional coordinates
    """

    def __init__(self, hkl, unit_cell: UnitCell):
        self.hkl = reduce_hkl(hkl)
        self.unit_cell = unit_cell
        self.basis = surface_basis(self.hkl, unit_cell)

    @property
    def reciprocal_vector(self) -> np.ndarray:
        "Cartesian reciprocal lattice vector of (hkl), length 1/d."
        return np.array(self.hkl, dtype=float) @ self.unit_cell.reciprocal_lattice

    @property
    def d_spacing(self) -> float:
        "Interplanar spacing in Angstroms."
        return 1.0 / np.linalg.norm(self.reciprocal_vector)

    @property
    def normal(self) -> np.ndarray:
        "Unit Cartesian normal, pointing to the side with h.x > 0."
        g = self.reciprocal_vector
        return g / np.linalg.norm(g)

    @property
    def vectors(self) -> np.ndarray:
        "Cartesian rows (a2, b2, w)."
        return self.basis @ self.unit_cell.direct

    @property
    def area(self) -> float:
        "Area of the 2D surface cell in Angstroms^2."
        a, b, _ = self.vectors
        return float(np.linalg.norm(np.cross(a, b)))

    @property
    def metric_2d(self) -> np.ndarray:
        "Metric tensor of the in-plane lattice (a2, b2)."
        v = self.vectors[:2]
        return v @ v.T

    def surface_frame(self) -> np.ndarray:
        """Orthonormal rows (x, y, z): x along a2, z along the normal.

        `cart @ surface_frame().T` expresses Cartesian coordinates in the
        surface frame.
        """
        a = self.vectors[0]
        x = a / np.linalg.norm(a)
        z = self.normal
        y = np.cross(z, x)
        return np.array([x, y, z])

    def to_surface_fractional(self, frac: np.ndarray) -> np.ndarray:
        "Bulk fractional coordinates expressed in the (a2, b2, w) basis."
        return np.asarray(frac) @ self.inverse_basis

    @property
    def inverse_basis(self) -> np.ndarray:
        "Exact integer inverse of `basis`."
        return np.round(np.linalg.inv(self.basis)).astype(int)

    def __repr__(self):
        h, k, l = self.hkl
        return (
            f"<MillerPlane ({h} {k} {l}) d={self.d_spacing:.4f} area={self.area:.4f}>"
        )


def _image(op: SymmetryOperation, hkl) -> tuple[int, int, int]:
    "Image of face hkl under op: R^-T h."
    r = np.round(op.rotation).astype(int)
    r_inv_t = np.round(np.linalg.inv(r)).astype(int).T
    return tuple(int(x) for x in r_inv_t @ np.asarray(hkl, dtype=int))


def face_images(
    space_group: SpaceGroup, hkl
) -> list[tuple[tuple[int, int, int], SymmetryOperation]]:
    """Distinct faces equivalent to `hkl` under the point group.

    Each is paired with the first operation (R|t) producing it, so a cut at
    offset o on `hkl` maps to offset o + image.t on the image.
    """
    result = {}
    for op in space_group.symmetry_operations:
        image = _image(op, hkl)
        if image not in result:
            result[image] = op
    return list(result.items())


def faces_are_equivalent(space_group: SpaceGroup, hkl1, hkl2) -> bool:
    "Whether two faces belong to the same form."
    target = tuple(int(x) for x in hkl2)
    return any(image == target for image, _ in face_images(space_group, hkl1))


def friedel_mate_is_equivalent(space_group: SpaceGroup, hkl) -> bool:
    """Whether (hkl) and (-h-k-l) belong to the same form.

    True for the centrosymmetric point groups; otherwise the two faces are
    distinct forms with possibly different terminations and energies.
    """
    return faces_are_equivalent(space_group, hkl, tuple(-int(x) for x in hkl))


def _representative(faces) -> tuple[int, int, int]:
    "Preferred face of a form: fewest negative indices, then largest."
    return min(faces, key=lambda f: (sum(x < 0 for x in f), tuple(-x for x in f)))


def unique_faces(
    space_group: SpaceGroup, unit_cell: UnitCell, d_min: float
) -> list[MillerPlane]:
    """One plane per form with d-spacing >= d_min, sorted by decreasing d.

    Only reduced (primitive) Miller indices are considered.
    """
    lengths = np.linalg.norm(unit_cell.direct, axis=1)
    limits = np.floor(lengths / d_min).astype(int)
    seen = set()
    result = []
    ranges = [range(-n, n + 1) for n in limits]
    for h in ranges[0]:
        for k in ranges[1]:
            for l in ranges[2]:
                if (h, k, l) == (0, 0, 0) or gcd(gcd(h, k), l) != 1:
                    continue
                if (h, k, l) in seen:
                    continue
                g = np.array((h, k, l), dtype=float) @ unit_cell.reciprocal_lattice
                if 1.0 / np.linalg.norm(g) < d_min - 1e-9:
                    continue
                form = [image for image, _ in face_images(space_group, (h, k, l))]
                seen.update(form)
                result.append(MillerPlane(_representative(form), unit_cell))
    result.sort(key=lambda p: (-round(p.d_spacing, 8), tuple(-x for x in p.hkl)))
    return result

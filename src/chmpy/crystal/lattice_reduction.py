"""Reduced bases of a 3D lattice: Niggli and Selling (Delaunay).

Both reductions take lattice vectors as rows and return the reduced vectors
with an integer matrix T, |det T| = 1, such that reduced = T @ direct. A
fractional coordinate f in the old basis is f @ inv(T) in the new one.

Niggli reduction gives the unique reduced cell of a lattice (Krivy & Gruber,
Acta Cryst. A32, 297 (1976)), with the tolerance handling of Grosse-Kunstleve,
Sauter & Adams, Acta Cryst. A60, 1 (2004).

Selling reduction works on the four vectors b1, b2, b3, b4 = -(b1 + b2 + b3)
and makes all six scalars b_i . b_j non-positive (Delaunay's reduced form).
It is simpler and needs no special cases on the boundaries between forms
(Andrews, Bernstein & Sauter, Acta Cryst. A75, 115 (2019)).
"""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np

_MAX_STEPS = 1000


class ReducedBasis(NamedTuple):
    """A reduced lattice basis.

    Attributes:
        vectors: (3, 3) reduced lattice vectors as rows
        transform: (3, 3) integer T with vectors = T @ original
    """

    vectors: np.ndarray
    transform: np.ndarray


def _default_eps(vectors: np.ndarray) -> float:
    "Tolerance on squared lengths from the volume, the same in every basis."
    return 1e-5 * float(np.cbrt(abs(np.linalg.det(vectors)))) ** 2


def _check_basis(direct) -> np.ndarray:
    v = np.asarray(direct, dtype=float)
    if v.shape != (3, 3):
        raise ValueError(f"lattice vectors must be a (3, 3) array, got {v.shape}")
    if abs(np.linalg.det(v)) < 1e-12 * np.prod(np.linalg.norm(v, axis=1)):
        raise ValueError("lattice vectors are linearly dependent")
    return v


def _right_handed(v: np.ndarray, t: np.ndarray) -> ReducedBasis:
    # negating all three keeps every dot product, so the reduced form survives
    if np.linalg.det(v) < 0:
        v, t = -v, -t
    return ReducedBasis(v, t)


def _sign(x: float, eps: float) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def niggli_reduce(direct, eps: float | None = None) -> ReducedBasis:
    """Niggli reduced basis of the lattice with the given row vectors.

    Args:
        direct: (3, 3) lattice vectors as rows
        eps: absolute tolerance on the metric (squared lengths); by default
            1e-5 V^(2/3) for cell volume V

    Returns:
        ReducedBasis: the right-handed reduced vectors and the transform

    Raises:
        RuntimeError: if the reduction does not finish
    """
    v = _check_basis(direct).copy()
    t = np.eye(3, dtype=int)
    if eps is None:
        eps = _default_eps(v)

    def swap(i, j):
        v[[i, j]] = v[[j, i]]
        t[[i, j]] = t[[j, i]]

    def add(i, j, s):
        v[i] += s * v[j]
        t[i] += s * t[j]

    for _ in range(_MAX_STEPS):
        g = v @ v.T
        A, B, C = g[0, 0], g[1, 1], g[2, 2]
        xi, eta, zeta = 2 * g[1, 2], 2 * g[0, 2], 2 * g[0, 1]

        # 1, 2: order by length, ties by the off-diagonal terms
        if A > B + eps or (abs(A - B) <= eps and abs(xi) > abs(eta) + eps):
            swap(0, 1)
            continue
        if B > C + eps or (abs(B - C) <= eps and abs(eta) > abs(zeta) + eps):
            swap(1, 2)
            continue

        # 3, 4: off-diagonal terms all positive (type I) or all non-positive
        # (type II); diag(i, j, k) sends xi -> jk xi, eta -> ik eta, zeta -> ij zeta
        signs = [_sign(x, eps) for x in (xi, eta, zeta)]
        if signs[0] * signs[1] * signs[2] == 1:
            flip = [-1 if s == -1 else 1 for s in signs]
        else:
            flip = [-1 if s == 1 else 1 for s in signs]
            if np.prod(flip) == -1:
                flip[max(i for i, s in enumerate(signs) if s == 0)] = -1
        if any(f == -1 for f in flip):
            v *= np.array(flip)[:, None]
            t *= np.array(flip)[:, None]
            continue

        # 5, 6, 7: shorten one vector by another
        if (
            abs(xi) > B + eps
            or (abs(xi - B) <= eps and 2 * eta < zeta - eps)
            or (abs(xi + B) <= eps and zeta < -eps)
        ):
            add(2, 1, -int(np.sign(xi)))
            continue
        if (
            abs(eta) > A + eps
            or (abs(eta - A) <= eps and 2 * xi < zeta - eps)
            or (abs(eta + A) <= eps and zeta < -eps)
        ):
            add(2, 0, -int(np.sign(eta)))
            continue
        if (
            abs(zeta) > A + eps
            or (abs(zeta - A) <= eps and 2 * xi < eta - eps)
            or (abs(zeta + A) <= eps and eta < -eps)
        ):
            add(1, 0, -int(np.sign(zeta)))
            continue

        # 8: the body diagonal a + b + c is shorter than c
        s = xi + eta + zeta + A + B
        if s < -eps or (abs(s) <= eps and 2 * (A + eta) + zeta > eps):
            add(2, 0, 1)
            add(2, 1, 1)
            continue

        return _right_handed(v, t)

    raise RuntimeError(f"Niggli reduction did not finish in {_MAX_STEPS} steps")


def selling_reduce(direct, eps: float | None = None) -> ReducedBasis:
    """Selling (Delaunay) reduced basis of the lattice with the given row vectors.

    The returned vectors are the three shortest of the reduced
    b1, b2, b3, b4, ordered by length, so every scalar between them is
    non-positive.

    Args:
        direct: (3, 3) lattice vectors as rows
        eps: absolute tolerance on the Selling scalars; by default 1e-5
            V^(2/3) for cell volume V

    Returns:
        ReducedBasis: the right-handed reduced vectors and the transform

    Raises:
        RuntimeError: if the reduction does not finish
    """
    v = _check_basis(direct)
    if eps is None:
        eps = _default_eps(v)
    b = np.vstack([v, -v.sum(axis=0)])
    t = np.vstack([np.eye(3, dtype=int), -np.ones((1, 3), dtype=int)])

    for _ in range(_MAX_STEPS):
        s = b @ b.T
        i, j = max(combinations(range(4), 2), key=lambda p: s[p])
        if s[i, j] <= eps:
            break
        # b_i -> -b_i and the other two (not b_j) gain b_i: the sum stays zero
        # and the sum of squared lengths falls by 4 b_i . b_j
        for k in range(4):
            if k not in (i, j):
                b[k] += b[i]
                t[k] += t[i]
        b[i] = -b[i]
        t[i] = -t[i]
    else:
        raise RuntimeError(f"Selling reduction did not finish in {_MAX_STEPS} steps")

    order = np.argsort(np.einsum("ij,ij->i", b, b), kind="stable")[:3]
    return _right_handed(b[order], t[order])

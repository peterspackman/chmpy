"""Vectorised sphere-cast root finders.

Given a centre ``origin`` and a batch of unit directions, find the radial
distance ``r`` at which a scalar field along each ray equals a target
``isovalue``. The shape-descriptor pipeline samples a Gauss-Legendre /
equispaced angular grid of directions and uses the resulting radii as the
input to a spherical-harmonic transform.

Both the promolecule density and the Hirshfeld stockholder weight are
monotonically decreasing along outward rays in the regime of interest, so
the unique root in the bracket ``[lower, upper]`` is found with a fully
vectorised bisection. After ``max_iter`` iterations the bracket width is
``(upper - lower) / 2**max_iter`` — at the default 50 iterations and a
20-Å bracket that's ~2e-14 Å, which is well below surface precision.
"""

from __future__ import annotations

import numpy as np

from . import _backends
from .density import PromoleculeDensity, StockholderWeight, _DOMAIN

__all__ = ["sphere_promolecule_radii", "sphere_stockholder_radii"]


def _bracket_root(eval_f, lower, upper, n_rays, max_iter):
    """Vectorised bisection: find roots of ``eval_f(t)`` per-ray.

    The promolecule density and stockholder weight are monotonically
    decreasing along outward rays in the regime of interest, so each ray has
    a unique root inside the bracket. Bisection has linear convergence (one
    bit per iteration), which is plenty given a 20-Å bracket and ~50 iters
    (final bracket ~2e-14 Å).

    ``eval_f`` takes a length-``n_rays`` array of t values and returns the
    signed function value (already shifted by the target) for each ray.
    Returns an array of roots; rays where ``[lower, upper]`` does not
    bracket a sign change get ``-1.0``.
    """
    a = np.full(n_rays, np.float64(lower))
    b = np.full(n_rays, np.float64(upper))
    fa = eval_f(a)
    fb = eval_f(b)
    no_bracket = (fa * fb) > 0

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = eval_f(m)
        same_side = (fa * fm) > 0
        a = np.where(same_side, m, a)
        fa = np.where(same_side, fm, fa)
        b = np.where(~same_side, m, b)
        fb = np.where(~same_side, fm, fb)

    return np.where(no_bracket, np.float64(-1.0), 0.5 * (a + b))


def _ray_points(origin, directions, t):
    """Build a ``(n_rays, 3)`` array of evaluation points at ``t`` per ray."""
    return origin[np.newaxis, :] + t[:, np.newaxis] * directions


def sphere_promolecule_radii(
    promol: PromoleculeDensity,
    origin,
    directions,
    lower: float,
    upper: float,
    tol: float = 1e-7,
    max_iter: int = 50,
    isovalue: float = 0.0002,
):
    """Find the per-ray distance at which the promolecule density equals ``isovalue``.

    Args:
        promol: A ``PromoleculeDensity`` (the python wrapper, not the cython
            cdef class).
        origin: ``(3,)`` array, the ray start point.
        directions: ``(N, 3)`` array of ray directions (typically unit vectors
            on a sphere).
        lower, upper: bracket the root in radial distance.
        tol: kept for parity with the cython signature; the bisection
            tolerance is set by ``max_iter``.
        max_iter: bisection iterations. 50 gives an absolute tolerance of
            ``(upper - lower) / 2**50``.
        isovalue: target density.

    Returns:
        ``(N,)`` array of radial distances. Rays where the bracket doesn't
        contain a sign change return ``-1.0``.
    """
    del tol  # bisection convergence is set by max_iter alone
    origin = np.asarray(origin, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    iso = np.float32(isovalue)
    positions = promol.positions
    rho_data = promol.rho_data
    n_rays = directions.shape[0]

    def eval_f(t):
        pts = _ray_points(origin, directions, t.astype(np.float32))
        return _backends.rho(positions, rho_data, _DOMAIN, pts).astype(np.float64) - iso

    return _bracket_root(eval_f, lower, upper, n_rays, max_iter)


def sphere_stockholder_radii(
    stock: StockholderWeight,
    origin,
    directions,
    lower: float,
    upper: float,
    tol: float = 1e-7,
    max_iter: int = 50,
    isovalue: float = 0.5,
):
    """Find the per-ray distance at which the stockholder weight equals ``isovalue``."""
    del tol
    origin = np.asarray(origin, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    iso = np.float32(isovalue)
    pos_a = stock.dens_a.positions
    rho_a = stock.dens_a.rho_data
    pos_b = stock.dens_b.positions
    rho_b = stock.dens_b.rho_data
    bg = stock.background
    n_rays = directions.shape[0]

    def eval_f(t):
        pts = _ray_points(origin, directions, t.astype(np.float32))
        w = _backends.weights(pos_a, rho_a, pos_b, rho_b, _DOMAIN, pts, bg)
        return w.astype(np.float64) - iso

    return _bracket_root(eval_f, lower, upper, n_rays, max_iter)

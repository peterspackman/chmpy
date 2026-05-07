"""Sobol quasi-random sequences via scipy.stats.qmc.

Provides the same `quasirandom_sobol(N, D)` and `quasirandom_sobol_batch(start,
end, D)` interface that chmpy used historically (1-indexed seeds, end inclusive),
backed by `scipy.stats.qmc.Sobol`. Output is bit-identical to the previous
cython implementation, which used the same Joe & Kuo direction numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats.qmc import Sobol

__all__ = ["quasirandom_sobol", "quasirandom_sobol_batch"]


def _draw(end: int, D: int) -> np.ndarray:
    """Draw the first ``end`` Sobol points in ``D`` dimensions."""
    sampler = Sobol(d=D, scramble=False)
    # Sobol emits a UserWarning when the requested length is not 2**m; that's
    # an advisory about balance, not correctness, and we don't want it leaking
    # out of every random() call.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return sampler.random(end)


def quasirandom_sobol(N: int, D: int) -> np.ndarray:
    """Return the ``N``th Sobol vector (1-indexed) in ``D`` dimensions."""
    if N < 1:
        raise ValueError("input seed for sobol_vector must be >= 1")
    pts = _draw(N, D)
    return pts[N - 1]


def quasirandom_sobol_batch(start: int, end: int, D: int) -> np.ndarray:
    """Return Sobol vectors for the inclusive seed range ``[start, end]``."""
    if start < 1:
        raise ValueError("input seed for sobol_vector must be >= 1")
    if end < start:
        raise ValueError("end must be >= start")
    pts = _draw(end, D)
    return pts[start - 1 : end]

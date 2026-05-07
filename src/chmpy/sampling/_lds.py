"""Korobov / R_d (a.k.a. KGF) low-discrepancy sequence in pure numpy.

Generates the additive recurrence
    x_n = (offset + n * alpha) mod 1
with ``alpha_i = (1/g)^(i+1)`` and ``g`` the unique positive root of
    g^(D+1) = g + 1
(see https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).
"""

from __future__ import annotations

import numpy as np

__all__ = ["quasirandom_kgf", "quasirandom_kgf_batch"]

_OFFSET = 0.5


def _phi(d: int, iterations: int = 30) -> float:
    """Iterate g <- (1 + g)^(1/(d+1)) to convergence; matches the cython."""
    x = 2.0
    for _ in range(iterations):
        x = pow(1 + x, 1.0 / (d + 1.0))
    return x


def _alpha(d: int) -> np.ndarray:
    g = _phi(d)
    return ((1.0 / g) ** np.arange(1, d + 1)) % 1.0


def quasirandom_kgf(N: int, D: int) -> np.ndarray:
    """Return the ``N``th KGF point (1-indexed) in ``D`` dimensions."""
    a = _alpha(D)
    return (_OFFSET + a * (N + 1)) % 1.0


def quasirandom_kgf_batch(L: int, U: int, D: int) -> np.ndarray:
    """Return KGF points for the inclusive seed range ``[L, U]``."""
    a = _alpha(D)
    n = np.arange(L, U + 1, dtype=np.float64)
    return (_OFFSET + a[np.newaxis, :] * (n[:, np.newaxis] + 1.0)) % 1.0

"""Pure-python rotation-invariant descriptors of spherical harmonic coefficients.

Public functions match the previous cython module:

- ``p_invariants_c(coeffs)`` — invariants from a complex SHT coefficient vector
- ``p_invariants_r(coeffs)`` — invariants from a real (m >= 0) coefficient vector
- ``clebsch_gordan(l1, m1, l2, m2, l, m)`` — convenience wrapper

The Racah formula for Clebsch-Gordan coefficients is computed in scalar python
with an LRU cache so that repeated triples in the invariant loops are free
after the first call. lmax is typically <= 20 for shape descriptors so even
without vectorisation this runs in a few ms.

References:
    Brink & Satchler, "Angular Momentum", Oxford, 1968 (Racah formula)
    Burel & Henocq, "Three-dimensional invariants and their application to
        object recognition", Sig. Proc. 45 (1995) 1-22 (P-invariants)
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial as _factorial, sqrt

import numpy as np

__all__ = ["clebsch_gordan", "p_invariants_c", "p_invariants_r"]

# Precomputed factorial lookup. The Racah formula calls factorial repeatedly
# inside its k-summation; for the lmax ranges chmpy's shape descriptors use
# (typically <= 25, so doubled-j <= 50) a flat lookup is dramatically faster
# than calling math.factorial each time.
_FACT_MAX = 100
_FACT = [float(_factorial(n)) for n in range(_FACT_MAX + 1)]


@lru_cache(maxsize=None)
def _clebsch_doubled(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> float:
    """Clebsch-Gordan coefficient with all arguments pre-doubled (matches cython)."""
    if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.0
    if j1 < 0 or j2 < 0 or j < 0:
        return 0.0
    if abs(j1 - j2) > j or j > j1 + j2:
        return 0.0
    if (m1 + m2) != m:
        return 0.0

    j1nm1 = (j1 - m1) // 2
    jnj2pm1 = (j - j2 + m1) // 2
    j2pm2 = (j2 + m2) // 2
    jnj1nm2 = (j - j1 - m2) // 2
    j1pj2nj = (j1 + j2 - j) // 2

    # Validity: each halved difference must come from an even integer.
    if (
        (j1nm1 * 2) != (j1 - m1)
        or (j2pm2 * 2) != (j2 + m2)
        or (j1pj2nj * 2) != (j1 + j2 - j)
    ):
        return 0.0

    mink = max(-jnj2pm1, -jnj1nm2, 0)
    maxk = min(j1nm1, j2pm2, j1pj2nj)

    iphase = -1 if (mink % 2) else 1
    if mink > maxk:
        res = 1.0
    else:
        res = 0.0
        fact = _FACT
        for k in range(mink, maxk + 1):
            denom = (
                fact[j1nm1 - k]
                * fact[jnj2pm1 + k]
                * fact[j2pm2 - k]
                * fact[jnj1nm2 + k]
                * fact[k]
                * fact[j1pj2nj - k]
            )
            res += iphase / denom
            iphase = -iphase

    fact = _FACT
    norm = sqrt(fact[j1pj2nj])
    norm *= sqrt(fact[(j1 + j - j2) // 2])
    norm *= sqrt(fact[(j2 + j - j1) // 2])
    norm /= sqrt(fact[(j1 + j2 + j) // 2 + 1])
    norm *= sqrt(j + 1)
    norm *= sqrt(fact[(j1 + m1) // 2])
    norm *= sqrt(fact[j1nm1])
    norm *= sqrt(fact[j2pm2])
    norm *= sqrt(fact[(j2 - m2) // 2])
    norm *= sqrt(fact[(j + m) // 2])
    norm *= sqrt(fact[(j - m) // 2])

    return res * norm


def clebsch_gordan(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:
    """Clebsch-Gordan coefficient ⟨l1 m1; l2 m2 | l m⟩."""
    return _clebsch_doubled(2 * l1, 2 * m1, 2 * l2, 2 * m2, 2 * l, 2 * m)


def _coefficient_c(l: int, m: int) -> int:
    """Index into a complex SHT coefficient vector (length (lmax+1)^2)."""
    return (l + 1) * (l + 1) - l + m - 1


def _coefficient_r(l: int, m: int) -> int:
    """Index into a real SHT coefficient vector (length (lmax+1)(lmax+2)/2)."""
    return (l + 1) * (l + 2) // 2 - l + m - 1


def _invariant_p(coeffs, l, l1, l2, *, real: bool):
    """Single ⟨l, l1, l2⟩ invariant; mirrors the cython invariant_P_c / _r."""
    res = 0.0 + 0.0j
    m_lo = 0 if real else -l
    coeff_idx = _coefficient_r if real else _coefficient_c
    for m in range(m_lo, l + 1):
        p = 0.0 + 0.0j
        m1_lo = 0 if real else -l1
        for m1 in range(m1_lo, l1 + 1):
            c = clebsch_gordan(l1, m1, l2, m - m1, l, m)
            if c == 0.0:
                continue
            p += c * coeffs[coeff_idx(l1, m1)] * coeffs[coeff_idx(l2, m - m1)]
        res += p * np.conjugate(coeffs[coeff_idx(l, m)])
    return res


def _signed_cuberoot(x):
    return np.sign(x) * np.cbrt(x)


def _p_invariants_impl(coeffs, l_max: int, *, real: bool):
    even_inv: list[complex] = []
    odd_inv: list[complex] = []
    for l2 in range(1, l_max + 1):
        for l1 in range(l2, l_max + 1):
            for l in range(l1, l_max + 1):
                if (l1 - l2) > l or (l1 + l2) < l:
                    continue
                # Selection: skip terms which are guaranteed zero by parity.
                if not (
                    ((l % 2 == 0) or (l2 != l1))
                    and ((l2 % 2 == 0) or (l1 != l))
                ):
                    continue
                inv = _invariant_p(coeffs, l, l1, l2, real=real)
                (even_inv if (l + l1 + l2) % 2 == 0 else odd_inv).append(inv)
    return np.hstack(
        [
            _signed_cuberoot(np.real(even_inv)),
            _signed_cuberoot(np.imag(odd_inv)),
        ]
    )


def p_invariants_c(coeffs):
    """P-invariants from a complex SHT coefficient vector (length (lmax+1)^2)."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    l_max = int(round(np.sqrt(len(coeffs)))) - 1
    return _p_invariants_impl(coeffs, l_max, real=False)


def p_invariants_r(coeffs):
    """P-invariants from a real (m >= 0) SHT coefficient vector."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    n = len(coeffs)
    l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
    return _p_invariants_impl(coeffs, l_max, real=True)

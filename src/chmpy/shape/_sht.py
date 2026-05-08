"""Numpy implementation of the spherical-harmonic transform kernels.

Replaces the previous cython ``_sht.pyx``. The hot paths are vectorised over
both ``(l, m)`` and the ``theta`` quadrature points: the SHT instance caches a
``(ntheta, nplm)`` matrix of associated-Legendre values once at construction
and the analysis / synthesis kernels become single matrix products.

The pre-existing kernel signatures
``analysis_kernel_real(sht, w, coeffs)`` / ``synthesis_kernel_real(sht, coeffs)``
etc. — which operated on one theta row at a time — are kept as thin
single-row wrappers for any callers that still rely on them, but the
``SHT.analysis`` / ``.synthesis`` paths now use the vectorised entry points
``analyze_real_full`` / ``analyze_cplx_full`` / ``synthesize_real_full`` /
``synthesize_cplx_full`` for speed.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft as scipy_fft
from scipy.fft import ifft as scipy_ifft

__all__ = [
    "AssocLegendre",
    "analysis_kernel_cplx",
    "analysis_kernel_real",
    "analyze_cplx_full",
    "analyze_real_full",
    "build_plm_cache",
    "build_sht_index_tables",
    "expand_coeffs_to_full",
    "synthesis_kernel_cplx",
    "synthesis_kernel_real",
    "synthesize_cplx_full",
    "synthesize_real_full",
]


# ---------------------------------------------------------------------------
# Associated Legendre polynomials (un-phased, normalised for SHT use)
# ---------------------------------------------------------------------------

class AssocLegendre:
    """Evaluate normalised associated Legendre polynomials P_l^m(x).

    Output ordering matches the previous cython class: index ``plm_idx``
    corresponds to ``(l, m)`` enumerated as ``for m in 0..lmax: for l in m..lmax``.
    No (-1)^m phase factor is included; the SHT kernels apply it externally.
    """

    def __init__(self, lmax: int):
        self.lmax = lmax
        n = lmax + 1
        a = np.zeros((n, n), dtype=np.float64)
        b = np.zeros((n, n), dtype=np.float64)
        for m in range(n):
            amm = 1.0
            for k in range(1, abs(m) + 1):
                amm *= (2.0 * k + 1) / (2.0 * k)
            a[m, m] = np.sqrt(amm / (4.0 * np.pi))
            for l in range(abs(m) + 1, n):
                a[l, m] = np.sqrt((4 * l * l - 1) / (1.0 * l * l - m * m))
                b[l, m] = -np.sqrt(
                    (2.0 * l + 1) * ((l - 1) * (l - 1) - m * m)
                    / ((2.0 * l - 3) * (l * l - m * m))
                )
        self._a = a
        self._b = b

    def evaluate_batch(self, x, result: np.ndarray | None = None) -> np.ndarray:
        """Evaluate all P_l^m(x) at scalar or vector ``x``.

        For scalar ``x`` returns a length-``nplm`` vector in plm_idx order
        (compatible with the previous cython API). For 1-D ``x`` it returns a
        ``(len(x), nplm)`` matrix; the underlying recurrence runs once over
        the entire batch using numpy element-wise ops.
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        n = self.lmax + 1
        nplm = n * (n + 1) // 2
        out = np.empty((x_arr.shape[0], nplm), dtype=np.float64)
        cache = np.zeros((x_arr.shape[0], n, n), dtype=np.float64)
        x2 = 1.0 - x_arr * x_arr
        idx = 0
        for m in range(n):
            for l in range(m, n):
                if l == m:
                    val = self._a[l, m] * (x2 ** (0.5 * m))
                elif l == m + 1:
                    val = self._a[l, m] * x_arr * cache[:, l - 1, m]
                else:
                    val = (
                        self._a[l, m] * x_arr * cache[:, l - 1, m]
                        + self._b[l, m] * cache[:, l - 2, m]
                    )
                cache[:, l, m] = val
                out[:, idx] = val
                idx += 1
        if np.ndim(x) == 0:
            scalar = out[0]
            if result is not None:
                result[:] = scalar
                return result
            return scalar
        return out


def build_plm_cache(lmax: int, cos_theta: np.ndarray) -> np.ndarray:
    """Evaluate Plm at every quadrature point. Returns shape ``(ntheta, nplm)``."""
    return AssocLegendre(lmax).evaluate_batch(cos_theta)


# ---------------------------------------------------------------------------
# Pre-computed index tables for vectorised SHT kernels
# ---------------------------------------------------------------------------

def build_sht_index_tables(lmax: int, nphi: int) -> dict[str, np.ndarray]:
    """Pre-compute index tables used by the vectorised kernels."""
    nplm = (lmax + 1) * (lmax + 2) // 2
    m_values = np.empty(nplm, dtype=np.int64)
    l_values = np.empty(nplm, dtype=np.int64)
    idx = 0
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            m_values[idx] = m
            l_values[idx] = l
            idx += 1

    signs = np.where(m_values & 1, -1.0, 1.0)
    m_factor_real = np.where(m_values == 0, 1.0, 2.0)
    cplx_l_offset_pos = l_values * (l_values + 1) + m_values
    cplx_l_offset_neg = l_values * (l_values + 1) - m_values
    fft_idx_pos = m_values % nphi
    fft_idx_neg = (nphi - m_values) % nphi

    return {
        "m_values": m_values,
        "l_values": l_values,
        "signs": signs,
        "m_factor_real": m_factor_real,
        "cplx_l_offset_pos": cplx_l_offset_pos,
        "cplx_l_offset_neg": cplx_l_offset_neg,
        "fft_idx_pos": fft_idx_pos,
        "fft_idx_neg": fft_idx_neg,
        "is_m0": m_values == 0,
        "is_mge1": m_values >= 1,
    }


# ---------------------------------------------------------------------------
# Fully-vectorised analysis / synthesis (the fast paths)
# ---------------------------------------------------------------------------

def analyze_real_full(values, weights, plm_cache, tables):
    """Real analysis vectorised across all theta. Returns (nplm,) complex."""
    fft_all = scipy_fft(values, norm="forward", axis=1)  # (ntheta, nphi) complex
    weighted = fft_all * weights[:, None]
    # For each plm_idx, gather fft at the right m bin: shape (ntheta, nplm)
    fft_at_m = weighted[:, tables["fft_idx_pos"]]
    # Multiply by Plm (real) and reduce over theta
    products = plm_cache * fft_at_m
    return tables["signs"] * products.sum(axis=0)


def analyze_cplx_full(values, weights, plm_cache, tables):
    """Complex analysis vectorised across all theta. Returns ((lmax+1)^2,) complex."""
    fft_all = scipy_fft(values, norm="forward", axis=1)
    weighted = fft_all * weights[:, None]
    nlm = (tables["m_values"].max() + 1) ** 2
    out = np.zeros(nlm, dtype=np.complex128)

    # m == 0
    is_m0 = tables["is_m0"]
    if is_m0.any():
        plm_g = plm_cache[:, is_m0]
        contrib = (plm_g * weighted[:, 0:1]).sum(axis=0)
        np.add.at(out, tables["cplx_l_offset_pos"][is_m0], contrib)

    # m >= 1
    is_mge1 = tables["is_mge1"]
    if is_mge1.any():
        signs = tables["signs"][is_mge1]
        m_vals = tables["m_values"][is_mge1]
        plm_g = plm_cache[:, is_mge1]

        fft_pos_g = weighted[:, tables["fft_idx_pos"][is_mge1]]
        fft_neg_g = weighted[:, tables["fft_idx_neg"][is_mge1]]
        rr = signs * (plm_g * fft_pos_g).sum(axis=0)
        ii = signs * (plm_g * fft_neg_g).sum(axis=0)
        ii = np.where(m_vals & 1, -ii, ii)

        np.add.at(out, tables["cplx_l_offset_pos"][is_mge1], rr)
        np.add.at(out, tables["cplx_l_offset_neg"][is_mge1], ii)

    return out


def synthesize_real_full(coeffs, plm_cache, tables, nphi):
    """Real synthesis vectorised across all theta. Returns (ntheta, nphi)."""
    weighted = tables["m_factor_real"] * tables["signs"] * coeffs
    # products[theta, plm_idx] = plm_cache[theta, plm_idx] * weighted[plm_idx]
    products = plm_cache * weighted[None, :]
    ntheta = plm_cache.shape[0]
    fft_buf = np.zeros((ntheta, nphi), dtype=np.complex128)
    np.add.at(fft_buf, (slice(None), tables["fft_idx_pos"]), products)
    return scipy_ifft(fft_buf, norm="forward", axis=1).real


def synthesize_cplx_full(coeffs, plm_cache, tables, nphi):
    """Complex synthesis vectorised across all theta. Returns (ntheta, nphi)."""
    ntheta = plm_cache.shape[0]
    fft_buf = np.zeros((ntheta, nphi), dtype=np.complex128)

    is_m0 = tables["is_m0"]
    if is_m0.any():
        l_off_m0 = tables["cplx_l_offset_pos"][is_m0]
        plm_g = plm_cache[:, is_m0]
        contrib_m0 = (coeffs[l_off_m0] * plm_g).sum(axis=1)
        fft_buf[:, 0] += contrib_m0

    is_mge1 = tables["is_mge1"]
    if is_mge1.any():
        signs = tables["signs"][is_mge1]
        m_vals = tables["m_values"][is_mge1]
        plm_g = plm_cache[:, is_mge1]

        rr = (signs * coeffs[tables["cplx_l_offset_pos"][is_mge1]]) * plm_g
        ii_signed = signs * coeffs[tables["cplx_l_offset_neg"][is_mge1]]
        ii_signed = np.where(m_vals & 1, -ii_signed, ii_signed)
        ii = ii_signed * plm_g

        np.add.at(fft_buf, (slice(None), tables["fft_idx_pos"][is_mge1]), rr)
        np.add.at(fft_buf, (slice(None), tables["fft_idx_neg"][is_mge1]), ii)

    return scipy_ifft(fft_buf, norm="forward", axis=1)


# ---------------------------------------------------------------------------
# Single-theta kernels — kept for any callers that still need the row API.
# ---------------------------------------------------------------------------

def analysis_kernel_real(sht, w, coeffs):
    """Single-theta real-analysis update: matches the legacy cython kernel."""
    tables = sht._sht_tables
    fft_at_m = sht.fft_work_array[tables["fft_idx_pos"]]
    coeffs += tables["signs"] * fft_at_m * sht.plm_work_array * w


def analysis_kernel_cplx(sht, w, coeffs):
    """Single-theta complex-analysis update."""
    tables = sht._sht_tables
    plm = sht.plm_work_array
    fft = sht.fft_work_array

    is_m0 = tables["is_m0"]
    if is_m0.any():
        l_off_m0 = tables["cplx_l_offset_pos"][is_m0]
        contrib_m0 = fft[0] * plm[is_m0] * w
        np.add.at(coeffs, l_off_m0, contrib_m0)

    is_mge1 = tables["is_mge1"]
    if is_mge1.any():
        signs = tables["signs"][is_mge1]
        plm_g = plm[is_mge1]
        fft_pos = fft[tables["fft_idx_pos"][is_mge1]]
        fft_neg = fft[tables["fft_idx_neg"][is_mge1]]
        m_vals = tables["m_values"][is_mge1]

        pw = plm_g * w
        rr = signs * fft_pos * pw
        ii = signs * fft_neg * pw
        ii = np.where(m_vals & 1, -ii, ii)

        np.add.at(coeffs, tables["cplx_l_offset_pos"][is_mge1], rr)
        np.add.at(coeffs, tables["cplx_l_offset_neg"][is_mge1], ii)


def synthesis_kernel_real(sht, coeffs):
    tables = sht._sht_tables
    contrib = tables["m_factor_real"] * tables["signs"] * coeffs * sht.plm_work_array
    np.add.at(sht.fft_work_array, tables["fft_idx_pos"], contrib)


def synthesis_kernel_cplx(sht, coeffs):
    tables = sht._sht_tables
    plm = sht.plm_work_array
    fft = sht.fft_work_array

    is_m0 = tables["is_m0"]
    if is_m0.any():
        l_off_m0 = tables["cplx_l_offset_pos"][is_m0]
        fft[0] += (coeffs[l_off_m0] * plm[is_m0]).sum()

    is_mge1 = tables["is_mge1"]
    if is_mge1.any():
        signs = tables["signs"][is_mge1]
        plm_g = plm[is_mge1]
        m_vals = tables["m_values"][is_mge1]

        rr = signs * coeffs[tables["cplx_l_offset_pos"][is_mge1]] * plm_g
        ii = signs * coeffs[tables["cplx_l_offset_neg"][is_mge1]] * plm_g
        ii = np.where(m_vals & 1, -ii, ii)

        np.add.at(fft, tables["fft_idx_pos"][is_mge1], rr)
        np.add.at(fft, tables["fft_idx_neg"][is_mge1], ii)


# ---------------------------------------------------------------------------
# Real -> complex coefficient expansion
# ---------------------------------------------------------------------------

def expand_coeffs_to_full(lmax: int, coeffs: np.ndarray) -> np.ndarray:
    """Expand a real-mode SHT coefficient vector to its complex equivalent."""
    out = np.zeros((lmax + 1) * (lmax + 1), dtype=np.complex128)
    plm_idx = 0
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            l_offset = l * (l + 1)
            out[l_offset + m] = coeffs[plm_idx]
            if m != 0:
                sign = -1 if m & 1 else 1
                out[l_offset - m] = sign * np.conjugate(coeffs[plm_idx])
            plm_idx += 1
    return out

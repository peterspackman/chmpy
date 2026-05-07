"""Pure-numpy backend for promolecule density evaluation.

The dispatch indirection is kept so that an alternative fast backend (e.g.
``occpy``) can be slotted in via ``CHMPY_INTERP_BACKEND`` without touching
``density.py``. For now there's only the numpy implementation.

Public entry points:

- ``rho(positions, rho_data, domain, pts)`` -> sum of atomic densities at ``pts``
- ``weights(pos_a, rho_a, pos_b, rho_b, domain, pts, background)`` -> Hirshfeld
  stockholder weights
"""

from __future__ import annotations

import os

import numpy as np

_BOHR_PER_ANGSTROM = 0.5291772108
_INV_BOHR2 = 1.0 / (_BOHR_PER_ANGSTROM * _BOHR_PER_ANGSTROM)


def _rho_numpy(positions, rho_data, domain, pts):
    """Pure-numpy sum of atomic densities sampled at ``pts``.

    Uniform-grid linear interpolation is computed manually so that index
    arithmetic happens once per atom; matches the behaviour of the legacy
    cython ``interp_f`` (lower fill ``yi[0]`` for ``j <= 0``, upper fill
    ``0`` for ``j >= ni - 1``).
    """
    positions = np.asarray(positions, dtype=np.float32)
    rho_data = np.asarray(rho_data, dtype=np.float32)
    domain = np.asarray(domain, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)

    lbound = np.float32(domain[0])
    inv_dx = np.float32(1.0 / (domain[1] - domain[0]))
    inv_a2 = np.float32(_INV_BOHR2)
    ni = domain.shape[0]
    one = np.float32(1.0)

    rho = np.zeros(pts.shape[0], dtype=np.float32)
    for i in range(positions.shape[0]):
        diff = pts - positions[i]
        r = (diff * diff).sum(axis=1) * inv_a2
        idx_f = inv_dx * (r - lbound)
        j = idx_f.astype(np.int32)
        j_clip = np.clip(j, 0, ni - 2)
        t = idx_f - j_clip
        yi = rho_data[i]
        contrib = (one - t) * yi[j_clip] + t * yi[j_clip + 1]
        np.putmask(contrib, j <= 0, yi[0])
        np.putmask(contrib, j >= ni - 1, np.float32(0.0))
        rho += contrib
    return rho


def _weights_numpy(pos_a, rho_data_a, pos_b, rho_data_b, domain, pts, background):
    rho_a = _rho_numpy(pos_a, rho_data_a, domain, pts)
    rho_b = _rho_numpy(pos_b, rho_data_b, domain, pts)
    return rho_a / (rho_a + rho_b + np.float32(background))


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _select_backend():
    forced = os.environ.get("CHMPY_INTERP_BACKEND", "").lower()
    if forced and forced != "numpy":
        raise ValueError(
            f"CHMPY_INTERP_BACKEND={forced!r} not recognised; only 'numpy' is "
            "currently supported."
        )
    return "numpy"


BACKEND = _select_backend()
rho = _rho_numpy
weights = _weights_numpy


__all__ = ["BACKEND", "rho", "weights"]

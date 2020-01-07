from .density import StockholderWeight
from .util import spherical_to_cartesian
from scipy.optimize import minimize_scalar
from ._density import sphere_stockholder_radii
from ._invariants import p_invariants_c, p_invariants_r
import logging
import numpy as np

LOG = logging.getLogger(__name__)
_HAVE_WARNED_ABOUT_LMAX_P = False


def make_N_invariants(coefficients, kind="real"):
    """Construct the 'N' type invariants from sht coefficients.
    If coefficients is of length n, the size of the result will be sqrt(n)

    Arguments:
    coefficients -- the set of spherical harmonic coefficients
    """
    if kind == "complex":
        size = int(np.sqrt(len(coefficients)))
        invariants = np.empty(shape=(size), dtype=np.float64)
        for i in range(0, size):
            lower, upper = i ** 2, (i + 1) ** 2
            invariants[i] = np.sum(
                coefficients[lower : upper + 1]
                * np.conj(coefficients[lower : upper + 1])
            ).real
        return np.sqrt(invariants)
    else:
        # n = (l_max +2)(l_max+1)/2
        n = len(coefficients)
        size = int((-3 + np.sqrt(8 * n + 1)) // 2) + 1
        lower = 0
        invariants = np.empty(shape=(size), dtype=np.float64)
        for i in range(0, size):
            x = i + 1
            upper = lower + x
            invariants[i] = np.sum(
                coefficients[lower : upper + 1]
                * np.conj(coefficients[lower : upper + 1])
            ).real
            lower += x
        return np.sqrt(invariants)


def make_invariants(l_max, coefficients, kinds="NP"):
    global _HAVE_WARNED_ABOUT_LMAX_P
    invariants = []
    if "N" in kinds:
        invariants.append(make_N_invariants(coefficients))
    if "P" in kinds:
        # Because we only have factorial precision in our
        # clebsch implementation up to 70! l_max for P type
        # invariants is restricted to < 23
        if l_max > 23:
            if not _HAVE_WARNED_ABOUT_LMAX_P:
                LOG.warn(
                    "P type invariants only supported up to l_max = 23: "
                    "will only using N type invariants beyond that."
                )
                _HAVE_WARNED_ABOUT_LMAX_P = True
            c = coefficients[:(25 * 24)//2]
            invariants.append(p_invariants_r(c))
        else:
            invariants.append(p_invariants_r(coefficients))
    return np.hstack(invariants)


def stockholder_weight_descriptor(sht, n_i, p_i, n_e, p_e, **kwargs):
    isovalue = kwargs.get("isovalue", 0.5)
    r_min, r_max = kwargs.get("bounds", (0.1, 20.0))
    s = StockholderWeight.from_arrays(n_i, p_i, n_e, p_e)
    g = np.empty(sht.grid.shape, dtype=np.float32)
    g[:, :] = sht.grid[:, :]
    o = kwargs.get("origin", np.mean(p_i, axis=0, dtype=np.float32))
    r = sphere_stockholder_radii(s.s, o, g, r_min, r_max, 1e-7, 30)
    l_max = sht.l_max
    return make_invariants(l_max, sht.analyse(r))


def stockholder_weight_descriptor_slow(sht, n_i, p_i, n_e, p_e, **kwargs):
    isovalue = kwargs.get("isovalue", 0.5)
    s = StockholderWeight.from_arrays(n_i, p_i, n_e, p_e)
    g = sht.grid

    r = np.empty(len(g), dtype=np.float32)
    for i, (phi, theta) in enumerate(g):
        rtp = np.array([[1.0, theta, phi]], dtype=np.float32)
        v = spherical_to_cartesian(rtp, dtype=np.float32)

        def f(r):
            xyz = np.array(p_i + r * v, dtype=np.float32)
            return abs(s.weights(xyz)[0] - 0.5)

        result = minimize_scalar(
            f,
            bounds=np.array([0.2, 4.0], dtype=np.float32),
            method="bounded",
            options=dict(xatol=1e-3),
        )
        r[i] = abs(result.x)

    print(np.min(r), np.max(r))
    return make_invariants(sht.analyse(np.array(r, dtype=np.float64)))

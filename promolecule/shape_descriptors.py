from .density import StockholderWeight
from .util import spherical_to_cartesian
from scipy.optimize import minimize_scalar
import numpy as np


def make_invariants(coefficients, kind='real'):
    """Construct the 'N' type invariants from sht coefficients.
    If coefficients is of length n, the size of the result will be sqrt(n)

    Arguments:
    coefficients -- the set of spherical harmonic coefficients
    """
    if kind == 'complex':
        size = int(np.sqrt(len(coefficients)))
        invariants = np.empty(shape=(size), dtype=np.float64)
        for i in range(0, size):
            lower, upper = i**2, (i+1)**2
            invariants[i] = np.sum(coefficients[lower:upper+1] *
                                   np.conj(coefficients[lower:upper+1])).real
        return invariants
    else:
        # n = (l_max +2)(l_max+1)/2
        n = len(coefficients)
        size = int((-3 + np.sqrt(8*n + 1))//2) + 1
        lower = 0
        invariants = np.empty(shape=(size), dtype=np.float64)
        for i in range(0, size):
            x = i + 1
            upper = lower + x
            invariants[i] = np.sum(
                coefficients[lower:upper+1] *
                np.conj(coefficients[lower:upper+1])
            ).real
            lower += x
        return invariants



def stockholder_weight_descriptor(sht, n_i, p_i, n_e, p_e, **kwargs):
    isovalue = kwargs.get("isovalue", 0.5)
    s = StockholderWeight.from_arrays(n_i, p_i, n_e, p_e)
    g = sht.grid
    l, u = 0.2, 5.0
    n = 100
    rvals = np.logspace(-2, 3, n, base=np.e)
    sep = (l - u) / n
    pts = np.vstack([
        np.c_[np.ones(len(g)) * r, g[:, 1], g[:, 0]]
        for r in rvals
    ])
    f = s.weight(spherical_to_cartesian(pts) + p_i[0]).reshape((n, -1))

    r = np.empty(len(g))
    for i in range(len(g)):
        idx = n - (np.searchsorted(f[::-1, i], 0.5, side='left') + 1)
        x2 = f[idx, i]
        x1 = f[idx - 1, i]
        y2 = rvals[idx]
        y1 = rvals[idx - 1]
        grad = (y2 - y1)/(x2 - x1)
        r[i] = y1 + grad * (0.5 - x1)
    print(np.min(r), np.max(r))
    return make_invariants(sht.analyse(r))



def stockholder_weight_descriptor_slow(sht, n_i, p_i, n_e, p_e, **kwargs):
    isovalue = kwargs.get("isovalue", 0.5)
    s = StockholderWeight.from_arrays(n_i, p_i, n_e, p_e)
    g = sht.grid

    r = np.empty(len(g))
    for i, (phi, theta) in enumerate(g):
        rtp = np.array([[1.0, theta, phi]])
        v = spherical_to_cartesian(rtp)
        def f(r):
            rtp = np.array([[r, theta, phi]])
            xyz = p_i + r * v
            return abs(s.weight(xyz)[0] - 0.5)

        result = minimize_scalar(f, bounds=np.array([0.2, 4.0]), method='bounded', options=dict(xatol=1e-3))
        r[i] = abs(result.x)

    print(np.min(r), np.max(r))
    return make_invariants(sht.analyse(r))

from .density import StockholderWeight
from .util import spherical_to_cartesian
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
    l, u = 0.2, 3.2
    n = 100
    rvals = np.linspace(l, u, n)
    sep = (l - u) / n
    pts = np.vstack([
        np.c_[np.ones(len(g)) * r, g[:, 1], g[:, 0]]
        for r in rvals
    ])
    f = s.weight(spherical_to_cartesian(pts) + p_i[0]).reshape((n, -1))

    r = np.empty(len(g))
    for i in range(len(g)):
        idx = n - np.searchsorted(f[::-1, i], 0.5, side='left')
        x2 = f[idx - 1, i]
        x1 = f[idx, i]
        y2 = rvals[idx]
        y1 = rvals[idx - 1]
        grad = (y2 - y1)/(x2 - x1)
        r[i] = y1 + grad * (0.5 - x1)
    print(r)
    return make_invariants(sht.analyse(r))

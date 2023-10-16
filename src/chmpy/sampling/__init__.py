"""
This module is dedicated to sampling points, sequences and
generation of random or quasi random object.
"""

from ._lds import quasirandom_kgf, quasirandom_kgf_batch
from ._sobol import quasirandom_sobol, quasirandom_sobol_batch
import numpy as np

__all__ = [
    "quasirandom_kgf",
    "quasirandom_kgf_batch",
    "quasirandom_sobol",
    "quasirandom_sobol_batch",
]

_BATCH = {
    "sobol": quasirandom_sobol_batch,
    "kgf": quasirandom_kgf_batch,
}

_SINGLE = {
    "sobol": quasirandom_sobol,
    "kgf": quasirandom_kgf,
}


def quasirandom(d1: int, d2=None, method="sobol", seed=1) -> np.ndarray:
    """
    Generate a quasirandom point, or sequence of points with coefficients
    in the interval [0, 1].

    Args:
        d1 (int): number of points to generate (or number of dimensions if d2 is not provided)
        d2 (int, optional): number of dimensions
        method (str, optional): use the 'sobol' or 'kgf' sequences to generate points
        seed (int, optional): start seed for the sequence of numbers. if more than 1 point is
            generated then the seeds will be in the range [seed, seed + d1 -1] corresponding
            to each point in the resulting sequence.

    Returns:
        np.ndarray: The sequence of quasirandom vectors
    """
    if d2 is None:
        return _SINGLE[method](seed, d1)
    else:
        return _BATCH[method](seed, seed + d1 - 1, d2)

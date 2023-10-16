# cython: language_level=3, boundscheck=False, wraparound=False
cimport numpy as cnp
cimport cython
import numpy as np
from libc.math cimport pow

cnp.import_array()

@cython.cdivision(True)
cdef double phi(const unsigned int d) noexcept nogil:
    cdef int iterations = 30
    cdef int i
    cdef double x = 2.0
    for i in range(iterations):
        x = pow(1 + x, 1.0 / (d + 1.0))
    return x

@cython.cdivision(True)
cdef void alpha(double[::1] a) noexcept nogil:
    cdef int dims = a.shape[0]
    cdef double g = phi(dims)
    cdef int i
    for i in range(dims):
        a[i] = pow(1 / g, i + 1) % 1


cpdef quasirandom_kgf(const unsigned int N, const unsigned int D):
    """
    Generate an D dimensional Korobov type quasi-random vector
    based on the generalized Fibonacci sequence.

    Based on the R_1, R_2 sequences available here:
        `https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/`

    Parameters:
        N (int): seed
        D (int): number of dimensions

    Returns:
        np.ndarray: an (D) dimensional sampling point
    """
    cdef double offset = 0.5
    cdef cnp.ndarray[cnp.float64_t, ndim=1] a = np.empty(D, dtype=np.float64)
    cdef double[::1] a_view = a
    with nogil:
        alpha(a_view)
    return (offset + a * (N + 1)) % 1

cpdef quasirandom_kgf_batch(const unsigned int L, const unsigned int U, const unsigned int D):
    """
    Generate a batch of D dimensional Korobov type quasi-random vectors
    based on the generalized Fibonacci sequence.

    Based on the R_1, R_2 sequences available here:
        `https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/`

    Parameters:
        L (int): Start seed
        U (int): End seed
        D (int): number of dimensions

    Returns:
        np.ndarray: an (U - L + 1, D) dimensional array of sampling points
    """
    cdef double offset = 0.5
    cdef cnp.ndarray[cnp.float64_t, ndim=1] a = np.empty(D, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.empty((U - L + 1, D), dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] N = np.arange(L, U + 1, dtype=np.int32)
    cdef double[::1] a_view = a
    with nogil:
        alpha(a_view)
    return (offset + a[np.newaxis, :] * (N[:, np.newaxis] + 1)) % 1

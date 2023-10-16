# cython: language_level=3, boundscheck=False, wraparound=False
from pathlib import Path
from libc.math cimport ceil, log, pow
cimport cython
cimport numpy as np
import numpy as np

__all__ = ["quasirandom_sobol", "quasirandom_sobol_batch"]

def _load_data():
    return np.load((Path(__file__).parent / "_sobol_parameters.npz"))["poly"]

_SOBOL_DATA = _load_data()

@cython.cdivision(True)
cpdef quasirandom_sobol(unsigned int N, unsigned int D):
    """
    Generate a sobol vector with dimension `D` and seed `N`.
    Sobol sequence generation based on`[1, 2]`.

    Parameters:
        N (int): sobol seed
        D (int): sobol vector dimension

    Returns:
        np.ndarray: the sampling vector corresponding to seed `N`

    Data in `_sobol_parameters.npz` is a modified version of polynomial
    coefficient table based on the D(6) criteria up to dimension 21201, where
    the $d$ value is implicit in the 1st index, the $a$ value is the at
    position 0 in the 2nd index and $s$ is implicit in the nonzero polynomial
    coefficients.

    Coefficient data retrieved from:
        `https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201`
    Retrieval date:
        2019/10/25

    References:
    ```
    S. Joe & F.Y. Kuo, ACM Trans. Math. Softw., 29, 49-57 (2003)
    S. Joe & F.Y. Kuo, SIAM J. Sci. Comput., 30, 2635-2654 (2008)
    ```
    """
    assert N > 0, "input seed for sobol_vector must be > 0"
    cdef double base = 2.0
    cdef unsigned int bitwidth = 32
    cdef unsigned int L = <unsigned>(ceil(log(<double>(N))/log(base)))
    cdef np.ndarray[np.uint32_t, ndim=1] C_arr = np.empty(N, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] V_arr = np.empty(L+1, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] X_arr = np.empty(N, dtype=np.uint32)
    cdef np.ndarray[np.float64_t, ndim=1] pts = np.zeros(D, dtype=np.float64)
    cdef unsigned int[::1] C = C_arr, V = V_arr, X = X_arr
    cdef double[::1] pts_view = pts
    cdef const unsigned int[:, ::1] poly = _SOBOL_DATA
    cdef const unsigned int[::1] m
    cdef unsigned int i, j, k
    cdef unsigned int value, s, a
    with nogil:
        # C[i] = index from the right of the first zero bit of i
        C[0] = 1
        for i in range(1, N):
            C[i] = 1
            value = i
            while ((value & 1U) != 0U):
                value >>= 1U
                C[i] += 1

        # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
        for i in range(1, L + 1):
            V[i] = 1U << (bitwidth - i)  # all m's = 1

        # Evalulate X[0] to X[N-1], scaled by pow(2,32)
        X[0] = 0
        for i in range(1, N):
            X[i] = X[i-1] ^ V[C[i-1]];

        pts_view[0] = <double>(X[N-1]) / pow(base, bitwidth) 
        for j in range(1, D):
            m = poly[j+1]
            # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
            a = m[0]
            for s in range(1, m.shape[0]):
                if m[s] == 0:
                    break
            s = s - 1
            if (L <= s):
                for i in range(1, L + 1):
                    V[i] = <unsigned int>(m[i]) << (bitwidth - i)
            else:
                for i in range(1, s + 1):
                    V[i] = <unsigned int>(m[i]) << (bitwidth - i)
                for i in range(s + 1, L + 1):
                    V[i] = V[i-s] ^ (V[i-s] >> s)
                    for k in range(1, s):
                        V[i] ^= (((a >> (s - 1U - k)) & 1U) * V[i-k]); 

            # Evalulate X[0] to X[N-1], scaled by pow(2,32)
            X[0] = 0
            for i in range(1, N):
                X[i] = X[i-1] ^ V[C[i-1]]

            pts_view[j] = <double>(X[N-1]) / pow(base, bitwidth)
    return pts



@cython.cdivision(True)
cpdef quasirandom_sobol_batch(unsigned int start, unsigned int end, unsigned int D):
    """
    Generate all sobol vectors with dimension `D` from seed `start` to seed `end` (including boundaries).
    Sobol sequence generation based on`[1, 2]`.

    Parameters:
        start (unsigned int): first sobol seed
        end (unsigned int): last sobol seed
        D (unsigned int): sobol vector dimension

    Returns:
        np.ndarray: (start - end + 1, D) array corresponding to the
            sampling vectors for each seed in the interval `[start, end]`

    Data in `_sobol_parameters.npz` is a modified version of polynomial
    coefficient table based on the D(6) criteria up to dimension 21201, where
    the $d$ value is implicit in the 1st index, the $a$ value is the at
    position 0 in the 2nd index and $s$ is implicit in the nonzero polynomial
    coefficients.

    Coefficient data retrieved from:
        `https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201`
    Retrieval date:
        2019/10/25

    References:
    ```
    S. Joe & F.Y. Kuo, ACM Trans. Math. Softw., 29, 49-57 (2003)
    S. Joe & F.Y. Kuo, SIAM J. Sci. Comput., 30, 2635-2654 (2008)
    ```
    """
    assert start > 0, "input seed for sobol_vector must be > 0"
    cdef double base = 2.0
    cdef unsigned int bitwidth = 32
    cdef unsigned int L = <unsigned>(ceil(log(<double>(end))/log(base)))
    cdef np.ndarray[np.uint32_t, ndim=1] C_arr = np.empty(end, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] V_arr = np.empty(L+1, dtype=np.uint32)
    cdef np.ndarray[np.uint32_t, ndim=1] X_arr = np.empty(end, dtype=np.uint32)
    cdef np.ndarray[np.float64_t, ndim=2] pts = np.zeros((end - start + 1, D), dtype=np.float64)
    cdef unsigned int[::1] C = C_arr, V = V_arr, X = X_arr
    cdef double[:, ::1] pts_view = pts
    cdef const unsigned int[:, ::1] poly = _SOBOL_DATA
    cdef const unsigned int[::1] m
    cdef unsigned int i, j, k, seed, idx
    cdef unsigned int value, s, a
    with nogil:
        # C[i] = index from the right of the first zero bit of i
        C[0] = 1
        for i in range(1, end):
            C[i] = 1
            value = i
            while ((value & 1U) != 0U):
                value >>= 1U
                C[i] += 1

        # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
        for i in range(1, L + 1):
            V[i] = 1U << (bitwidth - i)  # all m's = 1

        # Evalulate X[0] to X[N-1], scaled by pow(2,32)
        X[0] = 0
        for i in range(1, end):
            X[i] = X[i-1] ^ V[C[i-1]];

        idx = 0
        for seed in range(start - 1, end):
            pts_view[idx, 0] = <double>(X[seed]) / pow(base, bitwidth) 
            idx = idx + 1
        for j in range(1, D):
            m = poly[j+1]
            # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
            a = m[0]
            for s in range(1, m.shape[0]):
                if m[s] == 0:
                    break
            s = s - 1
            if (L <= s):
                for i in range(1, L + 1):
                    V[i] = <unsigned int>(m[i]) << (bitwidth - i)
            else:
                for i in range(1, s + 1):
                    V[i] = <unsigned int>(m[i]) << (bitwidth - i)
                for i in range(s + 1, L + 1):
                    V[i] = V[i-s] ^ (V[i-s] >> s)
                    for k in range(1, s):
                        V[i] ^= (((a >> (s - 1U - k)) & 1U) * V[i-k]); 

            # Evalulate X[0] to X[N-1], scaled by pow(2,32)
            X[0] = 0
            for i in range(1, end):
                X[i] = X[i-1] ^ V[C[i-1]]
            idx = 0
            for seed in range(start - 1, end):
                pts_view[idx, j] = <double>(X[seed]) / pow(base, bitwidth)
                idx = idx + 1
    return pts

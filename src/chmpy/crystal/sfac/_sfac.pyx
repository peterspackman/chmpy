# cython: language_level=3, boundscheck=False, wraparound=False
"""
 Atomic Form factors taken from the following sources:

 [1]  D. Waasmaier & A. Kirfel, Acta Cryst. (1995). A51, 416-413                      
 [2]  P. Rez & I. Grant, Acta Cryst. (1994), A50, 481-497

 Fit parameters of all atoms/ions (with the exception of O1-)           
 from [1], and the tit for O1- based on the tabulated values
 of Table 2 from [2]

"""

import numpy as np
import cython
from os.path import join, dirname
from libc.math cimport exp
_DATA_DIR = dirname(__file__)
_FORM_FACTOR_DATA = np.load(join(_DATA_DIR, "waaskirf.npz"))
_FORM_FACTOR_KEYS = _FORM_FACTOR_DATA.f.keys
_FORM_FACTOR_KEYS_LIST = _FORM_FACTOR_KEYS.tolist()
_FORM_FACTOR_VALUES = _FORM_FACTOR_DATA.f.values

cdef void fill_unique_plane_factors(const int[:, ::] hkl, const double[:] q, int[:] fac) noexcept nogil:
    cdef int i, j
    cdef int N
    cdef int nh, nk, nl
    cdef int hj, kj, lj
    cdef double qj, qi
    N = len(q)
    for i in range(N):
        if fac[i] < 1:
            continue
        qi = q[i]
        nh = - hkl[i, 0]
        nk = - hkl[i, 1]
        nl = - hkl[i, 2]
        for j in range(i + 1, N):
            qj = q[j]
            if (qj - qi) > 1e-7:
                continue
            hj = hkl[j, 0]
            kj = hkl[j, 1]
            lj = hkl[j, 2]
            if hj == nh and kj == nk and lj == nl:
                fac[i] += 1
                fac[j] = 0


cpdef calculate_unique_plane_factors(hkl, q):
    factors = np.ones(q.shape, dtype=np.int32)
    cdef const int[:, ::] hklview = hkl
    cdef const double[:] qview = q
    cdef int[:] facview = factors
    fill_unique_plane_factors(hklview, qview, facview)
    return factors


def get_form_factor_index(key):
    return _FORM_FACTOR_KEYS_LIST.index(key)

cpdef scattering_factors(idx, sintl):
    cdef Aff form_factor = Aff(idx)
    result = np.empty(sintl.shape, np.float64)
    form_factor.calc(sintl, result)
    return result

@cython.final
cdef class Aff:
    cdef double a1
    cdef double b1
    cdef double a2
    cdef double b2
    cdef double a3
    cdef double b3
    cdef double a4
    cdef double b4
    cdef double a5
    cdef double b5
    cdef double c
    def __init__(self, idx):
        "order = a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5"
        v = _FORM_FACTOR_VALUES[idx]
        self.a1 = v[0]
        self.a2 = v[1]
        self.a3 = v[2]
        self.a4 = v[3]
        self.a5 = v[4]
        self.c = v[5]
        self.b1 = -v[6]
        self.b2 = -v[7]
        self.b3 = -v[8]
        self.b4 = -v[9]
        self.b5 = -v[10]

    cdef void calc(self, const double[::1] s, double[::1] fac) nogil:
        cdef int i = 0
        cdef int N = s.shape[0]
        cdef double sintl2 = 0
        for i in range(N):
            sintl2 = s[i]
            fac[i] = (
                self.a1 * exp(self.b1 * sintl2) +
                self.a2 * exp(self.b2 * sintl2) +
                self.a3 * exp(self.b3 * sintl2) +
                self.a4 * exp(self.b4 * sintl2) + 
                self.a5 * exp(self.b5 * sintl2) + 
                self.c
            )


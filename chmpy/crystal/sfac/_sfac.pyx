# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language = c++
import numpy as np
cimport numpy as np
import cython
from os.path import join, dirname

# Enable low level memory management
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.math cimport exp, ceil, sqrt 
_DATA_DIR = dirname(__file__)
_FORM_FACTOR_DATA = np.load(join(_DATA_DIR, "atomic_form_factors.npz"))
_FORM_FACTOR_KEYS = _FORM_FACTOR_DATA.f.keys
_FORM_FACTOR_KEYS_LIST = _FORM_FACTOR_KEYS.tolist()
_FORM_FACTOR_VALUES = _FORM_FACTOR_DATA.f.values


def get_form_factor_index(key):
    return _FORM_FACTOR_KEYS_LIST.index(key)

cpdef scattering_factors(idx, sintl):
    cdef Aff form_factor = Aff(idx)
    result = np.empty(sintl.shape, np.float64)
    form_factor.calc(sintl, result)
    return result

cdef class Aff:
    cdef double a1
    cdef double b1
    cdef double a2
    cdef double b2
    cdef double a3
    cdef double b3
    cdef double a4
    cdef double b4
    cdef double c
    def __init__(self, idx):
        v = _FORM_FACTOR_VALUES[idx]
        self.a1 = v[0]
        self.b1 = -v[1]
        self.a2 = v[2]
        self.b2 = -v[3]
        self.a3 = v[4]
        self.b3 = -v[5]
        self.a4 = v[6]
        self.b4 = -v[7]
        self.c = v[8]

    cdef void calc(self, const double[::1] s, double[::1] fac) nogil:
        cdef int i = 0
        cdef int N = s.shape[0]
        for i in range(N):
            fac[i] = (
                self.a1 * exp(self.b1 * s[i]) +
                self.a2 * exp(self.b2 * s[i]) +
                self.a3 * exp(self.b3 * s[i]) +
                self.a4 * exp(self.b4 * s[i]) + self.c
            )


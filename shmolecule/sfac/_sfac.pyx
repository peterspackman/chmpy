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


cdef class MillerIndex:
    cdef int h, k, l
    cdef double mag
    def __init__(self, h, k, l):
        self.h = h
        self.k = k
        self.l = l
        self.mag = -1.0

    cdef bint is_zero(self) nogil:
        return self.h == 0 and self.k == 0 and self.l == 0

    cdef void plus(self, int h, int k, int l) nogil:
        self.h = self.h + h
        self.k = self.k + k
        self.l = self.l + l

    cdef void dot(self, int[:, ::1] mat) nogil:
        cdef int h = self.h
        cdef int k = self.k
        cdef int l = self.l
        self.h = mat[0, 0] * h + mat[0, 1] * k + mat[0, 1] * l
        self.k = mat[1, 0] * h + mat[1, 1] * k + mat[1, 1] * l
        self.l = mat[2, 0] * h + mat[2, 1] * k + mat[2, 1] * l

    cdef void magnitude(self, double[:, ::1] recip) nogil:
        cdef double q1, q2, q3
        q1 = recip[0, 0] * self.h + recip[0, 1] * self.k + recip[0, 2] * self.l
        q2 = recip[1, 0] * self.h + recip[1, 1] * self.k + recip[1, 2] * self.l
        q2 = recip[2, 0] * self.h + recip[2, 1] * self.k + recip[2, 2] * self.l
        self.mag = sqrt(q1 * q1 + q2 * q2 + q3 * q3)


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

    cdef void calc(self, const double[::1] s, double[::1] fac):
        cdef int i = 0
        for i in s.shape[0]:
            fac[i] = (
                self.a1 * exp(self.b1 * s[i]) +
                self.a2 * exp(self.b2 * s[i]) +
                self.a3 * exp(self.b3 * s[i]) +
                self.a4 * exp(self.b4 * s[i]) + self.c
            )


cdef void pruned_reflections_c(
        double a, double b, double c, double[:,::1] recip,
        int[:,::1] apexes, int[:,:, ::1] mats, double wavelength,
        vector[int]* hs, vector[int]* ks, vector[int]* ls,
        vector[double]* mags):
    cdef int h_max = int(ceil(a * wavelength))
    cdef int k_max = int(ceil(b * wavelength))
    cdef int l_max = int(ceil(c * wavelength))
    cdef int i, h1, k1, l1
    cdef int ah, ak, al
    cdef int[:, ::1] mat
    cdef MillerIndex m = MillerIndex(0, 0, 0)
    cdef double threshold

    with nogil:
        threshold = 1 / wavelength
        for i in range(apexes.shape[0]):
            ah = apexes[i, 0]
            ak = apexes[i, 1]
            al = apexes[i, 2]
            mat = mats[i, :, :]
            for h1 in range(0, h_max + 1):
                for k1 in range(0, k_max + 1):
                    for l1 in range(0, l_max + 1):
                        m.h = h1
                        m.k = k1
                        m.l = l1
                        m.mag = -1.0
                        m.dot(mat)
                        m.plus(ah, ak, al)
                        if m.is_zero():
                            continue
                        #if (m.is_centering_absent(centering)) continue;
                        m.magnitude(recip)
                        if (m.mag > threshold):
                            break
                        hs.push_back(m.h)
                        ks.push_back(m.k)
                        ls.push_back(m.l)
                        mags.push_back(m.mag)

cpdef pruned_reflections(a, b, c, recip, apexes, mats, wavelength):
    # minimum d spacing set to radius of ewald sphere
    cdef vector[int] hs
    cdef vector[int] ks
    cdef vector[int] ls
    cdef vector[double] mags
    cdef int i

    pruned_reflections_c(a, b, c, recip, apexes, mats, wavelength, &hs, &ks, &ls, &mags)
    idxs = []
    magnitudes = []
    for i in range(mags.size()):
        idxs.append((hs[i], ks[i], ls[i]))
        magnitudes.append(mags[i])
    return np.vstack(idxs), np.vstack(mags)

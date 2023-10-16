# cython: language_level=3, boundscheck=False, wraparound=False
cimport cython
import numpy as np
from scipy.fft import fft, ifft
from libc.math cimport sqrt, M_PI
from libc.stdlib cimport abs

@cython.cdivision(True)
cdef double amm(const int m) noexcept nogil:
    cdef double a = 1.0
    cdef int k
    for k in range(1, abs(m) + 1):
        a *= (2.0 * k + 1) / (2.0 * k)
    return sqrt( a / (4.0 * M_PI))



@cython.cdivision(True)
cdef double alm(const int l, const int m) noexcept nogil:
    return sqrt((4 * l * l - 1) / (1.0 * l * l - m * m))

@cython.cdivision(True)
cdef double blm(const int l, const int m) noexcept nogil:
    return -sqrt(
        (2.0 * l + 1) * ((l - 1)*(l - 1) - m * m) / 
        ((2.0 * l - 3) * (l * l - m * m))
    )

@cython.cdivision(True)
cdef void compute_ab(const int lmax, double[:, ::1] a, double[:, ::1] b) noexcept nogil:
    cdef int m, l
    for m in range(0, lmax + 1):
        a[m, m] = amm(m)
        for l in range(abs(m) + 1, lmax + 1):
            a[l, m] = alm(l, m)
            b[l, m] = blm(l, m)

@cython.final
cdef class AssocLegendre:
    cdef int lmax
    cdef double[:, ::1] cache
    cdef double[:, ::1] a
    cdef double[:, ::1] b

    def __init__(self, lm):
        self.lmax = lm
        shape = (lm + 1, lm + 1)
        self.a = np.zeros(shape)
        self.b = np.zeros(shape)
        self.cache = np.zeros(shape)
        compute_ab(self.lmax, self.a, self.b)


    @cython.cdivision(True)
    cdef void evaluate_batch_cython(self, double x, double[:] result) noexcept nogil:
        cdef int idx = 0
        cdef int l, m
        for m in range(0, self.lmax + 1):
            for l in range(m, self.lmax + 1):
                if(l == m):
                    result[idx] = self.a[l, m] * (1 - x * x) ** (0.5 * m)
                elif (l == (m + 1)):
                    result[idx] = self.a[l, m] * x * self.cache[l - 1, m]
                else:
                    result[idx] = (
                        self.a[l, m] * x * self.cache[l - 1, m] +
                        self.b[l, m] * self.cache[l - 2, m]
                    )
                self.cache[l, m] = result[idx]
                idx += 1

    def evaluate_batch(self, x, result=None):
        if result is None:
            result = np.zeros((self.lmax + 1) * (self.lmax + 2) // 2)
        self.evaluate_batch_cython(x, result)
        return result



@cython.cdivision(True)
cdef void analysis_cython_cplx(const int lmax, const int nphi,
                          const double complex[:] fft,
                          const double[:] plm_work_array,
                          const double w,
                          double complex[:] coeffs) noexcept nogil:
        cdef int plm_idx, l_offset, l, m, m_idx_neg, m_idx_pos, sign
        cdef double p
        cdef double complex pw, tmp, ii, rr
        plm_idx = 0

        for l in range(lmax + 1):
            l_offset = l * (l + 1)
            pw = plm_work_array[plm_idx] * w
            coeffs[l_offset] = coeffs[l_offset] + fft[0] * pw
            plm_idx += 1

        # because we don't include a phase factor (-1)^m in our
        # Associated Legendre Polynomials, we need a factor here.
        # which alternates with m
        for m in range(1, lmax + 1):
            sign = -1 if m & 1 else 1
            for l in range(m, lmax + 1):
                l_offset = l * (l + 1)
                pw = plm_work_array[plm_idx] * w
                m_idx_neg = nphi - m
                m_idx_pos = m
                rr = sign * fft[m_idx_pos] * pw
                ii = sign * fft[m_idx_neg] * pw
                if m & 1:
                    ii = - ii

                coeffs[l_offset - m] = coeffs[l_offset - m] + ii
                coeffs[l_offset + m] = coeffs[l_offset + m] + rr
                plm_idx += 1


@cython.cdivision(True)
cdef void synthesis_cython_cplx(const int lmax, const int nphi,
                                const double complex[:] coeffs,
                                const double[:] plm_work_array,
                                double complex[:] fft) noexcept nogil:
    cdef int plm_idx, l_offset, l, m, m_idx_neg, m_idx_pos, sign
    cdef double p
    cdef double complex ii, rr

    plm_idx = 0
    # m = 0 case
    for l in range(lmax + 1):
        l_offset = l * (l + 1)
        p = plm_work_array[plm_idx]
        fft[0] = fft[0] + coeffs[l_offset] * p
        plm_idx += 1

    # because we don't include a phase factor (-1)^m in our
    # Associated Legendre Polynomials, we need a factor here.
    # which alternates with m

    for m in range(1, lmax + 1):
        sign = -1 if m & 1 else 1
        for l in range(m, lmax + 1):

            l_offset = l * (l + 1)
            p = plm_work_array[plm_idx]
            m_idx_neg = nphi - m
            m_idx_pos = m
            rr = sign * coeffs[l_offset + m] * p
            ii = sign * coeffs[l_offset - m] * p
            if m & 1:
                ii = - ii
            fft[m_idx_neg] = fft[m_idx_neg] + ii
            fft[m_idx_pos] = fft[m_idx_pos] + rr
            plm_idx += 1



@cython.cdivision(True)
cdef void synthesis_cython_real(const int lmax, const int nphi,
                                const double complex[:] coeffs,
                                const double[:] plm_work_array,
                                double complex[:] fft) noexcept nogil:
    cdef int plm_idx, l_offset, l, m, m_idx_neg, m_idx_pos, sign

    cdef double p
    cdef double complex rr
    plm_idx = 0
    # m = 0 case
    for l in range(lmax + 1):
        p = plm_work_array[plm_idx]
        fft[0] = fft[0] + coeffs[plm_idx] * p
        plm_idx += 1

    for m in range(1, lmax + 1):
        sign = -1 if m & 1 else 1
        for l in range(m, lmax + 1):
            p = plm_work_array[plm_idx]
            rr = 2 * sign * coeffs[plm_idx] * p
            fft[m] = fft[m] + rr
            plm_idx += 1


@cython.cdivision(True)
cdef void analysis_cython_real(const int lmax, const int nphi,
                          const double complex[:] fft,
                          const double[:] plm_work_array,
                          const double w,
                          double complex[:] coeffs) noexcept nogil:
        cdef int plm_idx, l_offset, l, m, m_idx_neg, m_idx_pos
        cdef double complex pw, tmp
        cdef int sign

        plm_idx = 0
        # m = 0 case
        for l in range(lmax + 1):
            pw = plm_work_array[plm_idx] * w
            coeffs[plm_idx] = coeffs[plm_idx] + fft[0] * pw
            plm_idx += 1

        # because we don't include a phase factor (-1)^m in our
        # Associated Legendre Polynomials, we need a factor here.
        # which alternates with m and l
        for m in range(1, lmax + 1):
            sign = -1 if m & 1 else 1
            for l in range(m, lmax + 1):
                pw = plm_work_array[plm_idx] * w
                coeffs[plm_idx] = coeffs[plm_idx] + sign * fft[m] * pw
                plm_idx += 1



@cython.cdivision(True)
cdef void expand_coeffs_cython(const int lmax, const double complex[:] cin, double complex[:] cout) noexcept nogil:
    cdef int l, m, plm_idx, l_offset
    cdef int sign
    plm_idx = 0
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            l_offset = l * (l + 1)
            cout[l_offset + m] = cin[plm_idx]
            if m != 0:
                sign = -1 if m & 1 else 1
                cout[l_offset - m] = sign * cin[plm_idx].conjugate()
            plm_idx += 1


def analysis_kernel_cplx(sht, w, coeffs):
    analysis_cython_cplx(sht.lmax, sht.nphi, sht.fft_work_array, sht.plm_work_array, w, coeffs)

def analysis_kernel_real(sht, w, coeffs):
    analysis_cython_real(sht.lmax, sht.nphi, sht.fft_work_array, sht.plm_work_array, w, coeffs)

def synthesis_kernel_cplx(sht, coeffs):
    synthesis_cython_cplx(sht.lmax, sht.nphi, coeffs, sht.plm_work_array, sht.fft_work_array)

def synthesis_kernel_real(sht, coeffs):
    synthesis_cython_real(sht.lmax, sht.nphi, coeffs, sht.plm_work_array, sht.fft_work_array)



def expand_coeffs_to_full(lmax, coeffs):
    new_coeffs = np.empty((lmax + 1) * (lmax + 1), dtype=np.complex128)
    expand_coeffs_cython(lmax, coeffs, new_coeffs)
    return new_coeffs

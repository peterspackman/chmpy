cimport cython
cimport numpy as np
import numpy as np
from os.path import join, dirname
from libc.math cimport abs, log, sqrt

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))
_DOMAIN = _INTERPOLATOR_DATA.f.domain
_RHO = _INTERPOLATOR_DATA.f.rho
_GRAD_RHO = _INTERPOLATOR_DATA.f.grad_rho


cdef class PromoleculeDensity:
    cpdef public float[:, ::1] positions
    cpdef public int[::1] elements

    def __init__(self, pos, elements):
        self.positions = pos
        self.elements = elements

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef rho(self, float[:, ::1] positions):
        cdef int i, e, j
        cdef float diff
        cdef float[::1] pos
        cdef np.ndarray[np.float32_t, ndim=1] r = np.empty(positions.shape[0], dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1] rho = np.zeros(positions.shape[0], dtype=np.float32)
        cdef float[::1] xi = _DOMAIN
        cdef float[::1] yi
        cdef float[::1] rho_view = rho
        for i in range(self.elements.shape[0]):
            e = self.elements[i]
            pos = self.positions[i]
            yi = _RHO[e - 1, :]
            for j in range(positions.shape[0]):
                r[j] = 0.0
                for col in range(3):
                    diff = positions[j, col] - pos[col]
                    r[j] += diff*diff
                r[j] = sqrt(r[j]) / 0.5291772108 # bohr_per_angstrom
            log_interp_f(r, xi, yi, rho_view)
        return rho


cdef class StockholderWeight:
    cpdef public PromoleculeDensity dens_a, dens_b

    def __init__(self, dens_a, dens_b):
        self.dens_a = dens_a
        self.dens_b = dens_b

    def weights(self, float[:, ::1] positions):
        rho_a = self.dens_a.rho(positions)
        rho_b = self.dens_b.rho(positions)
        return rho_a / (rho_b + rho_a)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void log_interp_d(const double[::1] x, const double[::1] xi,
                       const double[::1] yi, double[::1] y) nogil:
    cdef double xval, lxval, guess
    cdef double slope
    cdef int ni = xi.shape[0]
    cdef double lbound = log(xi[0]), ubound = log(xi[ni - 1])
    cdef double lrange = ubound - lbound
    cdef double lfill = yi[0], rfill = yi[ni - 1]
    cdef int i, j
    for i in range(x.shape[0]):
        xval = x[i]
        lxval = log(xval)
        guess = ni * (lxval - lbound) / lrange
        j = <int>guess
        if j <= 0:
            y[i] = lfill
            continue
        if j >= ni - 1:
            y[i] = rfill
            continue

        while xi[j] < xval:
            j += 1

        slope = (yi[j] - yi[j-1]) / (xi[j] - xi[j-1])
        y[i] += yi[j-1] + (xval - xi[j-1]) * slope

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void log_interp_f(const float[::1] x, const float[::1] xi,
                       const float[::1] yi, float[::1] y) nogil:
    cdef float xval, lxval, guess
    cdef float slope
    cdef int ni = xi.shape[0]
    cdef float lbound = log(xi[0]), ubound = log(xi[ni - 1])
    cdef float lrange = ubound - lbound
    cdef float lfill = yi[0], rfill = yi[ni - 1]
    cdef int i, j
    for i in range(x.shape[0]):
        xval = x[i]
        lxval = log(xval)
        guess = ni * (lxval - lbound) / lrange
        j = <int>guess
        if j <= 0:
            y[i] = lfill
            continue
        if j >= ni - 1:
            y[i] = rfill
            continue

        while True:
            j += 1
            if xi[j] >= xval: break

        slope = (yi[j] - yi[j-1]) / (xi[j] - xi[j-1])
        y[i] += yi[j-1] + (xval - xi[j-1]) * slope



def log_interp(const double[::1] pts, const double[::1] xi, const double[::1] yi):
    y = np.zeros(pts.shape[0], dtype=np.float64)
    cdef double[::1] yview = y
    with nogil:
        log_interp_d(pts, xi, yi, yview)
    return y

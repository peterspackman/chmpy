cimport cython
cimport numpy as np
import numpy as np
from os.path import join, dirname
from libc.math cimport fabs, log, sqrt, cos, sin

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))
_DOMAIN = _INTERPOLATOR_DATA.f.domain
_RHO = _INTERPOLATOR_DATA.f.rho
_GRAD_RHO = _INTERPOLATOR_DATA.f.grad_rho


cdef class PromoleculeDensity:
    cpdef public float[:, ::1] positions
    cpdef public int[::1] elements
    cpdef public np.ndarray rho_data
    cpdef public np.ndarray domain

    def __init__(self, pos, elements):
        self.positions = pos
        self.elements = elements
        self.rho_data = np.empty((elements.shape[0], _DOMAIN.shape[0]), dtype=np.float32)
        self.domain = np.empty(_DOMAIN.shape, dtype=np.float32)
        self.domain[:] = _DOMAIN[:]
        for i in range(elements.shape[0]):
            self.rho_data[i, :] = _RHO[elements[i] - 1, :]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef rho(self, float[:, ::1] positions):
        cdef int i, e, j
        cdef float diff
        cdef float[::1] pos
        cdef np.ndarray[np.float32_t, ndim=1] r = np.empty(positions.shape[0], dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1] rho = np.zeros(positions.shape[0], dtype=np.float32)
        cdef float[::1] xi = self.domain
        cdef float[::1] yi
        cdef float[::1] rho_view = rho
        for i in range(self.elements.shape[0]):
            e = self.elements[i]
            pos = self.positions[i]
            yi = self.rho_data[i, :]
            for j in range(positions.shape[0]):
                r[j] = 0.0
                for col in range(3):
                    diff = positions[j, col] - pos[col]
                    r[j] += diff*diff
                r[j] = sqrt(r[j]) / 0.5291772108 # bohr_per_angstrom
            log_interp_f(r, xi, yi, rho_view)
        return rho

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float one_rho(self, float position[3]) nogil:
        cdef int i, e
        cdef float diff, r
        cdef const float[::1] pos
        cdef const float[::1] xi = self.domain
        cdef const float[:, ::1] rho_data = self.rho_data
        cdef const float[::1] yi
        cdef float rho = 0.0
        for i in range(self.elements.shape[0]):
            e = self.elements[i]
            pos = self.positions[i]
            yi = rho_data[i, :]
            r = 0.0
            for col in range(3):
                diff = position[col] - pos[col]
                r += diff*diff
            r = sqrt(r) / 0.5291772108 # bohr_per_angstrom
            rho += log_interp_f_one(r, xi, yi)
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
    cdef float one_weight(self, float position[3]) nogil:
        cdef float rho_a = self.dens_a.one_rho(position)
        cdef float rho_b = self.dens_b.one_rho(position)
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float log_interp_f_one(const float x, const float[::1] xi, const float[::1] yi) nogil:
    cdef float xval, lxval, guess
    cdef float slope
    cdef int ni = xi.shape[0]
    cdef float lbound = log(xi[0]), ubound = log(xi[ni - 1])
    cdef float lrange = ubound - lbound
    cdef float lfill = yi[0], rfill = yi[ni - 1]
    cdef int j
    lxval = log(x)
    guess = ni * (lxval - lbound) / lrange
    j = <int>guess
    if j <= 0:
        return lfill
    if j >= ni - 1:
        return rfill
    while True:
        j += 1
        if xi[j] >= x: break
    slope = (yi[j] - yi[j-1]) / (xi[j] - xi[j-1])
    return yi[j-1] + (x - xi[j-1]) * slope

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void fvmul(const float[::1] o, const float a, const float[::1] v, float dest[3]) nogil:
    dest[0] = o[0] + v[0] * a
    dest[1] = o[1] + v[1] * a
    dest[2] = o[2] + v[2] * a

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float brents(StockholderWeight stock, const float[::1] origin,
                   const float[::1] direction,
                   const float lower, const float upper,
                   const float tol, const int max_iter) nogil:
    cdef float a = lower
    cdef float b = upper
    cdef float v[3]
    fvmul(origin, a, direction, v)
    cdef float fa = stock.one_weight(v) - 0.5
    fvmul(origin, b, direction, v)
    cdef float fb = stock.one_weight(v) - 0.5
    cdef np.npy_bool mflag = False
    cdef float s = 0.0, d = 0.0, c = a, fc = fa
    cdef int i
    for i in range(max_iter):
        if fabs(b - a) < tol:
            return s
        if ((fa != fc) and (fb != fc)):
            s = (
                (a * fb * fc / ((fa - fb) * (fa - fc)) ) +
                (b * fa * fc / ((fb - fa) * (fb - fc)) ) +
                (c * fa * fb / ((fc - fa) * (fc - fb)) )
            )
        else:
            s = b - fb * (b - a) / (fb - fa)
        
        if (((s < (3 * a + b) * 0.25) or (s > b)) or
                (mflag and (fabs(s-b) >= (fabs(b -c) * 0.5))) or
                ((not mflag) and (fabs(s-b) >= (fabs(c -d) * 0.5))) or
                (mflag and (fabs(b-c) < tol)) or
                ((not mflag) and (fabs(c- d) < tol))):
            s = (a + b) * 0.5
            mflag = True
        else:
            mflag = False
        fvmul(origin, s, direction, v)
        fs = stock.one_weight(v) - 0.5
        d = c
        c = b
        fc = fb
        if (fa * fs < 0):
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        if fabs(fa) < fabs(fb):
            a, b = b, a
            fa, fb = fb, fa
    return -1.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sphere_stockholder_radii(
        StockholderWeight s, const float[::1] origin, const float[:, ::1] grid,
        const float l, const float u, const float tol, const int max_iter):
    cdef int i, N = grid.shape[0]
    cdef float[::1] d = np.empty(3, dtype=np.float32)
    
    r = np.empty(N, dtype=np.float64)
    for i in range(N):
        d[0] = sin(grid[i, 1]) * cos(grid[i, 0])
        d[1] = sin(grid[i, 1]) * sin(grid[i, 0])
        d[2] = cos(grid[i, 1])
        r[i] = brents(s, origin, d, l, u, tol, max_iter)
    return r


def log_interp(const double[::1] pts, const double[::1] xi, const double[::1] yi):
    y = np.zeros(pts.shape[0], dtype=np.float64)
    cdef double[::1] yview = y
    with nogil:
        log_interp_d(pts, xi, yi, yview)
    return y

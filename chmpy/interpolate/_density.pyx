# cython: language_level=3, boundscheck=False, wraparound=False
cimport cython
cimport numpy as np
import numpy as np
from os.path import join, dirname
from libc.math cimport fabs, log, sqrt, cos, sin
from cython.parallel import prange

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))
_DOMAIN = _INTERPOLATOR_DATA.f.domain
_RHO = _INTERPOLATOR_DATA.f.rho
_GRAD_RHO = _INTERPOLATOR_DATA.f.grad_rho


@cython.final
cdef class PromoleculeDensity:
    cpdef public float[:, ::1] positions
    cdef public int[::1] elements
    cdef const float[::1] domain
    cdef const float[:, ::1] rho_data

    def __init__(self, pos, const float[::1] domain, const float[:, ::1] rho_data):
        self.positions = pos
        self.domain = domain
        self.rho_data = rho_data

    cpdef rho(self, const float[:, ::1] pts):
        cdef int i, j, col
        cdef float diff
        cdef float[3] pos
        cdef np.ndarray[np.float32_t, ndim=1] r = np.empty(pts.shape[0], dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1] rho = np.zeros(pts.shape[0], dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1] tmp = np.empty(pts.shape[0], dtype=np.float32)
        cdef float[::1] rho_view = rho
        cdef float[::1] tmp_view = tmp
        cdef float[::1] r_view = r
        cdef const float[:, ::1] pos_view = self.positions
        cdef const float[:, ::1] rho_data_view = self.rho_data
        cdef const float[::1] domain_view = self.domain
        cdef int npos = self.positions.shape[0]
        cdef int npts = pts.shape[0]
        with nogil:
            for i in range(npos):
                pos[0] = pos_view[i, 0]
                pos[1] = pos_view[i, 1]
                pos[2] = pos_view[i, 2]
                for j in range(npts):
                    tmp_view[j] = 0.0
                    r_view[j] = (
                        (pts[j, 0] - pos[0]) * (pts[j, 0] - pos[0]) + 
                        (pts[j, 1] - pos[1]) * (pts[j, 1] - pos[1]) + 
                        (pts[j, 2] - pos[2]) * (pts[j, 2] - pos[2])
                    )
                    r_view[j] = sqrt(r_view[j]) / 0.5291772108 # bohr_per_angstrom
                log_interp_f(r_view, domain_view, rho_data_view[i], tmp_view)
                for j in range(pts.shape[0]):
                    rho_view[j] += tmp_view[j]
        return rho

    cdef float one_rho(self, const float position[3]) nogil:
        cdef int i
        cdef int N = self.positions.shape[0]
        cdef float diff, r
        cdef const float[::1] pos
        cdef const float[::1] xi = self.domain
        cdef const float[::1] yi
        cdef const float[:, ::1] pos_view = self.positions
        cdef const float[:, ::1] rho_data_view = self.rho_data
        cdef float rho = 0.0
        for i in range(N):
            r = 0.0
            for col in range(3):
                diff = position[col] - pos_view[i, col]
                r += diff*diff
            r = sqrt(r) / 0.5291772108 # bohr_per_angstrom
            rho += log_interp_f_one(r, xi, rho_data_view[i])
        return rho

@cython.final
cdef class StockholderWeight:
    cdef public PromoleculeDensity dens_a, dens_b
    cdef float background

    def __init__(self, dens_a, dens_b, background=0.0):
        self.dens_a = dens_a
        self.dens_b = dens_b
        self.background = background

    cpdef weights(self, const float[:, ::1] positions):
        cdef np.ndarray[np.float32_t, ndim=1] rho = np.empty(
                positions.shape[0], dtype=np.float32
        )
        rho_a = self.dens_a.rho(positions)
        rho_b = self.dens_b.rho(positions)
        result = rho_a / (rho_a + rho_b + self.background)
        return result
    
    cdef float one_weight(self, const float position[3]) nogil:
        cdef float rho_a = self.dens_a.one_rho(position)
        cdef float rho_b = self.dens_b.one_rho(position)
        return rho_a / (rho_b + rho_a + self.background)


@cython.cdivision(True)
cdef void log_interp_d(const double[::1] x, const double[::1] xi,
                       const double[::1] yi, double[::1] y) nogil:
    cdef double xval, lxval, guess
    cdef double slope
    cdef int ni = xi.shape[0]
    cdef double lbound = log(xi[0]), ubound = log(xi[ni - 1])
    cdef double lrange = ubound - lbound
    cdef double lfill = yi[0], rfill = yi[ni - 1]
    cdef int i, j
    for i in prange(x.shape[0]):
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
            j = j + 1

        slope = (yi[j] - yi[j-1]) / (xi[j] - xi[j-1])
        y[i] += yi[j-1] + (xval - xi[j-1]) * slope

@cython.cdivision(True)
cdef void log_interp_f(const float[::1] x, const float[::1] xi,
                       const float[::1] yi, float[::1] y) nogil:
    cdef float xval, lxval, guess
    cdef float slope
    cdef int ni = xi.shape[0]
    cdef float lbound = log(xi[0])
    cdef float ubound = log(xi[ni - 1])
    cdef float lrange = ubound - lbound
    cdef float lfill = yi[0], rfill = 0.0
    cdef int i, j
    for i in prange(x.shape[0]):
        xval = x[i]
        lxval = log(xval)
        guess = ni * (lxval - lbound) / lrange
        j = <int>guess
        if j <= 0:
            y[i] = lfill
            continue
        elif j >= ni - 1:
            y[i] = rfill
            continue

        while True:
            j = j + 1
            if xi[j] >= xval: break

        slope = (yi[j] - yi[j-1]) / (xi[j] - xi[j-1])
        y[i] = y[i] + yi[j-1] + (xval - xi[j-1]) * slope

@cython.cdivision(True)
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

cdef inline void fvmul(const float o[3], const float a, const float v[3], float dest[3]) nogil:
    dest[0] = o[0] + v[0] * a
    dest[1] = o[1] + v[1] * a
    dest[2] = o[2] + v[2] * a

@cython.cdivision(True)
cdef float brents_stock(StockholderWeight stock, const float origin[3],
                   const float direction[3],
                   const float lower, const float upper,
                   const float tol, const int max_iter,
                   const float isovalue) nogil:
    cdef float xpre, xcur, xblk, fpre, fcur, fblk, spre, scur, sbis, stry, dpre, dblk
    cdef float delta, xtol = 1e-5
    cdef int i
    cdef float v[3]
    xpre = lower
    xcur = upper
    xblk = 0.0; fblk = 0.0
    spre = 0.0; scur = 0.0
    fvmul(origin, xpre, direction, v)
    fpre = stock.one_weight(v) - isovalue
    fvmul(origin, xcur, direction, v)
    fcur = stock.one_weight(v) - isovalue
    if (fpre * fcur > 0):
        return -1.0
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    for i in range(max_iter):
        if (fpre * fcur) < 0:
            xblk = xpre
            fblk = fpre
            scur = xcur - xpre
            spre = scur
        if fabs(fblk) < fabs(fcur):
            xpre = xcur; xcur = xblk; xblk = xpre
            fpre = fcur; fcur = fblk; fblk = fpre
        delta = (xtol + tol * fabs(xcur))/2
        sbis = (xblk - xcur)/2
        if (fcur == 0) or (fabs(sbis) < delta):
            return xcur

        if (fabs(spre) > delta) and (fabs(fcur) < fabs(fpre)):
            if xpre == xblk:
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur*(fblk * dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

            if (2 * fabs(stry) < min(fabs(spre), 3*fabs(sbis) - delta)):
                spre = scur; scur = stry
            else:
                spre = sbis; scur = sbis
        else:
            spre = sbis; scur = sbis

        xpre = xcur; fpre = fcur
        if (fabs(scur) > delta):
            xcur += scur
        else:
            xcur += (delta if (sbis > 0) else -delta)

        fvmul(origin, xcur, direction, v)
        fcur = stock.one_weight(v) - isovalue

    return xcur


@cython.cdivision(True)
cdef float brents_pro(PromoleculeDensity pro, const float origin[3],
                   const float direction[3],
                   const float lower, const float upper,
                   const float tol, const int max_iter,
                   const float isovalue) nogil:
    cdef float xpre, xcur, xblk, fpre, fcur, fblk, spre, scur, sbis, stry, dpre, dblk
    cdef float delta, xtol = 1e-5
    cdef int i
    cdef float v[3]
    xpre = lower
    xcur = upper
    xblk = 0.0; fblk = 0.0
    spre = 0.0; scur = 0.0
    fvmul(origin, xpre, direction, v)
    fpre = pro.one_rho(v) - isovalue
    fvmul(origin, xcur, direction, v)
    fcur = pro.one_rho(v) - isovalue
    if (fpre * fcur > 0):
        return -1.0
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    for i in range(max_iter):
        if (fpre * fcur) < 0:
            xblk = xpre
            fblk = fpre
            scur = xcur - xpre
            spre = scur
        if fabs(fblk) < fabs(fcur):
            xpre = xcur; xcur = xblk; xblk = xpre
            fpre = fcur; fcur = fblk; fblk = fpre
        delta = (xtol + tol * fabs(xcur))/2
        sbis = (xblk - xcur)/2
        if (fcur == 0) or (fabs(sbis) < delta):
            return xcur

        if (fabs(spre) > delta) and (fabs(fcur) < fabs(fpre)):
            if xpre == xblk:
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur*(fblk * dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))

            if (2 * fabs(stry) < min(fabs(spre), 3*fabs(sbis) - delta)):
                spre = scur; scur = stry
            else:
                spre = sbis; scur = sbis
        else:
            spre = sbis; scur = sbis

        xpre = xcur; fpre = fcur
        if (fabs(scur) > delta):
            xcur += scur
        else:
            xcur += (delta if (sbis > 0) else -delta)

        fvmul(origin, xcur, direction, v)
        fcur = pro.one_rho(v) - isovalue

    return xcur

cpdef sphere_promolecule_radii(
        PromoleculeDensity s, const float[::1] origin, const float[:, ::1] grid,
        const float l, const float u, const float tol, const int max_iter,
        const float isovalue):
    cdef int i, N = grid.shape[0]
    cdef float d[3]
    cdef float o[3]
    r = np.empty(N, dtype=np.float64)
    cdef double[::1] rview  = r
    o[0] = origin[0]
    o[1] = origin[1]
    o[2] = origin[2]
    
    with nogil:
        for i in range(N):
            d[0] = grid[i, 0]
            d[1] = grid[i, 1]
            d[2] = grid[i, 2]
            rview[i] = brents_pro(s, o, d, l, u, tol, max_iter, isovalue)
    return r



cpdef sphere_stockholder_radii(
        StockholderWeight s, const float[::1] origin, const float[:, ::1] grid,
        const float l, const float u, const float tol, const int max_iter, const float isovalue):
    cdef int i, N = grid.shape[0]
    cdef float d[3]
    cdef float o[3]
    r = np.empty(N, dtype=np.float64)
    cdef double[::1] rview  = r
    o[0] = origin[0]
    o[1] = origin[1]
    o[2] = origin[2]
    
    with nogil:
        for i in range(N):
            d[0] = grid[i, 0]
            d[1] = grid[i, 1]
            d[2] = grid[i, 2]
            rview[i] = brents_stock(s, o, d, l, u, tol, max_iter, isovalue)
    return r


def log_interp(const double[::1] pts, const double[::1] xi, const double[::1] yi):
    y = np.zeros(pts.shape[0], dtype=np.float64)
    cdef double[::1] yview = y
    with nogil:
        log_interp_d(pts, xi, yi, yview)
    return y

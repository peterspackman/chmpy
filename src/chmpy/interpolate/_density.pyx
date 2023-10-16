# cython: language_level=3, boundscheck=False, wraparound=False
cimport cython
cimport numpy as cnp
import numpy as np
from os.path import join, dirname
from libc.math cimport fabs, log, cos, sin
from cython.parallel import prange

cnp.import_array()

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))
_DOMAIN = _INTERPOLATOR_DATA.f.domain
_RHO = _INTERPOLATOR_DATA.f.rho
_GRAD_RHO = _INTERPOLATOR_DATA.f.grad_rho


@cython.final
cdef class PromoleculeDensity:
    cdef public float[:, ::1] positions
    cdef public int[::1] elements
    cdef const float[::1] domain
    cdef const float[:, ::1] rho_data

    def __init__(self, pos, const float[::1] domain, const float[:, ::1] rho_data):
        self.positions = pos
        self.domain = domain
        self.rho_data = rho_data


    def rho(self, pts):
        r = np.empty(pts.shape[0], dtype=np.float32)
        rho = np.zeros(pts.shape[0], dtype=np.float32)
        tmp = np.empty(pts.shape[0], dtype=np.float32)
        self.evaluate_rho(pts, rho, tmp, r)
        return rho

    cdef void evaluate_rho(self, const float[:, ::1] pts, float[::1] rho_view, float[::1] tmp_view, float[::1] r_view) noexcept nogil:
        cdef int i, j, col
        cdef float diff
        cdef float[3] pos
        cdef const float[:, ::1] pos_view = self.positions
        cdef const float[:, ::1] rho_data_view = self.rho_data
        cdef const float[::1] domain_view = self.domain
        cdef int npos = self.positions.shape[0]
        cdef int npts = pts.shape[0]
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
                r_view[j] = r_view[j] / (0.5291772108 * 0.5291772108) # bohr_per_angstrom
            interp_f(r_view, domain_view, rho_data_view[i], tmp_view)
            for j in range(pts.shape[0]):
                rho_view[j] += tmp_view[j]

    cdef float one_rho(self, const float position[3]) noexcept nogil:
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
            r = r / (0.5291772108 * 0.5291772108) # bohr_per_angstrom
            rho += interp_f_one(r, xi, rho_data_view[i])
        return rho

@cython.final
cdef class StockholderWeight:
    cdef public PromoleculeDensity dens_a, dens_b
    cdef float background

    def __init__(self, dens_a, dens_b, background=0.0):
        self.dens_a = dens_a
        self.dens_b = dens_b
        self.background = background

    def weights(self, positions):
        rho_a = self.dens_a.rho(positions)
        rho_b = self.dens_b.rho(positions)
        result = rho_a / (rho_a + rho_b + self.background)
        return result
    
    cdef float one_weight(self, const float position[3]) noexcept nogil:
        cdef float rho_a = self.dens_a.one_rho(position)
        cdef float rho_b = self.dens_b.one_rho(position)
        return rho_a / (rho_b + rho_a + self.background)


@cython.cdivision(True)
cdef void interp_f(const float[::1] x, const float[::1] xi,
                   const float[::1] yi, float[::1] y) noexcept nogil:

    cdef int ni = xi.shape[0]
    cdef float lbound = xi[0]
    cdef float ubound = xi[ni - 1]
    cdef float ufill = yi[ni - 1]
    cdef float lfill = yi[0]
    cdef float dx = xi[1] - xi[0]
    cdef float inv_dx = 1.0 / dx
    cdef int i, j
    cdef float t

    for i in prange(x.shape[0]):
        j = <int>(inv_dx * (x[i] - lbound))
        if j <= 0:
            y[i] = lfill
        elif j >= ni - 1:
            y[i] = ufill
        else:
            t = (x[i] - xi[j]) * inv_dx 
            y[i] = (1.0 - t) * yi[j] + t * yi[j + 1]

@cython.cdivision(True)
cdef inline float interp_f_one(const float x, const float[::1] xi,
                               const float[::1] yi) noexcept nogil:

    cdef int ni = xi.shape[0]
    cdef float lbound = xi[0]
    cdef float ubound = xi[ni - 1]
    cdef float ufill = 0.0
    cdef float lfill = yi[0]
    cdef float dx = xi[1] - xi[0]
    cdef float inv_dx = 1.0 / dx
    cdef int j
    cdef float t

    j = <int>(inv_dx * (x - lbound))
    if j <= 0:
        return lfill
    elif j >= ni - 1:
        return ufill
    else:
        t = (x - xi[j]) * inv_dx 
        return (1.0 - t) * yi[j] + t * yi[j + 1]


cdef inline void fvmul(const float o[3], const float a, const float v[3], float dest[3]) noexcept nogil:
    dest[0] = o[0] + v[0] * a
    dest[1] = o[1] + v[1] * a
    dest[2] = o[2] + v[2] * a

@cython.cdivision(True)
cdef float brents_stock(StockholderWeight stock, const float origin[3],
                   const float direction[3],
                   const float lower, const float upper,
                   const float tol, const int max_iter,
                   const float isovalue) noexcept nogil:
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
                   const float isovalue) noexcept nogil:
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

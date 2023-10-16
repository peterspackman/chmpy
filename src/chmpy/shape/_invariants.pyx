# cython: language_level=3, boundscheck=False, wraparound=False
cimport cython
import numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport abs

cdef double * factorial = [
    1.000000000000000000e+00,
    1.000000000000000000e+00,
    2.000000000000000000e+00,
    6.000000000000000000e+00,
    2.400000000000000000e+01,
    1.200000000000000000e+02,
    7.200000000000000000e+02,
    5.040000000000000000e+03,
    4.032000000000000000e+04,
    3.628800000000000000e+05,
    3.628800000000000000e+06,
    3.991680000000000000e+07,
    4.790016000000000000e+08,
    6.227020800000000000e+09,
    8.717829120000000000e+10,
    1.307674368000000000e+12,
    2.092278988800000000e+13,
    3.556874280960000000e+14,
    6.402373705728000000e+15,
    1.216451004088320000e+17,
    2.432902008176640000e+18,
    5.109094217170944000e+19,
    1.124000727777607680e+21,
    2.585201673888497821e+22,
    6.204484017332394100e+23,
    1.551121004333098606e+25,
    4.032914611266056503e+26,
    1.088886945041835194e+28,
    3.048883446117138367e+29,
    8.841761993739700773e+30,
    2.652528598121910322e+32,
    8.222838654177922430e+33,
    2.631308369336935178e+35,
    8.683317618811885939e+36,
    2.952327990396041196e+38,
    1.033314796638614422e+40,
    3.719933267899011775e+41,
    1.376375309122634310e+43,
    5.230226174666010379e+44,
    2.039788208119744159e+46,
    8.159152832478976838e+47,
    3.345252661316380276e+49,
    1.405006117752879789e+51,
    6.041526306337383407e+52,
    2.658271574788448529e+54,
    1.196222208654801886e+56,
    5.502622159812088457e+57,
    2.586232415111681777e+59,
    1.241391559253607253e+61,
    6.082818640342675225e+62,
    3.041409320171337558e+64,
    1.551118753287382189e+66,
    8.065817517094387685e+67,
    4.274883284060025485e+69,
    2.308436973392413792e+71,
    1.269640335365827645e+73,
    7.109985878048634810e+74,
    4.052691950487722053e+76,
    2.350561331282878906e+78,
    1.386831185456898649e+80,
    8.320987112741391580e+81,
    5.075802138772248358e+83,
    3.146997326038793939e+85,
    1.982608315404440085e+87,
    1.268869321858841654e+89,
    8.247650592082471517e+90,
    5.443449390774430694e+92,
    3.647111091818868322e+94,
    2.480035542436830548e+96,
    1.711224524281412974e+98,
    1.197857166996989027e+100,
];


@cython.cdivision(True)
cdef double clebsch(int j1, int m1, int j2, int m2, int j, int m) noexcept nogil:
   # Calculation using Racah formula taken from "Angular Momentum",
   # D.M.Brink & G.R.Satchler, Oxford, 1968
   cdef double res = 0.0

   cdef int j1nm1, jnj2pm1, j2pm2, jnj1nm2, j1pj2nj
   cdef int k, mink, maxk, iphase
   cdef double tmp

   if (abs(m1) > j1): return res
   if (abs(m2) > j2): return res
   if (abs(m) > j): return res
   if ((j1 < 0) or (j2 < 0) or (j < 0)): return res
   if (abs(j1 - j2) > j): return res
   if (j > j1 + j2): return res

   if ((m1 + m2) != m): return res

   j1nm1 = (j1 - m1)/2
   jnj2pm1 = (j - j2 + m1)/2
   j2pm2 = (j2 + m2)/2
   jnj1nm2 = (j - j1 - m2)/2
   j1pj2nj = (j1 + j2 - j)/2

   # check if evenness is valid i.e. j1 and m1 both even/odd
   if (not (((j1nm1 * 2) == (j1 - m1)) and
            ((j2pm2 * 2) == (j2 + m2)) and
            ((j1pj2nj * 2) == (j1 + j2 - j)))):
       return res

   mink = max(max(-jnj2pm1, -jnj1nm2), 0)
   maxk = min(min(j1nm1, j2pm2), j1pj2nj)

   if ((mink/2)*2 != mink):
       iphase = -1
   else:
       iphase = 1

   for k in range(mink, maxk +1):
       tmp =  (factorial[j1nm1 - k] * factorial[jnj2pm1 + k] * factorial[j2pm2 - k]
           * factorial[jnj1nm2 + k] * factorial[k] * factorial[j1pj2nj - k])
       res = res + iphase/tmp
       iphase = - iphase

   if (mink > maxk):
       res = 1.0

   tmp = sqrt(1.0 * factorial[j1pj2nj])
   tmp = tmp * sqrt(factorial[(j1 + j - j2) / 2])
   tmp = tmp * sqrt(factorial[(j2 + j - j1)/2])
   tmp = tmp / sqrt(factorial[(j1 + j2 + j)/2 + 1])
   tmp = tmp * sqrt(1.0 * (j + 1))
   tmp = tmp * sqrt(factorial[(j1 + m1)/2])
   tmp = tmp * sqrt(factorial[j1nm1])
   tmp = tmp * sqrt(factorial[j2pm2])
   tmp = tmp * sqrt(factorial[(j2 - m2)/2])
   tmp = tmp * sqrt(factorial[(j + m)/2])
   tmp = tmp * sqrt(factorial[(j - m)/2])

   return res * tmp


@cython.cdivision(True)
cdef int coefficient_c(int l, int m) noexcept nogil:
    return ((l+1)*(l+1)) - l + m - 1


@cython.cdivision(True)
cdef int coefficient_r(int l, int m) noexcept nogil:
    return (l+1)*(l+2)/2 - l + m - 1


cdef double complex invariant_P_c(double complex[::1] coeffs, int l, int l1, int l2) noexcept nogil:
    cdef double complex res = 0.0
    cdef double complex p, coeff
    cdef double c
    cdef int m, idx1, idx2, m1
    for m in range(-l, l + 1):
        p = 0.0
        for m1 in range(-l1, l1 + 1):
            c = clebsch(2*l1, 2*m1, 2* l2, 2*(m - m1), 2*l, 2*m)
            if c == 0 or c != c:
                continue
            idx1 = coefficient_c(l1, m1)
            idx2 = coefficient_c(l2, m - m1)
            p += (c * coeffs[idx1] * coeffs[idx2])
        idx1 = coefficient_c(l,m)
        coeff = coeffs[idx1]
        res += p * coeff.conjugate()
    return res

cdef double complex invariant_P_r(double complex[::1] coeffs, int l, int l1, int l2) noexcept nogil:
    cdef double complex res = 0.0
    cdef double complex p, coeff
    cdef double c
    cdef int m, idx1, idx2, m1
    for m in range(0, l + 1):
        p = 0.0
        for m1 in range(0, l1 + 1):
            c = clebsch(2*l1, 2*m1, 2* l2, 2*(m - m1), 2*l, 2*m)
            if c == 0 or c != c:
                continue
            idx1 = coefficient_r(l1, m1)
            idx2 = coefficient_r(l2, m - m1)
            p += (c * coeffs[idx1] * coeffs[idx2])
        idx1 = coefficient_r(l,m)
        coeff = coeffs[idx1]
        res += p * coeff.conjugate()
    return res

# See Burel & Henocq, Sig Proc 1995 for details
cpdef p_invariants_c(coeffs):
    cdef int l_max = <int>(sqrt(len(coeffs))) - 1
    cdef int l, l1, l2
    cdef double complex i
    even_inv = []
    odd_inv = []
    for l2 in range(1, l_max + 1):
        for l1 in range(l2, l_max+1):
            for l in range(l1, l_max+1):
                if (l1 - l2 > l) or (l1 + l2  < l):
                    continue
                if (((l % 2 == 0) or (l2 != l1)) and
                        ((l2 % 2 == 0) or (l1 != l))):
                    i = invariant_P_c(coeffs, l, l1, l2)
                else:
                    continue
                if (l + l1 + l2) % 2 == 0:
                    even_inv.append(i)
                else:
                    odd_inv.append(i)
    R = np.real(even_inv)
    R = np.sign(R) * np.cbrt(R)
    J = np.imag(odd_inv)
    J = np.sign(J) * np.cbrt(J)
    return np.hstack([R, J])

cpdef p_invariants_r(coeffs):
    cdef int l_max = <int>((-3 + sqrt(8 * len(coeffs) + 1)) / 2)
    cdef int l, l1, l2
    cdef double complex i
    even_inv = []
    odd_inv = []
    for l2 in range(1, l_max + 1):
        for l1 in range(l2, l_max+1):
            for l in range(l1, l_max+1):
                if (l1 - l2 > l) or (l1 + l2  < l):
                    continue
                if (((l % 2 == 0) or (l2 != l1)) and
                        ((l2 % 2 == 0) or (l1 != l))):
                    i = invariant_P_r(coeffs, l, l1, l2)
                else:
                    continue
                if (l + l1 + l2) % 2 == 0:
                    even_inv.append(i)
                else:
                    odd_inv.append(i)
    R = np.real(even_inv)
    R = np.sign(R) * np.cbrt(R)
    J = np.imag(odd_inv)
    J = np.sign(J) * np.cbrt(J)
    return np.hstack([R, J])


cpdef double clebsch_gordan(l1, m1, l2, m2, l, m):
    return clebsch(2*l1, 2*m1, 2*l2, 2*m2, 2*l, 2*m)

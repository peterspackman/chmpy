import numpy as np
from chmpy.util.num import spherical_to_cartesian_mgrid
from ._sht import (
    AssocLegendre, analysis_kernel_real, analysis_kernel_cplx,
    synthesis_kernel_real, synthesis_kernel_cplx,
    expand_coeffs_to_full
)
from scipy.special import roots_legendre
from scipy.fft import fft, ifft

_SHT_CACHE = {}

def _next_power_of_2(n):
  i = 1
  while i < n: i *= 2
  return i


def _closest_int_with_only_prime_factors_up_to_fmax(n, fmax=7):
    if (n <= fmax):
        return n
    if (fmax < 2):
        return 0
    if (fmax == 2):
        return _next_power_of_2(n);

    n -= 2 - (n & 1)
    f = 2
    while (f != n):
        n += 2
        f = 2
        while ((2*f <= n) and ((n & f) == 0)):
            f *= 2 # no divisions for factor 2.
        k = 3
        while ((k<=fmax) and (k*f <= n)):
            while ((k*f <= n) and (n%(k*f)==0)):
                f *= k
            k += 2

    k = _next_power_of_2(n) # what is the closest power of 2 ?

    if ((k - n) * 33 < n):
        return k # rather choose power of 2 if not too far (3%)
    return n


class SHT:
    def __init__(self, lm, nphi=None, ntheta=None):
        self.lmax = lm
        self.plm = AssocLegendre(lm)

        if nphi is None:
            self.nphi = _closest_int_with_only_prime_factors_up_to_fmax(2 * lm + 1)
        else:
            self.nphi = nphi
        # avoid the poles
        self.phi = np.arange(0, self.nphi) * 2 * np.pi / self.nphi

        if ntheta is None:
            n = self.lmax + 1
            n += (n & 1)
            n = ((n + 7) // 8) * 8
            self.ntheta = n
        else:
            self.ntheta = ntheta
        
        self.cos_theta, self.weights, self.total_weight = roots_legendre(self.ntheta, mu=True)
        self.weights *= 4 * np.pi / self.total_weight
        self.theta = np.arccos(self.cos_theta)

        self.fft_work_array = np.empty(self.nphi, dtype=np.complex128)
        self.plm_work_array = np.empty(self.nplm())
        self._grid = None
        self._grid_cartesian = None

    def idx_c(self, l, m):
        return l * (l + 1) + m

    def nlm(self):
        return (self.lmax + 1) * (self.lmax + 1)

    def nplm(self):
        return (self.lmax + 1) * (self.lmax + 2) // 2

    def compute_on_grid(self, func):
        values = func(*self.grid)
        return values

    def analysis_pure_python(self, values):
        """much slower"""
        coeffs = np.zeros(self.nplm(), dtype=np.complex128)
        for itheta, (ct, w) in enumerate(zip(self.cos_theta, self.weights)):
            self.fft_work_array[:] = values[itheta, :]

            fft(self.fft_work_array, norm="forward", overwrite_x=True) 
            self.plm.evaluate_batch(ct, result=self.plm_work_array)
            plm_idx = 0
	        # m = 0 case
            for l in range(self.lmax + 1):
                p = self.plm_work_array[plm_idx]
                coeffs[plm_idx] += self.fft_work_array[0] * p * w
                plm_idx += 1

            # because we don't include a phase factor (-1)^m in our
            # Associated Legendre Polynomials, we need a factor here.
            # which alternates with m and l
            for m in range(1, self.lmax + 1):
                sign = -1 if m & 1 else 1
                for l in range(m, self.lmax + 1):
                    p = self.plm_work_array[plm_idx]
                    coeffs[plm_idx] += sign * self.fft_work_array[m] * p * w
                    plm_idx += 1
        return coeffs

    def analysis_pure_python_cplx(self, values):
        """much slower"""
        coeffs = np.zeros(self.nlm(), dtype=np.complex128)
        rlm = np.zeros(self.nlm(), dtype=np.complex128)
        ilm = np.zeros(self.nlm(), dtype=np.complex128)
        for itheta, (ct, w) in enumerate(zip(self.cos_theta, self.weights)):
            self.fft_work_array[:] = values[itheta, :]

            fft(self.fft_work_array, norm="forward", overwrite_x=True) 
            self.plm.evaluate_batch(ct, result=self.plm_work_array)

            plm_idx = 0
            for l in range(self.lmax + 1):
                l_offset = l * (l + 1)
                pw = self.plm_work_array[plm_idx] * w
                coeffs[l_offset] = coeffs[l_offset] + self.fft_work_array[0] * pw
                plm_idx += 1

            # because we don't include a phase factor (-1)^m in our
            # Associated Legendre Polynomials, we need a factor here.
            # which alternates with m
            for m in range(1, self.lmax + 1):
                sign = -1 if m & 1 else 1
                for l in range(m, self.lmax + 1):
                    l_offset = l * (l + 1)
                    pw = self.plm_work_array[plm_idx] * w
                    m_idx_neg = self.nphi - m
                    m_idx_pos = m
                    rr = sign * self.fft_work_array[m_idx_pos] * pw
                    ii = sign * self.fft_work_array[m_idx_neg] * pw
                    if m & 1:
                        ii = - ii

                    coeffs[l_offset - m] = coeffs[l_offset - m] + ii
                    coeffs[l_offset + m] = coeffs[l_offset + m] + rr
                    plm_idx += 1
        return coeffs


    def synthesis_pure_python_cplx(self, coeffs):
        values = np.zeros((self.ntheta, self.nphi), dtype=np.complex128)
        for itheta, ct in enumerate(self.cos_theta):
            self.fft_work_array[:] = 0
            self.plm.evaluate_batch(ct, result=self.plm_work_array)

            plm_idx = 0
	        # m = 0 case
            for l in range(self.lmax + 1):
                l_offset = l * (l + 1)
                p = self.plm_work_array[plm_idx]
                self.fft_work_array[0] += coeffs[l_offset] * p
                plm_idx += 1

            for m in range(1, self.lmax + 1):
                sign = -1 if m & 1 else 1
                for l in range(m, self.lmax + 1):

                    l_offset = l * (l + 1)
                    p = self.plm_work_array[plm_idx]
                    m_idx_neg = self.nphi - m
                    m_idx_pos = m
                    rr = sign * coeffs[l_offset + m] * p
                    ii = sign * coeffs[l_offset - m] * p
                    if m & 1:
                        ii = - ii
                    self.fft_work_array[m_idx_neg] += ii
                    self.fft_work_array[m_idx_pos] += rr
                    plm_idx += 1

            ifft(self.fft_work_array, norm="forward", overwrite_x=True) 
            values[itheta, :] = self.fft_work_array[:]
        return values

    def synthesis_pure_python(self, coeffs):
        values = np.zeros((self.ntheta, self.nphi))
        for itheta, ct in enumerate(self.cos_theta):
            self.fft_work_array[:] = 0
            self.plm.evaluate_batch(ct, result=self.plm_work_array)

            plm_idx = 0
	        # m = 0 case
            for l in range(self.lmax + 1):
                p = self.plm_work_array[plm_idx]
                self.fft_work_array[0] += coeffs[plm_idx] * p
                plm_idx += 1

            for m in range(1, self.lmax + 1):
                sign = -1 if m & 1 else 1
                for l in range(m, self.lmax + 1):
                    p = self.plm_work_array[plm_idx]
                    rr = 2 * sign * coeffs[plm_idx] * p
                    self.fft_work_array[m] += rr
                    plm_idx += 1

            ifft(self.fft_work_array, norm="forward", overwrite_x=True) 
            values[itheta, :] = self.fft_work_array[:].real
        return values


    def analysis(self, values):
        real = not np.iscomplexobj(values)
        if real:
            kernel = analysis_kernel_real
            coeffs = np.zeros(self.nplm(), dtype=np.complex128)
        else:
            kernel = analysis_kernel_cplx
            coeffs = np.zeros(self.nlm(), dtype=np.complex128)
        for itheta, (ct, w) in enumerate(zip(self.cos_theta, self.weights)):
            self.fft_work_array[:] = values[itheta, :]

            fft(self.fft_work_array, norm="forward", overwrite_x=True) 
            self.plm.evaluate_batch(ct, result=self.plm_work_array)
            kernel(self, w, coeffs)
        return coeffs

    def synthesis(self, coeffs):
        real = (coeffs.size == self.nplm())
        if real:
            kernel = synthesis_kernel_real
            values = np.zeros(self.grid[0].shape)
        else:
            kernel = synthesis_kernel_cplx
            values = np.zeros(self.grid[0].shape, dtype=np.complex128)

        for itheta, ct in enumerate(self.cos_theta):
            self.fft_work_array[:] = 0
            self.plm.evaluate_batch(ct, result=self.plm_work_array)
            kernel(self, coeffs)
            ifft(self.fft_work_array, norm="forward", overwrite_x=True) 

            if real:
                values[itheta, :] = self.fft_work_array[:].real
            else:
                values[itheta, :] = self.fft_work_array[:]
        return values

    def _eval_at_points_real(self, coeffs, theta, phi):
        # verified
        cos_theta = np.cos(theta)
        result = 0.0
        self.plm.evaluate_batch(cos_theta, result=self.plm_work_array)
        plm_idx = 0
        for l in range(0, self.lmax + 1):
            result += self.plm_work_array[plm_idx] * coeffs[plm_idx].real
            plm_idx += 1


        mv = 2 * np.exp(1j * np.arange(1, self.lmax + 1) * phi)
        sign = 1
        for m in range(1, self.lmax + 1):
            tmp = 0.0
            sign *= -1
            for l in range(m, self.lmax + 1):
                # m +ve and m -ve
                tmp += sign * self.plm_work_array[plm_idx] * coeffs[plm_idx]
                plm_idx += 1
            
            result += (tmp.real * mv[m - 1].real + tmp.imag * mv[m - 1].imag)
        return result

    def _eval_at_points_cplx(self, coeffs, theta, phi):
        cos_theta = np.cos(theta)
        result = 0.0
        self.plm.evaluate_batch(cos_theta, result=self.plm_work_array)

        plm_idx = 0
        for l in range(0, self.lmax + 1):
            l_offset = l * (l + 1)
            result += self.plm_work_array[plm_idx] * coeffs[l_offset]
            plm_idx += 1

        mv = np.exp(1j * np.arange(-self.lmax, self.lmax + 1) * phi)
        for m in range(1, self.lmax + 1):
            tmpp = 0.0
            tmpn = 0.0
            for l in range(m, self.lmax + 1):
                l_offset = l * (l + 1)
                tmpp += self.plm_work_array[plm_idx] * coeffs[l_offset + m]
                tmpn += self.plm_work_array[plm_idx] * coeffs[l_offset - m]
                plm_idx += 1
            result += tmpp * mv[self.lmax + m]
            result += tmpn * np.conj(mv[self.lmax - m])
        return result

    def evaluate_at_points(self, coeffs, theta, phi):
        real = (coeffs.size == self.nplm())
        if real:
            return self._eval_at_points_real(coeffs, theta, phi)
        # assumes coeffs are the real transform order
        else:
            return self._eval_at_points_cplx(coeffs, theta, phi)


    def complete_coefficients(self, coeffs):
        return expand_coeffs_to_full(self.lmax, coeffs)

    @property
    def grid(self):
        return np.meshgrid(
            self.theta,
            self.phi, indexing="ij"
        )

    @property
    def grid_cartesian(self):
        "The set of cartesian grid points for this SHT"
        theta, phi = self.grid
        r = np.ones_like(theta)
        return spherical_to_cartesian_mgrid(r, theta, phi)

def test_func(theta, phi):
    return (1.0 + 0.01 * np.cos(theta) +
                0.1 * (3.0 * np.cos(theta) * np.cos(theta) - 1.0) +
                (np.cos(phi) + 0.3 * np.sin(phi)) * np.sin(theta) +
                (np.cos(2.0 * phi) + 0.1 * np.sin(2.0 * phi)) * 
                 np.sin(theta) * np.sin(theta) * (7.0 * np.cos(theta) * np.cos(theta) - 1.0) * 3.0/8.0
           )

def plot_sphere(name, grid, values):
    """Plot a function on a spherical surface.

    Parameters
    ----------
    name: str
        used for the title and the output filename
    grid: array_like
        theta, phi values from an angular grid on a sphere
    values: array_like
        scalar values of the function associated with each grid point
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm, colors
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=plt.figaspect(1.0))
    theta, phi = grid
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    fmin, fmax = np.min(values), np.max(values)
    fcolors = (values - fmin) / (fmax - fmin)
    fcolors = fcolors.reshape(theta.shape)

    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, facecolors=cm.viridis(fcolors), shade=True
    )
    ax.set_axis_off()
    plt.title("Contours of {}".format(name))
    plt.savefig("{}.png".format(name), dpi=300, bbox_inches="tight")

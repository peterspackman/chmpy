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
    r"""
    A class to encapsulate the re-usable data for a Spherical Harmonic Transform (SHT).

    Attributes:
        lmax: the maximum angular momentum of the SHT, affects grid size etc.
        plm: class to evaluate associated Legendre polynomials
        nphi: the number of phi angular grid points
        ntheta: the number of theta angular grid points
        phi: the phi angular grid points (equispaced) between [i, 2 \pi]
        cos_theta: cos values of the theta grid (evaluated as Gauss-Legendre quadrature points)
        weights: the Gauss-Legendre grid weights
        theta: the theta angular grid points (derived from cos_theta)
        fft_work_array: an internal work array for the various FFTs done in the transform
        plm_work_array: an internal work array for the evaluate of plm values

    """

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
        "the number of complex SHT coefficients"
        return (self.lmax + 1) * (self.lmax + 1)

    def nplm(self):
        "the number of real SHT coefficients (i.e. legendre polynomial terms)"
        return (self.lmax + 1) * (self.lmax + 2) // 2

    def compute_on_grid(self, func):
        "compute the values of `func` on the SHT grid"
        values = func(*self.grid)
        return values

    def analysis_pure_python(self, values):
        """
        Perform the forward SHT i.e. evaluate the given SHT coefficients
        given the values of the (real-valued) function at the grid points
        used in the transform.

        *NOTE*: this is implemented in pure python so will be much slower than
        just calling analysis, but it is provided here as a reference implementation

        Arguments:
            values (np.ndarray): the evaluated function at the SHT grid points

        Returns:
            np.ndarray the set of spherical harmonic coefficients
        """
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
        """
        Perform the forward SHT i.e. evaluate the given SHT coefficients
        given the values of the (complex-valued) function at the grid points
        used in the transform.

        *NOTE*: this is implemented in pure python so will be much slower than
        just calling analysis, but it is provided here as a reference implementation

        Arguments:
            values (np.ndarray): the evaluated function at the SHT grid points

        Returns:
            np.ndarray the set of spherical harmonic coefficients
        """

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
        """
        Perform the inverse SHT i.e. evaluate the given (complex-valued) function at the
        grid points used in the transform.


        *NOTE*: this is implemented in pure python so will be much slower than
        just calling analysis, but it is provided here as a reference implementation

        Arguments:
            coeffs (np.ndarray): the set of spherical harmonic coefficients

        Returns:
            np.ndarray the evaluated function at the SHT grid points
        """

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
        """
        Perform the inverse SHT i.e. evaluate the given (real-valued) function at the
        grid points used in the transform.

        *NOTE*

        Arguments:
            coeffs (np.ndarray): the set of spherical harmonic coefficients

        Returns:
            np.ndarray the evaluated function at the SHT grid points
        """

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
        """
        Perform the forward SHT i.e. evaluate the given SHT coefficients
        given the values of the function at the grid points used in the transform.

        Arguments:
            values (np.ndarray): the evaluated function at the SHT grid points

        Returns:
            np.ndarray the set of spherical harmonic coefficients
        """

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
        """
        Perform the inverse SHT i.e. evaluate the given function at the
        grid points used in the transform.

        Arguments:
            coeffs (np.ndarray): the set of spherical harmonic coefficients

        Returns:
            np.ndarray the evaluated function at the SHT grid points
        """
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
        # very slow pure python implementation, should move to cython
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
        # very slow pure python implementation, should move to cython
        cos_theta = np.cos(theta)
        result = 0.0
        self.plm.evaluate_batch(cos_theta, result=self.plm_work_array)

        plm_idx = 0
        for l in range(0, self.lmax + 1):
            l_offset = l * (l + 1)
            result += self.plm_work_array[plm_idx] * coeffs[l_offset]
            plm_idx += 1

        mv = np.exp(1j * np.arange(-self.lmax, self.lmax + 1) * phi)
        cos_vals = 0.0
        sin_vals = 0.0
        for m in range(1, self.lmax + 1):
            tmpr = 0.0
            tmpc = 0.0
            for l in range(m, self.lmax + 1):
                l_offset = l * (l + 1)
                tmpr += self.plm_work_array[plm_idx] * coeffs[l_offset + m]
                tmpc += self.plm_work_array[plm_idx] * coeffs[l_offset - m]
                plm_idx += 1
            if m & 1:
                tmpc = -tmpc
            cos_vals += np.real(mv[self.lmax + m]) * (tmpr + tmpc)
            sin_vals += np.imag(mv[self.lmax - m]) * (tmpr - tmpc)
        result += np.real(cos_vals) - np.imag(sin_vals) + 1j * (np.imag(cos_vals) + np.real(sin_vals))
        return result

    def evaluate_at_points(self, coeffs, theta, phi):
        r"""
        Evaluate the value of the function described in terms of the provided SH
        coefficients at the provided (angular) points. 
        Will attempt to detect if the provided coefficients are from a real
        or a complex transform.

        Note that this can be quite slow, especially in comparison with just 
        synthesis step.

        Arguments:
            coeffs (np.ndarray): the set of spherical harmonic coefficients
            theta (np.ndarray): the angular coordinates \theta
            phi (np.ndarray): the angular coordinates \phi

        Returns:
            np.ndarray the evaluated function values
        """
        real = (coeffs.size == self.nplm())
        if real:
            return self._eval_at_points_real(coeffs, theta, phi)
        # assumes coeffs are the real transform order
        else:
            return self._eval_at_points_cplx(coeffs, theta, phi)

    def complete_coefficients(self, coeffs):
        """
        Construct the complete set of SHT coefficients
        for a given real analysis. Should be equivalent to performing
        a complex valued SHT with the imaginary values being zero.

        Arguments:
            coefficients (np.ndarray): the set of spherical harmonic coefficients

        Returns:
            np.ndarray the full set of spherical harmonic coefficients for a complext transform
        """
        return expand_coeffs_to_full(self.lmax, coeffs)

    @property
    def grid(self):
        "The set of grid points [\theta, \phi] for this SHT"
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


    def power_spectrum(self, coeffs) -> np.ndarray:
        r"""
        Evaluate the power spectrum of the function described in terms of the provided SH
        coefficients.

        Arguments:
            coeffs (np.ndarray): the set of spherical harmonic coefficients

        Returns:
            np.ndarray the evaluated power spectrum
        """

        real = (coeffs.size == self.nplm())
        if real:
            n = len(coeffs)
            l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
            spectrum = np.zeros(l_max + 1)
            
            pattern = np.concatenate([np.arange(m, l_max + 1) for m in range(l_max + 1)])
            boundary = l_max + 1
            np.add.at(spectrum, pattern[:boundary], np.abs(coeffs[:boundary])**2)
            np.add.at(spectrum, pattern[boundary:], 2 * np.abs(coeffs[boundary:])**2)
            spectrum /= (2 * pattern[:boundary] + 1)
            
            return spectrum
        else:
            l_max = int(np.sqrt(len(coeffs))) - 1
            spectrum = np.empty(l_max + 1)
            coeffs2 = np.abs(coeffs) **2
            idx = 0
            for l in range(l_max + 1):
                count = 2 * l + 1
                spectrum[l] = np.sum(coeffs2[idx:idx + count]) / count
                idx += count
            return spectrum

    def invariants_kazhdan(self, coeffs):
        r"""
        Evaluate the rotation invariants as detailed in Kazhdan et al.[1] for the
        provided set of SHT coefficients

        *NOTE* this is not a well-tested implementation, and is not complete
        for the set of invariants described in the work.

        Arguments:
            coeffs(np.ndarray): the set of spherical harmonic coefficients

        Returns:
            np.ndarray the evaluated rotation invariants

        References:
        ```
        [1] Kazhdan et al. Proc. 2003 Eurographics/ACM SIGGRAPH SGP, (2003)
            https://dl.acm.org/doi/10.5555/882370.882392
        ```

        """

        invariants = np.empty(self.lmax + 1)
        for lvalue in range(0, self.lmax + 1):
            values = np.zeros((self.ntheta, self.nphi))
            for itheta, ct in enumerate(self.cos_theta):
                self.fft_work_array[:] = 0
                self.plm.evaluate_batch(ct, result=self.plm_work_array)

                plm_idx = 0
                # m = 0 case
                for l in range(self.lmax + 1):
                    if l == lvalue:
                        p = self.plm_work_array[plm_idx]
                        self.fft_work_array[0] += coeffs[plm_idx] * p
                    plm_idx += 1

                for m in range(1, self.lmax + 1):
                    sign = -1 if m & 1 else 1
                    for l in range(m, self.lmax + 1):
                        if l == lvalue:
                            p = self.plm_work_array[plm_idx]
                            rr = 2 * sign * coeffs[plm_idx] * p
                            self.fft_work_array[m] += rr
                        plm_idx += 1

                ifft(self.fft_work_array, norm="forward", overwrite_x=True) 
                values[itheta, :] = self.fft_work_array[:].real
            invariants[lvalue] = np.sum(values ** 2) / values.size / (2 * lvalue + 1)
        return invariants


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

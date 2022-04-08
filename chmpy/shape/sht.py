import numpy as np
from chmpy.util.num import spherical_to_cartesian_mgrid
from .assoc_legendre import AssocLegendre
from scipy.special import roots_legendre
from scipy.fft import fft, ifft


_SHT_CACHE = {}

class SHT:
    def __init__(self, lm):
        self.lmax = lm
        self.plm = AssocLegendre(lm)

        self.nphi = 2 * lm + 1
        self.phi = np.arange(self.nphi) * 2 / self.nphi * np.pi - np.pi
        self.ntheta = 1
        while (self.ntheta <= self.nphi):
            self.ntheta *= 2
        
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

    def analysis(self, values):
        coeffs = np.zeros(self.nlm(), dtype=np.complex128)
        for itheta, (ct, w) in enumerate(zip(self.cos_theta, self.weights)):
            self.fft_work_array[:] = values[itheta, :]

            fft(self.fft_work_array, norm="forward", overwrite_x=True) 
            self.plm.evaluate_batch(ct, result=self.plm_work_array)
            plm_idx = 0
	        # m = 0 case
            for l in range(self.lmax + 1):
                l_offset = l * (l + 1)
                p = self.plm_work_array[plm_idx]
                coeffs[l_offset] += np.conj(self.fft_work_array[0]) * p * w
                plm_idx += 1

            for m in range(1, self.lmax + 1):
                for l in range(m, self.lmax + 1):
                    l_offset = l * (l + 1)
                    p = self.plm_work_array[plm_idx]
                    m_idx_neg = self.nphi - m
                    m_idx_pos = m
                    if m & 1:
                        coeffs[l_offset - m] += self.fft_work_array[m_idx_neg] * p * w
                        coeffs[l_offset + m] += self.fft_work_array[m_idx_pos] * p * w
                    else:
                        coeffs[l_offset - m] += np.conj(self.fft_work_array[m_idx_neg]) * p * w
                        coeffs[l_offset + m] += np.conj(self.fft_work_array[m_idx_pos]) * p * w
                    plm_idx += 1
        return coeffs

    def synthesis(self, coeffs):
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
                for l in range(m, self.lmax + 1):
                    l_offset = l * (l + 1)
                    p = self.plm_work_array[plm_idx]
                    m_idx_neg = self.nphi - m
                    m_idx_pos = m
                    if m & 1:
                        self.fft_work_array[m_idx_neg] += coeffs[l_offset - m] * p
                        self.fft_work_array[m_idx_pos] += coeffs[l_offset + m] * p
                    else:
                        self.fft_work_array[m_idx_neg] += np.conj(coeffs[l_offset - m]) * p
                        self.fft_work_array[m_idx_pos] += np.conj(coeffs[l_offset + m]) * p
                    plm_idx += 1

            ifft(self.fft_work_array, norm="forward", overwrite_x=True) 
            values[itheta, :] = self.fft_work_array[:]
        return values

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


if __name__ == "__main__":

    plm = AssocLegendre(4)

    expected_plm = np.array([
        0.28209479177387814,
        0.24430125595145993,
        -0.07884789131313,
        -0.326529291016351,
        -0.24462907724141,
        0.2992067103010745,
        0.33452327177864455,
        0.06997056236064664,
        -0.25606603842001846,
        0.2897056515173922,
        0.3832445536624809,
        0.18816934037548755,
        0.27099482274755193,
        0.4064922341213279,
        0.24892463950030275,
    ])

    print("Plm correct: ", np.allclose(expected_plm, plm.evaluate_batch(0.5)))


    s = SHT(4)
    coeffs = s.analysis(test_func)

    expected = np.array([
        np.sqrt(4 * np.pi), # 0 0 
        -np.sqrt(2 * np.pi / 3) - 0.3 * np.sqrt(2 * np.pi / 3) *1j, # 1 -1
	    0.01 * np.sqrt(4 * np.pi / 3.0), # 1 0
        -np.sqrt(2 * np.pi / 3) + 0.3 * np.sqrt(2 * np.pi / 3) *1j, # 1 1
        0.0, # 2 -2
        0.0, # 2 -1
        0.1 * np.sqrt(16 * np.pi / 5.0), # 2  0
        0.0, # 2  1
        0.0, # 2  2
        0.0, # 3 -3
        0.0, # 3 -2
        0.0, # 3 -1
        0.0, # 3  0
        0.0, # 3  1
        0.0, # 3  2
        0.0, # 3  3
        0.0, # 4 -4
        0.0, # 4 -3
        0.5 * np.sqrt(2 * np.pi / 5.0) - 0.05 * np.sqrt(2.0 * np.pi / 5.0) * 1j, # 4  2
        0.0, # 4 -1
        0.0, # 4  0
        0.0, # 4  1
        0.5 * np.sqrt(2 * np.pi / 5.0) + 0.05 * np.sqrt(2.0 * np.pi / 5.0) * 1j, # 4  2
        0.0, # 4  3
        0.0, # 4  4
    ], dtype=np.complex128)

    if not np.allclose(expected, coeffs):
        print(coeffs - expected)
    else:
        print("Coeffs correct: true")

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

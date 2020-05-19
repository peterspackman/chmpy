import numpy as np
from chmpy.util.num import spherical_to_cartesian

_SHT_CACHE = {}


class SHT:
    """Class encapsulating the logic of spherical harmonic transform implementations

    Parameters
    ----------
    l_max: int
        maximum angular momentum for the transform
    """

    _shtns = None
    _l_max = 2
    _grid = None
    _grid_cartesian = None

    def __init__(self, l_max):
        import shtns

        self._l_max = l_max
        if l_max not in _SHT_CACHE:
            sht = shtns.sht(l_max, l_max)
            ntheta, nphi = sht.set_grid()
            _SHT_CACHE[l_max] = (sht, ntheta, nphi)

        self._shtns, self.ntheta, self.nphi = _SHT_CACHE[l_max]

    @property
    def mgrid(self):
        return np.meshgrid(
            np.arccos(self._shtns.cos_theta),
            np.arange(self.nphi) * (2 * np.pi / self.nphi),
        )

    @property
    def grid(self):
        "The set of angular grid points for this SHT"
        if self._grid is None:
            nphi = self.nphi
            self.phi, self.theta = np.meshgrid(
                np.arccos(self._shtns.cos_theta), np.arange(nphi) * (2 * np.pi / nphi)
            )
            self.phi = self.phi.flatten()
            self.theta = self.theta.flatten()
            self._grid = np.vstack((self.theta, self.phi)).transpose()
            self._grid_cartesian = spherical_to_cartesian(
                np.c_[np.ones(self._grid.shape[0]), self._grid[:, 1], self._grid[:, 0]]
            )
        return self._grid

    @property
    def grid_cartesian(self):
        "The set of cartesian grid points for this SHT"
        if self._grid_cartesian is None:
            nphi = self.nphi
            self.phi, self.theta = np.meshgrid(
                np.arccos(self._shtns.cos_theta), np.arange(nphi) * (2 * np.pi / nphi)
            )
            self.phi = self.phi.flatten()
            self.theta = self.theta.flatten()
            self._grid = np.vstack((self.theta, self.phi)).transpose()
            self._grid_cartesian = spherical_to_cartesian(
                np.c_[np.ones(self._grid.shape[0]), self._grid[:, 1], self._grid[:, 0]]
            )
        return self._grid_cartesian

    def analyse(self, values):
        """Perform a spherical harmonic transform given a grid and a set of values

        Parameters
        ----------
        values: :obj:`np.ndarray`
            set of scalar function values associated with grid points
        """
        desired_shape = self._shtns.spat_shape[::-1]
        grid = self._grid
        if values.dtype == np.complex128:
            return self._shtns.analys_cplx(values.reshape(desired_shape).transpose())
        else:
            return self._shtns.analys(values.reshape(desired_shape).transpose())

    def synth_cplx(self, coefficients):
        """Perform an inverse spherical harmonic transform given a set of coefficients

        Arguments
        ----------
        coefficients: :obj:`np.ndarray`
            set of spherical harmonic coefficient
        """
        max_coeff = (self.l_max + 1) ** 2
        return self._shtns.synth_cplx(coefficients[:max_coeff]).transpose().flatten()

    def synth_real(self, coefficients):
        """Perform an inverse spherical harmonic transform given a set of coefficients

        Arguments
        ----------
        coefficients: :obj:`np.ndarray`
            set of spherical harmonic coefficient
        """

        max_coeff = (self._l_max + 2) * (self._l_max + 1) // 2
        return self._shtns.synth(coefficients[:max_coeff]).transpose().flatten()

    @property
    def l_max(self):
        """Maximum angular momenta used in this SHT"""
        return self._l_max


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

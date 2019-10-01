import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from os.path import join, dirname
from .element_data import vdw_radii
import numpy as np

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))
_DOMAIN = _INTERPOLATOR_DATA.f.domain
_RHO = _INTERPOLATOR_DATA.f.rho
_GRAD_RHO = _INTERPOLATOR_DATA.f.grad_rho


class PromoleculeDensity:
    def __init__(self, mol):
        n, pos = mol
        self.elements = np.asarray(n)
        self.positions = np.asarray(pos)

        if np.any(self.elements < 1) or np.any(self.elements > 103):
            raise ValueError("All elements must be atomic numbers between [1,103]")

        self.rho_interpolators = {
            el_number: interp1d(
                _DOMAIN,
                _RHO[el_number - 1, :],
                assume_sorted=True,
                fill_value="extrapolate",
            )
            for el_number in np.unique(self.elements)
        }
        self.grad_rho_interpolators = {
            el_number: interp1d(
                _DOMAIN,
                _GRAD_RHO[el_number - 1, :],
                assume_sorted=True,
                fill_value="extrapolate",
            )
            for el_number in np.unique(self.elements)
        }
        self.principal_axes, _, _ = np.linalg.svd((self.positions - self.centroid).T)
        self.aa_positions = np.dot(
            self.positions - self.centroid, self.principal_axes.T
        )
        self.vdw_radii = vdw_radii(self.elements)

    def rho(self, positions, frame="xyz", return_grad=False):
        if not isinstance(positions, np.ndarray):
            positions = np.asarray(positions)
        if frame == "molecule":
            r = cdist(self.aa_positions, positions)
        else:
            r = cdist(self.positions, positions)
        rho = np.zeros(positions.shape[0], dtype=np.float32)
        if return_grad:
            grad_rho = np.zeros(positions.shape[0], dtype=np.float32)
        for el in np.unique(self.elements):
            idxs = np.where(self.elements == el)[0]
            rho[:] += np.sum(self.rho_interpolators[el](r[idxs, :]), axis=0)
            if return_grad:
                grad_rho[:] += self.grad_rho_interpolators[n](r[i, :])

        if return_grad:
            return rho, grad_rho
        return rho

    @property
    def centroid(self):
        return np.mean(self.positions, axis=0)

    @property
    def natoms(self):
        return len(self.elements)

    def aabb(self, vdw_buffer=2.0):
        extra = self.vdw_radii[:, np.newaxis] + vdw_buffer
        return (
            np.min(self.aa_positions - extra, axis=0),
            np.max(self.aa_positions + extra, axis=0),
        )

    def __repr__(self):
        return "<PromoleculeDensity: {} atoms, centre={}>".format(
            self.natoms, tuple(self.centroid)
        )

    @classmethod
    def from_xyz_file(cls, filename):
        from .xyz_file import parse_xyz_file

        return cls(parse_xyz_file(filename))


class StockholderWeight:
    def __init__(self, dens_a, dens_b):
        assert isinstance(dens_a, PromoleculeDensity) and isinstance(
            dens_b, PromoleculeDensity
        ), "Must be PromoleculeDensity instances"
        self.dens_a = dens_a
        self.dens_b = dens_b

    @property
    def positions(self):
        return np.r_[self.dens_a.positions, self.dens_b.positions]

    @property
    def vdw_radii(self):
        return np.r_[self.dens_a.vdw_radii, self.dens_b.vdw_radii]

    def weight(self, positions, frame="xyz"):
        rho_a = self.dens_a.rho(positions, frame=frame, return_grad=False)
        rho_b = self.dens_b.rho(positions, frame=frame, return_grad=False)
        return rho_a / (rho_a + rho_b)

    @classmethod
    def from_xyz_files(cls, f1, f2):
        from .xyz_file import parse_xyz_file

        return cls(
            PromoleculeDensity(parse_xyz_file(f1)),
            PromoleculeDensity(parse_xyz_file(f2)),
        )

    def bb(self, vdw_buffer=2.0):
        extra = self.dens_a.vdw_radii[:, np.newaxis] + vdw_buffer
        return (
            np.min(self.dens_a.positions - extra, axis=0),
            np.max(self.dens_a.positions + extra, axis=0),
        )

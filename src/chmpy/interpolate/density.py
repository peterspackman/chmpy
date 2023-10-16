import numpy as np
from scipy.spatial import cKDTree as KDTree
from os.path import join, dirname
from chmpy.core.element import vdw_radii
from ._density import PromoleculeDensity as cPromol, StockholderWeight as cStock
import numpy as np

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))
_DOMAIN = _INTERPOLATOR_DATA.f.domain
_RHO = _INTERPOLATOR_DATA.f.rho
_GRAD_RHO = _INTERPOLATOR_DATA.f.grad_rho


class PromoleculeDensity:
    def __init__(self, mol):
        n, pos = mol
        self.elements = np.asarray(n, dtype=np.int32)
        self.positions = np.asarray(pos, dtype=np.float32)
        if np.any(self.elements < 1) or np.any(self.elements > 103):
            raise ValueError("All elements must be atomic numbers between [1,103]")
        self.rho_data = np.empty(
            (self.elements.shape[0], _DOMAIN.shape[0]), dtype=np.float32
        )
        for i, el in enumerate(self.elements):
            self.rho_data[i, :] = _RHO[el - 1, :]
        self.dens = cPromol(self.positions, _DOMAIN, self.rho_data)
        self.principal_axes, _, _ = np.linalg.svd((self.positions - self.centroid).T)
        self.vdw_radii = vdw_radii(self.elements)

    def rho(self, positions):
        positions = np.asarray(positions, dtype=np.float32)
        return self.dens.rho(positions)

    @property
    def centroid(self):
        return np.mean(self.positions, axis=0)

    @property
    def natoms(self):
        return len(self.elements)

    def bb(self, vdw_buffer=3.8):
        extra = self.vdw_radii[:, np.newaxis] + vdw_buffer
        return (
            np.min(self.positions - extra, axis=0),
            np.max(self.positions + extra, axis=0),
        )

    def __repr__(self):
        return "<PromoleculeDensity: {} atoms, centre={}>".format(
            self.natoms, tuple(self.centroid)
        )

    def d_norm(self, positions):
        pos = self.positions
        tree = KDTree(pos)
        # make sure k is enough should be enough for d_norm to be correct
        dists, idxs = tree.query(positions, k=min(6, self.natoms))
        d_norm = np.empty(dists.shape[0])
        vecs = np.empty(positions.shape)
        for j, (d, i) in enumerate(zip(dists, idxs)):
            i = i[i < pos.shape[0]]
            vdw = self.vdw_radii[i]
            d_n = (d - vdw) / vdw
            p = np.argmin(d_n)
            d_norm[j] = d_n[p]
            vecs[j] = (pos[p] - positions[j]) / vdw[p]
        if dists.ndim == 1:
            return dists, d_norm, vecs
        return dists[:, 0], d_norm, vecs

    @classmethod
    def from_xyz_file(cls, filename):
        from chmpy.fmt.xyz_file import parse_xyz_file

        els, pos = parse_xyz_file(filename)
        els = np.array([x.atomic_number for x in els])
        return cls((els, pos))


class StockholderWeight:
    def __init__(self, dens_a, dens_b, background=0.0):
        assert isinstance(dens_a, PromoleculeDensity) and isinstance(
            dens_b, PromoleculeDensity
        ), "Must be PromoleculeDensity instances"
        self.dens_a = dens_a
        self.dens_b = dens_b
        self.s = cStock(dens_a.dens, dens_b.dens, background=background)

    @property
    def positions(self):
        return np.r_[self.dens_a.positions, self.dens_b.positions]

    @property
    def vdw_radii(self):
        return np.r_[self.dens_a.vdw_radii, self.dens_b.vdw_radii]

    def weights(self, positions):
        pos = positions.astype(np.float32)
        results = []
        return self.s.weights(positions.astype(np.float32))

    def d_norm(self, positions):
        d_a, d_norm_a, vecs_a = self.dens_a.d_norm(positions)
        d_b, d_norm_b, vecs_b = self.dens_b.d_norm(positions)
        dp = np.einsum("ij,ij->i", vecs_a, vecs_b)
        angles = dp / (np.linalg.norm(vecs_a, axis=1) * np.linalg.norm(vecs_b, axis=1))
        return d_a, d_b, d_norm_a, d_norm_b, dp, angles

    @classmethod
    def from_xyz_files(cls, f1, f2):
        return cls(
            PromoleculeDensity.from_xyz_file(f1), PromoleculeDensity.from_xyz_file(f2)
        )

    @classmethod
    def from_arrays(cls, n1, p1, n2, p2, unit="angstrom", **kwargs):
        return cls(PromoleculeDensity((n1, p1)), PromoleculeDensity((n2, p2)), **kwargs)

    def bb(self, vdw_buffer=3.8):
        extra = self.dens_a.vdw_radii[:, np.newaxis] + vdw_buffer
        return (
            np.min(self.dens_a.positions - extra, axis=0),
            np.max(self.dens_a.positions + extra, axis=0),
        )

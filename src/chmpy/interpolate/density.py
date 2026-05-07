from os.path import dirname, join

import numpy as np
from scipy.spatial import cKDTree as KDTree

from chmpy.core.element import vdw_radii

from . import _backends

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
        self.principal_axes, _, _ = np.linalg.svd((self.positions - self.centroid).T)
        self.vdw_radii = vdw_radii(self.elements)
        self._cython_dens = None

    @property
    def dens(self):
        """Backwards-compatible cython density object.

        Built lazily so that the surface fast path doesn't require the cython
        extension to be present. The shape-descriptor code still uses this for
        `sphere_promolecule_radii`/`sphere_stockholder_radii` (a follow-up will
        port those to the python backend too).
        """
        if self._cython_dens is None:
            from ._density import PromoleculeDensity as _cPromol

            self._cython_dens = _cPromol(self.positions, _DOMAIN, self.rho_data)
        return self._cython_dens

    def rho(self, positions):
        return _backends.rho(self.positions, self.rho_data, _DOMAIN, positions)

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
        return f"<PromoleculeDensity: {self.natoms} atoms, centre={self.centroid}>"

    def d_norm(self, positions):
        pos = self.positions
        tree = KDTree(pos)
        # make sure k is enough should be enough for d_norm to be correct
        dists, idxs = tree.query(positions, k=min(6, self.natoms))
        d_norm = np.empty(dists.shape[0])
        vecs = np.empty(positions.shape)
        for j, (d, i) in enumerate(zip(dists, idxs, strict=False)):
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
        self.background = float(background)
        self._cython_stock = None

    @property
    def s(self):
        """Backwards-compatible cython stockholder object (lazy)."""
        if self._cython_stock is None:
            from ._density import StockholderWeight as _cStock

            self._cython_stock = _cStock(
                self.dens_a.dens, self.dens_b.dens, background=self.background
            )
        return self._cython_stock

    @property
    def positions(self):
        return np.r_[self.dens_a.positions, self.dens_b.positions]

    @property
    def vdw_radii(self):
        return np.r_[self.dens_a.vdw_radii, self.dens_b.vdw_radii]

    def weights(self, positions):
        return _backends.weights(
            self.dens_a.positions,
            self.dens_a.rho_data,
            self.dens_b.positions,
            self.dens_b.rho_data,
            _DOMAIN,
            positions,
            self.background,
        )

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

from pathlib import Path
import numpy as np
from io import StringIO
from chmpy.util.unit import units


class CubeData:
    _filename = "unknown cube file"

    def __init__(self, filename=None):
        self._interpolator = None
        if filename is not None:
            self._parse_cube_file(filename)

    def _parse_title_lines(self, l1, l2):
        self.title, self.subtitle = l1.strip(), l2.strip

    def _parse_atom_lines(self, lines):
        elements = []
        masses = []
        positions = []
        for line in lines:
            tokens = line.split()
            elements.append(int(tokens[0]))
            masses.append(float(tokens[1]))
            positions.append(np.fromstring(" ".join(tokens[2:]), sep=" "))
        self.elements = np.array(elements)
        self.masses = np.array(masses)
        self._positions_bohr = np.vstack(positions)
        self.positions = units.angstrom(self._positions_bohr)

    def _parse_cube_buf(self, buf):
        n_atom = -1
        self._parse_title_lines(buf.readline(), buf.readline())
        tokens = buf.readline().split()
        self.natom = natom = int(tokens[0])
        self._volume_origin_bohr = np.fromstring(" ".join(tokens[1:]), sep=" ")
        self.volume_origin = units.angstrom(self._volume_origin_bohr)
        atoms = []
        for ax in ("x", "y", "z"):
            tokens = buf.readline().split()
            setattr(self, f"n{ax}", int(tokens[0]))
            setattr(
                self, f"_{ax}_basis_bohr", np.fromstring(" ".join(tokens[1:]), sep=" ")
            )
            setattr(
                self, f"{ax}_basis", units.angstrom(getattr(self, f"_{ax}_basis_bohr"))
            )

        self.basis = np.vstack((self.x_basis, self.y_basis, self.z_basis))
        self._parse_atom_lines(buf.readline() for i in range(self.natom))
        self.data = np.fromstring(buf.read(), sep=" ")

    @classmethod
    def from_string(cls, string):
        cube = cls()
        cube._parse_cube_buf(StringIO(string))
        return cube

    def _parse_cube_file(self, filename):
        self._filename = filename
        with Path(filename).open() as f:
            self._parse_cube_buf(f)

    def shift_origin_to(self, new_origin):
        shift = new_origin - self.volume_origin
        self.volume_origin = new_origin
        self.positions -= shift

    @property
    def xyz(self):
        x, y, z = np.mgrid[0 : self.nx, 0 : self.ny, 0 : self.nz]
        return np.c_[x.ravel(), y.ravel(), z.ravel()] @ self.basis + self.volume_origin

    def molecule(self):
        from chmpy import Molecule

        return Molecule.from_arrays(
            self.elements, self.positions, source_file=self._filename
        )

    def interpolator(self):
        if self._interpolator is None:
            from sklearn.neighbors import KNeighborsRegressor

            self._interpolator = KNeighborsRegressor(n_neighbors=5, weights="distance")
            self._interpolator.fit(self.xyz, self.data)
        return self._interpolator

    def isosurface(self, isovalue=0.0):
        from trimesh import Trimesh
        from chmpy.mc import marching_cubes

        vol = self.data.reshape((self.nx, self.ny, self.nz))
        seps = (
            np.linalg.norm(self.x_basis),
            np.linalg.norm(self.y_basis),
            np.linalg.norm(self.z_basis),
        )
        verts, faces, normals, _ = marching_cubes(vol, level=isovalue, spacing=seps)
        return Trimesh(vertices=verts, faces=faces, normals=normals)

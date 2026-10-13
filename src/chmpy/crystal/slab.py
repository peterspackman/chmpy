"""Slabs cut from a crystal on a lattice plane (hkl).

A slab is `layers` lattice-plane periods of the crystal, cut at a fractional
offset along the surface normal, in a surface frame with x along a2 and z
along the normal. Cuts either keep molecules whole (assigning each molecule
by its centroid) or fall between atomic planes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from math import ceil

import numpy as np

from .surface_symmetry import SurfaceSymmetry
from .symmetry_operation import SymmetryOperation

_POSITION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Termination:
    """A symmetry-distinct cut of a face.

    Attributes:
        offset: cut height in fractions of the layer period
        gap: separation (Angstroms) of the planes of atoms or molecular
            centroids either side of the cut
    """

    offset: float
    gap: float


def is_molecular(crystal, bond_tolerance: float = 0.4) -> bool:
    """Whether every bonded unit of the crystal is a finite molecule.

    Bonds (d < r_cov_a + r_cov_b + bond_tolerance) are followed to every
    periodic image; a unit is infinite when two paths reach the same atom in
    different cells.
    """
    from collections import deque

    from scipy.spatial import cKDTree

    from chmpy.core.element import Element

    images = crystal.slab(bounds=((-1, -1, -1), (1, 1, 1)))
    n = images["n_uc"]
    cart = crystal.to_cartesian(images["frac_pos"])
    cells = np.asarray(images["cell"], dtype=int)
    radii = np.array([Element.from_atomic_number(x).cov for x in images["element"]])
    cutoff = 2 * radii.max() + bond_tolerance
    pairs = cKDTree(cart[:n]).sparse_distance_matrix(cKDTree(cart), cutoff)
    neighbours = [[] for _ in range(n)]
    for (i, j), d in pairs.items():
        if 1e-3 < d < radii[i] + radii[j] + bond_tolerance:
            neighbours[i].append((j % n, cells[j]))

    shift = [None] * n
    for root in range(n):
        if shift[root] is not None:
            continue
        shift[root] = np.zeros(3, dtype=int)
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j, cell in neighbours[i]:
                expected = shift[i] + cell
                if shift[j] is None:
                    shift[j] = expected
                    queue.append(j)
                elif not np.array_equal(shift[j], expected):
                    return False
    return True


@dataclass
class _Unit:
    "A molecule or atom of the primitive surface cell, in surface coordinates."

    elements: np.ndarray
    positions: np.ndarray
    unit_cell_atoms: np.ndarray
    asym_atoms: np.ndarray
    molecule: int

    @property
    def reference(self) -> np.ndarray:
        return self.positions.mean(axis=0)


def _surface_units(crystal, symmetry: SurfaceSymmetry, keep_molecules: bool):
    "Molecules or atoms of one primitive surface cell, references in [0, 1)."
    to_surface = np.linalg.inv(np.array(symmetry.surface_vectors, dtype=float))
    units = []
    if keep_molecules:
        for i, mol in enumerate(crystal.unit_cell_molecules()):
            frac = crystal.to_fractional(mol.positions)
            units.append(
                _Unit(
                    mol.atomic_numbers,
                    frac @ to_surface,
                    np.asarray(mol.properties["unit_cell_atoms"]),
                    np.asarray(mol.properties["asymmetric_unit_atoms"]),
                    i,
                )
            )
    else:
        atoms = crystal.unit_cell_atoms()
        surface_pos = atoms["frac_pos"] @ to_surface
        for i, el in enumerate(atoms["element"]):
            units.append(
                _Unit(
                    np.array([el]),
                    surface_pos[i : i + 1],
                    np.array([i]),
                    np.array([atoms["asym_atom"][i]]),
                    -1,
                )
            )

    unique = {}
    for unit in units:
        shift = np.floor(unit.reference + _POSITION_TOLERANCE)
        unit.positions = unit.positions - shift
        key = (
            tuple(np.round(unit.reference, 3) % 1.0),
            tuple(sorted(unit.elements)),
        )
        unique.setdefault(key, unit)
    return list(unique.values())


def terminations(crystal, hkl, keep_molecules: bool | None = None) -> list[Termination]:
    """Symmetry-distinct cuts of the face (hkl), widest gap first.

    Args:
        crystal: the bulk crystal
        hkl: Miller indices
        keep_molecules: cut between molecular centroids (True) or atomic
            planes (False); by default True for molecular crystals

    Returns:
        list of Termination
    """
    if keep_molecules is None:
        keep_molecules = is_molecular(crystal)
    symmetry = SurfaceSymmetry.from_crystal(crystal, hkl)
    d = symmetry.layer_spacing
    tol = _POSITION_TOLERANCE / d
    z = np.sort(
        [
            u.reference[2] % 1.0
            for u in _surface_units(crystal, symmetry, keep_molecules)
        ]
    )
    planes = [z[0]]
    for value in z[1:]:
        if value - planes[-1] > tol:
            planes.append(value)
    if len(planes) > 1 and planes[0] + 1.0 - planes[-1] <= tol:
        planes.pop()

    candidates = []
    for i, lower in enumerate(planes):
        upper = planes[i + 1] if i + 1 < len(planes) else planes[0] + 1.0
        offset = round(((lower + upper) / 2) % 1.0, 10) % 1.0
        candidates.append(Termination(float(offset), float((upper - lower) * d)))
    candidates.sort(key=lambda t: (-round(t.gap, 6), t.offset))

    result = []
    for candidate in candidates:
        images = symmetry.equivalent_offsets(candidate.offset)
        if not any(
            min(abs(x - t.offset), 1.0 - abs(x - t.offset)) < tol
            for t in result
            for x in images
        ):
            result.append(candidate)
    return result


def _same_sites(reference, image, elements, tol, periodic_z=False) -> bool:
    """Whether `image` sites coincide with `reference` sites of the same
    element, in fractional coordinates periodic in x and y (and z)."""
    from scipy.spatial import cKDTree

    reference = np.array(reference, dtype=float)
    image = np.array(image, dtype=float)
    reference[:, :2] %= 1.0
    image[:, :2] %= 1.0
    if periodic_z:
        reference[:, 2] %= 1.0
        image[:, 2] %= 1.0
        box = [1.0, 1.0, 1.0]
    else:
        low = min(reference[:, 2].min(), image[:, 2].min()) - 1.0
        reference[:, 2] -= low
        image[:, 2] -= low
        box = [1.0, 1.0, max(reference[:, 2].max(), image[:, 2].max()) + 2.0]
    reference %= box
    image %= box
    for element in np.unique(elements):
        mask = elements == element
        dist, _ = cKDTree(reference[mask], boxsize=box).query(image[mask])
        if np.any(dist > tol):
            return False
    return True


@dataclass
class _Cut:
    "How a slab was cut from its bulk crystal; gives its exact symmetry."

    crystal: object
    symmetry: SurfaceSymmetry
    surface_positions: np.ndarray
    centre: Fraction
    flip: object
    keep_molecules: bool


def _atomic_numbers(elements) -> np.ndarray:
    from chmpy.core.element import Element

    return np.array(
        [
            x.atomic_number if hasattr(x, "atomic_number") else Element[x].atomic_number
            for x in elements
        ],
        dtype=int,
    )


class Slab:
    """A slab: atoms periodic in two dimensions, in a cell with vacuum.

    The cell is held in the surface frame: a along x, a and b in the xy
    plane and c along the normal z. Slabs are cut from a crystal with
    `Slab.from_crystal`, or read with `Slab.load`.

    Attributes:
        cell: (3, 3) rows a, b, c of the periodic slab cell
        elements: (N,) atomic numbers
        positions: (N, 3) Cartesian positions in the surface frame
        fixed: (N,) boolean mask of atoms held fixed
        hkl: Miller indices of the top face, if known
        offset: cut height in fractions of the layer period, if known
        layers: number of layer periods, if known
        layer: (N,) layer index of each atom (0 at the bottom), if known
        molecule: (N,) bulk unit cell molecule index (-1 for atoms), if known
        fragment: (N,) index of the molecule or atom each atom belongs to
        unit_cell_atom: (N,) bulk unit cell atom index, for cut slabs
        asym_atom: (N,) asymmetric unit atom index, for cut slabs
    """

    def __init__(
        self,
        cell,
        elements,
        positions,
        *,
        fixed=None,
        hkl=None,
        offset=None,
        layers=None,
        layer=None,
        molecule=None,
        fragment=None,
    ):
        """
        Args:
            cell: (3, 3) lattice vectors as rows; a and b span the surface
                and c points out of it, in any orientation
            elements: atomic numbers, symbols or Elements
            positions: (N, 3) Cartesian positions in the frame of `cell`
            fixed, hkl, offset, layers, layer, molecule, fragment: as the
                attributes
        """
        cell = np.asarray(cell, dtype=float)
        a, b, c = cell
        x = a / np.linalg.norm(a)
        z = np.cross(a, b)
        z /= np.linalg.norm(z)
        height = c @ z
        if height <= 0:
            raise ValueError("Slab cell must be right-handed with c above the ab plane")
        frame = np.array([x, np.cross(z, x), z])
        self.cell = np.array([frame @ a, frame @ b, [0.0, 0.0, height]])
        self.cell[0, 1:] = 0.0
        self.cell[1, 2] = 0.0
        self.positions = np.asarray(positions, dtype=float) @ frame.T
        self.elements = _atomic_numbers(elements)
        n = len(self.elements)
        self.fixed = (
            np.zeros(n, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
        )
        self.hkl = None if hkl is None else tuple(int(i) for i in hkl)
        self.offset = offset
        self.layers = layers
        self.layer = None if layer is None else np.asarray(layer, dtype=int)
        self.molecule = None if molecule is None else np.asarray(molecule, dtype=int)
        self.fragment = (
            np.arange(n) if fragment is None else np.asarray(fragment, dtype=int)
        )
        self.unit_cell_atom = None
        self.asym_atom = None
        self._cut = None
        self._operations = None

    @classmethod
    def from_crystal(
        cls,
        crystal,
        hkl,
        termination: int | Termination = 0,
        layers: int | None = None,
        thickness: float = 10.0,
        vacuum: float = 15.0,
        keep_molecules: bool | None = None,
    ) -> Slab:
        """Cut a slab from a crystal.

        Args:
            crystal: the bulk crystal
            hkl: Miller indices of the top face
            termination: index into `terminations(crystal, hkl)` or a
                Termination
            layers: number of layer periods; by default the fewest giving
                at least `thickness`
            thickness: minimum slab thickness (Angstroms) when `layers` is
                not given
            vacuum: vacuum gap (Angstroms) in the periodic cell
            keep_molecules: keep molecules whole; by default True for
                molecular crystals
        """
        if keep_molecules is None:
            keep_molecules = is_molecular(crystal)
        if isinstance(termination, int):
            termination = terminations(crystal, hkl, keep_molecules)[termination]
        symmetry = SurfaceSymmetry.from_crystal(crystal, hkl)
        offset = termination.offset
        d = symmetry.layer_spacing
        if layers is None:
            layers = max(1, ceil(thickness / d - 1e-9))

        columns = {k: [] for k in ("pos", "el", "layer", "mol", "uc", "asym", "unit")}
        units = _surface_units(crystal, symmetry, keep_molecules)
        for u, unit in enumerate(units):
            base = unit.positions - [0.0, 0.0, np.floor(unit.reference[2] - offset)]
            n = len(unit.elements)
            for k in range(layers):
                columns["pos"].append(base + [0.0, 0.0, k])
                columns["el"].append(unit.elements)
                columns["layer"].append(np.full(n, k))
                columns["mol"].append(np.full(n, unit.molecule))
                columns["uc"].append(unit.unit_cell_atoms)
                columns["asym"].append(unit.asym_atoms)
                columns["unit"].append(np.full(n, u * layers + k))
        surface_pos = np.vstack(columns["pos"])
        data = {k: np.concatenate(v) for k, v in columns.items() if k != "pos"}
        order = np.lexsort((surface_pos[:, 2], data["layer"]))
        surface_pos = surface_pos[order]
        data = {k: v[order] for k, v in data.items()}

        flip = _find_flip(symmetry, surface_pos, data["el"])
        if flip is not None:
            centre = flip[1] / 2
        else:
            centre = Fraction(round((offset + layers / 2) * 12), 12)

        vectors = symmetry.plane.vectors @ symmetry.plane.surface_frame().T
        c = layers * d + vacuum
        cell = np.array([vectors[0], vectors[1], [0.0, 0.0, c]])
        shifted = surface_pos - [0.0, 0.0, float(centre)]
        cart = shifted @ vectors + [0.0, 0.0, 0.5 * c]
        frac = cart @ np.linalg.inv(cell)
        for unit in np.unique(data["unit"]):
            mask = data["unit"] == unit
            frac[mask, :2] -= np.floor(
                frac[mask, :2].mean(axis=0) + _POSITION_TOLERANCE
            )

        _, fragment = np.unique(data["unit"], return_inverse=True)
        slab = cls(
            cell,
            data["el"],
            frac @ cell,
            hkl=symmetry.hkl,
            offset=offset,
            layers=layers,
            layer=data["layer"],
            molecule=data["mol"],
            fragment=fragment,
        )
        slab.unit_cell_atom = data["uc"]
        slab.asym_atom = data["asym"]
        slab._cut = _Cut(crystal, symmetry, surface_pos, centre, flip, keep_molecules)
        return slab

    @classmethod
    def load(cls, filename, **kwargs) -> Slab:
        """Read a slab, choosing the format from the file name.

        geometry.in or *.in: FHI-aims; *.gin: GULP; POSCAR, CONTCAR or
        *.vasp: VASP; *.xyz: extended XYZ.
        """
        from pathlib import Path

        path = Path(filename)
        reader = _reader(path)
        return reader(path.read_text(), **kwargs)

    @classmethod
    def from_aims_string(cls, contents) -> Slab:
        from chmpy.fmt.aims import geometry_string_to_slab

        return geometry_string_to_slab(contents)

    @classmethod
    def from_poscar_string(cls, contents) -> Slab:
        from chmpy.fmt.vasp import poscar_string_to_slab

        return poscar_string_to_slab(contents)

    @classmethod
    def from_extxyz_string(cls, contents) -> Slab:
        from chmpy.fmt.xyz_file import extxyz_string_to_slab

        return extxyz_string_to_slab(contents)

    @classmethod
    def from_gulp_string(cls, contents, vacuum: float = 15.0) -> Slab:
        from chmpy.fmt.gulp import gulp_string_to_slab

        return gulp_string_to_slab(contents, vacuum=vacuum)

    @property
    def crystal(self):
        "The bulk crystal this slab was cut from, or None."
        return None if self._cut is None else self._cut.crystal

    @property
    def symmetry(self) -> SurfaceSymmetry | None:
        "SurfaceSymmetry of the face for cut slabs, or None."
        return None if self._cut is None else self._cut.symmetry

    @property
    def frac_positions(self) -> np.ndarray:
        "(N, 3) fractional coordinates in the periodic slab cell."
        return self.positions @ np.linalg.inv(self.cell)

    @property
    def area(self) -> float:
        return float(np.linalg.norm(np.cross(self.cell[0], self.cell[1])))

    @property
    def thickness(self) -> float:
        """Slab thickness (Angstroms): the layer periods of a cut slab,
        otherwise the extent of the atoms along the normal."""
        if self._cut is not None:
            return self.layers * self._cut.symmetry.layer_spacing
        z = self.positions[:, 2]
        return float(z.max() - z.min())

    @property
    def vacuum(self) -> float:
        "Vacuum (Angstroms) between periodic images of the slab."
        return self.cell[2, 2] - self.thickness

    @property
    def plane_group(self):
        """Plane group of the top surface.

        Exact for cut slabs; for other slabs, that of the in-plane
        operations of the whole slab found from its atoms.
        """
        if self._cut is not None:
            return self._cut.symmetry.plane_group
        from .plane_group import identify_plane_group

        in_plane = [
            (op.rotation[:2, :2], tuple(Fraction(x) for x in op.translation[:2]))
            for op in self._rational_operations()
            if op.rotation[2, 2] == 1
        ]
        return identify_plane_group(in_plane, self.cell[:2] @ self.cell[:2].T)

    @property
    def is_symmetric(self) -> bool:
        "Whether an operation exchanges the top and bottom faces."
        if self._cut is not None:
            return self._cut.flip is not None
        return any(op.rotation[2, 2] == -1 for op in self.symmetry_operations())

    def symmetry_operations(self, tolerance: float = 0.01) -> list[SymmetryOperation]:
        """Operations of the periodic slab model, in slab fractional
        coordinates.

        Exact for cut slabs; otherwise found from the atoms to within
        `tolerance` (Angstroms).
        """
        if self._cut is not None:
            return self._cut_operations()
        if self._operations is None or self._operations[0] != tolerance:
            from .symmetry_finder import find_symmetry_operations
            from .unit_cell import UnitCell

            ops = find_symmetry_operations(
                UnitCell(self.cell),
                self.frac_positions % 1.0,
                self.elements,
                atol=tolerance,
            )
            self._operations = tolerance, ops
        return self._operations[1]

    def _cut_operations(self) -> list[SymmetryOperation]:
        cut = self._cut
        ops = list(cut.symmetry.surface_operations)
        if cut.flip is not None:
            ops += cut.symmetry.flip_operations
        result = []
        for op in ops:
            rotation = np.zeros((3, 3), dtype=int)
            rotation[:2, :2] = op.rotation[:2, :2]
            rotation[2, 2] = op.normal_sign
            t_xy = [
                op.translation[i] + cut.centre * int(op.rotation[i, 2])
                for i in range(2)
            ]
            translation = np.array([float(x % 1) for x in t_xy] + [0.0])
            candidate = SymmetryOperation(rotation, translation)
            if self._is_slab_operation(candidate):
                result.append(candidate)
        return result

    def _rational_operations(self) -> list[SymmetryOperation]:
        """Symmetry operations about an origin on symmetry elements, with
        translations exact multiples of 1/12."""
        from .plane_group import rational_origin

        ops = self.symmetry_operations()
        origin = np.zeros(3)
        origin[:2] = rational_origin(
            [(op.rotation[:2, :2], op.translation[:2]) for op in ops]
        )
        flips = [op for op in ops if op.rotation[2, 2] == -1]
        if flips:
            origin[2] = 0.5 * flips[0].translation[2]
        result = []
        for op in ops:
            t = op.translation + op.rotation @ origin - origin
            snapped = np.round(t * 12) / 12
            residual = (t - snapped) @ self.cell
            if np.linalg.norm(residual) > 0.05:
                raise ValueError(
                    f"Operation {op} has no rational form about a common origin"
                )
            result.append(SymmetryOperation(op.rotation, snapped))
        return result

    def _is_slab_operation(self, op: SymmetryOperation) -> bool:
        frac = self.frac_positions % 1.0
        tol = 1e-3 / np.linalg.norm(self.cell, axis=1).max()
        return _same_sites(frac, op.apply(frac), self.elements, tol, periodic_z=True)

    def space_group(self):
        """Space group type of the periodic slab model.

        Returns:
            StandardSettingResult from `identify_standard_setting`, or None
        """
        from .subgroup import identify_standard_setting

        ops = (
            self._cut_operations()
            if self._cut is not None
            else self._rational_operations()
        )
        return identify_standard_setting(ops)

    def dipole(self, charges=None) -> np.ndarray:
        """Dipole moment (e Angstrom) of the slab cell about its centre.

        Args:
            charges: per atom charges, or for cut slabs per asymmetric unit
                atom charges; cut slabs default to the crystal's EEQ charges
        """
        if charges is None:
            if self._cut is None:
                raise ValueError(
                    "Charges are required for a slab not cut from a crystal"
                )
            charges = self._cut.crystal.asymmetric_unit_partial_charges()
        q = np.asarray(charges, dtype=float)
        if len(q) != len(self):
            q = q[self.asym_atom]
        return q @ (self.positions - 0.5 * self.cell[2])

    def fixed_mask(self, layers: int, side: str = "bottom") -> np.ndarray:
        "Atoms in the outermost `layers` layer periods on the given side."
        if self.layer is None:
            raise ValueError("Slab has no layer information")
        if side == "bottom":
            return self.layer < layers
        if side == "top":
            return self.layer >= self.layers - layers
        raise ValueError(f"side must be 'bottom' or 'top', not {side!r}")

    def oriented_bulk(self):
        """The bulk crystal in the slab's repeat unit: P1, lattice vectors
        (a2, b2, w) in the slab's frame and the atoms of one layer period.

        As a reference for surface energies it shares the slab's in-plane
        lattice, so k-point sampling errors cancel in E_slab - n E_bulk.
        """
        if self._cut is None:
            raise ValueError("Only slabs cut from a crystal have an oriented bulk")
        from chmpy.core.element import Element

        from .asymmetric_unit import AsymmetricUnit
        from .crystal import Crystal
        from .space_group import SpaceGroup
        from .unit_cell import UnitCell

        plane = self._cut.symmetry.plane
        vectors = plane.vectors @ plane.surface_frame().T
        mask = self.layer == 0
        frac = self._cut.surface_positions[mask]
        return Crystal(
            UnitCell(vectors),
            SpaceGroup(1),
            AsymmetricUnit([Element[x] for x in self.elements[mask]], frac),
            titl=self.title.replace(" ", "_") + "_bulk",
        )

    @property
    def title(self) -> str:
        "One-line description, with the metadata the slab readers recover."
        if self.hkl is None:
            return "slab"
        h, k, l = self.hkl
        title = f"({h} {k} {l}) slab"
        if self.offset is not None:
            title += f" offset {self.offset:.6f}"
        if self.layers is not None:
            title += f" layers {self.layers}"
        return title

    def to_crystal(self):
        "The periodic slab model as a P1 Crystal."
        from chmpy.core.element import Element

        from .asymmetric_unit import AsymmetricUnit
        from .crystal import Crystal
        from .space_group import SpaceGroup
        from .unit_cell import UnitCell

        return Crystal(
            UnitCell(self.cell),
            SpaceGroup(1),
            AsymmetricUnit([Element[x] for x in self.elements], self.frac_positions),
            titl=self.title.replace(" ", "_"),
        )

    def to_aims_geometry_string(self, fixed=None) -> str:
        from chmpy.fmt.aims import slab_to_geometry_string

        return slab_to_geometry_string(self, fixed=fixed)

    def to_gulp_input(self, **kwargs) -> str:
        from chmpy.fmt.gulp import slab_to_gulp_input

        return slab_to_gulp_input(self, **kwargs)

    def to_poscar_string(self, fixed=None) -> str:
        from chmpy.fmt.vasp import slab_to_poscar_string

        return slab_to_poscar_string(self, fixed=fixed)

    def to_extxyz_string(self) -> str:
        from chmpy.fmt.xyz_file import slab_to_extxyz_string

        return slab_to_extxyz_string(self)

    def save(self, filename, **kwargs):
        """Write the slab, choosing the format from the file name.

        geometry.in or *.in: FHI-aims; *.gin: GULP; POSCAR or *.vasp: VASP;
        *.xyz: extended XYZ; anything else via `to_crystal().save`.
        """
        from pathlib import Path

        path = Path(filename)
        writers = {
            "aims": self.to_aims_geometry_string,
            "gulp": self.to_gulp_input,
            "vasp": self.to_poscar_string,
            "extxyz": self.to_extxyz_string,
        }
        kind = _format(path)
        if kind is None:
            return self.to_crystal().save(filename, **kwargs)
        path.write_text(writers[kind](**kwargs))

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        description = self.title
        if self._cut is not None:
            description += f" {self.plane_group.symbol}"
        return f"<Slab {description}, {len(self)} atoms>"


_TITLE = re.compile(
    r"\((-?\d+) (-?\d+) (-?\d+)\) slab(?: offset (\S+))?(?: layers (\d+))?"
)


def parse_slab_title(text: str) -> dict:
    "hkl, offset and layers from a `Slab.title` embedded in `text`."
    match = _TITLE.search(text)
    if match is None:
        return {}
    h, k, l, offset, layers = match.groups()
    result = {"hkl": (int(h), int(k), int(l))}
    if offset is not None:
        result["offset"] = float(offset)
    if layers is not None:
        result["layers"] = int(layers)
    return result


def _find_flip(symmetry, surface_positions, elements):
    "A normal-reversing operation mapping the slab onto itself, with its t_z."
    z = surface_positions[:, 2]
    for op in symmetry.flip_operations:
        t_z = op.translation[2]
        k = round(z.min() + z.max() - float(t_z))
        image = op.apply(surface_positions) + [0.0, 0.0, k]
        if _same_sites(surface_positions, image, elements, 1e-4):
            return op, Fraction(t_z) + k
    return None


def _format(path) -> str | None:
    name, suffix = path.name, path.suffix.lower()
    if name == "geometry.in" or suffix == ".in":
        return "aims"
    if suffix == ".gin":
        return "gulp"
    if name in ("POSCAR", "CONTCAR") or suffix == ".vasp":
        return "vasp"
    if suffix == ".xyz":
        return "extxyz"
    return None


def _reader(path):
    readers = {
        "aims": Slab.from_aims_string,
        "gulp": Slab.from_gulp_string,
        "vasp": Slab.from_poscar_string,
        "extxyz": Slab.from_extxyz_string,
    }
    kind = _format(path)
    if kind is None:
        raise ValueError(f"Unknown slab file format: {path}")
    return readers[kind]

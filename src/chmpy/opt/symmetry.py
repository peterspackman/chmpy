"""Degrees of freedom that cannot break a crystal's symmetry.

The degrees of freedom here *are* the asymmetric unit. Symmetry operations are
applied to produce the unit cell, so every geometry the calculator sees is
exactly symmetric by construction, and the optimiser carries roughly `3N / |G|`
atomic variables plus one per allowed strain rather than `3N + 9`. For acetic
acid in Pna2_1 that is 24 atomic and 3 cell freedoms in place of 102.

Being exact rather than approximate matters most when the forces are noisy,
which is to say whenever they come from a machine-learned potential: a
relaxation that projects a structure back towards symmetry after each step
accumulates the residual, while one that cannot express an asymmetric geometry
has nothing to accumulate.

Atoms on special positions get fewer than three freedoms each rather than three
followed by a projection. The site-symmetry group's invariant subspace is
computed once and the atom is parameterised by its coordinates in it: two
freedoms on a mirror plane, one on a three-fold axis, none on an inversion
centre.
"""

from __future__ import annotations

import numpy as np

from chmpy.calc.result import ENERGY, FORCES, STRESS
from chmpy.calc.system import System
from chmpy.crystal.symmetry_operation import decode_symm_int

from .coordinates import Coordinates, Strain
from .strain import cartesian_rotations, invariant_strain_basis


def site_displacement_basis(rotations, tolerance: float = 1e-8) -> np.ndarray:
    """The displacements a site may make without leaving its special position.

    The site-symmetry group fixes the site, so a displacement `d` keeps it
    there exactly when `R d = d` for every `R` in that group. Averaging the
    group gives `P = (1/|G|) sum_R R`, which is idempotent, and the fixed
    vectors are exactly its range: `P d = d` for any `d = P v`, and any fixed
    `d` is `P d`. So an orthonormal basis of the range of `P` is the answer.

    It has to be the range, not an eigenvector calculation. In fractional
    coordinates `R` is an integer matrix that is only orthogonal when the cell
    is, so for a trigonal, hexagonal or rhombohedral cell `P` is idempotent but
    not symmetric, and neither symmetrising it nor taking the eigenvectors of
    `P^T P` recovers the space it projects onto. Both give a direction that
    leaves the special position: for triazine in R-3c they move an atom off a
    two-fold axis, the site multiplicity doubles from 18 to 36, and the cell
    comes back with 72 atoms in it instead of 54, some of them 0.37 A apart.

    Args:
        rotations: (M, 3, 3) fractional rotation parts of the operations that
            map the site onto itself
        tolerance: how far the rank of `P` may sit from an integer before the
            averaged operator is rejected as not a projector

    Returns:
        (3, d) orthonormal columns spanning the allowed displacements, with `d`
        from 0 (a fixed site, such as an inversion centre) to 3 (a general one)
    """
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    if len(rotations) <= 1:
        return np.eye(3)
    projector = rotations.mean(axis=0)
    # an idempotent matrix has only 0 and 1 for eigenvalues, so its trace is
    # its rank exactly -- no threshold to pick, and nothing to get wrong
    rank = projector.trace()
    if abs(rank - round(rank)) > tolerance:
        raise ValueError(
            f"the averaged site-symmetry operator has trace {rank}, which is "
            "not an integer, so these rotations are not a group"
        )
    rank = int(round(rank))
    if rank <= 0:
        return np.zeros((3, 0))
    if rank >= 3:
        return np.eye(3)
    left, _, _ = np.linalg.svd(projector)
    return left[:, :rank]


class SymmetryAdapted(Strain):
    """The asymmetric unit and the allowed cell strains, and nothing else.

    The atomic degrees of freedom are displacements of the asymmetric unit in
    *fractional* coordinates, which is the natural choice once the cell can
    deform: fractional coordinates are unchanged by a strain, so the two kinds
    of freedom do not fight each other, and the strain gradient is exactly the
    stress.

    Args:
        crystal: the structure to relax
        cell: whether to vary the cell
        fixed: (n_asym,) boolean mask of asymmetric-unit atoms to hold still
        tolerance: site-merging tolerance passed to `Crystal.unit_cell_atoms`
        info: model inputs that are not geometry, carried on the `System` --
            a charge, a spin, an external field

    Attributes:
        crystal: the crystal this was built from
        system: the P1 unit cell handed to the calculator
    """

    wanted = frozenset({ENERGY, FORCES, STRESS})

    def __init__(
        self,
        crystal,
        cell: bool = True,
        fixed=None,
        tolerance: float = 1e-2,
        info=None,
    ):
        self.crystal = crystal
        self._build_orbit(crystal, tolerance)
        self._build_site_bases(crystal, fixed)

        system = System(
            self._numbers,
            self._fractional() @ np.asarray(crystal.unit_cell.direct),
            np.asarray(crystal.unit_cell.direct),
            True,
            dict(info) if info else {},
        )
        basis = (
            invariant_strain_basis(cartesian_rotations(crystal))
            if cell
            else np.zeros((0, 3, 3))
        )
        super().__init__(system, basis=basis)
        self.atom_map = crystal_atom_map(crystal, self.system, tolerance)

    # -- construction --------------------------------------------------------

    def _build_orbit(self, crystal, tolerance) -> None:
        """How each unit-cell atom is generated from an asymmetric-unit one."""
        atoms = crystal.unit_cell_atoms(tolerance=tolerance)
        self._numbers = np.asarray(atoms["element"])
        self._asym_index = np.asarray(atoms["asym_atom"], dtype=int)

        rotations, translations = [], []
        reference = np.asarray(crystal.asymmetric_unit.positions, dtype=float)
        for uc_index, code in enumerate(atoms["symop"]):
            rotation, translation = decode_symm_int(int(code))
            generated = rotation @ reference[self._asym_index[uc_index]] + translation
            # unit_cell_atoms wraps into [0, 1); keep the lattice translation
            # that wrapping applied, so the map stays exact as atoms move
            translation = translation + np.round(
                atoms["frac_pos"][uc_index] - generated
            )
            rotations.append(rotation)
            translations.append(translation)

        self._rotations = np.array(rotations)
        self._translations = np.array(translations)
        self.reference_fractional = reference.copy()

    def _build_site_bases(self, crystal, fixed) -> None:
        """One displacement basis per asymmetric-unit atom, from its site symmetry."""
        operations = crystal.space_group.symmetry_operations
        reference = self.reference_fractional
        fixed = (
            np.zeros(len(reference), dtype=bool)
            if fixed is None
            else np.asarray(fixed, dtype=bool)
        )

        bases, offsets, total = [], [], 0
        for index, position in enumerate(reference):
            if fixed[index]:
                bases.append(np.zeros((3, 0)))
                offsets.append(total)
                continue
            stabiliser = [
                operation.rotation
                for operation in operations
                if _fixes(operation, position)
            ]
            basis = site_displacement_basis(stabiliser)
            bases.append(basis)
            offsets.append(total)
            total += basis.shape[1]

        self.site_bases = bases
        self.site_offsets = offsets
        self._n_atomic = total
        self.displacements = np.zeros(total)

        # The per-atom bases are a block-diagonal map from degrees of freedom
        # to fractional displacements. Flattened to one direction per degree of
        # freedom it becomes a gather and a weighted sum, which is the
        # difference between a Python loop over the unit cell on every gradient
        # and a couple of array operations: measured at 1.08 ms against 0.05 ms
        # for a 1098-atom cell, when the whole energy evaluation was 0.01 ms.
        self._dof_atom = np.concatenate(
            [
                np.full(basis.shape[1], index, dtype=np.int64)
                for index, basis in enumerate(bases)
            ]
            or [np.zeros(0, dtype=np.int64)]
        )
        self._dof_vector = np.concatenate(
            [basis.T for basis in bases] or [np.zeros((0, 3))]
        )

    # -- the interface -------------------------------------------------------

    @property
    def n_atomic(self) -> int:
        "Atomic degrees of freedom, after site symmetry has removed the rest"
        return self._n_atomic

    @property
    def n_dof(self) -> int:
        return self._n_atomic + len(self.basis)

    def get(self) -> np.ndarray:
        return np.concatenate([self.displacements, self.amplitudes])

    def set(self, x) -> None:
        x = np.asarray(x, dtype=float)
        self.displacements = x[: self._n_atomic].copy()
        self.amplitudes = x[self._n_atomic :].copy()
        cell = self.reference_cell @ self.deformation_gradient
        self.system.set_cell(cell)
        self.system.set_positions(self._fractional() @ cell)

    def gradient(self, result) -> np.ndarray:
        cell = np.asarray(self.system.cell)
        # cart = frac @ cell, so dE/dfrac = dE/dcart @ cell^T
        unit_cell_gradient = -result.forces @ cell.T

        # each unit-cell atom pushes its gradient back onto the asymmetric-unit
        # atom that generated it, rotated by the operation that generated it:
        # g_asym += g_uc R, summed over the orbit
        rotated = np.einsum("ui,uij->uj", unit_cell_gradient, self._rotations)
        asymmetric = self._scatter_to_asymmetric(rotated, self._asym_index)

        atomic = np.einsum("kd,kd->k", self._dof_vector, asymmetric[self._dof_atom])
        if len(self.basis) == 0:
            return atomic
        return np.concatenate([atomic, self.strain_gradient(result)])

    def _scatter_to_asymmetric(self, vectors, index) -> np.ndarray:
        """Sum (M, 3) vectors onto the asymmetric-unit atoms `index` names.

        Used twice with different indices -- over the unit cell to fold a
        gradient back, and over the degrees of freedom to build a displacement
        -- so the index is an argument rather than an attribute.
        """
        n_asym = len(self.reference_fractional)
        return np.stack(
            [
                np.bincount(index, weights=vectors[:, d], minlength=n_asym)
                for d in range(3)
            ],
            axis=1,
        )

    def scale(self) -> np.ndarray:
        """Angstroms moved per unit degree of freedom.

        A fractional displacement of one along a basis direction moves the atom
        by the length of that direction's Cartesian image -- roughly a cell
        edge. Getting this wrong is what makes an unscaled optimiser take
        wildly different effective step sizes along `a` and along `c`.
        """
        cell = np.asarray(self.system.cell)
        atomic = np.maximum(np.linalg.norm(self._dof_vector @ cell, axis=1), 1e-3)
        if len(self.basis) == 0:
            return atomic
        return np.concatenate([atomic, Strain.scale(self)])

    def step_fraction(self, x, dx) -> float:
        if len(self.basis) == 0:
            return 1.0
        return Strain.step_fraction(self, x[self._n_atomic :], dx[self._n_atomic :])

    def reanchor(self, x, force: bool = False):
        """Fold the current strain into the reference cell.

        The atomic freedoms are fractional, so they are untouched by this --
        only the cell reference and the amplitudes change.
        """
        if len(self.basis) == 0:
            return None
        from .coordinates import REANCHOR_AMPLITUDE

        amplitudes = np.asarray(x)[self._n_atomic :]
        if not force and np.abs(amplitudes).max(initial=0.0) < REANCHOR_AMPLITUDE:
            return None
        self.set(x)
        self.reference_cell = np.array(self.system.cell)
        self.amplitudes = np.zeros(len(self.basis))
        return self.get()

    def measures(self, result) -> dict:
        return symmetric_measures(result, self.atom_map, self.system.cell, self.basis)

    # -- results -------------------------------------------------------------

    def asymmetric_fractional(self) -> np.ndarray:
        "(n_asym, 3) current fractional coordinates of the asymmetric unit"
        if self._n_atomic == 0:
            return self.reference_fractional.copy()
        displaced = self._dof_vector * self.displacements[:, None]
        return self.reference_fractional + self._scatter_to_asymmetric(
            displaced, self._dof_atom
        )

    def to_crystal(self):
        """The relaxed structure, in its original space group."""
        from chmpy.crystal import AsymmetricUnit, Crystal, UnitCell

        asymmetric_unit = AsymmetricUnit(
            self.crystal.asymmetric_unit.elements,
            self.asymmetric_fractional(),
            labels=self.crystal.asymmetric_unit.labels,
        )
        return Crystal(
            UnitCell(np.asarray(self.system.cell)),
            self.crystal.space_group,
            asymmetric_unit,
            properties=dict(getattr(self.crystal, "properties", {}) or {}),
        )

    def _fractional(self) -> np.ndarray:
        """Unit-cell fractional coordinates from the current asymmetric unit."""
        asymmetric = self.asymmetric_fractional()
        return (
            np.einsum("uab,ub->ua", self._rotations, asymmetric[self._asym_index])
            + self._translations
        )

    def __repr__(self) -> str:
        return (
            f"<SymmetryAdapted {self.crystal.space_group.symbol}: "
            f"{self._n_atomic} atomic + {len(self.basis)} cell dof "
            f"for {len(self.system)} atoms>"
        )


def _fixes(operation, position, tolerance: float = 1e-6) -> bool:
    """Whether a symmetry operation maps a fractional position onto itself."""
    image = operation.rotation @ position + operation.translation
    difference = image - position
    return bool(np.abs(difference - np.round(difference)).max() < tolerance)


# -- symmetry on a P1 system ---------------------------------------------------
#
# These work from a list of operations and a P1 `System` rather than a
# `Crystal`, so they handle subgroups in non-standard settings (e.g. what a
# strained cell keeps), which have no space group table entry.


def cartesian_operations(rotations, cell) -> np.ndarray:
    """Fractional rotations as Cartesian ones, `A^T R A^-T`, with lattice
    vectors as the rows of `A`.

    Unlike `chmpy.opt.strain.cartesian_rotations` this keeps duplicates, so the
    result lines up with an `AtomMap`.
    """
    direct = np.asarray(cell, dtype=float).T
    return np.einsum(
        "ij,gjk,kl->gil", direct, np.asarray(rotations, float), np.linalg.inv(direct)
    )


class AtomMap:
    """How each of a set of symmetry operations permutes a system's atoms.

    Operation `g` takes atom `i` onto atom `permutation[g, i]` in the cell
    displaced by the integer lattice vector `shifts[g, i]`:

        R_g x_i + t_g = x_(permutation[g, i]) + shifts[g, i]

    in fractional coordinates.

    Attributes:
        rotations: (G, 3, 3) fractional rotation parts
        translations: (G, 3) fractional translation parts
        permutation: (G, N) image atom of each atom under each operation
        shifts: (G, N, 3) integer lattice shift of each image
    """

    def __init__(self, rotations, translations, permutation, shifts):
        self.rotations = np.asarray(rotations, dtype=float)
        self.translations = np.asarray(translations, dtype=float)
        self.permutation = np.asarray(permutation, dtype=np.int64)
        self.shifts = np.asarray(shifts, dtype=np.int64)

    @classmethod
    def identity(cls, n_atoms: int) -> AtomMap:
        "The trivial group, for a structure with no symmetry to use"
        return cls(
            np.eye(3)[None],
            np.zeros((1, 3)),
            np.arange(n_atoms)[None],
            np.zeros((1, n_atoms, 3)),
        )

    def __len__(self) -> int:
        return len(self.rotations)

    def __repr__(self) -> str:
        return f"<AtomMap {len(self)} operations on {self.permutation.shape[1]} atoms>"

    def subset(self, keep) -> AtomMap:
        "The operations selected by a boolean mask or an index array"
        keep = np.asarray(keep)
        return AtomMap(
            self.rotations[keep],
            self.translations[keep],
            self.permutation[keep],
            self.shifts[keep],
        )

    def cartesian(self, cell) -> np.ndarray:
        "(G, 3, 3) Cartesian rotations in a given cell"
        return cartesian_operations(self.rotations, cell)

    def symmetrise(self, vectors, cell) -> np.ndarray:
        """Symmetric part of per-atom vectors (e.g. forces).

        Group average of `v_i -> R_g v_i` placed on atom `g(i)`, i.e. the
        orthogonal projection onto the invariant subspace.

        Args:
            vectors: (N, 3) Cartesian vectors, one per atom
            cell: (3, 3) lattice vectors as rows, to make the rotations
                Cartesian in

        Returns:
            (N, 3) symmetric part
        """
        vectors = np.asarray(vectors, dtype=float)
        rotated = np.einsum("gij,nj->gni", self.cartesian(cell), vectors)
        symmetric = np.zeros_like(vectors)
        np.add.at(symmetric, self.permutation.ravel(), rotated.reshape(-1, 3))
        return symmetric / len(self)

    def preserving_strain(self, strain, cell, tolerance: float = 1e-8) -> AtomMap:
        """The operations with `R eps R^T = eps`, i.e. those a strained cell keeps.

        Fractional coordinates are unchanged by a strain, so the permutations
        and shifts carry over.
        """
        strain = np.asarray(strain, dtype=float)
        rotated = np.einsum(
            "gij,jk,glk->gil", self.cartesian(cell), strain, self.cartesian(cell)
        )
        scale = max(float(np.abs(strain).max()), 1e-300)
        kept = np.abs(rotated - strain).max(axis=(1, 2)) <= tolerance * scale
        return self.subset(kept)


def symmetric_measures(result, atom_map, cell, basis=None) -> dict:
    """Convergence measures for a symmetry-constrained relaxation.

    `fmax` and `smax` are taken on the symmetric parts of the forces and
    stress, since those are all the relaxation can change. Models that are not
    exactly equivariant (e.g. PET) leave a small symmetry-forbidden force that
    would otherwise make tight tolerances unreachable. Its size is reported as
    `forbidden` but is not a convergence criterion.

    Args:
        result: a calculator `Result` with forces, and stress if `basis` is given
        atom_map: the operations the relaxation preserves
        cell: the current lattice vectors
        basis: (k, 3, 3) the allowed strains, or None at fixed cell

    Returns:
        {"fmax", "forbidden"}, plus "smax" when the cell varies
    """
    from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

    symmetric = atom_map.symmetrise(result.forces, cell)
    measures = {
        "fmax": float(np.abs(symmetric).max(initial=0.0)),
        "forbidden": float(np.abs(result.forces - symmetric).max(initial=0.0)),
    }
    if basis is not None and len(basis):
        # projection onto the (orthonormal) allowed strain basis
        stress = np.einsum(
            "k,kab->ab", np.einsum("kab,ab->k", basis, result.stress), basis
        )
        measures["smax"] = float(np.abs(stress).max()) * EV_PER_ANGSTROM3_TO_GPA
    return measures


def map_atoms(system, operations, tolerance: float = 1e-2) -> AtomMap:
    """Find how each operation permutes the atoms of a periodic system.

    Args:
        system: a periodic `System`
        operations: `SymmetryOperation`s, or `(rotation, translation)` pairs,
            in the fractional coordinates of `system.cell`
        tolerance: max distance in Angstroms between an image and its match;
            the default matches `Crystal.unit_cell_atoms`

    Raises:
        ValueError: if an operation is not a symmetry of the structure
    """
    from scipy.spatial import cKDTree

    if not system.periodic:
        raise ValueError("an AtomMap needs a periodic system")
    cell = np.asarray(system.cell)
    numbers = np.asarray(system.numbers)
    fractional = np.asarray(system.scaled_positions)
    tree = cKDTree(_wrap(fractional), boxsize=1.0)

    rotations, translations, permutations, shifts = [], [], [], []
    for operation in operations:
        if hasattr(operation, "rotation"):
            rotation, translation = operation.rotation, operation.translation
        else:
            rotation, translation = operation
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)

        images = fractional @ rotation.T + translation
        _, match = tree.query(_wrap(images))
        difference = images - fractional[match]
        shift = np.round(difference)
        miss = np.linalg.norm((difference - shift) @ cell, axis=1)
        wrong_element = numbers[match] != numbers
        if miss.max(initial=0.0) > tolerance or wrong_element.any():
            raise ValueError(
                f"the structure is not symmetric under the operation with "
                f"rotation {rotation.astype(int).tolist()} and translation "
                f"{np.round(translation, 4).tolist()}: an image misses its "
                f"nearest atom by {miss.max():.3g} A (tolerance {tolerance} A)"
                + (", or lands on a different element" if wrong_element.any() else "")
            )
        if len(np.unique(match)) != len(match):
            raise ValueError(
                "two atoms map onto the same site; the structure has "
                f"overlapping atoms closer than {tolerance} A"
            )
        rotations.append(rotation)
        translations.append(translation)
        permutations.append(match)
        shifts.append(shift.astype(np.int64))

    return AtomMap(rotations, translations, permutations, shifts)


def crystal_atom_map(crystal, system, tolerance: float = 1e-2) -> AtomMap:
    "A crystal's space group, as it acts on a P1 system built from that crystal"
    return map_atoms(system, crystal.space_group.symmetry_operations, tolerance)


def _wrap(fractional) -> np.ndarray:
    """Fractional coordinates strictly in [0, 1), as cKDTree requires
    (`x - floor(x)` can round to 1.0 for tiny negative `x`)."""
    wrapped = np.asarray(fractional) - np.floor(fractional)
    wrapped[wrapped >= 1.0] = 0.0
    return wrapped


class InvariantAtomic(Coordinates):
    """Cartesian atomic positions at a fixed cell, restricted by a symmetry.

    One degree of freedom per symmetry-unique atom and allowed site direction;
    the rest of each orbit follows by symmetry. Like `SymmetryAdapted` but
    needs only a P1 `System` and an `AtomMap`, so it works for any subgroup.
    A unit amplitude moves each atom in the orbit by 1 Angstrom.

    Args:
        system: the structure to vary
        atom_map: the operations to preserve, from `map_atoms`
    """

    wanted = frozenset({ENERGY, FORCES})

    def __init__(self, system, atom_map: AtomMap):
        super().__init__(system)
        self.atom_map = atom_map
        self.reference_positions = np.array(system.positions)
        cartesian = atom_map.cartesian(system.cell)
        permutation = atom_map.permutation

        dof, atom, vector = [], [], []
        seen = np.zeros(len(system), dtype=bool)
        n_dof = 0
        for representative in range(len(system)):
            if seen[representative]:
                continue
            images = permutation[:, representative]
            stabiliser = cartesian[images == representative]
            basis = site_displacement_basis(stabiliser)
            # any operation reaching a given image will do: they differ by a
            # stabiliser element, which leaves `basis` fixed
            members, first = np.unique(images, return_index=True)
            seen[members] = True
            for column in basis.T:
                dof.extend([n_dof] * len(members))
                atom.extend(members)
                vector.extend(cartesian[first] @ column)
                n_dof += 1

        self._n_dof = n_dof
        self._dof = np.array(dof, dtype=np.int64)
        self._atom = np.array(atom, dtype=np.int64)
        self._vector = np.array(vector, dtype=float).reshape(-1, 3)
        self.amplitudes = np.zeros(n_dof)

    @property
    def n_dof(self) -> int:
        return self._n_dof

    def get(self) -> np.ndarray:
        return self.amplitudes.copy()

    def set(self, x) -> None:
        self.amplitudes = np.asarray(x, dtype=float).copy()
        moved = self._vector * self.amplitudes[self._dof, None]
        displacement = np.zeros_like(self.reference_positions)
        np.add.at(displacement, self._atom, moved)
        self.system.set_positions(self.reference_positions + displacement)

    def gradient(self, result) -> np.ndarray:
        along = np.einsum("ed,ed->e", self._vector, -result.forces[self._atom])
        return np.bincount(self._dof, weights=along, minlength=self._n_dof)

    def scale(self) -> np.ndarray:
        return np.ones(self._n_dof)

    def measures(self, result) -> dict:
        return symmetric_measures(result, self.atom_map, self.system.cell)

    def __repr__(self) -> str:
        return (
            f"<InvariantAtomic {self._n_dof} dof under {len(self.atom_map)} "
            f"operations for {len(self.system)} atoms>"
        )

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

from .coordinates import Strain
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
        if len(self.basis) == 0:
            return {"fmax": result.fmax}
        return {"fmax": result.fmax, "smax": result.smax}

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

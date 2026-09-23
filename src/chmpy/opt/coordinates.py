"""What an optimiser varies.

A `Coordinates` object is a parameterisation of a structure. It hands the
optimiser a vector, writes a new vector back into the structure, converts
forces and stress into a gradient with respect to that vector, and reports how
many Angstroms of atomic motion one unit of each degree of freedom is worth.

That last part is `scale`, and it is what lets one trust radius and one set of
tolerances cover degrees of freedom measured in different units. A cell strain
is dimensionless, a fractional coordinate spans a lattice vector, and a
rotation is in radians; scaled to Angstroms of displacement they are
comparable, and an optimiser step means the same thing in each. Convergence is
then tested on the quantities themselves -- force in eV/A, stress in GPa --
rather than on a norm over a mixed-unit array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chmpy.calc.result import ENERGY, EV_PER_ANGSTROM3_TO_GPA, FORCES, STRESS

from .strain import deformation, invariant_strain_basis, step_fraction

#: below this the strain has drifted far enough from its reference that the
#: linearised parameterisation is worth re-anchoring
REANCHOR_AMPLITUDE = 0.5


class Coordinates(ABC):
    """A set of degrees of freedom, and the structure they act on.

    Attributes:
        system: the `System` written to by `set`
        wanted: the properties the gradient needs from a calculator
    """

    wanted: frozenset = frozenset({ENERGY, FORCES})

    def __init__(self, system):
        self.system = system

    @property
    @abstractmethod
    def n_dof(self) -> int:
        "How many degrees of freedom"

    @abstractmethod
    def get(self) -> np.ndarray:
        "The current degrees of freedom"

    @abstractmethod
    def set(self, x) -> None:
        "Write degrees of freedom into the structure"

    @abstractmethod
    def gradient(self, result) -> np.ndarray:
        "dE/dx from a calculator result, in eV per unit degree of freedom"

    @abstractmethod
    def scale(self) -> np.ndarray:
        "Angstroms of atomic displacement per unit of each degree of freedom"

    def step_fraction(self, x, dx) -> float:
        """How much of a proposed step keeps the parameterisation valid.

        One factor for the whole step, never a per-component clamp: the
        optimiser has to evaluate the step its model predicted a reduction for.
        """
        return 1.0

    def reanchor(self, x, force: bool = False) -> np.ndarray | None:
        """Re-express the current geometry about a new reference, or None.

        The geometry is unchanged, only the variables describing it, so a
        result computed at `x` is still valid at the vector this returns.

        Args:
            x: the current degrees of freedom
            force: re-anchor even if the parameterisation has not drifted far.
                The optimiser passes this when a step has been clipped, which
                is the situation the re-anchor exists for.

        Returns:
            the new degree-of-freedom vector, or None if nothing was done
        """
        return None

    def measures(self, result) -> dict:
        """Physical convergence measures, each in its own units."""
        return {"fmax": result.fmax}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.n_dof} dof>"


class Atomic(Coordinates):
    """Cartesian atomic positions at a fixed cell.

    Args:
        system: the structure to vary
        fixed: (N,) boolean mask of atoms to hold still, or None
    """

    wanted = frozenset({ENERGY, FORCES})

    def __init__(self, system, fixed=None):
        super().__init__(system)
        self.free = (
            np.ones(len(system), dtype=bool)
            if fixed is None
            else ~np.asarray(fixed, dtype=bool)
        )

    @property
    def n_dof(self) -> int:
        return 3 * int(self.free.sum())

    def get(self) -> np.ndarray:
        return self.system.positions[self.free].ravel()

    def set(self, x) -> None:
        positions = np.array(self.system.positions)
        positions[self.free] = np.asarray(x).reshape(-1, 3)
        self.system.set_positions(positions)

    def gradient(self, result) -> np.ndarray:
        return -result.forces[self.free].ravel()

    def scale(self) -> np.ndarray:
        return np.ones(self.n_dof)


class Strain(Coordinates):
    """The cell alone, as amplitudes along a symmetry-adapted strain basis.

    The atoms are carried along affinely, which is exactly the deformation the
    stress is the derivative with respect to.

    Args:
        system: the structure to vary
        basis: (k, 3, 3) strain basis, or None for all six directions. Build a
            symmetry-adapted one with `chmpy.opt.strain.invariant_strain_basis`.
    """

    wanted = frozenset({ENERGY, STRESS})

    def __init__(self, system, basis=None):
        super().__init__(system)
        if not system.periodic:
            raise ValueError("a cell degree of freedom needs a periodic system")
        self.basis = (
            invariant_strain_basis([]) if basis is None else np.asarray(basis, float)
        )
        self._anchor()

    def _anchor(self) -> None:
        """Take the current geometry as the undeformed reference."""
        self.reference_cell = np.array(self.system.cell)
        self.reference_positions = np.array(self.system.positions)
        self.amplitudes = np.zeros(len(self.basis))

    @property
    def n_dof(self) -> int:
        return len(self.basis)

    def get(self) -> np.ndarray:
        return self.amplitudes.copy()

    def set(self, x) -> None:
        self.amplitudes = np.asarray(x, dtype=float).copy()
        gradient = self.deformation_gradient
        self.system.set_cell(self.reference_cell @ gradient)
        self.system.set_positions(self.reference_positions @ gradient)

    @property
    def deformation_gradient(self) -> np.ndarray:
        "(3, 3) `F = I + eps`, the deformation from the reference"
        return deformation(self.amplitudes, self.basis)

    def strain_gradient(self, result) -> np.ndarray:
        """dE/dq for the strain amplitudes.

        A further strain `d` about the *current* geometry takes `F` to
        `F (I + d)`, so `dE = <F^T dE/dF, d> = V <sigma, d>` and therefore
        `dE/dF = F^-T V sigma`. Contracting with the basis, which is symmetric,
        symmetrises it. At `F = I` this is just `V <sigma, B>`.
        """
        cartesian = np.linalg.solve(
            self.deformation_gradient.T, result.stress * self.system.volume
        )
        return np.einsum("kab,ab->k", self.basis, cartesian)

    def gradient(self, result) -> np.ndarray:
        return self.strain_gradient(result)

    def scale(self) -> np.ndarray:
        # a unit amplitude is a strain of unit Frobenius norm, which moves an
        # atom at the far side of the cell by about the longest cell vector
        length = float(np.linalg.norm(self.system.cell, axis=1).max())
        return np.full(len(self.basis), max(length, 1e-3))

    def step_fraction(self, x, dx) -> float:
        return step_fraction(x, dx, self.basis)

    def reanchor(self, x, force: bool = False):
        """Move the reference to the current cell once the strain is large.

        The bounds in `strain` exist to stop one linearised step producing a
        degenerate cell, not to cap how far a cell may travel over a whole
        relaxation. Without a re-anchor a structure that reaches a bound
        proposes zero-length steps for ever after, which a trust region reads
        as failure and answers by shrinking the radius until the structure is
        dead. That is why the optimiser also calls this with `force` the moment
        a step is clipped, rather than only after one is accepted -- at a bound,
        no step is ever accepted, so waiting for one waits for ever.

        This is not a pure change of origin: deformations compose
        multiplicatively, so a model built in the old parameterisation is
        slightly wrong in the new one. It self-heals on the next gradient, and
        it only fires when the strain is already large.
        """
        if not force and np.abs(np.asarray(x)).max(initial=0.0) < REANCHOR_AMPLITUDE:
            return None
        self.set(x)
        self._anchor()
        return self.amplitudes.copy()

    def measures(self, result) -> dict:
        return {"smax": result.smax}


class AtomicStrain(Strain):
    """Atomic positions and the cell together.

    The variables are the atomic positions *in the undeformed frame* plus the
    strain amplitudes. Holding the atoms in the undeformed frame is what keeps
    the two kinds of freedom from fighting: a change in strain moves them
    affinely, which is exactly the deformation the stress is the derivative
    with respect to, so neither gradient contains a term belonging to the
    other.

    Args:
        system: the structure to vary
        basis: (k, 3, 3) strain basis, or None for all six directions
        fixed: (N,) boolean mask of atoms to hold still in the undeformed
            frame. They still move with the cell.
    """

    wanted = frozenset({ENERGY, FORCES, STRESS})

    def __init__(self, system, basis=None, fixed=None):
        self.free = (
            np.ones(len(system), dtype=bool)
            if fixed is None
            else ~np.asarray(fixed, dtype=bool)
        )
        super().__init__(system, basis)

    @property
    def n_atomic(self) -> int:
        "How many degrees of freedom belong to the atoms"
        return 3 * int(self.free.sum())

    @property
    def n_dof(self) -> int:
        return self.n_atomic + len(self.basis)

    def get(self) -> np.ndarray:
        return np.concatenate(
            [self.reference_positions[self.free].ravel(), self.amplitudes]
        )

    def set(self, x) -> None:
        x = np.asarray(x, dtype=float)
        self.reference_positions = np.array(self.reference_positions)
        self.reference_positions[self.free] = x[: self.n_atomic].reshape(-1, 3)
        self.amplitudes = x[self.n_atomic :].copy()
        gradient = self.deformation_gradient
        self.system.set_cell(self.reference_cell @ gradient)
        self.system.set_positions(self.reference_positions @ gradient)

    def gradient(self, result) -> np.ndarray:
        # r_i = p_i F, so dE/dp_i = (dE/dr_i) F^T = -f_i F^T
        atomic = -result.forces @ self.deformation_gradient.T
        return np.concatenate([atomic[self.free].ravel(), self.strain_gradient(result)])

    def scale(self) -> np.ndarray:
        return np.concatenate([np.ones(self.n_atomic), Strain.scale(self)])

    def step_fraction(self, x, dx) -> float:
        return step_fraction(x[self.n_atomic :], dx[self.n_atomic :], self.basis)

    def reanchor(self, x, force: bool = False):
        amplitudes = np.asarray(x)[self.n_atomic :]
        if not force and np.abs(amplitudes).max(initial=0.0) < REANCHOR_AMPLITUDE:
            return None
        self.set(x)
        self.reference_cell = np.array(self.system.cell)
        self.reference_positions = np.array(self.system.positions)
        self.amplitudes = np.zeros(len(self.basis))
        return self.get()

    def measures(self, result) -> dict:
        return {"fmax": result.fmax, "smax": result.smax}


def strain_basis_for(structure, symmetry: bool = True) -> np.ndarray:
    """The strain basis to use for a structure.

    Args:
        structure: a `Crystal`, or anything else
        symmetry: when False, or when the structure carries no space group,
            all six strain directions are allowed

    Returns:
        (k, 3, 3) orthonormal strain basis
    """
    from .strain import cartesian_rotations

    if not symmetry or not hasattr(structure, "space_group"):
        return invariant_strain_basis([])
    return invariant_strain_basis(cartesian_rotations(structure))


def pressure_term(coordinates, pressure_gpa: float):
    """The gradient contribution of an external hydrostatic pressure.

    Minimising `E + P V` rather than `E` relaxes to a structure in equilibrium
    with a pressure. `dV/dq = V tr(F^-1 B)` for a strain basis direction `B`.

    Args:
        coordinates: a `Strain` or `AtomicStrain`
        pressure_gpa: applied pressure in GPa

    Returns:
        (n_dof,) gradient contribution in eV per unit degree of freedom
    """
    pressure = pressure_gpa / EV_PER_ANGSTROM3_TO_GPA
    inverse = np.linalg.inv(coordinates.deformation_gradient)
    volume = coordinates.system.volume
    strain_part = volume * np.einsum("ab,kba->k", inverse, coordinates.basis)
    gradient = np.zeros(coordinates.n_dof)
    gradient[coordinates.n_dof - len(coordinates.basis) :] = pressure * strain_part
    return gradient

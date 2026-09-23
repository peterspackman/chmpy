"""Harmonic force constants, and the dynamical matrix they give.

The second derivative of the energy with respect to atomic displacements,
`Phi_ab(i, j, S) = -dF_jb / du_ia`, where `j` sits in the cell displaced by the
integer lattice vector `S` from `i`'s. Those triples are indexed the same way a
neighbour list is, which is the point: the dynamical matrix is then a phase sum
over exactly the pairs that exist,

    D_ab(q) = sum_(i,j,S) w(i,j,S) Phi_ab(i,j,S) exp(2 pi i q.S) / sqrt(m_i m_j)

and no supercell index ever appears in it. A supercell is used to *measure*
`Phi` -- far enough that a displaced atom does not feel its own image -- and
then discarded.

`w` is the multiplicity weight. A pair `(i, j)` may be reachable through
several lattice translations that are equally short, and each of those images
is physically present in the infinite crystal while the finite supercell
measured their sum once. The weight shares it back out. The degeneracy is exact
-- it comes from lattice symmetry, not from floating point -- so the ties can
be found by comparing squared distances against the minimum with a tolerance
far below any real spacing.

Three symmetries constrain `Phi`, and they do not commute, so the order is
fixed and documented in `symmetrise`:

* permutation, `Phi_ab(i,j,S) = Phi_ba(j,i,-S)`;
* the acoustic sum rule, `sum_(j,S) Phi_ab(i,j,S) = 0`, which is the statement
  that translating the whole crystal costs nothing;
* the space group, which is applied by `chmpy.vib.symmetry`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from chmpy.calc.system import System
from chmpy.core import Element

LOG = logging.getLogger(__name__)

#: displacement for the finite difference, Angstroms. Large enough to clear a
#: calculator's force noise, small enough to stay harmonic; see the module
#: docstring of `chmpy.opt.elastic` for the same trade-off in strain.
DEFAULT_DISPLACEMENT = 0.01

#: two images count as equally distant within this many Angstroms squared
TIE_TOLERANCE = 1e-6

#: eV / (Angstrom^2 amu) -> (rad/s)^2. One conversion, in one place.
#: 1 eV = 1.602176634e-19 J, 1 A = 1e-10 m, 1 amu = 1.66053906660e-27 kg.
EV_PER_ANGSTROM2_AMU_TO_RAD2_S2 = 1.602176634e-19 / (1e-20 * 1.66053906660e-27)

#: (rad/s) -> THz, and -> cm^-1
RAD_S_TO_THZ = 1e-12 / (2 * np.pi)
RAD_S_TO_PER_CM = 1.0 / (2 * np.pi * 2.99792458e10)


@dataclass
class ForceConstants:
    """Harmonic force constants of a periodic structure.

    Attributes:
        pairs: (P, 2) the atom indices `i`, `j` of each block, within the cell
        shifts: (P, 3) integer lattice translation taking `j`'s cell from `i`'s
        blocks: (P, 3, 3) the force-constant matrices, eV/Angstrom^2
        weights: (P,) multiplicity weights for the phase sum
        masses: (N,) atomic masses in amu
        cell: (3, 3) lattice vectors as rows, Angstroms
        positions: (N, 3) Cartesian positions the constants were measured at
    """

    pairs: np.ndarray
    shifts: np.ndarray
    blocks: np.ndarray
    weights: np.ndarray
    masses: np.ndarray
    cell: np.ndarray
    positions: np.ndarray

    @property
    def n_atoms(self) -> int:
        return len(self.masses)

    def __repr__(self) -> str:
        return (
            f"<ForceConstants {self.n_atoms} atoms, {len(self.blocks)} blocks, "
            f"shifts to {int(np.abs(self.shifts).max())}>"
        )

    # -- the dynamical matrix ------------------------------------------------

    def dynamical_matrix(self, q) -> np.ndarray:
        """The (3N, 3N) dynamical matrix at a fractional wavevector.

        Args:
            q: (3,) wavevector in fractional reciprocal coordinates, so that
                the phase is `exp(2 pi i q.S)` with `S` the integer shift

        Returns:
            (3N, 3N) complex Hermitian matrix in eV/(Angstrom^2 amu)
        """
        q = np.asarray(q, dtype=float)
        n = self.n_atoms
        matrix = np.zeros((3 * n, 3 * n), dtype=complex)

        phases = np.exp(2j * np.pi * (self.shifts @ q)) * self.weights
        root_mass = np.sqrt(self.masses)
        for (i, j), block, phase in zip(self.pairs, self.blocks, phases, strict=True):
            matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += (
                phase * block / (root_mass[i] * root_mass[j])
            )

        # the sum is Hermitian in exact arithmetic; make it so in practice
        return 0.5 * (matrix + matrix.conj().T)

    def frequencies(self, q, units: str = "thz") -> np.ndarray:
        """Phonon frequencies at a wavevector, ordered low to high.

        A negative frequency is reported for an imaginary mode, which is the
        usual convention and keeps the array real.

        Args:
            q: (3,) fractional wavevector
            units: "thz", "cm-1", or "rad/s"

        Returns:
            (3N,) frequencies
        """
        eigenvalues = np.linalg.eigvalsh(self.dynamical_matrix(q))
        omega_squared = eigenvalues * EV_PER_ANGSTROM2_AMU_TO_RAD2_S2
        omega = np.sign(omega_squared) * np.sqrt(np.abs(omega_squared))
        factor = {
            "rad/s": 1.0,
            "thz": RAD_S_TO_THZ,
            "cm-1": RAD_S_TO_PER_CM,
        }
        if units not in factor:
            raise ValueError(f"unknown units {units!r}, expected one of {list(factor)}")
        return omega * factor[units]

    def acoustic_error(self, units: str = "cm-1") -> float:
        """How far the three acoustic modes at Gamma are from zero.

        The cleanest single check on a set of force constants: translating the
        crystal costs no energy, so three frequencies at `q = 0` must vanish.
        Anything left is the residue of the acoustic sum rule and of noise in
        the finite differences.
        """
        return float(np.abs(self.frequencies(np.zeros(3), units=units)[:3]).max())


# -- measuring them -----------------------------------------------------------


def force_constants(
    structure,
    calculator,
    supercell=None,
    displacement: float = DEFAULT_DISPLACEMENT,
    cutoff: float | None = None,
    info=None,
    logger=None,
) -> ForceConstants:
    """Measure force constants by displacing atoms in a supercell.

    Args:
        structure: a `Crystal` or a periodic `System`, already relaxed
        calculator: a `chmpy.calc.Calculator`
        supercell: (3,) repetitions, or None to choose one from `cutoff`
        displacement: finite-difference step in Angstroms
        cutoff: interaction range in Angstroms used to size the supercell when
            one is not given; defaults to a third of the smallest supercell
            width that `supercell` provides
        info: model inputs that are not geometry, carried on every `System`
        logger: called with a line per displacement

    Returns:
        ForceConstants
    """
    system = _as_system(structure, info)
    if not system.periodic:
        raise ValueError("force constants need a periodic structure")

    repeats = (
        np.array(supercell, dtype=int)
        if supercell is not None
        else _supercell_for(system, cutoff or 8.0)
    )
    if np.any(repeats < 1):
        raise ValueError(f"supercell repetitions must be positive, got {repeats}")

    big, origin, cell_index = _build_supercell(system, repeats)
    n = len(system)

    blocks = np.zeros((n, n, int(np.prod(repeats)), 3, 3))

    scratch = big.copy()
    for atom in range(n):
        for axis in range(3):
            shifted = []
            for sign in (1, -1):
                positions = np.array(big.positions)
                positions[origin[atom], axis] += sign * displacement
                scratch.set_positions(positions)
                shifted.append(calculator.forces(scratch))
            # Phi_ab(i, j) = -dF_jb/du_ia
            derivative = -(shifted[0] - shifted[1]) / (2 * displacement)
            for image, (j, cell) in enumerate(cell_index):
                blocks[atom, j, cell, axis, :] = derivative[image]
            if logger is not None:
                logger(f"  atom {atom} axis {axis}")

    return symmetrise(_assemble(system, repeats, blocks))


def _as_system(structure, info=None) -> System:
    kwargs = dict(info) if info else {}
    if isinstance(structure, System):
        system = structure.copy()
        system.info.update(kwargs)
        return system
    if hasattr(structure, "space_group"):
        return System.from_crystal(structure, **kwargs)
    return System.from_molecule(structure, **kwargs)


def _supercell_for(system, cutoff) -> np.ndarray:
    """Repetitions giving a supercell at least `2 * cutoff` wide in each direction.

    The displaced atom must not feel its own periodic image, so the supercell
    has to be wider than twice the interaction range along every direction --
    measured as the interplanar spacing, not the lattice parameter, since a
    skewed cell is thinner than its edges suggest.
    """
    reciprocal = np.linalg.inv(np.asarray(system.cell))
    spacing = 1.0 / np.linalg.norm(reciprocal, axis=0)
    return np.maximum(1, np.ceil(2.0 * cutoff / spacing).astype(int))


def _build_supercell(system, repeats):
    """The supercell, where each primitive atom went, and what each atom is.

    Returns:
        (supercell system, origin[i] -> supercell index of primitive atom i,
         cell_index[k] -> (primitive atom, flat cell number) for supercell atom k)
    """
    offsets = (
        np.array(np.meshgrid(*[np.arange(r) for r in repeats], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    n = len(system)
    cell = np.asarray(system.cell)

    positions = (system.positions[None, :, :] + (offsets @ cell)[:, None, :]).reshape(
        -1, 3
    )
    numbers = np.tile(np.asarray(system.numbers), len(offsets))
    big = System(numbers, positions, (cell.T * repeats).T, True, dict(system.info))

    cell_index = [(atom, image) for image in range(len(offsets)) for atom in range(n)]
    origin = np.array([_flat(0, atom, n) for atom in range(n)])
    return big, origin, cell_index


def _flat(image, atom, n_atoms) -> int:
    return image * n_atoms + atom


def _assemble(system, repeats, blocks) -> ForceConstants:
    """Turn per-image blocks into (i, j, shift) triples with multiplicity weights."""
    n = len(system)
    cell = np.asarray(system.cell)
    offsets = (
        np.array(np.meshgrid(*[np.arange(r) for r in repeats], indexing="ij"))
        .reshape(3, -1)
        .T
    )

    pairs, shifts, kept, weights = [], [], [], []
    for i in range(n):
        for j in range(n):
            for image, offset in enumerate(offsets):
                block = blocks[i, j, image]
                if not np.any(block):
                    continue
                images, weight = _equivalent_images(
                    system.positions[i], system.positions[j], offset, repeats, cell
                )
                for shift in images:
                    pairs.append((i, j))
                    shifts.append(shift)
                    kept.append(block)
                    weights.append(weight)

    masses = np.array([Element.from_atomic_number(int(z)).mass for z in system.numbers])
    return ForceConstants(
        pairs=np.array(pairs, dtype=int).reshape(-1, 2),
        shifts=np.array(shifts, dtype=float).reshape(-1, 3),
        blocks=np.array(kept).reshape(-1, 3, 3),
        weights=np.array(weights),
        masses=masses,
        cell=cell,
        positions=np.array(system.positions),
    )


def symmetrise(constants: ForceConstants) -> ForceConstants:
    """Impose permutation symmetry and the acoustic sum rule, in that order.

    The two do not commute and neither is idempotent with respect to the
    other, so the order is fixed here rather than left to the caller:

    1. **Permutation.** `Phi_ab(i,j,S)` and `Phi_ba(j,i,-S)` are the same
       second derivative measured from opposite ends. Finite differences give
       them slightly differently, and their average is the better estimate --
       the same free error signal the elastic tensor gets from its asymmetry.
    2. **Acoustic sum rule.** Translating the whole crystal costs nothing, so
       `sum_(j,S) w Phi(i,j,S) = 0` for every `i`. Whatever is left over is
       put on the self block `Phi(i,i,0)`, which is the one the finite
       difference measures least well, since it is the difference of a large
       cancelling sum.

    Doing the sum rule first and the permutation second would break the sum
    rule again, by an amount that shows up directly in the acoustic branches.
    """
    blocks = np.array(constants.blocks)
    pairs, shifts = constants.pairs, constants.shifts

    index = {
        (int(i), int(j), *np.rint(shift).astype(int)): position
        for position, ((i, j), shift) in enumerate(zip(pairs, shifts, strict=True))
    }
    for position, ((i, j), shift) in enumerate(zip(pairs, shifts, strict=True)):
        key = (int(j), int(i), *(-np.rint(shift).astype(int)))
        partner = index.get(key)
        if partner is None or partner < position:
            continue
        averaged = 0.5 * (blocks[position] + blocks[partner].T)
        blocks[position] = averaged
        blocks[partner] = averaged.T

    weighted = constants.weights[:, None, None] * blocks
    for i in range(constants.n_atoms):
        belongs = pairs[:, 0] == i
        residual = weighted[belongs].sum(axis=0)
        self_block = np.flatnonzero(
            belongs & (pairs[:, 1] == i) & ~np.any(shifts != 0, axis=1)
        )
        if len(self_block) == 0:
            LOG.warning(
                "atom %d has no self block, so the acoustic sum rule cannot be "
                "imposed; its acoustic modes will not come out at zero",
                i,
            )
            continue
        target = self_block[0]
        blocks[target] -= residual / constants.weights[target]

    return ForceConstants(
        pairs=pairs,
        shifts=shifts,
        blocks=blocks,
        weights=constants.weights,
        masses=constants.masses,
        cell=constants.cell,
        positions=constants.positions,
    )


def _equivalent_images(position_i, position_j, offset, repeats, cell):
    """The shortest lattice translations reaching `j` from `i`, and their share.

    The supercell measured one number for what is, in the infinite crystal,
    several equally distant images of the same pair. Each gets `1 / M` of it.
    The degeneracy is exact -- it comes from the lattice -- so the tie test is
    a comparison of squared distances well below any real spacing.
    """
    candidates = (
        np.array(np.meshgrid(*[np.arange(-1, 2) for _ in range(3)], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    shifts = offset + candidates * repeats
    vectors = position_j - position_i + shifts @ cell
    squared = np.einsum("ij,ij->i", vectors, vectors)
    closest = squared.min()
    tied = squared <= closest + TIE_TOLERANCE
    return shifts[tied], 1.0 / float(tied.sum())

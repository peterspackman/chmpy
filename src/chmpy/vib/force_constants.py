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
* the space group, which reduces what is measured in the first place; see
  `force_constants`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.calc.system import System
from chmpy.core import Element
from chmpy.opt.progress import reporter

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

#: largest size of the per-shift real matrices `dynamical_matrices` keeps
BY_SHIFT_BYTES = 512 * 2**20

#: the acoustic sum rule is imposed until each atom's residual is below this
#: fraction of its largest block, or this many sweeps have been made
SUM_RULE_TOLERANCE = 1e-12
SUM_RULE_SWEEPS = 200


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
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

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
        return self.dynamical_matrices(np.reshape(q, (1, 3)))[0]

    def dynamical_matrices(self, qs) -> np.ndarray:
        """Dynamical matrices at several fractional wavevectors at once.

        The blocks are first summed into one mass-weighted real matrix per
        distinct shift `S`, which needs doing once, and then each `D(q)` is a
        phase-weighted sum of those -- one matrix product for every q at once,
        instead of a scatter over every block for each. Where the per-shift
        matrices would be too large to hold, each q is scattered instead.

        Args:
            qs: (M, 3) fractional wavevectors

        Returns:
            (M, 3N, 3N) complex Hermitian matrices in eV/(Angstrom^2 amu)
        """
        qs = np.asarray(qs, dtype=float).reshape(-1, 3)
        n3 = 3 * self.n_atoms
        by_shift = self._by_shift()
        if by_shift is not None:
            shifts, matrices = by_shift
            phases = np.exp(2j * np.pi * (qs @ shifts.T))
            result = (phases @ matrices.reshape(len(shifts), -1)).reshape(-1, n3, n3)
        else:
            result = np.stack([self._scatter(q) for q in qs])
        # the sum is Hermitian in exact arithmetic; make it so in practice
        return 0.5 * (result + result.conj().transpose(0, 2, 1))

    def _mass_weighted(self) -> np.ndarray:
        "(P, 3, 3) blocks times their weights over sqrt(m_i m_j)"
        root_mass = np.sqrt(self.masses)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        scale = self.weights / (root_mass[i] * root_mass[j])
        return scale[:, None, None] * self.blocks

    def _flat_index(self) -> np.ndarray:
        "(P, 3, 3) index of each block element in a flattened (3N, 3N) matrix"
        n3 = 3 * self.n_atoms
        rows = 3 * self.pairs[:, 0, None] + np.arange(3)
        cols = 3 * self.pairs[:, 1, None] + np.arange(3)
        return rows[:, :, None] * n3 + cols[:, None, :]

    def _by_shift(self):
        """(shifts, (S, 3N, 3N) real matrices), or None if too large to keep."""
        if self._cache.get("by_shift", False) is not False:
            return self._cache["by_shift"]
        n3 = 3 * self.n_atoms
        shifts, which = np.unique(
            np.rint(self.shifts).astype(np.int64), axis=0, return_inverse=True
        )
        which = which.ravel()
        if len(shifts) * n3 * n3 * 8 > BY_SHIFT_BYTES:
            self._cache["by_shift"] = None
            return None
        index = which[:, None, None] * (n3 * n3) + self._flat_index()
        matrices = np.bincount(
            index.ravel(),
            weights=self._mass_weighted().ravel(),
            minlength=len(shifts) * n3 * n3,
        ).reshape(len(shifts), n3, n3)
        self._cache["by_shift"] = (shifts.astype(float), matrices)
        return self._cache["by_shift"]

    def _scatter(self, q) -> np.ndarray:
        "One dynamical matrix, summed straight from the blocks"
        n3 = 3 * self.n_atoms
        phases = np.exp(2j * np.pi * (self.shifts @ q))
        values = phases[:, None, None] * self._mass_weighted()
        index = self._flat_index().ravel()
        real = np.bincount(index, weights=values.real.ravel(), minlength=n3 * n3)
        imag = np.bincount(index, weights=values.imag.ravel(), minlength=n3 * n3)
        return (real + 1j * imag).reshape(n3, n3)

    def frequencies(self, q, units: str = "thz") -> np.ndarray:
        """Phonon frequencies at one or more wavevectors, ordered low to high.

        A negative frequency is reported for an imaginary mode, which is the
        usual convention and keeps the array real.

        Args:
            q: (3,) fractional wavevector, or (M, 3) for several
            units: "thz", "cm-1", or "rad/s"

        Returns:
            (3N,) frequencies, or (M, 3N) for several wavevectors
        """
        factor = _unit_factor(units)
        q = np.asarray(q, dtype=float)
        eigenvalues = np.linalg.eigvalsh(self.dynamical_matrices(q.reshape(-1, 3)))
        omega = _omega(eigenvalues) * factor
        return omega[0] if q.ndim == 1 else omega

    def optical_frequencies(self, units: str = "thz") -> np.ndarray:
        """The 3N - 3 optical frequencies at Gamma, ordered low to high.

        The three uniform translations are projected out of the dynamical
        matrix before it is diagonalised, rather than dropped as the three
        lowest frequencies or those under a threshold. That way a soft optical
        mode is never mistaken for an acoustic one, and an imaginary optical
        mode is still reported as one.
        """
        factor = _unit_factor(units)
        complement = self._translation_complement()
        matrix = self.dynamical_matrix(np.zeros(3)).real
        projected = complement.T @ matrix @ complement
        eigenvalues = np.linalg.eigvalsh(0.5 * (projected + projected.T))
        return _omega(eigenvalues) * factor

    def _translation_complement(self) -> np.ndarray:
        """(3N, 3N - 3) orthonormal basis orthogonal to uniform translations.

        A translation by `t` displaces every atom by `t`; in mass-weighted
        coordinates that is `sqrt(m_i) t`, which is what the dynamical matrix
        acts on.
        """
        root_mass = np.sqrt(self.masses)
        translations = np.kron(root_mass[:, None], np.eye(3))
        basis, _ = np.linalg.qr(translations, mode="complete")
        return basis[:, 3:]

    def acoustic_error(self, units: str = "cm-1") -> float:
        """How far the three acoustic modes at Gamma are from zero.

        The cleanest single check on a set of force constants: translating the
        crystal costs no energy, so three frequencies at `q = 0` must vanish.
        Anything left is the residue of the acoustic sum rule and of noise in
        the finite differences.
        """
        return float(np.abs(self.frequencies(np.zeros(3), units=units)[:3]).max())


def _unit_factor(units: str) -> float:
    factor = {
        "rad/s": 1.0,
        "thz": RAD_S_TO_THZ,
        "cm-1": RAD_S_TO_PER_CM,
    }
    if units not in factor:
        raise ValueError(f"unknown units {units!r}, expected one of {list(factor)}")
    return factor[units]


def _omega(eigenvalues) -> np.ndarray:
    "Signed angular frequencies, rad/s, from dynamical-matrix eigenvalues"
    omega_squared = np.asarray(eigenvalues) * EV_PER_ANGSTROM2_AMU_TO_RAD2_S2
    return np.sign(omega_squared) * np.sqrt(np.abs(omega_squared))


# -- measuring them -----------------------------------------------------------


def force_constants(
    structure,
    calculator,
    supercell=None,
    displacement: float = DEFAULT_DISPLACEMENT,
    cutoff: float | None = None,
    info=None,
    progress=None,
    symmetry: bool = True,
) -> ForceConstants:
    """Measure force constants by displacing atoms in a supercell.

    With symmetry, one atom per orbit is displaced, along only as many
    directions as its site symmetry needs to span all three (one on a cubic
    site, three on a general position). The other blocks follow from
    `Phi(g(r), g(k)) = R_g Phi(r, k) R_g^T`.

    Args:
        structure: a `Crystal` or a periodic `System`, already relaxed
        calculator: a `chmpy.calc.Calculator`
        supercell: (3,) repetitions, or None to choose one from `cutoff`
        displacement: finite-difference step in Angstroms
        cutoff: interaction range in Angstroms used to size the supercell when
            one is not given; defaults to a third of the smallest supercell
            width that `supercell` provides
        info: model inputs that are not geometry, carried on every `System`
        progress: True to print a line per displacement, or a callable given
            a `chmpy.opt.progress.Progress` for each
        symmetry: use a `Crystal`'s space group to reduce the displacements.
            A `System` carries no symmetry and every atom is displaced.

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

    atom_map = _symmetry_for(structure, system, repeats, symmetry)
    big, origin, _ = _build_supercell(system, repeats)
    n = len(system)
    n_cells = int(np.prod(repeats))
    cartesian = atom_map.cartesian(system.cell)

    # plan the displacements first so progress can report a total
    plan, measured = [], np.zeros(n, dtype=bool)
    for representative in range(n):
        if measured[representative]:
            continue
        images = atom_map.permutation[:, representative]
        stabiliser = np.flatnonzero(images == representative)
        plan.append(
            (representative, stabiliser, _directions_to_displace(cartesian[stabiliser]))
        )
        measured[images] = True
    total = sum(len(directions) for _, _, directions in plan)

    report = reporter(progress)
    report(
        "force constants",
        "setup",
        f"force constants: {total} displacements of {len(plan)} of {n} atoms, "
        f"{len(atom_map)} operations, supercell {repeats.tolist()} "
        f"({len(big)} atoms)",
    )

    blocks = np.zeros((n, n, n_cells, 3, 3))
    scratch = big.copy()
    done = 0
    for representative, stabiliser, directions in plan:
        rows, responses = [], []
        for direction in directions:
            stage = f"atom {representative} along {np.round(direction, 4).tolist()}"
            report("force constants", stage, stage, index=done, total=total)
            done += 1
            shifted = []
            for sign in (1, -1):
                positions = np.array(big.positions)
                positions[origin[representative]] += sign * displacement * direction
                scratch.set_positions(positions)
                shifted.append(calculator.forces(scratch))
            # d . Phi(r, k) = -dF_k / du along d, for every supercell atom k
            response = -(shifted[0] - shifted[1]) / (2 * displacement)
            # each site-symmetry operation s gives another equation:
            # (R^T d) . Phi(r, k) = (d . Phi(r, s(k))) R
            for operation in stabiliser:
                rotation = cartesian[operation]
                target = _supercell_image(atom_map, operation, representative, repeats)
                rows.append(rotation.T @ direction)
                responses.append(response[target] @ rotation)

        # least squares for every supercell atom at once
        solved = np.einsum(
            "am,mkb->kab", np.linalg.pinv(np.array(rows)), np.array(responses)
        )
        _fill_orbit(blocks, solved, atom_map, cartesian, representative, repeats)

    return symmetrise(_assemble(system, repeats, blocks))


def _symmetry_for(structure, system, repeats, symmetry):
    """The space-group operations usable with this supercell.

    Only operations that map the supercell lattice onto itself are kept, since
    the measured blocks are sums over supercell images (e.g. a 2x2x3 cell of a
    cubic crystal keeps the tetragonal subgroup).
    """
    from chmpy.opt.symmetry import AtomMap, crystal_atom_map

    if not symmetry or not hasattr(structure, "space_group"):
        return AtomMap.identity(len(system))
    atom_map = crystal_atom_map(structure, system)
    scaled = atom_map.rotations * repeats[None, None, :] / repeats[None, :, None]
    keep = np.all(np.abs(scaled - np.round(scaled)) < 1e-8, axis=(1, 2))
    if not keep.all():
        LOG.info(
            "supercell %s keeps %d of %d operations",
            repeats.tolist(),
            int(keep.sum()),
            len(keep),
        )
    return atom_map.subset(keep)


def _directions_to_displace(stabiliser) -> list:
    """Axes to displace along so their site-symmetry images span 3D."""
    directions, span = [], np.zeros((0, 3))
    for direction in np.eye(3):
        images = np.einsum("gij,j->gi", stabiliser, direction)
        extended = np.vstack([span, images])
        if np.linalg.matrix_rank(extended, tol=1e-8) > np.linalg.matrix_rank(
            span, tol=1e-8
        ):
            directions.append(direction)
            span = extended
        if np.linalg.matrix_rank(span, tol=1e-8) == 3:
            break
    return directions


def _supercell_image(atom_map, operation, atom, repeats) -> np.ndarray:
    """Where an operation sends every supercell atom, relative to `g(atom)`.

    Supercell atom `k` (primitive atom `j`, cell `c`) goes to `perm(j)` in cell
    `shift(j) + R c`; subtracting `shift(atom)` puts `g(atom)` in cell zero.

    Returns:
        (n * n_cells,) supercell index of the image of each supercell atom
    """
    n = atom_map.permutation.shape[1]
    rotation = np.rint(atom_map.rotations[operation]).astype(np.int64)
    permutation = atom_map.permutation[operation]
    shifts = atom_map.shifts[operation]

    cells = _cell_offsets(repeats)
    images = (
        shifts[None, :, :]
        + (cells @ rotation.T)[:, None, :]
        - shifts[atom][None, None, :]
    )
    images = np.mod(images, repeats)
    flat = np.ravel_multi_index(images.reshape(-1, 3).T, repeats).reshape(len(cells), n)
    return (flat * n + permutation[None, :]).ravel()


def _fill_orbit(blocks, solved, atom_map, cartesian, representative, repeats):
    """Write `Phi(g(r), g(k)) = R Phi(r, k) R^T` for every image of `r`,
    using the first operation that reaches each image."""
    n = atom_map.permutation.shape[1]
    images = atom_map.permutation[:, representative]
    _, first = np.unique(images, return_index=True)
    for operation in first:
        rotation = cartesian[operation]
        target = _supercell_image(atom_map, operation, representative, repeats)
        rotated = np.einsum("ij,kjl,ml->kim", rotation, solved, rotation)
        atom = images[operation]
        blocks[atom, target % n, target // n] = rotated


def _cell_offsets(repeats) -> np.ndarray:
    "(n_cells, 3) integer cell offsets, in the supercell's flat order"
    return (
        np.array(np.meshgrid(*[np.arange(r) for r in repeats], indexing="ij"))
        .reshape(3, -1)
        .T
    )


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
    cell = np.asarray(system.cell)
    positions = np.asarray(system.positions)
    offsets = _cell_offsets(repeats)

    i, j, image = np.nonzero(np.any(blocks != 0, axis=(3, 4)))
    pairs, shifts, kept, weights = [], [], [], []
    for chunk in range(0, len(i), ASSEMBLE_CHUNK):
        part = slice(chunk, chunk + ASSEMBLE_CHUNK)
        tied, share = _equivalent_images(
            positions[i[part]], positions[j[part]], offsets[image[part]], repeats, cell
        )
        row, candidate = np.nonzero(tied)
        pairs.append(np.c_[i[part][row], j[part][row]])
        shifts.append(
            offsets[image[part]][row] + _IMAGE_CANDIDATES[candidate] * repeats
        )
        kept.append(blocks[i[part][row], j[part][row], image[part][row]])
        weights.append(share[row])

    masses = np.array([Element.from_atomic_number(int(z)).mass for z in system.numbers])
    return ForceConstants(
        pairs=np.concatenate(pairs or [np.zeros((0, 2))]).astype(int),
        shifts=np.concatenate(shifts or [np.zeros((0, 3))]).astype(float),
        blocks=np.concatenate(kept or [np.zeros((0, 3, 3))]),
        weights=np.concatenate(weights or [np.zeros(0)]),
        masses=masses,
        cell=cell,
        positions=np.array(positions),
    )


def symmetrise(constants: ForceConstants) -> ForceConstants:
    """Impose permutation symmetry and the acoustic sum rule together.

    * **Permutation.** `Phi_ab(i,j,S)` and `Phi_ba(j,i,-S)` are the same
      second derivative measured from opposite ends. Finite differences give
      them slightly differently, and their average is the better estimate --
      the same free error signal the elastic tensor gets from its asymmetry.
    * **Acoustic sum rule.** Translating the whole crystal costs nothing, so
      `sum_(j,S) w Phi(i,j,S) = 0` for every `i`.

    Each is a linear constraint, and the answer wanted is the closest set of
    constants satisfying both. Imposing them once each in turn does not get
    there: a residual that is not symmetric cannot be removed by changing only
    the symmetric self block, and the part left over reappears as a
    sum-rule error in the dynamical matrix, which is Hermitian, of a size
    that shows up as acoustic modes at Gamma that are not quite zero.
    Alternating the two orthogonal projections converges to the projection
    onto both, so that is what is done until the residual is at round-off.

    The sum-rule projection is the least-squares one: each block of atom
    `i`'s row takes a share of the residual in proportion to its weight.
    """
    blocks = np.array(constants.blocks)
    pairs, shifts, weights = constants.pairs, constants.shifts, constants.weights
    n = constants.n_atoms
    partner = _permutation_partners(pairs, shifts)
    has_partner = partner >= 0
    row = pairs[:, 0]
    norm = np.bincount(row, weights=weights * weights, minlength=n)
    if np.any(norm == 0):
        LOG.warning(
            "atoms %s have no force constants, so the acoustic sum rule cannot "
            "be imposed on them",
            np.flatnonzero(norm == 0).tolist(),
        )
    norm[norm == 0] = np.inf

    scale = max(float(np.abs(blocks).max(initial=0.0)), 1e-300)
    for _ in range(SUM_RULE_SWEEPS):
        mirrored = blocks[partner[has_partner]].transpose(0, 2, 1)
        blocks[has_partner] = 0.5 * (blocks[has_partner] + mirrored)
        residual = np.zeros((n, 3, 3))
        np.add.at(residual, row, weights[:, None, None] * blocks)
        if np.abs(residual).max(initial=0.0) <= SUM_RULE_TOLERANCE * scale:
            break
        blocks -= (weights / norm[row])[:, None, None] * residual[row]
    else:
        LOG.warning(
            "the acoustic sum rule converged only to %.1e of the largest block",
            np.abs(residual).max() / scale,
        )

    return ForceConstants(
        pairs=pairs,
        shifts=shifts,
        blocks=blocks,
        weights=constants.weights,
        masses=constants.masses,
        cell=constants.cell,
        positions=constants.positions,
    )


def _permutation_partners(pairs, shifts) -> np.ndarray:
    """For each block `(i, j, S)` the index of `(j, i, -S)`, or -1 if absent."""
    shifts = np.rint(shifts).astype(np.int64)
    n = int(pairs.max(initial=0)) + 1
    span = int(np.abs(shifts).max(initial=0))
    width = 2 * span + 1

    def encode(i, j, s):
        cells = ((s[:, 0] + span) * width + s[:, 1] + span) * width + s[:, 2] + span
        return (i * n + j) * width**3 + cells

    keys = encode(pairs[:, 0], pairs[:, 1], shifts)
    wanted = encode(pairs[:, 1], pairs[:, 0], -shifts)
    order = np.argsort(keys, kind="stable")
    found = np.searchsorted(keys, wanted, sorter=order)
    found = np.minimum(found, len(keys) - 1)
    partner = order[found]
    return np.where(keys[partner] == wanted, partner, -1)


def _equivalent_images(position_i, position_j, offset, repeats, cell):
    """The shortest lattice translations reaching `j` from `i`, and their share.

    The supercell measured one number for what is, in the infinite crystal,
    several equally distant images of the same pair. Each gets `1 / M` of it.
    The degeneracy is exact -- it comes from the lattice -- so the tie test is
    a comparison of squared distances well below any real spacing.

    Args:
        position_i, position_j: (P, 3) Cartesian positions of each pair
        offset: (P, 3) supercell offset each block was measured at
        repeats: (3,) supercell repetitions
        cell: (3, 3) primitive lattice vectors as rows

    Returns:
        ((P, 27) mask of which of `_IMAGE_CANDIDATES` are among the shortest,
         (P,) the share each of them gets)
    """
    shifts = offset[:, None, :] + _IMAGE_CANDIDATES[None, :, :] * repeats
    vectors = (position_j - position_i)[:, None, :] + shifts @ cell
    squared = np.einsum("pci,pci->pc", vectors, vectors)
    tied = squared <= squared.min(axis=1, keepdims=True) + TIE_TOLERANCE
    return tied, 1.0 / tied.sum(axis=1)


#: supercell images considered for the shortest translation, in a fixed order
_IMAGE_CANDIDATES = _cell_offsets((3, 3, 3)) - 1

#: blocks handled at once when assembling, to bound the memory used
ASSEMBLE_CHUNK = 65536

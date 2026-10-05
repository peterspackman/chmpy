"""The atomic configuration a calculator is asked about.

`System` holds atomic numbers, Cartesian positions, lattice vectors,
periodicity, and a dictionary of whatever else a model wants (a charge, a spin,
an external field). Everything chemical -- bonding, symmetry, molecules --
stays in `Crystal` and `Molecule`, which convert to and from a `System`.

Two invariants are worth knowing before using it.

**The arrays are read-only, and changing a geometry goes through a setter**
which installs a new array and increments `version`. That makes
`(id(system), version)` a complete identifier for a geometry, so a cached
result can be validated in constant time, and it makes a stale read impossible
rather than merely unlikely -- `system.positions[0] += 0.1` raises rather than
quietly leaving a cache valid.

**The cell is always a (3, 3) array with the lattice vectors as rows**,
matching `chmpy.crystal.UnitCell.direct`. Non-periodic directions carry a zero
row, so "is there a cell" and "is it periodic" are the same question.
"""

from __future__ import annotations

import numpy as np

_EMPTY_CELL = np.zeros((3, 3))
_EMPTY_CELL.flags.writeable = False


def _frozen(array, dtype, shape=None, name="array"):
    """Return a read-only contiguous copy of `array`, validating its shape.

    The copy is unconditional. `np.ascontiguousarray` hands back its argument
    when no conversion is needed, and marking *that* read-only would reach out
    and freeze an array the caller still owns -- which is how a finite
    difference that writes into a scratch buffer comes to fail on its second
    displacement. A (N, 3) copy costs nothing next to an energy evaluation.
    """
    out = np.array(array, dtype=dtype, order="C", copy=True)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    out.flags.writeable = False
    return out


class System:
    """An atomic configuration: numbers, positions, cell, periodicity.

    Lengths are Angstroms and positions are Cartesian.

    Args:
        numbers: (N,) atomic numbers
        positions: (N, 3) Cartesian positions in Angstroms
        cell: (3, 3) lattice vectors as rows, or None for an isolated system
        pbc: (3,) booleans, or a single bool for all three directions. Defaults
            to True in every direction when a cell is given, False otherwise.
        info: model inputs that are not geometry, e.g. ``{"charge": 0}``

    Examples:
        >>> s = System([8, 1, 1], [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
        >>> len(s), s.periodic
        (3, False)
    """

    __slots__ = ("_numbers", "_positions", "_cell", "_pbc", "_version", "info")

    def __init__(self, numbers, positions, cell=None, pbc=None, info=None):
        self._numbers = _frozen(numbers, np.int32, name="numbers")
        if self._numbers.ndim != 1:
            raise ValueError(f"numbers must be 1D, got shape {self._numbers.shape}")
        n = len(self._numbers)
        self._positions = _frozen(positions, np.float64, (n, 3), "positions")

        if cell is None:
            self._cell = _EMPTY_CELL
        else:
            self._cell = _frozen(cell, np.float64, (3, 3), "cell")

        if pbc is None:
            pbc = cell is not None
        if np.ndim(pbc) == 0:
            pbc = np.repeat(bool(pbc), 3)
        self._pbc = _frozen(pbc, bool, (3,), "pbc")

        if np.any(self._pbc) and not np.any(self._cell):
            raise ValueError("a periodic system needs a cell with non-zero vectors")

        self.info = dict(info) if info else {}
        self._version = 0

    # -- state ---------------------------------------------------------------

    @property
    def numbers(self) -> np.ndarray:
        "(N,) atomic numbers, read-only"
        return self._numbers

    @property
    def positions(self) -> np.ndarray:
        "(N, 3) Cartesian positions in Angstroms, read-only"
        return self._positions

    @property
    def cell(self) -> np.ndarray:
        "(3, 3) lattice vectors as rows in Angstroms, read-only; zeros if isolated"
        return self._cell

    @property
    def pbc(self) -> np.ndarray:
        "(3,) which directions are periodic, read-only"
        return self._pbc

    @property
    def version(self) -> int:
        """Bumped by every geometry change.

        `(id(system), system.version)` identifies a geometry, so a cached
        result can be validated in constant time.
        """
        return self._version

    def __len__(self) -> int:
        return len(self._numbers)

    @property
    def n_atoms(self) -> int:
        "Number of atoms"
        return len(self._numbers)

    @property
    def periodic(self) -> bool:
        "True if every direction is periodic"
        return bool(np.all(self._pbc))

    @property
    def volume(self) -> float:
        "Cell volume in Angstrom^3, or 0.0 for an isolated system"
        if not np.any(self._pbc):
            return 0.0
        return float(abs(np.linalg.det(self._cell)))

    def __repr__(self) -> str:
        formula = "".join(
            f"{sym}{count}" if count > 1 else sym
            for sym, count in _formula_counts(self._numbers)
        )
        if self.periodic:
            a, b, c = np.linalg.norm(self._cell, axis=1)
            return f"<System {formula} periodic a={a:.3f} b={b:.3f} c={c:.3f}>"
        return f"<System {formula} isolated>"

    # -- geometry changes ----------------------------------------------------

    def set_positions(self, positions) -> None:
        """Replace the Cartesian positions (Angstroms) and bump `version`."""
        self._positions = _frozen(
            positions, np.float64, (len(self._numbers), 3), "positions"
        )
        self._version += 1

    def set_cell(self, cell, scale_atoms: bool = False) -> None:
        """Replace the lattice vectors (rows, Angstroms) and bump `version`.

        Args:
            cell: (3, 3) new lattice vectors as rows
            scale_atoms: carry the atoms along in fractional coordinates rather
                than leaving them where they are in Cartesian space
        """
        new = _frozen(cell, np.float64, (3, 3), "cell")
        if scale_atoms:
            fractional = self.scaled_positions
            self._cell = new
            self._positions = _frozen(
                fractional @ new, np.float64, (len(self._numbers), 3), "positions"
            )
        else:
            self._cell = new
        self._version += 1

    def set_scaled_positions(self, scaled) -> None:
        """Replace positions given in fractional coordinates."""
        scaled = np.asarray(scaled, dtype=np.float64)
        self.set_positions(scaled @ self._cell)

    @property
    def scaled_positions(self) -> np.ndarray:
        """(N, 3) fractional coordinates. Raises for an isolated system."""
        if not np.any(self._pbc):
            raise ValueError("an isolated system has no fractional coordinates")
        return np.linalg.solve(self._cell.T, self._positions.T).T

    def wrapped(self) -> System:
        """A copy with every atom brought into the [0, 1) fractional cell."""
        if not self.periodic:
            return self.copy()
        out = self.copy()
        out.set_scaled_positions(self.scaled_positions % 1.0)
        return out

    def copy(self) -> System:
        """An independent `System` with the same geometry, at version 0."""
        return System(
            self._numbers, self._positions, self._cell, self._pbc, dict(self.info)
        )

    # -- conversions ---------------------------------------------------------

    @classmethod
    def from_crystal(cls, crystal, **info) -> System:
        """The P1 unit cell of a `Crystal`, with no symmetry attached."""
        atoms = crystal.unit_cell_atoms()
        return cls(
            atoms["element"],
            atoms["cart_pos"],
            crystal.unit_cell.direct,
            True,
            info,
        )

    @classmethod
    def from_molecule(cls, molecule, **info) -> System:
        """An isolated `Molecule`."""
        return cls(molecule.atomic_numbers, molecule.positions, info=info)

    def to_crystal(self, space_group=1):
        """A `Crystal` in the given space group, the atoms taken as its asymmetric unit.

        The default, P1, is the honest reading of a `System`: it carries no
        symmetry, so anything higher is a claim the caller is making.
        """
        from chmpy.core import Element
        from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell

        if not self.periodic:
            raise ValueError("an isolated system cannot become a Crystal")
        cell = UnitCell(self._cell)
        return Crystal(
            cell,
            space_group
            if isinstance(space_group, SpaceGroup)
            else SpaceGroup(space_group),
            AsymmetricUnit(
                [Element.from_atomic_number(n) for n in self._numbers],
                cell.to_fractional(self._positions),
            ),
        )

    def to_molecule(self):
        """A `Molecule` holding these atoms."""
        from chmpy.core import Molecule

        return Molecule.from_arrays(np.asarray(self._numbers), self._positions)

    @classmethod
    def from_ase(cls, atoms) -> System:
        """Convert an `ase.Atoms`."""
        return cls(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            np.asarray(atoms.cell),
            atoms.pbc,
            dict(atoms.info),
        )

    def to_ase(self):
        """Convert to an `ase.Atoms`."""
        from chmpy.util.optional import require

        ase = require("ase", "converting a System to ase.Atoms")
        return ase.Atoms(
            numbers=np.asarray(self._numbers),
            positions=np.asarray(self._positions),
            cell=np.asarray(self._cell),
            pbc=np.asarray(self._pbc),
            info=dict(self.info),
        )


def _formula_counts(numbers):
    """(symbol, count) pairs in Hill order, for repr."""
    from chmpy.core import Element

    unique, counts = np.unique(numbers, return_counts=True)
    symbols = {int(z): Element.from_atomic_number(int(z)).symbol for z in unique}
    pairs = {symbols[int(z)]: int(c) for z, c in zip(unique, counts, strict=True)}
    order = []
    for sym in ("C", "H"):
        if sym in pairs:
            order.append((sym, pairs.pop(sym)))
    order.extend(sorted(pairs.items()))
    return order

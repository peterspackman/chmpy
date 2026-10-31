"""Character tables of the crystallographic point groups.

Computed from a group's own rotations, in whatever setting they are given:
conjugacy classes from the exact multiplication table, and the irreducible
characters as the simultaneous eigenvectors of the class multiplication
coefficients (Burnside), found from one random combination of them (Dixon).

Vibrations are real, so a complex-conjugate pair of irreducible
representations is merged into one physically irreducible representation of
twice the dimension (an E), and the table holds those, with `norms` 2.

Labels follow Mulliken. Where the rules leave a choice:

- the principal axis is the unique axis of highest order, counting -4 as 4 and
  -3, -6 as 3; A or B is the character of its highest-order operation, proper
  if there is one
- subscripts 1, 2 on A and B use a two-fold perpendicular to the principal
  axis, or failing one a mirror containing it; with two classes of these, the
  class that fixes the next lattice vector after the principal axis (a for an
  axis along c), so 1 is symmetric under C2' along a, or a mirror containing a
- with no principal axis (222, mmm), B1, B2, B3 are symmetric under the
  two-fold along c, b and a
- cubic subscripts come from the 4 (or -4) operations, E1 and E2 from the
  principal two-fold, g and u from the inversion, ' and '' from the mirror
  normal to the principal axis
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .hall_symbol import parse_hall_symbol
from .symmetry_operation import _IMPROPER_ROTATION_SYMBOLS, _PROPER_ROTATION_SYMBOLS

_POWERS = 3 ** np.arange(9)
_PROPER_ORDER = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}
# order of the axis an improper operation lies on, by its proper part's order
_IMPROPER_AXIS_ORDER = {1: 0, 2: 0, 3: 3, 4: 4, 6: 3}
_LETTERS = {1: "A", 2: "E", 3: "T"}


def rotation_codes(rotations) -> np.ndarray:
    "A unique integer for each (..., 3, 3) rotation with entries in {-1, 0, 1}"
    rotations = np.asarray(rotations, dtype=np.int64)
    return (rotations.reshape(*rotations.shape[:-2], 9) + 1) @ _POWERS


@dataclass(frozen=True, eq=False)
class CharacterTable:
    """Characters of the physically irreducible representations of a group.

    Attributes:
        labels: Mulliken label of each representation, e.g. "B2g"
        classes: symbol of each conjugacy class, its operation's
            Hermann-Mauguin symbol and axis, e.g. "4[001]", "m[110]"
        class_sizes: (k,) number of operations in each class
        characters: (n, k) integer characters, a row per representation
        norms: (n,) 1, or 2 for a merged complex-conjugate pair
        class_index: rotation code (see `rotation_codes`) -> class index
    """

    labels: tuple[str, ...]
    classes: tuple[str, ...]
    class_sizes: np.ndarray
    characters: np.ndarray
    norms: np.ndarray
    class_index: dict[int, int]

    @property
    def order(self) -> int:
        return int(self.class_sizes.sum())

    def classes_of(self, rotations) -> np.ndarray:
        "Class index of each of a (..., 3, 3) array of the group's rotations"
        codes = rotation_codes(rotations)
        return np.vectorize(self.class_index.__getitem__, otypes=[int])(codes)

    def reduce(self, characters) -> np.ndarray:
        """How many times each representation occurs in a reducible one.

        Args:
            characters: (k,) or (k, m) characters of the representation(s) on
                each class

        Returns:
            (n,) or (n, m) multiplicities, as floats (integers for a true
            representation)
        """
        weighted = self.characters * self.class_sizes
        return (weighted @ np.asarray(characters)) / (
            self.order * self.norms.reshape(-1, *[1] * (np.ndim(characters) - 1))
        )

    def __str__(self) -> str:
        width = max(map(len, self.classes + self.labels)) + 1
        head = " " * width + "".join(
            f"{f'{n}x{c}':>{width + 2}}"
            for n, c in zip(self.class_sizes, self.classes, strict=True)
        )
        rows = (
            f"{label:<{width}}" + "".join(f"{x:>{width + 2}}" for x in row)
            for label, row in zip(self.labels, self.characters, strict=True)
        )
        return "\n".join((head, *rows))


@cache
def hall_character_table(hall: str) -> CharacterTable:
    "The character table of the point group of a (symmorphic) Hall symbol"
    rotations = np.array([s.rotation for s in parse_hall_symbol(hall)])
    return character_table(np.rint(rotations).astype(np.int64))


def character_table(rotations) -> CharacterTable:
    """The character table of a point group, given as its rotation matrices.

    Args:
        rotations: (G, 3, 3) integer rotations forming a group, in any basis
            where their entries are -1, 0 or 1 (any crystallographic setting)

    Returns:
        CharacterTable
    """
    rotations = np.asarray(rotations, dtype=np.int64)
    table, inverse = _multiplication_table(rotations)
    members = _conjugacy_classes(table, inverse)
    geometry = _Geometry(rotations)
    # each class shown by a member fixing a lattice vector if one does
    representatives = [
        next((g for g in m if geometry.fixes_x(g)), m[0]) for m in members
    ]
    by_key = sorted(
        range(len(members)), key=lambda k: geometry.class_key(representatives[k])
    )
    members = [members[k] for k in by_key]
    representatives = [representatives[k] for k in by_key]
    class_of = np.empty(len(rotations), dtype=np.int64)
    for k, m in enumerate(members):
        class_of[m] = k
    sizes = np.array([len(m) for m in members])

    characters, norms = _real_characters(table, inverse, class_of, sizes)
    labels = _labels(characters, representatives, geometry)
    rows = np.lexsort((labels, [_label_key(x) for x in labels]))
    codes = rotation_codes(rotations)
    return CharacterTable(
        labels=tuple(labels[i] for i in rows),
        classes=tuple(geometry.symbol(g) for g in representatives),
        class_sizes=sizes,
        characters=characters[rows],
        norms=norms[rows],
        class_index={int(c): int(k) for c, k in zip(codes, class_of, strict=True)},
    )


def _multiplication_table(rotations):
    "table[a, b] = index of rotations[a] @ rotations[b], and each inverse"
    codes = rotation_codes(rotations)
    order = np.argsort(codes)
    products = rotation_codes(np.einsum("aij,bjk->abik", rotations, rotations))
    table = order[np.searchsorted(codes[order], products).clip(0, len(codes) - 1)]
    if not np.array_equal(codes[table], products):
        raise ValueError("the rotations do not form a group")
    identity = int(np.flatnonzero(codes == rotation_codes(np.eye(3)))[0])
    inverse = np.argmax(table == identity, axis=1)
    return table, inverse


def _conjugacy_classes(table, inverse) -> list[np.ndarray]:
    "The classes, as arrays of operation indices, in order of first member"
    n = len(table)
    # conjugates[g, h] = h g h^-1
    conjugates = table[table.T, inverse[None, :]]
    seen = np.zeros(n, dtype=bool)
    classes = []
    for g in range(n):
        if not seen[g]:
            members = np.unique(conjugates[g])
            seen[members] = True
            classes.append(members)
    return classes


def _real_characters(table, inverse, class_of, sizes):
    """Characters of the physically irreducible representations.

    With K_r the class sums, K_r K_s = sum_t c_rst K_t, and the central
    characters w_t = |C_t| chi_t / chi(1) of each irreducible representation
    satisfy w_r w_s = sum_t c_rst w_t: they are common eigenvectors of every
    matrix c_r.., so of a random combination of them.
    """
    n, k = len(table), len(sizes)
    representatives = np.array([np.flatnonzero(class_of == t)[0] for t in range(k)])
    # c[r, s, t] = #{x in C_r : x^-1 z in C_s}, for z the representative of C_t
    c = np.zeros((k, k, k))
    x = np.arange(n)
    for t, z in enumerate(representatives):
        np.add.at(c, (class_of[x], class_of[table[inverse[x], z]], t), 1)
    weights = np.random.default_rng(0).uniform(1, 2, k)
    values, vectors = np.linalg.eig(np.einsum("r,rst->st", weights, c))
    gaps = np.abs(values[:, None] - values[None, :]) + np.eye(k)
    if gaps.min() < 1e-6:
        raise ValueError("could not separate the irreducible representations")

    central = vectors / vectors[0]  # the identity is class 0
    dimension = np.sqrt(n / np.sum(np.abs(central) ** 2 / sizes[:, None], axis=0))
    complex_chi = (central * dimension / sizes[:, None]).T

    rows, norms, used = [], [], set()
    for i, chi in enumerate(complex_chi):
        if i in used:
            continue
        if np.abs(chi.imag).max() < 1e-6:
            rows.append(chi.real)
            norms.append(1)
            continue
        partner = next(
            j
            for j in range(i + 1, k)
            if j not in used and np.allclose(complex_chi[j], chi.conj(), atol=1e-6)
        )
        used.add(partner)
        rows.append(2 * chi.real)
        norms.append(2)
    characters = np.rint(rows).astype(np.int64)
    if not np.allclose(characters, rows, atol=1e-6):
        raise ValueError("characters of a real representation are not integers")
    return characters, np.array(norms)


class _Geometry:
    "The axis and kind of each operation, as Mulliken's rules need them"

    def __init__(self, rotations):
        self.rotations = rotations
        self.det = np.rint(np.linalg.det(rotations)).astype(int)
        proper = rotations * self.det[:, None, None]
        self.proper_order = np.array([_PROPER_ORDER[int(np.trace(p))] for p in proper])
        self.axis_order = np.where(
            self.det > 0,
            np.where(self.proper_order > 1, self.proper_order, 0),
            [_IMPROPER_AXIS_ORDER[o] for o in self.proper_order],
        )
        self.axes = np.array([_axis(p) for p in proper])
        # metric invariant under the group: perpendicularity without a cell
        self.metric = np.einsum("gji,gjk->ik", rotations, rotations)
        self.principal = self._principal()
        self.x = self._next_lattice_vector()

    def dot(self, u, v) -> float:
        return float(u @ self.metric @ v)

    def parallel(self, u, v) -> bool:
        return abs(abs(self.dot(u, v)) ** 2 - self.dot(u, u) * self.dot(v, v)) < 1e-8

    def perpendicular(self, u, v) -> bool:
        return abs(self.dot(u, v)) < 1e-8

    def _principal(self):
        top = self.axis_order.max()
        if top < 2:
            return None
        axes = self.axes[self.axis_order == top]
        if not all(self.parallel(axes[0], a) for a in axes):
            return None
        return axes[0]

    def _next_lattice_vector(self):
        "The lattice vector after the principal axis's (a after c), if one is"
        if self.principal is None:
            return None
        basis = np.eye(3, dtype=np.int64)
        along = [i for i in range(3) if self.parallel(basis[i], self.principal)]
        start = along[0] if along else 2
        for i in (start + 1, start + 2):
            if self.perpendicular(basis[i % 3], self.principal):
                return basis[i % 3]
        return None

    def along_principal(self, g) -> bool:
        return (
            self.principal is not None
            and self.proper_order[g] > 1
            and self.parallel(self.axes[g], self.principal)
        )

    def fixes_x(self, g) -> bool:
        return self.x is not None and np.array_equal(self.rotations[g] @ self.x, self.x)

    def class_key(self, g):
        "Proper first; then E (or i), the principal axis, others; by order"
        return (
            self.det[g] < 0,
            self.proper_order[g] > 1,
            not self.along_principal(g),
            -self.axis_order[g],
            not self.fixes_x(g),
            g,
        )

    def symbol(self, g) -> str:
        trace = int(np.trace(self.rotations[g]))
        if self.det[g] > 0:
            symbol = _PROPER_ROTATION_SYMBOLS[trace]
        else:
            symbol = _IMPROPER_ROTATION_SYMBOLS[trace]
        if self.proper_order[g] == 1:
            return symbol
        return f"{symbol}[{''.join(map(str, self.axes[g]))}]"


def _axis(proper) -> np.ndarray:
    "The rotation axis of a proper rotation, as the smallest integer vector"
    if np.array_equal(proper, np.eye(3)):
        return np.zeros(3, dtype=np.int64)
    # summing every power of R projects onto its axis
    fixed, power = np.eye(3, dtype=np.int64), proper
    for _ in range(_PROPER_ORDER[int(np.trace(proper))] - 1):
        fixed, power = fixed + power, power @ proper
    column = fixed[:, np.argmax(np.abs(fixed).sum(axis=0))]
    column = column // np.gcd.reduce(column[column != 0])
    first = column[np.flatnonzero(column)[0]]
    return column * np.sign(first)


def _labels(characters, first, geometry: _Geometry) -> list[str]:
    "Mulliken labels of each row, given each class's representative"
    g = geometry

    def find(test):
        "The first class whose first operation passes `test`, or None"
        return next((k for k, op in enumerate(first) if test(op)), None)

    inversion = find(lambda op: g.det[op] < 0 and g.proper_order[op] == 1)
    if g.principal is None:
        # only Cs has primes without a principal axis: its mirror is sigma_h
        mirror_h = None
        if g.axis_order.max() < 2:
            mirror_h = find(lambda op: g.det[op] < 0 and g.proper_order[op] == 2)
        principal_op = None
    else:
        mirror_h = find(
            lambda op: (
                g.det[op] < 0
                and g.proper_order[op] == 2
                and g.parallel(g.axes[op], g.principal)
            )
        )
        top = g.axis_order.max()
        on_axis = [
            k
            for k, op in enumerate(first)
            if g.axis_order[op] == top and g.along_principal(op)
        ]
        principal_op = min(on_axis, key=lambda k: g.det[first[k]] < 0)
    cubic = g.principal is None and g.axis_order.max() >= 3
    twofolds = [k for k, op in enumerate(first) if g.axis_order[op] == 2]
    # 222 and mmm: three perpendicular two-folds and no principal axis
    orthorhombic = g.principal is None and not cubic and bool(twofolds)

    def base(row) -> tuple[str, str, str]:
        dim = int(row[0])
        letter = _LETTERS[dim]
        if dim == 1 and principal_op is not None and row[principal_op] < 0:
            letter = "B"
        if dim == 1 and orthorhombic and row[twofolds].min() < 0:
            letter = "B"
        parity = "" if inversion is None else ("g" if row[inversion] > 0 else "u")
        prime = ""
        if inversion is None and mirror_h is not None:
            prime = "'" if row[mirror_h] > 0 else "''"
        return letter, parity, prime

    bases = [base(row) for row in characters]
    clash = [bases.count(b) > 1 for b in bases]
    subscript = _subscripter(first, geometry, cubic) if any(clash) else None
    labels = []
    for row, b, c in zip(characters, bases, clash, strict=True):
        number = subscript(row, b[0]) if c else ""
        labels.append(f"{b[0]}{number}{b[1]}{b[2]}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"could not tell representations apart: {labels}")
    return labels


def _subscripter(first, g: _Geometry, cubic: bool):
    "The function giving a representation's numeric subscript"

    def classes(test):
        return [k for k, op in enumerate(first) if test(op)]

    if cubic:
        fourfold = classes(lambda op: g.axis_order[op] == 4)
        fourfold = min(fourfold, key=lambda k: g.det[first[k]] < 0)
        return lambda row, letter: "1" if row[fourfold] > 0 else "2"

    if g.principal is None:  # 222 and mmm: the two-folds along c, b, a
        basis = np.eye(3, dtype=np.int64)
        along = [
            classes(
                lambda op, axis=axis: (
                    g.det[op] > 0
                    and g.proper_order[op] == 2
                    and g.parallel(g.axes[op], axis)
                )
            )[0]
            for axis in basis[::-1]
        ]
        return lambda row, letter: str(1 + int(np.argmax(row[along] > 0)))

    twofold = classes(
        lambda op: g.det[op] > 0 and g.along_principal(op) and g.proper_order[op] == 2
    )

    def perpendicular(op):
        return g.proper_order[op] == 2 and g.perpendicular(g.axes[op], g.principal)

    side = classes(lambda op: g.det[op] > 0 and perpendicular(op)) or classes(
        lambda op: g.det[op] < 0 and perpendicular(op)
    )
    if len(side) > 1:
        side = [k for k in side if g.fixes_x(first[k])] or side
    side = side[0] if side else None

    def subscript(row, letter):
        if letter == "E":
            return "1" if row[twofold[0]] < 0 else "2"
        return "1" if row[side] > 0 else "2"

    return subscript


def _label_key(label: str) -> int:
    "Mulliken order: g before u, ' before '', then A, B, E, T"
    return 10 * ("u" in label) + 5 * label.endswith("''") + "ABET".index(label[0])

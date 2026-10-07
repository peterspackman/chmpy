"""Hall space group symbols.

A Hall symbol (Hall, Acta Cryst. A37, 517, 1981) lists generators explicitly:

    [-]L  N1 [N2 [N3 [N4]]]  [V]

L is the lattice centring (P A B C I R S T F), with a leading '-' adding the
inversion at the origin. Each N is a matrix symbol: an optional '-' for an
improper rotation, the order (1 2 3 4 6), an optional axis (x y z, ' or "
for 2-fold face diagonals, * for the 3-fold body diagonal) and translation
symbols (a b c n u v w d, or a screw digit). V is a change of basis, either
an origin shift in twelfths "(0 0 1)" or an operation "(x,y,z+1/4)"
(Grosse-Kunstleve, Acta Cryst. A55, 383, 1999).

All arithmetic is exact: rotations are integer matrices and translations
`Fraction`s.
"""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np

from .symmetry_operation import SymmetryOperation

_HALF, _THIRD, _QUARTER = Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)

_CENTRING = {
    "P": [],
    "A": [(0, _HALF, _HALF)],
    "B": [(_HALF, 0, _HALF)],
    "C": [(_HALF, _HALF, 0)],
    "I": [(_HALF, _HALF, _HALF)],
    "R": [(2 * _THIRD, _THIRD, _THIRD), (_THIRD, 2 * _THIRD, 2 * _THIRD)],
    "S": [(_THIRD, _THIRD, 2 * _THIRD), (2 * _THIRD, 2 * _THIRD, _THIRD)],
    "T": [(_THIRD, 2 * _THIRD, _THIRD), (2 * _THIRD, _THIRD, 2 * _THIRD)],
    "F": [(0, _HALF, _HALF), (_HALF, 0, _HALF), (_HALF, _HALF, 0)],
}

_TRANSLATION = {
    "a": (_HALF, 0, 0),
    "b": (0, _HALF, 0),
    "c": (0, 0, _HALF),
    "n": (_HALF, _HALF, _HALF),
    "u": (_QUARTER, 0, 0),
    "v": (0, _QUARTER, 0),
    "w": (0, 0, _QUARTER),
    "d": (_QUARTER, _QUARTER, _QUARTER),
}

# Proper rotations about z; x and y follow by cycling the axes.
_ROTATION_Z = {
    1: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    2: ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    3: ((0, -1, 0), (1, -1, 0), (0, 0, 1)),
    4: ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    6: ((1, -1, 0), (1, 0, 0), (0, 0, 1)),
}
# 2-fold axes along the face diagonals a-b (') and a+b ("), relative to z.
_FACE_DIAGONAL_Z = {
    "'": ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
    '"': ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
}
_BODY_DIAGONAL = ((0, 0, 1), (1, 0, 0), (0, 1, 0))
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

_MATRIX_SYMBOL = re.compile(r"^(-?)([12346])([xyz'\"*]?)([abcnuvwd12345]*)$")


def _on_axis(matrix, axis: int) -> np.ndarray:
    "A z-axis matrix re-expressed about axis 0 (x), 1 (y) or 2 (z)."
    m = np.array(matrix, dtype=int)
    order = [(i - axis - 1) % 3 for i in range(3)]
    return m[np.ix_(order, order)]


def _default_axis(position: int, order: int, previous_order: int | None) -> str:
    "The implied axis of the matrix symbol at `position` (0-based)."
    if position == 0 or order == 1:
        return "z"
    if position == 1 and order == 2:
        return "x" if previous_order in (2, 4) else "'"
    if position == 2 and order == 3:
        return "*"
    raise ValueError(f"Matrix symbol {position + 1} of order {order} needs an axis")


def _matrix(token, position, previous_order, previous_axis):
    "Seitz pair (R, t) of one matrix symbol, with its order and axis."
    match = _MATRIX_SYMBOL.match(token)
    if match is None:
        raise ValueError(f"Invalid matrix symbol {token!r}")
    improper, order, axis, translations = match.groups()
    order = int(order)
    axis = axis or _default_axis(position, order, previous_order)

    if axis in _AXIS_INDEX:
        index = _AXIS_INDEX[axis]
        rotation = _on_axis(_ROTATION_Z[order], index)
    elif axis in _FACE_DIAGONAL_Z:
        if order != 2:
            raise ValueError(f"Face-diagonal axis needs a 2-fold rotation: {token!r}")
        index = None
        rotation = _on_axis(_FACE_DIAGONAL_Z[axis], _AXIS_INDEX[previous_axis or "z"])
    else:
        if order != 3:
            raise ValueError(f"Body-diagonal axis needs a 3-fold rotation: {token!r}")
        index = None
        rotation = np.array(_BODY_DIAGONAL, dtype=int)
    if improper:
        rotation = -rotation

    translation = [Fraction(0)] * 3
    for symbol in translations:
        if symbol.isdigit():
            if index is None:
                raise ValueError(f"Screw translation needs a principal axis: {token!r}")
            translation[index] += Fraction(int(symbol), order)
        else:
            for i, x in enumerate(_TRANSLATION[symbol]):
                translation[i] += x
    return (rotation, tuple(t % 1 for t in translation)), order, axis


def _as_twelfths(op):
    "A Seitz pair as (9 rotation ints, 3 translation ints in twelfths)."
    rotation, translation = op
    twelfths = []
    for x in translation:
        value = Fraction(x) * 12
        if value.denominator != 1:
            raise ValueError(f"Translation {x} is not a multiple of 1/12")
        twelfths.append(int(value) % 12)
    return tuple(int(x) for x in np.asarray(rotation).flat), tuple(twelfths)


def _compose(a, b):
    "a after b, with translations in twelfths reduced mod 12."
    (ra, ta), (rb, tb) = a, b
    r = tuple(
        ra[3 * i] * rb[j] + ra[3 * i + 1] * rb[3 + j] + ra[3 * i + 2] * rb[6 + j]
        for i in range(3)
        for j in range(3)
    )
    t = tuple(
        (ra[3 * i] * tb[0] + ra[3 * i + 1] * tb[1] + ra[3 * i + 2] * tb[2] + ta[i]) % 12
        for i in range(3)
    )
    return r, t


def _translation_group(vectors):
    "All sums of the centring vectors, mod 1, in twelfths."
    group = {(0, 0, 0)}
    frontier = list(vectors)
    while frontier:
        t = frontier.pop()
        if t in group:
            continue
        group.add(t)
        frontier.extend(
            tuple((x + y) % 12 for x, y in zip(t, v, strict=True)) for v in vectors
        )
    return sorted(group)


def _expand(generators):
    """Every operation, one per coset of the integer lattice.

    The point group cosets are generated by right multiplication with the
    generators, holding each translation as its smallest representative
    modulo the centred lattice; the centring translations are added last.
    """
    identity_rotation = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    centring = _translation_group(
        [t for r, t in generators if r == identity_rotation and any(t)]
    )
    rotations = [g for g in generators if g[0] != identity_rotation]

    def canonical(op):
        r, t = op
        return r, min(
            tuple((x + c) % 12 for x, c in zip(t, v, strict=True)) for v in centring
        )

    identity = (identity_rotation, (0, 0, 0))
    cosets = {identity[0]: identity}
    frontier = [identity]
    while frontier:
        op = frontier.pop()
        for generator in rotations:
            new = canonical(_compose(op, generator))
            existing = cosets.get(new[0])
            if existing is None:
                cosets[new[0]] = new
                frontier.append(new)
                if len(cosets) > 48:
                    raise ValueError(
                        "Generators do not form a crystallographic space group"
                    )
            elif existing != new:
                raise ValueError(
                    "Generators do not form a crystallographic space group"
                )
    return [
        (r, tuple((x + c) % 12 for x, c in zip(t, v, strict=True)))
        for r, t in cosets.values()
        for v in centring
    ]


def _change_of_basis(text):
    "The change of basis operation of a Hall symbol's trailing '(...)'."
    body = text.strip()[1:-1].strip()
    if "," in body:
        op = SymmetryOperation.from_string_code(body)
        rotation = np.round(op.rotation).astype(int)
        translation = tuple(Fraction(round(float(x) * 12), 12) for x in op.translation)
        return rotation, translation
    parts = body.split()
    if len(parts) != 3:
        raise ValueError(f"Origin shift must have three components: {text!r}")
    return np.eye(3, dtype=int), tuple(Fraction(int(x), 12) for x in parts)


def _inverse(op):
    "Inverse of a Seitz pair in twelfths."
    rotation, translation = op
    r_inv = np.round(np.linalg.inv(np.reshape(rotation, (3, 3)))).astype(int)
    t = tuple(
        -sum(int(r_inv[i, j]) * translation[j] for j in range(3)) % 12 for i in range(3)
    )
    return tuple(int(x) for x in r_inv.flat), t


def hall_generators(symbol: str):
    """The generators of a Hall symbol as exact Seitz pairs.

    Returns:
        the generators, as (rotation, translation) pairs of an integer
        (3, 3) matrix and three `Fraction`s, in the basis before any change
        of basis; and the change of basis as such a pair, or None
    """
    symbol = symbol.strip()
    change = None
    if "(" in symbol:
        start = symbol.index("(")
        if not symbol.endswith(")"):
            raise ValueError(f"Unbalanced change of basis in {symbol!r}")
        change = _change_of_basis(symbol[start:])
        symbol = symbol[:start]
    tokens = symbol.split()
    if not tokens:
        raise ValueError("Empty Hall symbol")
    lattice = tokens[0]
    centrosymmetric = lattice.startswith("-")
    letter = lattice.lstrip("-").upper()
    if letter not in _CENTRING or len(lattice) - centrosymmetric != 1:
        raise ValueError(f"Invalid Hall lattice symbol {lattice!r}")
    if not 1 <= len(tokens) - 1 <= 4:
        raise ValueError(f"A Hall symbol has one to four matrix symbols: {symbol!r}")

    identity = np.eye(3, dtype=int)
    generators = [(identity, tuple(Fraction(x) for x in t)) for t in _CENTRING[letter]]
    if centrosymmetric:
        generators.append((-identity, (Fraction(0),) * 3))
    previous_order = previous_axis = None
    for position, token in enumerate(tokens[1:]):
        op, order, axis = _matrix(token, position, previous_order, previous_axis)
        if order != 1:
            previous_order = order
            if axis in _AXIS_INDEX:
                previous_axis = axis
        generators.append(op)
    return generators, change


def parse_hall_symbol(symbol: str) -> list[SymmetryOperation]:
    """All operations of the space group a Hall symbol describes.

    One operation per coset of the integer lattice, so centring translations
    are included (as in `SpaceGroup.symmetry_operations`).

    Args:
        symbol: Hall symbol, e.g. "-P 2ybc" or "P 31 2 (0 0 4)"

    Returns:
        list of SymmetryOperation, sorted by integer code
    """
    generators, change = hall_generators(symbol)
    ops = _expand([_as_twelfths(g) for g in generators])
    if change is not None:
        change = _as_twelfths(change)
        change_inv = _inverse(change)
        ops = [_compose(change, _compose(op, change_inv)) for op in ops]
    result = [
        SymmetryOperation(np.reshape(r, (3, 3)), np.array(t, dtype=float) / 12)
        for r, t in ops
    ]
    return sorted(result, key=lambda op: op.integer_code)

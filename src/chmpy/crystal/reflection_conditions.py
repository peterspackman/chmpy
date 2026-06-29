"""
General reflection conditions (systematic absences) in International Tables form.

Given the symmetry operations of a space group, derive the reflection
conditions grouped by reflection class (hkl, 0kl, h0l, ..., h00, 0k0, 00l, ...)
as printed in the International Tables for Crystallography.

The derivation is exact and operation-based. The reflections left invariant by
an operation (R, t) -- those with R^T h = h -- form an integer sublattice S of
reciprocal space; these sublattices ARE the reflection classes (rank 3 = the
general class hkl, rank 2 = a zone like 0kl, rank 1 = an axis like 00l). Every
operation fixing such a class pointwise forces

    t . h  in Z   for h in S

for the reflection to be present. Per class these congruences are collected,
reduced to a minimal set, and expressed relative to the more general classes
already listed -- so each printed line carries only the *additional* condition,
as in the International Tables. Classes are derived from the operations, so
hexagonal and cubic zones are handled without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import gcd, lcm

import numpy as np

_AXIS_SYMBOLS = ("h", "k", "l")


@dataclass(frozen=True)
class ReflectionCondition:
    """A reflection condition for a single reflection class.

    Attributes:
        reflection_class: the class label, e.g. "0kl"
        condition: the additional human-readable condition, e.g. "k+l=4n"
        basis: (3, d) integer basis B; reflections in the class are h = B p
        forms: tuple of (coefficients, modulus) congruences on (h, k, l); a
            reflection in the class is present iff coefficients . h == 0
            (mod modulus) for every form (the full condition on the class)
    """

    reflection_class: str
    condition: str
    basis: tuple = field(repr=False)
    forms: tuple = field(repr=False)

    def __str__(self) -> str:
        return f"{self.reflection_class}: {self.condition}"

    def applies_to(self, hkl) -> bool:
        "Whether `hkl` belongs to this reflection class."
        return _in_lattice(np.asarray(hkl, dtype=int), np.array(self.basis))

    def is_present(self, hkl) -> bool:
        "Whether `hkl` satisfies the full condition on this class."
        h = np.asarray(hkl, dtype=int)
        return all(int(np.dot(coeffs, h)) % m == 0 for coeffs, m in self.forms)


def _canonical(coeffs, modulus):
    """Reduce a congruence coeffs . x == 0 (mod modulus) to canonical form.

    Returns (primitive_coeffs, modulus) with gcd(coeffs) == 1 and the leading
    non-zero coefficient positive, or None if the congruence is trivial.
    """
    coeffs = [int(c) for c in coeffs]
    g = modulus
    for c in coeffs:
        g = gcd(g, abs(c))
    if g == 0:
        return None
    coeffs = [c // g for c in coeffs]
    modulus //= g
    if modulus == 1:
        return None
    gc = 0
    for c in coeffs:
        gc = gcd(gc, abs(c))
    if gc > 1:
        coeffs = [c // gc for c in coeffs]
    for c in coeffs:
        if c != 0:
            if c < 0:
                coeffs = [-x for x in coeffs]
            break
    return tuple(coeffs), modulus


def _in_lattice(hkl: np.ndarray, basis: np.ndarray) -> bool:
    "Whether reflection hkl lies in the integer column span of basis."
    if basis.shape[1] == 3:
        return True
    solution, _, _, _ = np.linalg.lstsq(
        basis.astype(float), hkl.astype(float), rcond=None
    )
    rounded = np.round(solution).astype(int)
    return np.array_equal(basis @ rounded, hkl)


def _fixed_sublattice(rotation: np.ndarray) -> np.ndarray:
    """Integer basis (3, d) of the reflections fixed by `rotation`: R^T h = h.

    Returns a canonical primitive basis (shortest independent fixed vectors),
    deterministic so that identical sublattices compare equal.
    """
    matrix = rotation.T - np.eye(3, dtype=int)
    fixed = []
    for vec in product(range(-2, 3), repeat=3):
        if any(vec) and not np.any(matrix @ np.array(vec)):
            fixed.append(_primitive(np.array(vec)))
    if not fixed:
        return np.empty((3, 0), dtype=int)

    unique = sorted({tuple(v) for v in fixed}, key=lambda v: (sum(map(abs, v)), v))
    chosen: list[np.ndarray] = []
    for vec in unique:
        candidate = np.array(vec)
        if np.linalg.matrix_rank(np.array([*chosen, candidate])) > len(chosen):
            chosen.append(candidate)
    return np.array(chosen).T


def _primitive(vec: np.ndarray) -> tuple:
    "Reduce an integer vector to a primitive one with a positive leading entry."
    g = 0
    for x in vec:
        g = gcd(g, abs(int(x)))
    vec = vec // g
    for x in vec:
        if x != 0:
            if x < 0:
                vec = -vec
            break
    return tuple(int(x) for x in vec)


def _same_lattice(a: np.ndarray, b: np.ndarray) -> bool:
    "Whether two integer bases span the same sublattice."
    if a.shape[1] != b.shape[1]:
        return False
    return all(_in_lattice(c, b) for c in a.T) and all(_in_lattice(c, a) for c in b.T)


def _is_sublattice(small: np.ndarray, large: np.ndarray) -> bool:
    "Whether every reflection spanned by `small` lies in `large`."
    return all(_in_lattice(col, large) for col in small.T)


def _axis_symbols(basis: np.ndarray) -> list[str]:
    "Assign a unique axis symbol (h/k/l) to each class parameter (column)."
    symbols = []
    used = set()
    for column in basis.T:
        leading = next(i for i in range(3) if column[i] != 0)
        if _AXIS_SYMBOLS[leading] in used:
            leading = next(i for i in range(3) if _AXIS_SYMBOLS[i] not in used)
        used.add(_AXIS_SYMBOLS[leading])
        symbols.append(_AXIS_SYMBOLS[leading])
    return symbols


def _render(coeffs, symbols, ordered=True) -> str:
    "Render an integer linear form over `symbols` in h,k,l order, sign-fixed."
    terms = list(zip(symbols, (int(c) for c in coeffs), strict=False))
    if ordered:
        terms.sort(key=lambda t: _AXIS_SYMBOLS.index(t[0]))
        first = next((c for _, c in terms if c != 0), 0)
        if first < 0:
            terms = [(s, -c) for s, c in terms]
    out = ""
    for symbol, coeff in terms:
        if coeff == 0:
            continue
        sign = "-" if coeff < 0 else "+"
        out += sign + (f"{abs(coeff)}{symbol}" if abs(coeff) != 1 else symbol)
    return out.lstrip("+")


def _label(basis: np.ndarray) -> str:
    "Build an ITA-style class label, e.g. '0kl', 'hhl', from a basis."
    symbols = _axis_symbols(basis)
    return "".join(_render(row, symbols, ordered=False) or "0" for row in basis)


def _subgroup(generators, modulus: int, d: int) -> set:
    "All elements of the subgroup of (Z/modulus)^d spanned by `generators`."
    gens = [tuple(int(x) % modulus for x in g) for g in generators]
    elements = {(0,) * d}
    stack = [(0,) * d]
    while stack:
        element = stack.pop()
        for gen in gens:
            new = tuple((element[i] + gen[i]) % modulus for i in range(d))
            if new not in elements:
                elements.add(new)
                stack.append(new)
    return elements


def _minimal_conditions(own_forms, inherited_forms, d: int):
    """Minimal generators of own_forms beyond inherited_forms (parameter space).

    Each form is (coeffs, modulus). Returns a list of canonical (coeffs,
    modulus) conditions that, together with the inherited ones, generate the
    same present-set, or None if `own_forms` adds nothing (fully implied).
    """
    if not own_forms:
        return None
    modulus = 1
    for _, m in (*own_forms, *inherited_forms):
        modulus = lcm(modulus, m)

    def chars(forms):
        return [np.array(c) * (modulus // m) for c, m in forms]

    inherited_group = _subgroup(chars(inherited_forms), modulus, d)
    full_group = _subgroup(chars(own_forms) + chars(inherited_forms), modulus, d)
    if len(full_group) == len(inherited_group):
        return None  # implied by a more general class

    # greedily pick the "nicest" own generators until the group is reached
    candidates = []
    for coeffs, m in own_forms:
        reduced = _canonical(np.array(coeffs) * (modulus // m), modulus)
        if reduced is not None:
            candidates.append(reduced)
    # prefer simple forms (fewer terms, small coefficients) but, for the same
    # direction, the strongest (largest modulus) so weaker ones drop out
    candidates.sort(
        key=lambda cm: (sum(c != 0 for c in cm[0]), -cm[1], max(map(abs, cm[0])), cm[0])
    )

    chosen = []
    current = set(inherited_group)
    for coeffs, m in candidates:
        if tuple(int(c) * (modulus // m) % modulus for c in coeffs) in current:
            continue
        chosen.append((coeffs, m))
        current = _subgroup(chars(chosen) + chars(inherited_forms), modulus, d)
        if len(current) == len(full_group):
            break

    # drop any generator made redundant by a later one
    pruned = list(chosen)
    for condition in chosen:
        trial = [c for c in pruned if c is not condition]
        if len(_subgroup(chars(trial) + chars(inherited_forms), modulus, d)) == len(
            full_group
        ):
            pruned = trial
    return pruned


def reflection_conditions(symmetry_operations) -> list[ReflectionCondition]:
    """Derive the reflection conditions grouped by reflection class.

    Args:
        symmetry_operations: the symmetry operations of a space group

    Returns:
        list of ReflectionCondition ordered general -> zonal -> axial, excluding
        classes with no condition or whose condition is already implied by a
        more general class. An empty list means there are no reflection
        conditions (every reflection is allowed).
    """
    ops = [
        (
            np.round(op.rotation).astype(int),
            np.round(np.asarray(op.translation) * 12).astype(int),
        )
        for op in symmetry_operations
    ]

    classes: list[np.ndarray] = []
    for rotation, _ in ops:
        sublattice = _fixed_sublattice(rotation)
        if sublattice.shape[1] and not any(
            _same_lattice(sublattice, existing) for existing in classes
        ):
            classes.append(sublattice)
    classes.sort(key=lambda b: (-b.shape[1], int(np.count_nonzero(b)), _label(b)))

    results: list[ReflectionCondition] = []
    emitted: list[tuple[np.ndarray, list]] = []

    for basis in classes:
        d = basis.shape[1]
        # full condition on this class (hkl space) and projected to parameters
        hkl_forms, param_forms = set(), set()
        for rotation, translation in ops:
            if not np.array_equal(rotation.T @ basis, basis):
                continue
            hkl_form = _canonical(translation, 12)  # t . h == 0 (mod 1)
            if hkl_form is None:
                continue
            hkl_forms.add(hkl_form)
            projected = _canonical(basis.T @ np.array(hkl_form[0]), hkl_form[1])
            if projected is not None:
                param_forms.add(projected)
        if not param_forms:
            continue

        inherited = [
            projected
            for emitted_basis, emitted_forms in emitted
            if _is_sublattice(basis, emitted_basis)
            for projected in _project(emitted_basis, emitted_forms, basis)
        ]
        minimal = _minimal_conditions(sorted(param_forms), inherited, d)
        if not minimal:
            continue  # implied by a more general class

        symbols = _axis_symbols(basis)
        condition = ", ".join(
            sorted(_render(coeffs, symbols) + f"={m}n" for coeffs, m in minimal)
        )
        results.append(
            ReflectionCondition(
                reflection_class=_label(basis),
                condition=condition,
                basis=tuple(map(tuple, basis)),
                forms=tuple(sorted(hkl_forms)),
            )
        )
        emitted.append((basis, sorted(hkl_forms)))

    return results


def _project(source_basis, source_hkl_forms, target_basis):
    "Project a source class's hkl congruences onto the target class parameters."
    projected = []
    for coeffs, m in source_hkl_forms:
        reduced = _canonical(target_basis.T @ np.array(coeffs), m)
        if reduced is not None:
            projected.append(reduced)
    return projected

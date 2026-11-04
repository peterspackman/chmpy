"""COMPACK-style packing similarity: do two crystals pack the same way?

Build a coordination cluster in each crystal -- a central molecule and its
nearest neighbours by centroid distance -- superimpose the two, and match
neighbours by their RMSD in the overlaid frame, position and orientation
together. The answer is how many of the reference neighbours (15 by default)
found a counterpart; all of them means the same packing (Chisholm and
Motherwell, J. Appl. Cryst. 38, 228 (2005)).

This follows klasp's implementation, so that the two agree:

* Every molecule is fitted, as a rigid body, to one reference molecule: the
  first molecule of the first crystal. The fit is seeded from sorted
  intramolecular distances (blind to a symmetric molecule's own symmetry) and
  from matching triangles of atoms (which is not), then refined by alternating
  Kabsch and assignment. A mirror-image fit is accepted.
* With every molecule a placement (centroid c, orthogonal orientation M) of
  one body, two placements differ by
      RMSD^2 = |c_a - c_b|^2 + (2 / m) (tr G - max_Q tr(M_a^T M_b Q G))
  with G the body's gyration tensor and Q its own symmetry operations. Q
  matters: a planar body is carried onto itself by reflection in its plane,
  so its proper and improper placements are the same molecule.
* The overlay is seeded by superimposing the central molecules through each Q
  (and through an inversion, if allowed), then the neighbours are assigned by
  least total RMSD, those within `tolerance` refitted, and repeated.
* Only heavy atoms count, unless they lie on one line that hydrogens off it
  could orient (ethylene, methanol) or there are fewer than two (water).
* The reference cluster has `n_neighbours`; the other offers a larger pool, so
  a counterpart just beyond its own Nth-nearest can still match.
* When molecules are exactly as far away as the Nth, which of them would make
  the N is not a property of the structure, so the whole tied shell is kept:
  the nearer neighbours must each match, and the shell fills the places left.
  The result is still out of N.

For crystals with more than one symmetry-independent environment (Z' > 1),
each distinct one is tried as the centre of the reference cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

#: cost of an assignment that must not be made: finite, as the assignment
#: does arithmetic with it
FORBIDDEN = 1e6
#: molecules this far apart in centroid distance are tied, Angstroms
SHELL_TIE = 1e-4
#: most reference neighbours and candidates a comparison takes, as klasp's
#: kernel does
MAX_REFERENCES, MAX_POOL = 24, 40


class Cluster(NamedTuple):
    """A central molecule (first) and its neighbours, nearest first.

    Attributes:
        centroids: (1 + n, 3)
        orientations: (1 + n, 3, 3)
        untied: the first neighbour tied with the last place, when the cluster
            was extended to keep a tied shell whole; -1 otherwise
    """

    centroids: np.ndarray
    orientations: np.ndarray
    untied: int = -1


@dataclass(frozen=True)
class PackingOptions:
    """Settings for a packing comparison.

    Attributes:
        n_neighbours: molecules in the reference cluster (COMPACK uses 15)
        tolerance: per-molecule RMSD within which a neighbour matches, Angstroms
        radius: centroid distance within which neighbours are looked for
        heavy_only: compare heavy atoms only (see `heavy_atoms_suffice`)
        allow_inversion: also try overlays through an inversion
        self_symmetry_tolerance: how close an atom must land on a same-element
            atom for an operation to count as the molecule's own symmetry
        max_fit_rmsd: a molecule fitting the reference worse than this is not
            the same molecule
        max_iterations: rounds of matching and refitting
    """

    n_neighbours: int = 15
    tolerance: float = 1.0
    radius: float = 14.0
    heavy_only: bool = True
    allow_inversion: bool = True
    self_symmetry_tolerance: float = 0.05
    max_fit_rmsd: float = 1.0
    max_iterations: int = 8

    @property
    def pool_size(self) -> int:
        "Neighbours the compared cluster offers"
        return self.n_neighbours + max(self.n_neighbours // 2, 8)


@dataclass(frozen=True, eq=False)
class PackingComparison:
    """The result of overlaying one crystal's cluster on another's.

    Attributes:
        n_matched: reference neighbours that found a counterpart
        n_neighbours: reference neighbours
        rmsd: over the central molecule and the matched neighbours
        inverted: whether the overlay went through an inversion
        rotation, translation: the overlay, A's cluster onto B's
        matches: (reference neighbour, candidate, rmsd) for each match
    """

    n_matched: int
    n_neighbours: int
    rmsd: float
    inverted: bool
    rotation: np.ndarray
    translation: np.ndarray
    matches: list = field(default_factory=list)

    @property
    def same_packing(self) -> bool:
        return self.n_matched == self.n_neighbours

    def __repr__(self) -> str:
        return (
            f"<PackingComparison {self.n_matched}/{self.n_neighbours} "
            f"rmsd={self.rmsd:.3f}>"
        )


def heavy_atoms_suffice(positions, numbers, off_line: float = 0.05) -> bool:
    """Whether a molecule's heavy atoms alone fix its orientation.

    Not when there are fewer than two of them, or when they lie on one line
    and a hydrogen lies off it. A molecule linear as a whole is the same
    either way, so it keeps its heavy atoms.
    """
    positions = np.asarray(positions, dtype=float)
    heavy = np.asarray(numbers) > 1
    if heavy.sum() < 2:
        return False
    a, b = positions[np.flatnonzero(heavy)[:2]]
    axis = (b - a) / np.linalg.norm(b - a)
    rel = positions - a
    off = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1) > off_line
    if (off & heavy).any():
        return True
    return not off.any()


def _rotation(h, allow_improper: bool) -> np.ndarray:
    """The orthogonal R maximising tr(R H), H the covariance sum source target^T.

    With `allow_improper`, a reflection is taken if it fits better -- except
    for a rank-deficient H (planar or collinear points), which both fit
    equally; then the proper one is returned, so the answer does not depend on
    the signs the SVD happens to choose.
    """
    u, s, vt = np.linalg.svd(h)
    rank = int(np.sum(s > 1e-6 * max(s[0], 1e-300)))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0 and (not allow_improper or rank < 3):
        vt[2] = -vt[2]
        r = vt.T @ u.T
    return r


def _kabsch(source, target, allow_improper: bool):
    "(R, t, rmsd) with target ~ source R^T + t"
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    r = _rotation((source - cs).T @ (target - ct), allow_improper)
    t = ct - r @ cs
    rmsd = np.sqrt(np.mean(np.sum((source @ r.T + t - target) ** 2, axis=1)))
    return r, t, float(rmsd)


@dataclass(eq=False)
class RigidBody:
    """The reference molecule: the atoms a comparison counts, about their
    centroid, and the molecule's own symmetry operations.

    Attributes:
        positions: (m, 3) counted atoms, centroid at the origin
        numbers: (m,) their atomic numbers
        heavy_only: whether hydrogens were left out
        operations: (n, 3, 3) the body's own orthogonal symmetry operations,
            the identity first; improper ones included
    """

    positions: np.ndarray
    numbers: np.ndarray
    heavy_only: bool
    operations: np.ndarray = field(default_factory=lambda: np.eye(3)[None])

    @classmethod
    def from_molecule(cls, positions, numbers, heavy_only=True, tolerance=0.05):
        positions = np.asarray(positions, dtype=float)
        numbers = np.asarray(numbers)
        heavy_only = heavy_only and heavy_atoms_suffice(positions, numbers)
        counted = numbers > 1 if heavy_only else np.ones(len(numbers), bool)
        body = cls(
            positions[counted] - positions[counted].mean(axis=0),
            numbers[counted],
            heavy_only,
        )
        body.operations = _self_symmetry(body.positions, body.numbers, tolerance)
        return body

    @property
    def m(self) -> int:
        return len(self.numbers)

    @property
    def gyration(self) -> np.ndarray:
        return self.positions.T @ self.positions

    def counted(self, numbers) -> np.ndarray:
        "Which of a molecule's atoms this body counts"
        numbers = np.asarray(numbers)
        return numbers > 1 if self.heavy_only else np.ones(len(numbers), bool)

    def fit(self, positions, numbers, max_iterations=8):
        """Place this body onto a molecule's atoms.

        Returns:
            (centroid, orientation, rmsd) with the molecule's counted atoms
            near `body.positions @ orientation.T + centroid`, or None if the
            molecule is not this one
        """
        keep = self.counted(numbers)
        q = np.asarray(positions, dtype=float)[keep]
        z = np.asarray(numbers)[keep]
        p = self.positions
        if len(z) != self.m or sorted(z) != sorted(self.numbers):
            return None
        forbidden = self.numbers[:, None] != z[None, :]

        def refine(assignment):
            for _ in range(max_iterations):
                r, t, _ = _kabsch(p, q[assignment], allow_improper=True)
                moved = p @ r.T + t
                cost = np.sum((moved[:, None] - q[None]) ** 2, axis=-1)
                cost[forbidden] = FORBIDDEN
                _, new = linear_sum_assignment(cost)
                if np.array_equal(new, assignment):
                    break
                assignment = new
            r, t, rmsd = _kabsch(p, q[assignment], allow_improper=True)
            return rmsd, r, t

        best = None

        def consider(assignment):
            nonlocal best
            result = refine(assignment)
            if best is None or result[0] < best[0]:
                best = result

        # seed: the sorted distances from each atom to the others
        cost = np.linalg.norm(
            _sorted_distances(p)[:, None] - _sorted_distances(q)[None], axis=-1
        )
        cost[forbidden] = FORBIDDEN
        consider(linear_sum_assignment(cost)[1])

        # seed: a well-spread triangle of the body against every congruent
        # triangle of the molecule, for a symmetric molecule whose distances
        # cannot tell its atoms apart
        if best[0] > 1e-8 and self.m >= 3:
            for triangle in _congruent_triangles(p, self.numbers, q, z, 0.25):
                anchor, image = triangle
                r, t, _ = _kabsch(p[anchor], q[image], allow_improper=True)
                cost = np.sum(((p @ r.T + t)[:, None] - q[None]) ** 2, axis=-1)
                cost[forbidden] = FORBIDDEN
                consider(linear_sum_assignment(cost)[1])
                if best[0] <= 1e-8:
                    break
        rmsd, r, t = best
        return t, r, rmsd


def _sorted_distances(points) -> np.ndarray:
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return np.sort(d, axis=1)[:, 1:]


def _anchor(points):
    "Atom 0, the atom furthest from it, and the atom furthest off that line"
    a1 = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    u = points[a1] - points[0]
    area = np.linalg.norm(np.cross(u, points - points[0]), axis=1)
    area[[0, a1]] = -1
    a2 = int(np.argmax(area))
    return (0, a1, a2) if area[a2] > 1e-6 else None


def _reference_triangle(points):
    "The atom furthest from the centroid, the furthest from it, and the best spread"
    i0 = int(np.argmax(np.linalg.norm(points, axis=1)))
    d = np.linalg.norm(points - points[i0], axis=1)
    d[i0] = -1
    i1 = int(np.argmax(d))
    area = np.linalg.norm(
        np.cross(points[i1] - points[i0], points - points[i0]), axis=1
    )
    area[[i0, i1]] = -1
    i2 = int(np.argmax(area))
    return (i0, i1, i2) if area[i2] > 1e-6 else None


def _congruent_triangles(p, pz, q, qz, tolerance):
    "The body's anchor triangle, with every congruent same-element triple of q"
    anchor = _anchor(p)
    if anchor is None:
        return
    i0, i1, i2 = anchor
    dp = np.linalg.norm(p[:, None] - p[None], axis=-1)
    dq = np.linalg.norm(q[:, None] - q[None], axis=-1)
    for i in np.flatnonzero(qz == pz[i0]):
        for j in np.flatnonzero(qz == pz[i1]):
            if j == i or abs(dq[i, j] - dp[i0, i1]) > tolerance:
                continue
            for k in np.flatnonzero(qz == pz[i2]):
                if k in (i, j) or abs(dq[i, k] - dp[i0, i2]) > tolerance:
                    continue
                if abs(dq[j, k] - dp[i1, i2]) > tolerance:
                    continue
                yield list(anchor), [i, j, k]


def _self_symmetry(p, numbers, tolerance) -> np.ndarray:
    """The orthogonal operations carrying every atom onto one of its element.

    Found from the inertia axes (a planar body's mirror, a linear body's) and
    from where a reference triangle can go, then closed under composition.
    """
    operations = [np.eye(3)]
    if len(p) < 2:
        return np.array(operations)
    gyration = p.T @ p
    radius = np.sqrt(np.trace(gyration) / len(p)) if np.trace(gyration) > 0 else 1.0
    limit = tolerance / max(radius, 1e-9)

    def is_symmetry(q):
        moved = p @ q.T
        d = np.linalg.norm(moved[:, None] - p[None], axis=-1)
        d[numbers[:, None] != numbers[None, :]] = np.inf
        return bool(np.all(d.min(axis=1) <= tolerance))

    def consider(q):
        if len(operations) >= 96:
            return
        if not np.allclose(q @ q.T, np.eye(3), atol=1e-6):
            return
        if any(np.linalg.norm(q - o) < limit for o in operations):
            return
        if is_symmetry(q):
            operations.append(q)

    def mirror(normal):
        return np.eye(3) - 2 * np.outer(normal, normal)

    # sums over the atoms of squared distances from the best plane and line
    values, vectors = np.linalg.eigh(gyration)
    values = np.maximum(values, 0.0)
    if np.sqrt(values[0]) < tolerance:  # planar: the mirror in its own plane
        consider(mirror(vectors[:, 0]))
    if np.sqrt(values[0] + values[1]) < tolerance:  # linear: mirrors with it, across it
        consider(mirror(vectors[:, 1]))
        consider(mirror(vectors[:, 2]))
        consider(-np.eye(3))

    anchor = _reference_triangle(p)
    if anchor is not None:
        i0, i1, i2 = anchor
        r = np.linalg.norm(p, axis=1)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        source = p[list(anchor)]
        u, v = source[1] - source[0], source[2] - source[0]
        normal = np.cross(u, v)
        reflect = mirror(normal / np.linalg.norm(normal))
        for j0 in np.flatnonzero(
            (numbers == numbers[i0]) & (abs(r - r[i0]) <= tolerance)
        ):
            for j1 in np.flatnonzero(numbers == numbers[i1]):
                if j1 == j0 or abs(d[j0, j1] - d[i0, i1]) > tolerance:
                    continue
                for j2 in np.flatnonzero(numbers == numbers[i2]):
                    if j2 in (j0, j1) or abs(d[j0, j2] - d[i0, i2]) > tolerance:
                        continue
                    if abs(d[j1, j2] - d[i1, i2]) > tolerance:
                        continue
                    q, _, _ = _kabsch(source, p[[j0, j1, j2]], allow_improper=True)
                    consider(q)
                    # three points never decide what happens across their own
                    # plane: the other map is through its reflection
                    consider(q @ reflect)

    for _ in range(4):
        n = len(operations)
        for a in list(operations):
            for b in list(operations):
                consider(a @ b)
        if len(operations) == n:
            break
    return np.array(operations)


def _molecule_cost(body, ca, ma, cb, mb):
    """(n_a, n_b) per-molecule RMSD between placements, and the body operation
    each pair used."""
    g = body.gyration
    qgt = np.einsum("oij,jk->oki", body.operations, g)  # (Q G)^T
    p = np.einsum("aji,bjk->abik", ma, mb)  # M_a^T M_b
    traces = np.einsum("abij,oij->abo", p, qgt)
    best = traces.argmax(axis=-1)
    orient = 2 * (np.trace(g) - traces.max(axis=-1)) / body.m
    dc2 = np.sum((ca[:, None] - cb[None]) ** 2, axis=-1)
    return np.sqrt(np.maximum(dc2 + np.maximum(orient, 0), 0)), best


def _compare_clusters(body, a: Cluster, b: Cluster, options) -> PackingComparison:
    """Overlay cluster a's references on cluster b's candidates."""
    kept = len(a.centroids) - 1
    if a.untied >= 0:
        n_ref = min(kept, MAX_REFERENCES)
        places = min(options.n_neighbours, n_ref)
        required = a.untied
    else:
        n_ref = places = required = min(options.n_neighbours, kept, MAX_REFERENCES)
    b = Cluster(b.centroids[: MAX_POOL + 1], b.orientations[: MAX_POOL + 1], b.untied)
    ca, ma = a.centroids, a.orientations
    cb, mb = b.centroids, b.orientations
    # Seeds lay central A on central B through each body operation Q. That
    # overlay is improper when the two were fitted with opposite handedness
    # or Q is a mirror, so with inversion allowed it is refitted both ways; its
    # negation (A inverted in place) is a further starting point.
    exact = [mb[0] @ q @ ma[0].T for q in body.operations]
    seeds = [(r, False) for r in exact]
    if options.allow_inversion:
        seeds += [(r, True) for r in exact if np.linalg.det(r) < 0]
        seeds += [(-r, True) for r in exact]
    best = None
    for r, improper in seeds:
        matched, rmsd, r, t = _overlay(
            body, a, b, n_ref, r, cb[0] - r @ ca[0], improper, options
        )
        # the nearer neighbours each count; the tied shell fills what is left
        nearer = sum(i < required for i, *_ in matched)
        score = nearer + min(len(matched) - nearer, places - required)
        if best is None or score > best.n_matched:
            best = PackingComparison(
                score,
                places,
                rmsd,
                bool(np.linalg.det(r) < 0),
                r,
                t,
                [(i, j, c) for i, j, c, _ in matched],
            )
        if best.same_packing:
            break
    return best


def _overlay(body, a, b, n_ref, r, t, improper, options):
    """From the transform (r, t), match a's first `n_ref` neighbours to b's and
    refit until the matched set stops changing.

    Returns:
        (matched, rmsd, r, t), matched as (reference, candidate, rmsd, operation)
    """
    ca, ma = a.centroids, a.orientations
    cb, mb = b.centroids, b.orientations
    ref_c, ref_m = ca[1 : n_ref + 1], ma[1 : n_ref + 1]
    pool_c, pool_m = cb[1:], mb[1:]
    g = body.gyration
    previous, matched, rmsd = None, [], 0.0
    for _ in range(options.max_iterations):
        cost, ops = _molecule_cost(body, ref_c @ r.T + t, r @ ref_m, pool_c, pool_m)
        rows, cols = linear_sum_assignment(cost)
        matched = [
            (int(i), int(j), float(cost[i, j]), int(ops[i, j]))
            for i, j in zip(rows, cols, strict=True)
            if cost[i, j] <= options.tolerance
        ]
        # refit over the centrals and every matched neighbour, from centroids
        # and orientations alone
        pa = [(ca[0], ma[0])] + [(ref_c[i], ref_m[i]) for i, *_ in matched]
        pb = [(cb[0], mb[0])] + [
            (pool_c[j], pool_m[j] @ body.operations[q]) for _, j, _, q in matched
        ]
        abar = np.mean([x for x, _ in pa], axis=0)
        bbar = np.mean([x for x, _ in pb], axis=0)
        h = sum(
            body.m * np.outer(xa - abar, xb - bbar) + m_a @ g @ m_b.T
            for (xa, m_a), (xb, m_b) in zip(pa, pb, strict=True)
        )
        r = _rotation(h, allow_improper=improper)
        t = bbar - r @ abar
        residual = sum(
            np.sum((r @ xa + t - xb) ** 2)
            + max(2 * (np.trace(g) - np.sum((r @ m_a) * (m_b @ g))) / body.m, 0)
            for (xa, m_a), (xb, m_b) in zip(pa, pb, strict=True)
        )
        rmsd = float(np.sqrt(residual / len(pa)))
        signature = [(i, j) for i, j, *_ in matched]
        if signature == previous:
            break
        previous = signature
    return matched, rmsd, r, t


class PackingComparer:
    """Compares crystals of one compound by their molecular packing.

    Args:
        reference: a `Crystal` whose first molecule every molecule is fitted to
        **options: `PackingOptions` fields
    """

    def __init__(self, reference, **options):
        self.options = PackingOptions(**options)
        molecule = _molecules(reference)[0]
        self.body = RigidBody.from_molecule(
            molecule.positions,
            molecule.atomic_numbers,
            heavy_only=self.options.heavy_only,
            tolerance=self.options.self_symmetry_tolerance,
        )

    def place(self, crystal) -> Packing:
        "Fit every molecule of a crystal's unit cell to the reference"
        centroids, orientations = [], []
        for k, molecule in enumerate(_molecules(crystal)):
            fit = self.body.fit(molecule.positions, molecule.atomic_numbers)
            if fit is None or fit[2] > self.options.max_fit_rmsd:
                raise ValueError(
                    f"molecule {k} of {crystal} is not the reference molecule"
                    + ("" if fit is None else f" (fits only to {fit[2]:.3f} A)")
                )
            centroids.append(fit[0])
            orientations.append(fit[1])
        packing = Packing(
            np.asarray(crystal.unit_cell.direct),
            np.array(centroids),
            np.array(orientations),
        )
        packing.representatives = self._representatives(packing)
        return packing

    def _representatives(self, packing) -> list[int]:
        "One molecule of each distinct environment"
        kept, pools = [], []
        options = self.options
        for i in range(len(packing.centroids)):
            cluster = packing.cluster(i, options.n_neighbours, options.radius)
            if any(
                _compare_clusters(self.body, cluster, pool, options).same_packing
                for pool in pools
            ):
                continue
            kept.append(i)
            pools.append(packing.cluster(i, options.pool_size, options.radius))
        return kept

    def compare(self, a, b) -> PackingComparison:
        """Overlay a's clusters on b's. `a` and `b` are `Crystal`s or `Packing`s
        from `place`."""
        a = a if isinstance(a, Packing) else self.place(a)
        b = b if isinstance(b, Packing) else self.place(b)
        options = self.options
        pool = b.cluster(0, options.pool_size, options.radius)
        best = None
        for centre in a.representatives:
            cluster = a.cluster(centre, options.n_neighbours, options.radius)
            result = _compare_clusters(self.body, cluster, pool, options)
            if (
                best is None
                or result.n_matched > best.n_matched
                or (result.n_matched == best.n_matched and result.rmsd < best.rmsd)
            ):
                best = result
            if best.same_packing:
                break
        return best

    def matrix(self, crystals) -> np.ndarray:
        """(n, n) matched neighbours between every pair, each compared once."""
        packings = [self.place(c) for c in crystals]
        n = len(packings)
        result = np.full((n, n), self.options.n_neighbours, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                result[i, j] = result[j, i] = self.compare(
                    packings[i], packings[j]
                ).n_matched
        return result


@dataclass(eq=False)
class Packing:
    """A crystal's molecules as placements of the reference body.

    Attributes:
        cell: (3, 3) lattice vectors as rows
        centroids: (Z, 3) Cartesian centroids of the counted atoms
        orientations: (Z, 3, 3) orthogonal body-to-crystal orientations
        representatives: one molecule of each distinct environment
    """

    cell: np.ndarray
    centroids: np.ndarray
    orientations: np.ndarray
    representatives: list = field(default_factory=lambda: [0])

    def cluster(self, central: int, n: int, radius: float) -> Cluster:
        """The central molecule and its `n` nearest others within `radius`,
        nearest first, with every further molecule tied with the nth."""
        spacing = 1 / np.linalg.norm(np.linalg.inv(self.cell), axis=0)
        reach = np.ceil(radius / spacing).astype(int)
        shifts = np.array(list(product(*(range(-r, r + 1) for r in reach))))
        images = self.centroids[None] + (shifts @ self.cell)[:, None]
        which = np.broadcast_to(np.arange(len(self.centroids)), images.shape[:2])
        home = np.all(shifts == 0, axis=1)[:, None] & (which == central)
        d2 = np.sum((images - self.centroids[central]) ** 2, axis=-1)
        keep = (d2 <= radius**2) & ~home
        order = np.argsort(d2[keep], kind="stable")
        distance = np.sqrt(d2[keep][order])
        n = min(n, len(order))
        kept, untied = n, n
        if n > 0:
            tied = np.abs(distance - distance[n - 1]) <= SHELL_TIE
            while kept < len(order) and tied[kept]:
                kept += 1
            if kept > n:
                while untied > 0 and tied[untied - 1]:
                    untied -= 1
        order = order[:kept]
        return Cluster(
            np.vstack([self.centroids[central][None], images[keep][order]]),
            np.concatenate(
                [
                    self.orientations[central][None],
                    self.orientations[which[keep][order]],
                ]
            ),
            untied if kept > n else -1,
        )


def _molecules(crystal):
    from chmpy.crystal.slab import is_molecular

    if not is_molecular(crystal):
        raise ValueError(f"{crystal} is not molecular: some bonded unit is infinite")
    return crystal.unit_cell_molecules()


def compare_packing(a, b, **options) -> PackingComparison:
    "Overlay crystal a's cluster on crystal b's, fitting molecules to a's first"
    return PackingComparer(a, **options).compare(a, b)


def packing_matrix(crystals, **options) -> np.ndarray:
    "(n, n) matched neighbours between every pair, fitted to the first's molecule"
    return PackingComparer(crystals[0], **options).matrix(crystals)

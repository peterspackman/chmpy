"""Continuous shape measures of coordination polyhedra.

How far a set of ligand positions is from an ideal polyhedron (Pinsky and
Avnir, Inorg. Chem. 37, 5575 (1998); Alvarez et al., Coord. Chem. Rev. 249,
1693 (2005)):

    S(Q, P) = 100 min over R, s, pi of sum_i |q_i - s R p_pi(i)|^2 / sum_i |q_i|^2

over rotations R, scales s and assignments pi of ligands to vertices, with the
ligands Q and the vertices P each centred on their own centroid. S is 0 for
the ideal shape and at most 100. Only the ligands count: the central atom is
not a vertex, so where it sits plays no part.

With the optimal scale the measure is `100 (1 - c^2)`, where `c` is the
largest correlation `sum q . R p / (|Q| |P|)` over R and pi. So S(Q, P) is
symmetric in Q and P, and for a fixed assignment the best rotation is Kabsch's,
one 3x3 SVD. Up to `EXACT_MAX` vertices every assignment is tried. Beyond, the
assignment is the linear assignment problem for a fixed rotation, so the
two are alternated to a local optimum from many random rotations.

The ideal shapes in `IDEAL_SHAPES` are centred and scaled to a mean vertex
distance of 1, and named as in SHAPE where they are the same shape. Johnson
solids (names starting with J) have all their edges equal.
"""

from __future__ import annotations

from itertools import islice, permutations

import numpy as np

#: largest vertex count measured exactly, by trying every assignment
EXACT_MAX = 8
#: random starting rotations for the heuristic beyond `EXACT_MAX`
STARTS = 500


def _normalise(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    p = p - p.mean(axis=0)
    return p / np.linalg.norm(p, axis=1).mean()


def _ring(n, z=0.0, radius=1.0, phase=0.0) -> np.ndarray:
    angles = 2 * np.pi * np.arange(n) / n + phase
    return np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)]
    )


def _ideal_shapes() -> dict[str, np.ndarray]:
    from scipy.spatial.transform import Rotation

    root3, golden = np.sqrt(3), (1 + np.sqrt(5)) / 2
    poles = [[0, 0, 1], [0, 0, -1]]
    # prism and antiprism with unit edges
    prism = np.vstack([_ring(3, 0.5, 1 / root3), _ring(3, -0.5, 1 / root3)])
    r = 1 / np.sqrt(2)
    h = 0.5 * np.sqrt(1 - (2 * r * np.sin(np.pi / 8)) ** 2)
    antiprism = np.vstack([_ring(4, h, r), _ring(4, -h, r, np.pi / 4)])
    # a unit-edge square pyramid's apex is 1/sqrt2 above the square
    cap = h + 1 / np.sqrt(2)
    cuboctahedron = [
        v for a in (1, -1) for b in (1, -1) for v in ([a, b, 0], [a, 0, b], [0, a, b])
    ]
    # turn one triangular cupola of the cuboctahedron by 60 degrees about the
    # three-fold axis it sits on
    axis = np.ones(3) / root3
    coc = np.asarray(cuboctahedron, dtype=float)
    top = coc @ axis > 0
    turned = Rotation.from_rotvec(axis * np.pi / 3).apply(coc[top])

    shapes = {
        "L-2": poles,
        "TP-3": _ring(3),
        "T-4": [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        "SP-4": _ring(4),
        "TBPY-5": np.vstack([_ring(3), poles]),
        # vacant octahedron: also the Johnson square pyramid, ligands only
        "vOC-5": np.vstack([_ring(4), [[0, 0, 1]]]),
        "OC-6": np.vstack([_ring(4), poles]),
        "TPR-6": prism,
        "PBPY-7": np.vstack([_ring(5), poles]),
        "CU-8": [[x, y, z] for x in (1, -1) for y in (1, -1) for z in (1, -1)],
        "SAPR-8": antiprism,
        # caps over the square faces, a unit-edge pyramid's height out
        "JTCTPR-9": np.vstack(
            [prism, _ring(3, 0.0, 1 / (2 * root3) + 1 / np.sqrt(2), np.pi / 3)]
        ),
        "JCSAPR-9": np.vstack([antiprism, [[0, 0, cap]]]),
        "JBCSAPR-10": np.vstack([antiprism, [[0, 0, cap], [0, 0, -cap]]]),
        "IC-12": [
            v
            for a in (1, -1)
            for b in (1, -1)
            for v in ([0, a, b * golden], [a, b * golden, 0], [b * golden, 0, a])
        ],
        "COC-12": cuboctahedron,
        "ACOC-12": np.vstack([turned, coc[~top]]),
    }
    return {name: _normalise(v) for name, v in shapes.items()}


#: ideal polyhedra: (n, 3) vertices, centred, mean vertex distance 1
IDEAL_SHAPES = _ideal_shapes()


def continuous_shape_measure(
    ligands, shape, *, exact_max: int = EXACT_MAX, starts: int = STARTS, seed=0
) -> float:
    """The continuous shape measure of ligands against an ideal polyhedron.

    Args:
        ligands: (n, 3) ligand positions, e.g. relative to the central atom;
            any origin, orientation and scale give the same answer
        shape: a name from `IDEAL_SHAPES`, or (n, 3) vertices
        exact_max: try every assignment up to this many vertices
        starts: random starting rotations for the heuristic beyond that
        seed: for the starting rotations

    Returns:
        S, from 0 (the ideal shape) to 100
    """
    q, p = _centred(ligands, shape)
    if len(q) <= exact_max:
        best = _best_exact(q, p)
    else:
        best = _best_heuristic(q, p, starts, seed)
    c = best / (np.linalg.norm(q) * np.linalg.norm(p))
    return float(100 * (1 - min(c, 1.0) ** 2))


def shape_measures(ligands, **kwargs) -> dict[str, float]:
    """The measure against every ideal shape with as many vertices, best first.

    Args:
        ligands: (n, 3) ligand positions
        **kwargs: passed to `continuous_shape_measure`
    """
    n = len(ligands)
    measures = {
        name: continuous_shape_measure(ligands, vertices, **kwargs)
        for name, vertices in IDEAL_SHAPES.items()
        if len(vertices) == n
    }
    if not measures:
        raise ValueError(f"no ideal shape has {n} vertices")
    return dict(sorted(measures.items(), key=lambda item: item[1]))


def _centred(ligands, shape):
    q = np.asarray(ligands, dtype=float)
    if isinstance(shape, str):
        if shape not in IDEAL_SHAPES:
            raise KeyError(f"unknown shape {shape!r}; one of {list(IDEAL_SHAPES)}")
        shape = IDEAL_SHAPES[shape]
    p = np.asarray(shape, dtype=float)
    if q.shape != p.shape or q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"{q.shape} ligands against a {p.shape} shape")
    return q - q.mean(axis=0), p - p.mean(axis=0)


def _best_trace(h) -> np.ndarray:
    "max over proper rotations R of tr(R H), for (..., 3, 3) H"
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    return s[..., 0] + s[..., 1] + d * s[..., 2]


def _best_exact(q, p, chunk: int = 50_000) -> float:
    "The best correlation sum over every assignment of vertices to ligands"
    n = len(q)
    every = permutations(range(n))
    best = -np.inf
    while block := list(islice(every, chunk)):
        assignments = np.array(block)
        h = np.einsum("mni,nj->mij", p[assignments], q)
        best = max(best, float(_best_trace(h).max()))
    return best


def _best_heuristic(q, p, starts, seed) -> float:
    "Assignment and rotation alternated to a local optimum, from many rotations"
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.transform import Rotation

    rotations = Rotation.random(starts, random_state=seed).as_matrix()
    rotations[0] = np.eye(3)
    best = -np.inf
    for rotation in rotations:
        assignment = None
        for _ in range(100):
            _, new = linear_sum_assignment(-(q @ (p @ rotation.T).T))
            if assignment is not None and np.array_equal(new, assignment):
                break
            assignment = new
            u, _, vt = np.linalg.svd(p[assignment].T @ q)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rotation = vt.T @ np.diag([1, 1, d]) @ u.T
        best = max(best, float(_best_trace(p[assignment].T @ q)))
    return best

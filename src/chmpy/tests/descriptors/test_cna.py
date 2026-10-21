import itertools

import numpy as np
import pytest

from chmpy.calc.nearest import nearest_neighbors
from chmpy.descriptors.cna import (
    PATTERNS,
    StructureType,
    adaptive_cna,
    classify,
    cna_signatures,
    fixed_cna,
    interval_cna,
    neighbor_bonds,
)

FCC, HCP, BCC, ICO, OTHER = (
    StructureType.FCC,
    StructureType.HCP,
    StructureType.BCC,
    StructureType.ICO,
    StructureType.OTHER,
)
METHODS = [adaptive_cna, interval_cna]


def lattice(cell, basis, reps):
    grid = np.stack(np.meshgrid(*map(np.arange, reps), indexing="ij"), -1)
    fractional = (grid.reshape(-1, 1, 3) + np.asarray(basis)).reshape(-1, 3)
    return fractional @ cell, cell * np.asarray(reps)[:, None]


def fcc(reps=(4, 4, 4), a=3.6):
    basis = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    return lattice(np.eye(3) * a, basis, reps)


def bcc(reps=(5, 5, 5), a=2.87):
    return lattice(np.eye(3) * a, [[0, 0, 0], [0.5, 0.5, 0.5]], reps)


def hcp(reps=(5, 5, 3), a=2.5):
    c = a * np.sqrt(8 / 3)
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    return lattice(cell, [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]], reps)


def icosahedron(d=2.5):
    """A centred 13-atom icosahedron with edges of length `d`."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            vertices += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]]
    vertices = np.array(vertices) * d / 2
    return np.vstack([np.zeros(3), vertices])


def jitter(positions, scale, seed=0):
    rng = np.random.default_rng(seed)
    return positions + rng.normal(scale=scale, size=positions.shape)


def stacked(sequence, a=2.5, reps=(4, 3)):
    """Close-packed layers stacked in the given ABC sequence."""
    offsets = {"A": 0, "B": 1, "C": 2}
    layer = np.array(
        [[i * a, j * a * np.sqrt(3)] for i in range(reps[0]) for j in range(reps[1])]
    )
    layer = np.vstack([layer, layer + [a / 2, a * np.sqrt(3) / 2]])
    height = a * np.sqrt(2 / 3)
    shift = np.array([a / 2, a * np.sqrt(3) / 6])
    positions = [
        np.c_[layer + offsets[s] * shift, np.full(len(layer), z * height)]
        for z, s in enumerate(sequence)
    ]
    cell = np.diag([reps[0] * a, reps[1] * a * np.sqrt(3), len(sequence) * height])
    return np.vstack(positions), cell


def bonds_from_edges(k, edges):
    """Bitmasks for a neighbour shell with the given bonds among neighbours."""
    rows = np.zeros(k, dtype=np.uint32)
    for a, b in edges:
        rows[a] |= 1 << b
        rows[b] |= 1 << a
    return rows[None, :]


# -- structures ---------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "build,expected",
    [(fcc, FCC), (bcc, BCC), (hcp, HCP)],
)
def test_perfect_crystals(method, build, expected):
    positions, cell = build()
    assert np.all(method(positions, cell) == expected)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "cell,expected",
    [
        (3.6 / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float), FCC),
        (2.87 / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float), BCC),
    ],
)
def test_one_atom_primitive_cells(method, cell, expected):
    assert method(np.zeros((1, 3)), cell)[0] == expected


@pytest.mark.parametrize("method", METHODS)
def test_icosahedral_centre(method):
    types = method(icosahedron(), None)
    assert types[0] == ICO
    assert np.all(types[1:] == OTHER)


@pytest.mark.parametrize("method", METHODS)
def test_open_structures_are_other(method):
    diamond = lattice(
        np.eye(3) * 3.57,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        + [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        (3, 3, 3),
    )
    simple_cubic = lattice(np.eye(3) * 2.6, [[0, 0, 0]], (4, 4, 4))
    for positions, cell in (diamond, simple_cubic):
        assert np.all(method(positions, cell) == OTHER)


@pytest.mark.parametrize("method", METHODS)
def test_stacking_fault_layers_are_hcp(method):
    """An intrinsic stacking fault leaves two close-packed layers in HCP order."""
    positions, cell = stacked("ABCABCACABCABC")
    types = method(positions, cell)
    per_layer = len(positions) // 14
    layer_types = types.reshape(14, per_layer)
    assert np.all(layer_types == layer_types[:, :1])
    assert np.sum(layer_types[:, 0] == HCP) == 2
    assert np.sum(layer_types[:, 0] == FCC) == 12


@pytest.mark.parametrize("method", METHODS)
def test_free_surface_atoms_are_other(method):
    positions, cell = fcc((4, 4, 4))
    types = method(positions, cell, [True, True, False])
    z = positions[:, 2]
    surface = (z < 1e-6) | (z > z.max() - 1e-6)
    assert np.all(types[surface] == OTHER)
    assert np.all(types[(z > 2.0) & (z < z.max() - 2.0)] == FCC)


@pytest.mark.parametrize("method", METHODS)
def test_invariant_to_atom_order_and_rigid_motion(method):
    positions, cell = hcp((5, 5, 4))
    positions = jitter(positions, 0.12)
    rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    perm = np.random.default_rng(2).permutation(len(positions))
    moved = (positions[perm] + 7.3) @ rotation.T
    np.testing.assert_array_equal(
        method(moved, cell @ rotation.T), method(positions, cell)[perm]
    )


def test_interval_tolerates_more_disorder_than_adaptive():
    positions, cell = fcc((6, 6, 6))
    positions = jitter(positions, 0.06 * 3.6 / np.sqrt(2))
    adaptive = np.mean(adaptive_cna(positions, cell) == FCC)
    interval = np.mean(interval_cna(positions, cell) == FCC)
    assert interval > adaptive


def test_interval_ignores_rounding_error_between_equal_bonds():
    """A BCC environment missing one second neighbour has bonds of equal
    length on either side of where an FCC interval would be. Rounding error
    must not open that interval for some orientations and not others."""
    a = 2.87
    first = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))
    second = np.array([[-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    environment = np.vstack([np.zeros(3), first, second, [1, 1, 0]]) * a
    types = set()
    for seed in range(60):
        rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        types.add(interval_cna(environment @ rotation.T, None)[0])
    assert types == {OTHER}


def test_fixed_cutoff():
    positions, cell = fcc()
    nearest = 3.6 / np.sqrt(2)
    assert np.all(fixed_cna(positions, 1.2 * nearest, cell) == FCC)
    assert np.all(fixed_cna(positions, 0.9 * nearest, cell) == OTHER)
    positions, cell = bcc()
    assert np.all(fixed_cna(positions, 1.2 * 2.87, cell) == BCC)


@pytest.mark.parametrize("method", [adaptive_cna, interval_cna])
def test_reuses_a_neighbor_search(method):
    positions, cell = bcc()
    neighbors = nearest_neighbors(positions, 15, cell)
    np.testing.assert_array_equal(
        method(None, neighbors=neighbors), method(positions, cell)
    )
    with pytest.raises(ValueError):
        method(None, neighbors=nearest_neighbors(positions, 12, cell))


def test_threads_do_not_change_the_answer(monkeypatch):
    from chmpy.descriptors import cna

    monkeypatch.setattr(cna, "CHUNK", 64)
    positions, cell = fcc((5, 5, 5))
    positions = jitter(positions, 0.15)
    for method in METHODS:
        np.testing.assert_array_equal(
            method(positions, cell, workers=1), method(positions, cell, workers=4)
        )


# -- signatures ---------------------------------------------------------------


def signature_counts(signatures):
    rows = [tuple(s) for s in signatures.reshape(-1, 3)]
    return {s: rows.count(s) for s in set(rows)}


@pytest.mark.parametrize(
    "build,shell,cutoff,expected",
    [
        (fcc, 12, 3.0, {(4, 2, 1): 12}),
        (hcp, 12, 3.0, {(4, 2, 1): 6, (4, 2, 2): 6}),
        (bcc, 14, 3.3, {(6, 6, 6): 8, (4, 4, 4): 6}),
    ],
)
def test_signatures_of_perfect_crystals(build, shell, cutoff, expected):
    positions, cell = build((3, 3, 3))
    neighbors = nearest_neighbors(positions, shell, cell)
    bonds = neighbor_bonds(neighbors.vectors[:1], cutoff)
    assert signature_counts(cna_signatures(bonds)) == expected


@pytest.mark.parametrize(
    "edges,expected",
    [
        # two separate triangles, or one hexagon, among six common neighbours
        ([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)], (6, 6, 3)),
        ([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)], (6, 6, 6)),
        # a triangle and a separate bond, which needs a cluster search
        ([(1, 2), (2, 3), (1, 3), (4, 5)], (5, 4, 3)),
        # a path of four bonds through five neighbours
        ([(1, 2), (2, 3), (3, 4), (4, 5)], (5, 4, 4)),
        ([(1, 2), (3, 4)], (4, 2, 1)),
        ([(1, 2), (2, 3)], (4, 2, 2)),
    ],
)
def test_signature_of_one_bond(edges, expected):
    """Neighbour 0 shares every other neighbour; the bonds among those decide
    its signature."""
    common = sorted({x for edge in edges for x in edge} | {1, 2, 3, 4})
    bonds = bonds_from_edges(8, edges + [(0, c) for c in common])
    assert tuple(cna_signatures(bonds)[0, 0]) == expected


def test_classify_rejects_inconsistent_patterns():
    bonds = np.zeros((1, 12), dtype=np.uint32)
    with pytest.raises(ValueError):
        classify(bonds, PATTERNS[14])

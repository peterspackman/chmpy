import numpy as np
import pytest

from chmpy.calc import NeighborList, System
from chmpy.calc._celllist import cell_list_pairs


def simple_cubic(a=3.0, n=1):
    positions = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)], dtype=float
    )
    return System([18] * len(positions), positions * a, np.eye(3) * a * n, True)


def random_crystal(n_atoms, length, seed=0, displaced=False):
    rng = np.random.default_rng(seed)
    cell = np.array(
        [
            [length, 0, 0],
            [0.3 * length, 0.95 * length, 0],
            [0.2 * length, -0.1 * length, 0.8 * length],
        ]
    )
    fractional = rng.random((n_atoms, 3))
    if displaced:
        fractional += rng.integers(-2, 3, size=(n_atoms, 3))
    return System(rng.integers(1, 20, n_atoms), fractional @ cell, cell, True)


def canonical(i, j, shifts):
    """A pair list as a comparable set; either representative is valid."""
    i, j, shifts = np.asarray(i), np.asarray(j), np.asarray(shifts, dtype=int)
    total = shifts.sum(axis=1)
    flip = (i > j) | (
        (i == j)
        & (
            (total < 0)
            | (
                (total == 0)
                & ((shifts[:, 2] < 0) | ((shifts[:, 2] == 0) & (shifts[:, 1] < 0)))
            )
        )
    )
    key = np.c_[
        np.where(flip, j, i),
        np.where(flip, i, j),
        np.where(flip[:, None], -shifts, shifts),
    ]
    return {tuple(row) for row in key}


def test_simple_cubic_coordination():
    """Six neighbours at the lattice parameter, three of them in a half list."""
    system = simple_cubic(3.0)
    half = NeighborList(3.1).compute(system)
    assert len(half) == 3
    np.testing.assert_allclose(half.distances, 3.0)

    full = NeighborList(3.1, full=True).compute(system)
    assert len(full) == 6


def test_vectors_point_from_i_to_j():
    system = System([1, 1], [[0, 0, 0], [1.5, 0, 0]], np.eye(3) * 10.0, True)
    pairs = NeighborList(2.0).compute(system)
    assert len(pairs) == 1
    expected = system.positions[pairs.j[0]] - system.positions[pairs.i[0]]
    np.testing.assert_allclose(
        pairs.vectors[0], expected + pairs.shifts[0] @ system.cell
    )
    np.testing.assert_allclose(pairs.distances[0], 1.5)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(
    "system,cutoff",
    [
        (random_crystal(40, 10.0), 4.0),
        (random_crystal(20, 5.0, seed=1), 9.0),  # cutoff larger than the cell
        (random_crystal(60, 12.0, seed=2, displaced=True), 5.5),
        (random_crystal(30, 7.0, seed=3), 7.2),
    ],
)
def test_cell_list_matches_vesin(system, cutoff, full):
    vesin = pytest.importorskip("vesin")
    box = np.asarray(system.cell)
    reference = vesin.NeighborList(cutoff=cutoff, full_list=full).compute(
        points=np.asarray(system.positions), box=box, periodic=True, quantities="ijS"
    )
    ours = cell_list_pairs(system.positions, system.cell, system.pbc, cutoff, full)
    assert canonical(*ours) == canonical(*reference)


def test_cell_list_matches_vesin_for_an_isolated_system():
    vesin = pytest.importorskip("vesin")
    rng = np.random.default_rng(7)
    system = System(rng.integers(1, 10, 80), rng.random((80, 3)) * 12.0)
    reference = vesin.NeighborList(cutoff=5.0, full_list=False).compute(
        points=np.asarray(system.positions),
        box=np.zeros((3, 3)),
        periodic=False,
        quantities="ijS",
    )
    ours = cell_list_pairs(system.positions, system.cell, system.pbc, 5.0)
    assert canonical(*ours) == canonical(*reference)


def test_self_pairs_can_be_excluded():
    system = simple_cubic(3.0)
    assert len(NeighborList(3.1, self_pairs=True).compute(system)) == 3
    assert len(NeighborList(3.1, self_pairs=False).compute(system)) == 0


def test_skin_reuses_the_pair_list_without_changing_the_answer():
    system = random_crystal(50, 11.0, seed=5)
    plain = NeighborList(5.0, skin=0.0)
    skinned = NeighborList(5.0, skin=1.0)

    rng = np.random.default_rng(0)
    for _ in range(6):
        system.set_positions(system.positions + rng.normal(scale=0.02, size=(50, 3)))
        assert canonical(
            plain.compute(system).i,
            plain.compute(system).j,
            plain.compute(system).shifts,
        ) == canonical(
            skinned.compute(system).i,
            skinned.compute(system).j,
            skinned.compute(system).shifts,
        )
    assert skinned.reuses > 0
    assert skinned.rebuilds < plain.rebuilds


def test_repeated_queries_of_one_geometry_are_cached():
    system = simple_cubic(3.0)
    neighbors = NeighborList(3.1)
    first = neighbors.compute(system)
    assert neighbors.compute(system) is first
    system.set_positions(system.positions)
    assert neighbors.compute(system) is not first


def test_the_skin_survives_a_slowly_changing_cell():
    """Variable-cell relaxation moves the cell every step; the list should hold."""
    system = random_crystal(40, 10.0, seed=9)
    plain = NeighborList(4.5, skin=0.0)
    skinned = NeighborList(4.5, skin=0.6)

    for step in range(6):
        scale = 1.0 + 0.001 * (step + 1)
        system.set_cell(np.asarray(system.cell) * scale, scale_atoms=True)
        reference = plain.compute(system)
        ours = skinned.compute(system)
        assert canonical(reference.i, reference.j, reference.shifts) == canonical(
            ours.i, ours.j, ours.shifts
        )
    assert skinned.reuses > 0


def test_a_large_cell_change_forces_a_rebuild():
    system = random_crystal(40, 10.0, seed=10)
    skinned = NeighborList(4.5, skin=0.3)
    skinned.compute(system)
    rebuilds = skinned.rebuilds
    system.set_cell(np.asarray(system.cell) * 1.2, scale_atoms=True)
    skinned.compute(system)
    assert skinned.rebuilds == rebuilds + 1

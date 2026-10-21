import numpy as np
import pytest

from chmpy.calc.nearest import nearest_neighbors

FCC_PRIMITIVE = 3.6 / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


def random_crystal(n_atoms, length, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.array(
        [
            [length, 0, 0],
            [0.3 * length, 0.95 * length, 0],
            [0.2 * length, -0.1 * length, 0.8 * length],
        ]
    )
    # some atoms sit outside the cell, as they do mid-simulation
    fractional = rng.random((n_atoms, 3)) + rng.integers(-2, 3, size=(n_atoms, 3))
    return fractional @ cell, cell


def brute_force(positions, cell, pbc, k, reach=6):
    """Every image within `reach` cells along the periodic directions."""
    ranges = [np.arange(-reach, reach + 1) if p else [0] for p in pbc]
    shifts = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
    images = positions[None, :, :] + (shifts @ cell)[:, None, :]
    distances = []
    for i, r in enumerate(positions):
        d = np.linalg.norm(images - r, axis=-1)
        d[shifts.tolist().index([0, 0, 0]), i] = np.inf
        distances.append(np.sort(d.ravel())[:k])
    return np.array(distances)


def test_one_atom_cell_finds_its_own_images():
    """The 12 nearest images in FCC are a/sqrt(2) away, then 6 at a."""
    result = nearest_neighbors(np.zeros((1, 3)), 14, FCC_PRIMITIVE)
    np.testing.assert_allclose(result.distances[0, :12], 3.6 / np.sqrt(2))
    np.testing.assert_allclose(result.distances[0, 12:], 3.6)
    assert np.all(result.indices == 0)


@pytest.mark.parametrize("k", [1, 6, 20])
def test_periodic_matches_brute_force(k):
    positions, cell = random_crystal(30, 6.0)
    result = nearest_neighbors(positions, k, cell)
    expected = brute_force(positions, cell, [True] * 3, k)
    np.testing.assert_allclose(result.distances, expected)


def test_vectors_point_from_atom_to_neighbor_image():
    positions, cell = random_crystal(30, 6.0, seed=1)
    result = nearest_neighbors(positions, 8, cell)
    np.testing.assert_allclose(
        np.linalg.norm(result.vectors, axis=-1), result.distances
    )
    # each vector is a lattice translation away from r_j - r_i
    direct = positions[result.indices] - positions[:, None, :]
    shift = (result.vectors - direct) @ np.linalg.inv(cell)
    np.testing.assert_allclose(shift, np.round(shift), atol=1e-9)


def test_mixed_periodicity_matches_brute_force():
    positions, cell = random_crystal(40, 7.0, seed=2)
    pbc = [True, False, True]
    result = nearest_neighbors(positions, 10, cell, pbc)
    expected = brute_force(positions, cell, pbc, 10)
    np.testing.assert_allclose(result.distances, expected)


def test_zero_row_along_non_periodic_direction():
    positions, cell = random_crystal(40, 7.0, seed=3)
    cell[2] = 0.0
    result = nearest_neighbors(positions, 10, cell, [True, True, False])
    expected = brute_force(positions, cell, [True, True, False], 10)
    np.testing.assert_allclose(result.distances, expected)


def test_isolated_atoms_with_too_few_neighbors():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
    result = nearest_neighbors(positions, 3)
    np.testing.assert_allclose(result.distances[0], [1.0, 2.0, np.inf])
    assert result.indices[0, 2] == -1
    assert np.all(np.isnan(result.vectors[:, 2]))


def test_input_order_does_not_matter():
    positions, cell = random_crystal(50, 7.0, seed=4)
    perm = np.random.default_rng(5).permutation(len(positions))
    first = nearest_neighbors(positions, 12, cell)
    second = nearest_neighbors(positions[perm], 12, cell)
    np.testing.assert_allclose(second.distances, first.distances[perm])


def test_sparse_cell_grows_the_padding():
    """A density estimate far too low for a clustered cell still gives an
    exact answer."""
    rng = np.random.default_rng(6)
    positions = rng.random((20, 3))
    cell = np.eye(3) * 30.0
    result = nearest_neighbors(positions, 5, cell)
    expected = brute_force(positions, cell, [True] * 3, 5, reach=1)
    np.testing.assert_allclose(result.distances, expected)


def test_coincident_atoms_drop_only_themselves():
    positions = np.array([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    result = nearest_neighbors(positions, 2)
    assert sorted(result.indices[0]) == [1, 2]
    assert sorted(result.indices[1]) == [0, 2]


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        nearest_neighbors(np.zeros((2, 3)), 0)
    with pytest.raises(ValueError):
        nearest_neighbors(np.zeros((2, 3)), 1, pbc=True)

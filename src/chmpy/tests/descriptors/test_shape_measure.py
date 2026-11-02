import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chmpy.descriptors.shape_measure import (
    IDEAL_SHAPES,
    continuous_shape_measure,
    shape_measures,
)

#: Johnson and uniform polyhedra, by their number of (equal) edges
EDGES = {
    "TPR-6": 9,
    "SAPR-8": 16,
    "JTCTPR-9": 21,
    "JCSAPR-9": 20,
    "JBCSAPR-10": 24,
    "IC-12": 30,
    "COC-12": 24,
    "ACOC-12": 24,
}


def distorted(shape, sigma, seed):
    rng = np.random.default_rng(seed)
    vertices = IDEAL_SHAPES[shape]
    return vertices + rng.normal(scale=sigma, size=vertices.shape)


@pytest.mark.parametrize("name", IDEAL_SHAPES)
def test_an_ideal_shape_measures_zero_against_itself(name):
    vertices = IDEAL_SHAPES[name]
    assert continuous_shape_measure(vertices, name) == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(vertices.mean(axis=0), 0, atol=1e-12)
    assert np.linalg.norm(vertices, axis=1).mean() == pytest.approx(1)


@pytest.mark.parametrize("name", EDGES)
def test_polyhedra_have_all_their_edges_equal(name):
    vertices = IDEAL_SHAPES[name]
    distances = np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)
    pairs = distances[np.triu_indices(len(vertices), 1)]
    assert np.isclose(pairs, pairs.min()).sum() == EDGES[name]


def test_the_anticuboctahedron_is_not_the_cuboctahedron():
    assert continuous_shape_measure(IDEAL_SHAPES["COC-12"], "ACOC-12") > 1


@pytest.mark.parametrize(
    "a, b, expected",
    [("OC-6", "TPR-6", 16.737), ("T-4", "SP-4", 33.333)],
)
def test_published_values(a, b, expected):
    assert continuous_shape_measure(IDEAL_SHAPES[a], b) == pytest.approx(
        expected, abs=1e-3
    )


def test_the_measure_is_symmetric():
    a, b = IDEAL_SHAPES["CU-8"], IDEAL_SHAPES["SAPR-8"]
    assert continuous_shape_measure(a, b) == pytest.approx(
        continuous_shape_measure(b, a)
    )


def test_rotation_scale_origin_and_order_do_not_matter():
    ligands = distorted("OC-6", 0.1, seed=3)
    moved = 2.3 * Rotation.random(random_state=4).apply(ligands) + [1.0, -2.0, 0.5]
    moved = moved[np.random.default_rng(5).permutation(len(moved))]
    assert continuous_shape_measure(moved, "OC-6") == pytest.approx(
        continuous_shape_measure(ligands, "OC-6"), abs=1e-10
    )


@pytest.mark.parametrize(
    "shape, other, seeds", [("CU-8", "SAPR-8", 3), ("JTCTPR-9", "JCSAPR-9", 1)]
)
def test_the_heuristic_finds_the_exact_minimum(shape, other, seeds):
    for seed in range(seeds):
        ligands = distorted(shape, 0.15, seed)
        for reference in (shape, other):
            exact = continuous_shape_measure(ligands, reference, exact_max=9)
            heuristic = continuous_shape_measure(ligands, reference, exact_max=0)
            assert heuristic == pytest.approx(exact, abs=1e-8)


def test_the_closest_shape_comes_first():
    measures = shape_measures(distorted("TPR-6", 0.05, seed=1))
    assert list(measures) == ["TPR-6", "OC-6"]
    assert measures["TPR-6"] < measures["OC-6"]


def test_a_mismatched_count_is_refused():
    with pytest.raises(ValueError):
        continuous_shape_measure(IDEAL_SHAPES["OC-6"], "CU-8")
    with pytest.raises(ValueError, match="no ideal shape"):
        shape_measures(np.zeros((11, 3)))

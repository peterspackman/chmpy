import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chmpy import Crystal
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.crystal.packing_similarity import (
    Cluster,
    Packing,
    PackingComparer,
    PackingOptions,
    RigidBody,
    _compare_clusters,
    _kabsch,
    _molecule_cost,
    compare_packing,
    heavy_atoms_suffice,
)

from .. import TEST_FILES


@pytest.fixture(scope="module")
def acetic():
    return Crystal.load(TEST_FILES["acetic_acid.cif"])


def p1(cell, cartesian, numbers):
    cell = UnitCell(np.asarray(cell, dtype=float))
    asym = AsymmetricUnit.from_records(
        [
            {"label": f"X{k}", "element": int(z), "position": x}
            for k, (z, x) in enumerate(
                zip(numbers, cell.to_fractional(cartesian) % 1, strict=True)
            )
        ]
    )
    return Crystal(cell, SpaceGroup(1), asym)


def resettled(crystal, transform):
    "The same crystal in another cell setting: new lattice rows = transform @ old"
    atoms = crystal.unit_cell_atoms()
    cell = np.asarray(transform) @ crystal.unit_cell.direct
    return p1(cell, atoms["cart_pos"], atoms["element"])


@pytest.mark.parametrize(
    "positions, numbers, expected",
    [
        ([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], [8, 1, 1], False),  # water
        (
            [[0, 0, 0], [1.2, 0, 0], [-0.55, 0.94, 0], [-0.55, -0.94, 0]],
            [8, 6, 1, 1],
            False,
        ),
        ([[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]], [8, 6, 8], True),  # CO2
        ([[0, 0, 0], [1.5, 0, 0], [2.1, 1.1, 0], [2.1, -1.1, 0]], [6, 6, 8, 8], True),
    ],
)
def test_when_heavy_atoms_suffice(positions, numbers, expected):
    assert heavy_atoms_suffice(positions, numbers) is expected


def test_benzene_has_its_full_symmetry():
    ring = 1.39 * np.column_stack(
        [
            np.cos(np.arange(6) * np.pi / 3),
            np.sin(np.arange(6) * np.pi / 3),
            np.zeros(6),
        ]
    )
    body = RigidBody.from_molecule(ring, [6] * 6)
    assert len(body.operations) == 24


def test_a_nearly_planar_body_has_only_orthogonal_operations():
    # ethyl carbamate's heavy atoms, 0.02 A from their best plane
    atoms = np.array(
        [
            [0.0, 0.0, 0.02],
            [1.21, 0.0, -0.02],
            [-0.68, 1.15, 0.0],
            [-0.68, -1.17, 0.01],
            [-2.10, -1.20, -0.01],
            [-2.62, -2.62, 0.0],
        ]
    )
    body = RigidBody.from_molecule(atoms, [6, 8, 7, 8, 6, 6])
    for q in body.operations:
        np.testing.assert_allclose(q @ q.T, np.eye(3), atol=1e-9)


def test_a_slightly_bent_linear_body_gets_the_linear_operations():
    co2 = np.array([[-1.16, 0.0, 0.0], [0.0, 0.01, 0.0], [1.16, 0.0, 0.0]])
    body = RigidBody.from_molecule(co2, [8, 6, 8], heavy_only=True)
    assert len(body.operations) == 8


def test_collinear_points_fit_along_their_axis():
    rng = np.random.default_rng(7)
    line = np.array([[-1.2], [0.0], [1.2]]) * [1.0, 0.0, 0.0]
    axis = Rotation.random(random_state=8).as_matrix()
    target = line @ axis.T + rng.normal(size=3)
    _, _, rmsd = _kabsch(line, target, allow_improper=False)
    assert rmsd < 1e-10


def test_a_moved_mirrored_relabelled_copy_fits_exactly():
    rng = np.random.default_rng(3)
    atoms = rng.normal(size=(7, 3))
    numbers = np.array([6, 6, 7, 8, 8, 6, 9])
    body = RigidBody.from_molecule(atoms, numbers)
    rotation = Rotation.random(random_state=4).as_matrix() @ np.diag([1, 1, -1])
    order = rng.permutation(7)
    copy = (atoms @ rotation.T + [3.0, -1.0, 2.0])[order]
    centroid, orientation, rmsd = body.fit(copy, numbers[order])
    assert rmsd < 1e-8
    assert np.linalg.det(orientation) == pytest.approx(-1)


def test_the_closed_form_is_the_rmsd_of_the_atoms():
    rng = np.random.default_rng(5)
    body = RigidBody.from_molecule(rng.normal(size=(6, 3)), [6, 7, 8, 9, 16, 17])
    assert len(body.operations) == 1
    c = rng.normal(size=(2, 3))
    m = (
        Rotation.random(2, random_state=6).as_matrix()
        * np.array([1, -1])[:, None, None]
    )
    cost, _ = _molecule_cost(body, c[:1], m[:1], c[1:], m[1:])
    a = body.positions @ m[0].T + c[0]
    b = body.positions @ m[1].T + c[1]
    assert cost[0, 0] == pytest.approx(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_a_crystal_packs_like_itself(acetic):
    assert compare_packing(acetic, acetic).same_packing


@pytest.mark.parametrize(
    "transform", [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[1, 0, 0], [0, 1, 0], [1, 0, 1]]]
)
def test_the_cell_setting_does_not_matter(acetic, transform):
    comparer = PackingComparer(acetic)
    result = comparer.compare(acetic, resettled(acetic, transform))
    assert result.same_packing
    assert result.rmsd < 1e-6


def test_turned_molecules_pack_differently(acetic):
    atoms = acetic.unit_cell_atoms()
    turned = atoms["cart_pos"].copy()
    for molecule in acetic.unit_cell_molecules():
        index = np.asarray(molecule.properties["unit_cell_atoms"])
        centre = molecule.positions.mean(axis=0)
        spin = Rotation.from_rotvec([0, 0, np.pi / 3]).as_matrix()
        turned[index] = (molecule.positions - centre) @ spin.T + centre
    other = p1(acetic.unit_cell.direct, turned, atoms["element"])
    assert compare_packing(acetic, other).n_matched < 15


def test_a_framework_is_refused():
    cell = UnitCell.cubic(5.64)
    asym = AsymmetricUnit.from_records(
        [
            {"label": "Na", "element": "Na", "position": (0, 0, 0)},
            {"label": "Cl", "element": "Cl", "position": (0.5, 0.5, 0.5)},
        ]
    )
    salt = Crystal(cell, SpaceGroup(225), asym)
    with pytest.raises(ValueError, match="not molecular"):
        compare_packing(salt, salt)


@pytest.fixture(scope="module")
def chiral():
    "A chiral CHFClBr-like molecule in P2_1, whose mirror image is another crystal"
    cell = UnitCell.from_lengths_and_angles(
        [6.5, 7.0, 8.0], [90, 100, 90], unit="degrees"
    )
    bonds = (
        1.5 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    )
    centre = cell.to_cartesian(np.array([0.21, 0.13, 0.32]))
    cartesian = np.vstack([centre, centre + bonds[:3] * [1.0, 1.1, 1.2]])
    asym = AsymmetricUnit.from_records(
        [
            {"label": f"X{k}", "element": e, "position": x}
            for k, (e, x) in enumerate(
                zip(("C", "N", "O", "F"), cell.to_fractional(cartesian), strict=True)
            )
        ]
    )
    return Crystal(cell, SpaceGroup(4), asym)


def test_an_enantiomorph_needs_an_improper_overlay(chiral):
    atoms = chiral.unit_cell_atoms()
    mirror = np.diag([-1.0, 1.0, 1.0])
    image = p1(
        chiral.unit_cell.direct @ mirror, atoms["cart_pos"] @ mirror, atoms["element"]
    )
    result = compare_packing(chiral, image)
    assert result.same_packing
    assert result.inverted
    assert not compare_packing(chiral, image, allow_inversion=False).same_packing


@pytest.fixture(scope="module")
def lattice():
    "One asymmetric molecule per 6 A cubic cell, and the body it is placed as"
    rng = np.random.default_rng(7)
    body = RigidBody.from_molecule(rng.normal(size=(5, 3)), [6, 7, 8, 9, 16])
    packing = Packing(6.0 * np.eye(3), np.zeros((1, 3)), np.eye(3)[None])
    return body, packing


def test_a_cluster_keeps_the_whole_shell_tied_with_its_last_place(lattice):
    _, packing = lattice
    cluster = packing.cluster(0, 15, 14.0)
    # 6 at 6 A, then 12 at 6 sqrt(2) A: the 15th is tied with three more
    assert len(cluster.centroids) == 1 + 18
    assert cluster.untied == 6
    assert packing.cluster(0, 6, 14.0).untied == -1


def _reference(pool, stray, nearer_moved=False):
    "The pool's first 15 neighbours, then a 16th tied with the 15th"
    centroids = pool.centroids[:17].copy()
    orientations = pool.orientations[:17].copy()
    centroids[16] = centroids[15] + [3.0, 0, 0] if stray else pool.centroids[16]
    if nearer_moved:
        centroids[1] += [3.0, 0, 0]
    return Cluster(centroids, orientations, untied=14)


def test_either_tied_molecule_may_take_the_last_place(lattice):
    body, packing = lattice
    pool = packing.cluster(0, 40, 14.0)
    reference = _reference(pool, stray=True)
    assert _compare_clusters(body, reference, pool, PackingOptions()).n_matched == 15
    swapped = Cluster(
        reference.centroids[[*range(15), 16, 15]],
        reference.orientations[[*range(15), 16, 15]],
        untied=14,
    )
    assert _compare_clusters(body, swapped, pool, PackingOptions()).n_matched == 15


def test_a_tied_molecule_does_not_stand_in_for_a_nearer_one(lattice):
    body, packing = lattice
    pool = packing.cluster(0, 40, 14.0)
    reference = _reference(pool, stray=False, nearer_moved=True)
    result = _compare_clusters(body, reference, pool, PackingOptions())
    assert result.n_matched == 14
    assert result.n_neighbours == 15

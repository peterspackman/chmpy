import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chmpy import Crystal
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.crystal.packing_similarity import (
    PackingComparer,
    RigidBody,
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

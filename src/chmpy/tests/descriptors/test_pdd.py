import numpy as np
import pytest

from chmpy import Crystal
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.descriptors.pdd import amd, pdd, pdd_emd, pdd_rms

from .. import TEST_FILES


@pytest.fixture(scope="module")
def acetic():
    return Crystal.load(TEST_FILES["acetic_acid.cif"])


def cubic(a, positions):
    cell = UnitCell.cubic(a)
    asym = AsymmetricUnit.from_records(
        [
            {"label": f"C{i}", "element": "C", "position": p}
            for i, p in enumerate(positions)
        ]
    )
    return Crystal(cell, SpaceGroup(1), asym)


def test_a_simple_cubic_lattice():
    rows = pdd(cubic(2.0, [(0, 0, 0)]), k=26).rows
    expected = [2.0] * 6 + [2 * np.sqrt(2)] * 12 + [2 * np.sqrt(3)] * 8
    np.testing.assert_allclose(rows, [expected])


def test_one_weighted_row_per_asymmetric_atom(acetic):
    p = pdd(acetic, k=10)
    assert p.rows.shape == (len(acetic.asymmetric_unit), 10)
    assert p.weights.sum() == pytest.approx(1)
    assert np.all(np.diff(p.rows, axis=1) >= 0)


def test_atoms_outside_the_cell_are_wrapped_in():
    inside = pdd(cubic(5.0, [(0.1, 0.2, 0.3), (0.6, 0.5, 0.4)]), k=20)
    outside = pdd(cubic(5.0, [(1.1, -0.8, 0.3), (0.6, 2.5, -1.6)]), k=20)
    np.testing.assert_allclose(outside.rows, inside.rows)


def test_a_p1_copy_has_the_same_rows_when_grouped_by_atom(acetic):
    p1 = acetic.as_P1()
    groups = acetic.unit_cell_atoms()["asym_atom"]
    assert pdd_rms(pdd(p1, k=50, groups=groups), pdd(acetic, k=50)) < 1e-9


def test_a_supercell_is_no_distance_away(acetic):
    original = pdd(acetic, k=30)
    supercell = pdd(acetic.as_P1_supercell((2, 1, 1)), k=30)
    assert len(supercell.rows) != len(original.rows)
    assert pdd_emd(original, supercell) == pytest.approx(0, abs=1e-8)
    np.testing.assert_allclose(supercell.amd, original.amd, atol=1e-9)


def test_the_emd_is_symmetric_and_bounded_by_the_largest_row_difference():
    a = pdd(cubic(5.0, [(0.1, 0.2, 0.3), (0.6, 0.5, 0.4)]), k=20)
    b = pdd(cubic(5.2, [(0.1, 0.25, 0.3), (0.6, 0.5, 0.45)]), k=20)
    d = pdd_emd(a, b)
    assert d == pytest.approx(pdd_emd(b, a))
    assert 0 < d <= np.abs(a.rows - b.rows).max()


def test_amd_is_the_weighted_mean_of_the_rows(acetic):
    p = pdd(acetic, k=40)
    np.testing.assert_allclose(amd(acetic, k=40), p.weights @ p.rows)


def test_rows_that_do_not_correspond_are_refused(acetic):
    with pytest.raises(ValueError, match="pdd_emd"):
        pdd_rms(pdd(acetic, k=20), pdd(acetic.as_P1(), k=20))
    with pytest.raises(ValueError, match="k differs"):
        pdd_emd(pdd(acetic, k=20), pdd(acetic, k=30))

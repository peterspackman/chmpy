"""Writing CRYSTAL17 input, whose cell line depends on the crystal system."""

import numpy as np
import pytest

from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell
from chmpy.fmt.crystal17 import to_crystal17_input


def cell_line(lengths, angles, space_group):
    cell = UnitCell.from_lengths_and_angles(lengths, angles, unit="degrees")
    asym = AsymmetricUnit.from_records(
        [{"label": "C1", "element": "C", "position": (0.1, 0.2, 0.3)}]
    )
    crystal = Crystal(cell, SpaceGroup(space_group), asym)
    lines = to_crystal17_input(crystal).splitlines()
    index = next(i for i, x in enumerate(lines) if x.strip() == str(space_group))
    return [float(x) for x in lines[index + 1].split()]


def test_orthorhombic_cell_with_equal_lengths_writes_a_b_c():
    params = cell_line([5, 5, 7], [90, 90, 90], 62)
    assert params == pytest.approx([5, 5, 7])


def test_monoclinic_cell_with_equal_lengths_writes_a_b_c_beta():
    params = cell_line([5, 6, 5], [90, 100, 90], 14)
    assert params == pytest.approx([5, 6, 5, 100])


def test_triclinic_cell_writes_all_parameters():
    params = cell_line([5, 6, 7], [70, 100, 70], 2)
    np.testing.assert_allclose(params, [5, 6, 7, 70, 100, 70])

"""Reading what GULP writes, including when it writes less than expected."""

import numpy as np
import pytest

from chmpy.fmt.gulp import parse_drv_file

ENERGY_ONLY = "energy                53.2366418000 eV\n"

NO_GRADIENTS = """energy   -1.5 eV
coordinates cartesian Angstroms      2
     1   6      0.00000000      0.00000000      0.00000000
     2   8      1.20000000      0.00000000      0.00000000
"""

COMPLETE = """energy   -1.5 eV
coordinates cartesian Angstroms      2
     1   6      0.00000000      0.00000000      0.00000000
     2   8      1.20000000      0.00000000      0.00000000
gradients cartesian eV/Ang      2
     1      0.10000000      0.20000000      0.30000000
     2     -0.10000000     -0.20000000     -0.30000000
"""


def write(tmp_path, contents):
    path = tmp_path / "job.drv"
    path.write_text(contents)
    return path


def test_a_file_with_only_an_energy_is_read(tmp_path):
    """GULP writes this when the run declared nothing variable.

    It says so only in a warning, so a parser that assumes the gradient
    section is present fails with an index error a long way from the cause.
    """
    result = parse_drv_file(write(tmp_path, ENERGY_ONLY))
    assert result["energy"] == pytest.approx(53.2366418)
    assert result["gradients"] is None
    assert result["stress_raw"] is None


def test_coordinates_without_gradients_are_read(tmp_path):
    result = parse_drv_file(write(tmp_path, NO_GRADIENTS))
    assert result["energy"] == pytest.approx(-1.5)
    assert result["natoms"] == 2
    assert result["gradients"] is None


def test_a_complete_file_still_reads(tmp_path):
    result = parse_drv_file(write(tmp_path, COMPLETE))
    assert result["energy"] == pytest.approx(-1.5)
    assert result["natoms"] == 2
    np.testing.assert_allclose(
        result["gradients"], [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    )


def test_a_truncated_gradient_block_is_an_error(tmp_path):
    """Missing entirely is fine; half-written is not, and says which."""
    truncated = COMPLETE.rsplit("\n", 2)[0] + "\n"
    with pytest.raises(ValueError, match="promises 2 gradient rows"):
        parse_drv_file(write(tmp_path, truncated))


def test_an_empty_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Empty"):
        parse_drv_file(write(tmp_path, ""))


def _origin_lines(crystal):
    from chmpy.fmt.gulp import crystal_to_gulp_input

    return [
        line
        for line in crystal_to_gulp_input(crystal).splitlines()
        if line.startswith("origin")
    ]


def _fddd(choice):
    from chmpy import Crystal
    from chmpy.core import Element
    from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell

    cell = UnitCell.from_lengths_and_angles(
        (10.46, 12.87, 24.49), np.radians((90, 90, 90))
    )
    return Crystal(
        cell,
        SpaceGroup(70, choice=choice),
        AsymmetricUnit([Element["S"]], np.array([[0.8554, 0.9526, 0.9516]])),
    )


def test_origin_choice_2_is_written():
    assert _origin_lines(_fddd("2")) == ["origin 2"]


def test_origin_choice_1_needs_no_line():
    assert _origin_lines(_fddd("1")) == []


def test_a_rhombohedral_setting_is_not_an_origin_choice():
    from chmpy import Crystal
    from chmpy.tests import TEST_FILES

    crystal = Crystal.load(TEST_FILES["r3c_example.cif"])
    assert crystal.space_group.choice == "H"
    assert _origin_lines(crystal) == []

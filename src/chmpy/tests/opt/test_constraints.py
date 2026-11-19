"""Constrained relaxation: each constraint holds, and the rest relaxes."""

import numpy as np
import pytest

from chmpy.calc import LennardJones, System
from chmpy.opt import Angle, Bond, Constrained, Dihedral, Fixed, relax

#: the pair minimum of this potential, 2^(1/6) sigma
R_MIN = 2 ** (1 / 6) * 3.4


@pytest.fixture(scope="module")
def calc():
    return LennardJones(epsilon=0.0103, sigma=3.4, cutoff=12.0)


def argon(positions):
    positions = np.asarray(positions, dtype=float)
    return System([18] * len(positions), positions)


def trimer():
    return argon([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [2.0, 3.5, 0.3]])


def tetramer():
    return argon([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [4.9, 3.6, 0.0], [5.6, 4.6, 3.1]])


def relaxed(system, calc, constraints, **kwargs):
    result = relax(
        system, calc, constraints=constraints, fmax=1e-5, steps=500, **kwargs
    )
    assert result.converged
    return np.array(result.structure.positions)


def test_a_bond_keeps_its_starting_length(calc):
    positions = relaxed(trimer(), calc, [Bond(0, 1)])
    assert Bond(0, 1).measure(positions) == pytest.approx(4.5, abs=1e-8)
    # the free atom sits symmetrically between the held pair
    assert Bond(0, 2).measure(positions) == pytest.approx(
        Bond(1, 2).measure(positions), abs=1e-4
    )


def test_a_bond_reaches_its_target(calc):
    positions = relaxed(trimer(), calc, [Bond(0, 1, 4.0)])
    assert Bond(0, 1).measure(positions) == pytest.approx(4.0, abs=1e-8)


def test_unconstrained_the_bond_goes_to_the_pair_minimum(calc):
    """The control: without the constraint the same bond does move."""
    positions = np.array(relax(trimer(), calc, fmax=1e-5).structure.positions)
    assert Bond(0, 1).measure(positions) == pytest.approx(R_MIN, abs=1e-2)


def test_an_angle_reaches_its_target(calc):
    positions = relaxed(trimer(), calc, [Angle(0, 1, 2, 80.0)])
    assert Angle(0, 1, 2).measure(positions) == pytest.approx(80.0, abs=1e-6)


def test_a_dihedral_keeps_its_starting_value(calc):
    start = Dihedral(0, 1, 2, 3).measure(tetramer().positions)
    positions = relaxed(tetramer(), calc, [Dihedral(0, 1, 2, 3)])
    assert Dihedral(0, 1, 2, 3).measure(positions) == pytest.approx(start, abs=1e-6)


def test_a_dihedral_target_across_180(calc):
    positions = relaxed(tetramer(), calc, [Dihedral(0, 1, 2, 3, 175.0)])
    assert Dihedral(0, 1, 2, 3).measure(positions) == pytest.approx(175.0, abs=1e-6)


def test_several_constraints_hold_together(calc):
    constraints = [Bond(0, 1, 4.2), Angle(0, 1, 2, 95.0), Dihedral(0, 1, 2, 3, -60.0)]
    positions = relaxed(tetramer(), calc, constraints)
    for entry in constraints:
        assert entry.measure(positions) == pytest.approx(entry.value, abs=1e-6)


def test_fixed_atoms_and_components_stay(calc):
    start = tetramer().positions
    positions = relaxed(tetramer(), calc, [Fixed([0]), Fixed([3], axes="z")])
    np.testing.assert_allclose(positions[0], start[0], atol=1e-12)
    assert positions[3, 2] == pytest.approx(start[3, 2], abs=1e-12)
    assert np.abs(positions[3, :2] - start[3, :2]).max() > 1e-3


def test_fixed_atoms_match_the_fixed_mask(calc):
    by_constraint = relaxed(tetramer(), calc, [Fixed([0, 2])])
    mask = np.array([True, False, True, False])
    by_mask = relax(tetramer(), calc, fixed=mask, fmax=1e-5, steps=500)
    np.testing.assert_allclose(by_constraint, by_mask.structure.positions, atol=1e-4)


def test_the_result_is_a_constrained_minimum(calc):
    """At convergence the force lies along the constraint gradient alone,
    and nudging along an allowed direction cannot lower the energy."""
    system = trimer()
    relax(system, calc, constraints=[Bond(0, 1)], fmax=1e-6, steps=500)
    coordinates = Constrained(system, [Bond(0, 1)])
    energy = calc(system, ("energy",)).energy
    for direction in np.eye(coordinates.n_dof):
        for sign in (1, -1):
            coordinates.set(sign * 1e-3 * direction)
            assert calc(system, ("energy",)).energy >= energy - 1e-9
    coordinates.set(np.zeros(coordinates.n_dof))


def test_the_degrees_of_freedom_are_counted():
    # 4 atoms: 12 components, less one per constraint, less 3 + 1 fixed
    coordinates = Constrained(
        tetramer(), [Bond(0, 1), Angle(0, 1, 2), Fixed([3]), Fixed([2], axes="x")]
    )
    assert coordinates.n_dof == 12 - 2 - 4


@pytest.mark.parametrize(
    "constraints, message",
    [
        ([Bond(0, 5)], "names atoms \\[5\\]"),
        ([Angle(0, 1, 1)], "repeats an atom"),
        ([Fixed([0, 1]), Bond(0, 1)], "only fixed atoms"),
        ([Bond(0, 1, 3.0), Bond(1, 2, 3.0), Bond(0, 2, 9.0)], "could not be satisfied"),
    ],
)
def test_impossible_constraints_are_refused(constraints, message):
    with pytest.raises(ValueError, match=message):
        Constrained(trimer(), constraints)


def test_a_linear_angle_is_refused():
    system = argon([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="linear"):
        Constrained(system, [Angle(0, 1, 2)])


def test_a_periodic_cell_must_be_fixed(calc):
    system = System([18], np.zeros((1, 3)), np.eye(3) * 5.3, True)
    with pytest.raises(ValueError, match="cell=False"):
        relax(system, calc, constraints=[Fixed([0])])

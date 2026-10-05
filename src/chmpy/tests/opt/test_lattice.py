"""Lattice energy: the lattice sum it must reproduce, and the decomposition."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones, PairPotential
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt import lattice_energy
from chmpy.util.unit import EV_TO_KJ_PER_MOL

#: the 12-6 lattice sum for a relaxed fcc crystal, in units of epsilon. A
#: truncated sum approaches it from above, so this is the limit the calculation
#: has to be walking towards rather than a number it should hit.
FCC_LATTICE_SUM = -8.6102


def cubic(a):
    return UnitCell.from_lengths_and_angles((a, a, a), np.radians((90, 90, 90)))


def fcc_argon(a=5.3):
    return Crystal(
        cubic(a),
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )


class BondedDiatomics(PairPotential):
    """A harmonic bond between unlike close neighbours, packing between the rest.

    The bond has to come from something with no attractive tail. A Morse or a
    Lennard-Jones deep enough to hold the molecule together also binds the
    neighbouring molecules, and the crystal relaxes into a covalent chain with
    no molecules left in it to take apart. A harmonic well cannot do that: it
    is purely repulsive away from its minimum, so it holds the two atoms it is
    applied to and pulls nothing else in.

    The radii differ by element so the crystal field on one end of the molecule
    is not the field on the other, which is what makes the bond length in the
    crystal differ from the bond length in the gas.
    """

    cutoff = 9.0
    force_constant = 2.0
    length = 1.45
    #: an unlike pair closer than this is a bond. The packing keeps unlike
    #: atoms of different molecules several Angstroms apart, so no pair ever
    #: crosses it and the discontinuity there is never visited.
    bonded_below = 2.0
    epsilon = 0.01
    radii = {6: 3.6, 8: 2.9}

    def pair(self, r, zi, zj):
        sigma = 0.5 * (
            np.where(zi == 6, self.radii[6], self.radii[8])
            + np.where(zj == 6, self.radii[6], self.radii[8])
        )
        offset = r - self.length
        y = (sigma / r) ** 6
        packing = 4 * self.epsilon * (y * y - y)
        dpacking = 4 * self.epsilon * (-12 * y * y + 6 * y) / r
        bonded = (zi != zj) & (r < self.bonded_below)
        return (
            np.where(bonded, self.force_constant * offset * offset, packing),
            np.where(bonded, 2 * self.force_constant * offset, dpacking),
        )


def co_crystal(a=6.0, d=1.45):
    return Crystal(
        cubic(a),
        SpaceGroup(1),
        AsymmetricUnit(
            [Element[6], Element[8]],
            np.array([[0.5, 0.5, 0.5 - 0.5 * d / a], [0.5, 0.5, 0.5 + 0.5 * d / a]]),
        ),
    )


def bond_length(molecule):
    return float(np.linalg.norm(np.diff(molecule.positions, axis=0)))


@pytest.fixture(scope="module")
def diatomic():
    calculator = BondedDiatomics()
    return lattice_energy(co_crystal(), calculator, fmax=1e-6, smax=1e-5, steps=500)


def test_argon_approaches_the_lennard_jones_lattice_sum():
    """The cohesive energy of an atomic solid, against a number known exactly.

    Every part of the calculation is in this one number: the relaxation of the
    cell, the count of atoms per cell, and the energy of the free atom.
    """
    epsilon = 0.0103
    previous = None
    for cutoff in (12.0, 16.0, 24.0):
        result = lattice_energy(
            fcc_argon(),
            LennardJones(epsilon=epsilon, sigma=3.4, cutoff=cutoff, shift_energy=False),
            fmax=1e-7,
            smax=1e-6,
            steps=300,
        )
        reduced = result.energy / epsilon
        assert FCC_LATTICE_SUM < reduced < 0.0
        if previous is not None:
            assert reduced < previous  # a longer cutoff can only add attraction
        previous = reduced
    assert reduced == pytest.approx(FCC_LATTICE_SUM, abs=0.03)


def test_argon_relaxes_to_the_lennard_jones_spacing():
    result = lattice_energy(
        fcc_argon(),
        LennardJones(epsilon=0.0103, sigma=3.4, cutoff=24.0, shift_energy=False),
        fmax=1e-7,
        smax=1e-6,
        steps=300,
    )
    # nearest neighbour at 1.09 sigma, so a = sqrt(2) * 1.09 * sigma
    assert result.structure.unit_cell.lengths[0] == pytest.approx(5.2436, abs=0.01)


def test_an_atomic_solid_has_no_strain():
    """A single atom has no conformation to pay for, so the split is trivial."""
    result = lattice_energy(fcc_argon(), LennardJones(cutoff=12.0), steps=300)
    assert result.z == 4
    assert result.z_prime == 1
    assert result.multiplicities == [4]
    assert result.strain == 0.0
    assert result.energy == result.interaction


def test_the_decomposition_adds_up(diatomic):
    assert diatomic.energy == pytest.approx(
        diatomic.interaction + diatomic.strain, abs=1e-12
    )
    assert diatomic.kj_per_mol == pytest.approx(diatomic.energy * EV_TO_KJ_PER_MOL)


def test_a_bound_crystal_has_a_negative_lattice_energy(diatomic):
    assert diatomic.energy < 0.0
    assert diatomic.z == 1
    assert diatomic.z_prime == 1


def test_strain_is_positive_and_the_conformation_differs(diatomic):
    """The molecule is not in its gas-phase geometry once it is packed."""
    assert diatomic.strain > 0.0
    in_crystal = bond_length(diatomic.structure.symmetry_unique_molecules()[0])
    in_gas = bond_length(diatomic.molecules[0])
    assert in_gas == pytest.approx(BondedDiatomics.length, abs=1e-5)
    assert abs(in_crystal - in_gas) > 1e-3


def test_strain_falls_as_the_bond_stiffens():
    """A stiffer bond pays less for the same crystal field, as ~1/k."""
    strains = {}
    for constant in (2.0, 20.0):
        calculator = BondedDiatomics()
        calculator.force_constant = constant
        strains[constant] = lattice_energy(
            co_crystal(), calculator, fmax=1e-6, smax=1e-5, steps=500
        ).strain
    assert strains[2.0] > strains[20.0] > 0.0
    assert strains[2.0] / strains[20.0] == pytest.approx(10.0, rel=0.25)


def test_frozen_molecules_leave_the_interaction_energy():
    """With no gas-phase relaxation there is nothing to pay, by construction."""
    result = lattice_energy(
        co_crystal(),
        BondedDiatomics(),
        relax_molecules=False,
        fmax=1e-6,
        smax=1e-5,
        steps=500,
    )
    assert result.strain == 0.0
    assert result.energy == result.interaction
    assert result.molecule_energies == result.frozen_energies


def test_an_unrelaxed_crystal_keeps_its_cell():
    """`relax_crystal=False` reports the structure as supplied.

    Built with a bond stretched past its gas-phase length, so the strain the
    supplied geometry carries is there to be measured rather than relaxed away.
    """
    stretched = 1.60
    result = lattice_energy(
        co_crystal(d=stretched),
        BondedDiatomics(),
        relax_crystal=False,
        fmax=1e-6,
        steps=500,
    )
    assert result.structure.unit_cell.lengths[0] == pytest.approx(6.0)
    assert bond_length(
        result.structure.symmetry_unique_molecules()[0]
    ) == pytest.approx(stretched, abs=1e-6)
    # k * (1.60 - 1.45)^2 for the one bond in the cell
    expected = (
        BondedDiatomics.force_constant * (stretched - BondedDiatomics.length) ** 2
    )
    assert result.strain == pytest.approx(expected, rel=1e-3)


def test_a_box_leaves_an_isolated_molecule_alone():
    """A box wider than the cutoff cannot change the gas-phase energy."""
    calculator = BondedDiatomics()
    loose = lattice_energy(
        co_crystal(), calculator, box=30.0, fmax=1e-6, smax=1e-5, steps=500
    )
    free = lattice_energy(co_crystal(), calculator, fmax=1e-6, smax=1e-5, steps=500)
    assert loose.frozen_energies[0] == pytest.approx(free.frozen_energies[0], abs=1e-9)


def test_counting_survives_a_larger_cell():
    """Z doubles with the cell; the energy per molecule does not move."""
    one = lattice_energy(co_crystal(), BondedDiatomics(), fmax=1e-6, steps=500)
    doubled = co_crystal().as_P1()
    doubled = Crystal(
        UnitCell(np.diag([6.0, 6.0, 12.0])),
        SpaceGroup(1),
        AsymmetricUnit(
            [Element[6], Element[8], Element[6], Element[8]],
            np.array(
                [
                    [0.5, 0.5, 0.25 - 0.25 * 1.45 / 6.0],
                    [0.5, 0.5, 0.25 + 0.25 * 1.45 / 6.0],
                    [0.5, 0.5, 0.75 - 0.25 * 1.45 / 6.0],
                    [0.5, 0.5, 0.75 + 0.25 * 1.45 / 6.0],
                ]
            ),
        ),
    )
    two = lattice_energy(doubled, BondedDiatomics(), fmax=1e-6, steps=500)
    assert two.z == 2
    assert two.energy == pytest.approx(one.energy, rel=1e-3)


def test_a_framework_is_refused():
    """Nothing comes apart, so there is no lattice energy in this sense."""
    diamond = Crystal(
        cubic(3.57),
        SpaceGroup(227),
        AsymmetricUnit([Element[6]], np.array([[0.125, 0.125, 0.125]])),
    )
    with pytest.raises(ValueError, match="break 9 covalent"):
        lattice_energy(diamond, LennardJones())


def test_a_chain_is_refused():
    """Four parallel chains count the same as four molecules; the bonds do not."""
    chains = Crystal(
        UnitCell(np.diag([8.0, 8.0, 1.5])),
        SpaceGroup(1),
        AsymmetricUnit(
            [Element[6]] * 4,
            np.array(
                [[0.2, 0.2, 0.0], [0.7, 0.2, 0.0], [0.2, 0.7, 0.0], [0.7, 0.7, 0.0]]
            ),
        ),
    )
    assert len(chains.unit_cell_molecules()) == 4
    with pytest.raises(ValueError, match="chain, sheet or framework"):
        lattice_energy(chains, LennardJones())


def test_the_convenience_method_agrees():
    calculator = LennardJones(cutoff=12.0)
    direct = lattice_energy(fcc_argon(), calculator, steps=300)
    via_calculator = calculator.lattice_energy(fcc_argon(), steps=300)
    assert via_calculator.energy == pytest.approx(direct.energy)


def test_a_relaxation_that_falls_short_is_reported():
    """A lattice energy from unconverged structures is not a lattice energy.

    It has to be visible in the result rather than only in a log line, since
    the number itself looks perfectly ordinary.
    """
    result = lattice_energy(co_crystal(), BondedDiatomics(), fmax=1e-12, steps=2)
    assert not result.converged
    assert result.unconverged
    assert "NOT CONVERGED" in repr(result)


def test_a_converged_run_says_so(diatomic):
    assert diatomic.converged
    assert diatomic.unconverged == []
    assert "NOT CONVERGED" not in repr(diatomic)


def test_a_collapsed_cell_is_refused():
    """What a relaxation looks like when it destroys the thing it relaxed.

    A machine-learned potential without a repulsive wall does not blow up when
    a cell collapses; it runs off to a large negative energy, and the molecules
    cut out of the wreckage give a confident nonsense lattice energy. The cell
    here is squeezed until each molecule bonds to its own image along c.
    """
    from chmpy.opt.lattice import _check_still_molecular

    _check_still_molecular(co_crystal(a=6.0), z=1)  # the intact case passes
    with pytest.raises(ValueError, match="collapsed into an extended"):
        _check_still_molecular(co_crystal(a=2.6, d=1.3), z=1)


def test_a_relaxation_that_changes_z_is_refused():
    from chmpy.opt.lattice import _check_still_molecular

    with pytest.raises(ValueError, match="changed the bonding"):
        _check_still_molecular(co_crystal(a=6.0), z=4)

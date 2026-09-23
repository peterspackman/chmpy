"""Degrees of freedom that are the asymmetric unit, not a projection onto it."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import LennardJones, PairPotential, System
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.opt.symmetry import SymmetryAdapted, site_displacement_basis
from chmpy.opt.trust_region import TrustRegion

from .. import TEST_FILES

ACETIC_ACID = TEST_FILES["acetic_acid.cif"]
R3C = TEST_FILES["r3c_example.cif"]


class GaussianPairs(PairPotential):
    """A soft, everywhere-smooth pair potential. Two lines of physics.

    Being bounded at short range makes it the right thing to differentiate
    against for a gradient test: a 12-6 potential at molecular bond distances
    gives energies of 1e7 eV, where a finite difference says nothing.
    """

    cutoff = 6.0

    def pair(self, r, zi, zj):
        gaussian = 0.35 * np.exp(-0.4 * r * r)
        return gaussian, -0.8 * r * gaussian


def fcc_argon(a=5.3):
    """One atom on the m-3m site of Fm-3m: no atomic freedom, one cell freedom."""
    cell = UnitCell.from_lengths_and_angles((a, a, a), np.radians((90, 90, 90)))
    return Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )


def numeric_gradient(coordinates, calc, step=1e-6):
    x = coordinates.get()
    gradient = np.zeros_like(x)
    for k in range(len(x)):
        shifted = x.copy()
        shifted[k] = x[k] + step
        coordinates.set(shifted)
        plus = calc.energy(coordinates.system)
        shifted[k] = x[k] - step
        coordinates.set(shifted)
        minus = calc.energy(coordinates.system)
        gradient[k] = (plus - minus) / (2 * step)
    coordinates.set(x)
    return gradient


def symmetry_error(crystal, tolerance=1e-2):
    """How far the structure is from being invariant under its own space group."""
    atoms = crystal.unit_cell_atoms(tolerance=tolerance)
    fractional = atoms["frac_pos"]
    numbers = atoms["element"]
    worst = 0.0
    for operation in crystal.space_group.symmetry_operations:
        image = (fractional @ np.asarray(operation.rotation).T) + operation.translation
        image = image % 1.0
        for position, number in zip(image, numbers, strict=True):
            difference = np.abs(fractional - position)
            difference = np.minimum(difference, 1.0 - difference)
            distance = np.linalg.norm(difference, axis=1)
            distance = np.where(numbers == number, distance, np.inf)
            worst = max(worst, float(distance.min()))
    return worst


def test_the_p1_expansion_is_the_one_the_crystal_gives():
    crystal = Crystal.load(ACETIC_ACID)
    coordinates = SymmetryAdapted(crystal)
    reference = System.from_crystal(crystal)
    np.testing.assert_allclose(
        np.sort(coordinates.system.positions, axis=0),
        np.sort(reference.positions, axis=0),
        atol=1e-10,
    )


@pytest.mark.parametrize("path", [ACETIC_ACID, R3C])
def test_gradient_matches_finite_differences(path):
    crystal = Crystal.load(path)
    coordinates = SymmetryAdapted(crystal)
    calc = GaussianPairs()
    analytic = coordinates.gradient(calc(coordinates.system, coordinates.wanted))
    numeric = numeric_gradient(coordinates, calc)
    scale = max(float(np.abs(numeric).max()), 1e-12)
    assert np.abs(analytic - numeric).max() < 1e-6 * scale


def test_the_degrees_of_freedom_are_the_asymmetric_unit():
    crystal = Crystal.load(ACETIC_ACID)
    coordinates = SymmetryAdapted(crystal)
    assert len(coordinates.system) == 32
    assert coordinates.n_atomic == 24  # eight atoms, general positions
    assert len(coordinates.basis) == 3  # orthorhombic
    assert coordinates.n_dof == 27
    # what an unconstrained P1 relaxation would have carried
    assert 3 * len(coordinates.system) + 6 == 102


def test_a_special_position_has_fewer_freedoms():
    coordinates = SymmetryAdapted(fcc_argon())
    assert len(coordinates.system) == 4
    assert coordinates.n_atomic == 0  # the m-3m site cannot move at all
    assert len(coordinates.basis) == 1  # cubic: one isotropic strain
    assert coordinates.n_dof == 1


def test_site_basis_dimensions():
    identity = np.eye(3)
    mirror_z = np.diag([1.0, 1.0, -1.0])
    two_fold_z = np.diag([-1.0, -1.0, 1.0])
    inversion = -np.eye(3)

    assert site_displacement_basis([identity]).shape[1] == 3
    assert site_displacement_basis([identity, mirror_z]).shape[1] == 2
    assert site_displacement_basis([identity, two_fold_z]).shape[1] == 1
    assert site_displacement_basis([identity, inversion]).shape[1] == 0


def test_a_fixed_site_cannot_be_moved():
    coordinates = SymmetryAdapted(fcc_argon())
    before = np.array(coordinates.system.positions)
    coordinates.set(np.array([0.05]))  # a pure strain
    scaled_before = before @ np.linalg.inv(np.eye(3) * 5.3)
    np.testing.assert_allclose(
        coordinates.system.scaled_positions, scaled_before, atol=1e-12
    )


def test_relaxation_keeps_the_space_group_exactly():
    crystal = Crystal.load(ACETIC_ACID)
    coordinates = SymmetryAdapted(crystal)
    TrustRegion(coordinates, GaussianPairs()).run(fmax=1e-3, smax=1e-2, steps=60)

    relaxed = coordinates.to_crystal()
    assert relaxed.space_group == crystal.space_group
    assert symmetry_error(relaxed) < 1e-12


def test_relaxation_reaches_the_same_minimum_as_an_unconstrained_one():
    """Starting symmetric, a free relaxation should find the symmetric minimum."""
    from chmpy.opt.coordinates import AtomicStrain

    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)

    symmetric = SymmetryAdapted(fcc_argon(5.0))
    constrained = TrustRegion(symmetric, calc).run(fmax=1e-4, smax=1e-3, steps=50)

    free_system = System.from_crystal(fcc_argon(5.0))
    free = TrustRegion(AtomicStrain(free_system), calc).run(
        fmax=1e-4, smax=1e-3, steps=50
    )

    assert constrained.converged and free.converged
    assert constrained.energy == pytest.approx(free.energy, abs=1e-6)
    assert constrained.evaluations <= free.evaluations


def test_noisy_forces_cannot_break_the_symmetry():
    """The case that matters: a machine-learned potential's forces are noisy.

    Relaxing every atom of the P1 cell and projecting afterwards lets that
    noise walk the structure off its symmetry. Here the symmetry is in the
    parameterisation, so no amount of noise can break it.
    """
    from chmpy.opt.coordinates import Atomic

    class Noisy(GaussianPairs):
        """A potential with a small, reproducible error on its forces."""

        def compute(self, system, want):
            result = super().compute(system, want)
            if result.forces is None:
                return result
            rng = np.random.default_rng(abs(hash(result.forces.tobytes())) % 2**32)
            noise = rng.normal(scale=1e-3, size=result.forces.shape)
            return type(result)(
                energy=result.energy,
                forces=result.forces + noise,
                stress=result.stress,
                energies=result.energies,
                volume=result.volume,
            )

    crystal = Crystal.load(ACETIC_ACID)

    symmetric = SymmetryAdapted(crystal, cell=False)
    TrustRegion(symmetric, Noisy()).run(fmax=1e-6, steps=12)
    assert symmetry_error(symmetric.to_crystal()) < 1e-12

    free_system = System.from_crystal(crystal)
    TrustRegion(Atomic(free_system), Noisy()).run(fmax=1e-6, steps=12)
    assert symmetry_error(free_system.to_crystal(crystal.space_group)) > 1e-6


def test_to_crystal_round_trips_through_the_asymmetric_unit():
    crystal = Crystal.load(ACETIC_ACID)
    coordinates = SymmetryAdapted(crystal)
    rebuilt = coordinates.to_crystal()
    np.testing.assert_allclose(
        rebuilt.asymmetric_unit.positions, crystal.asymmetric_unit.positions, atol=1e-12
    )
    np.testing.assert_allclose(
        rebuilt.unit_cell.direct, crystal.unit_cell.direct, atol=1e-12
    )


def test_a_fixed_cell_run_has_no_strain_freedoms():
    crystal = Crystal.load(ACETIC_ACID)
    coordinates = SymmetryAdapted(crystal, cell=False)
    assert coordinates.n_dof == coordinates.n_atomic == 24
    assert set(
        coordinates.measures(GaussianPairs()(coordinates.system, coordinates.wanted))
    ) == {"fmax"}


def trigonal_triazine():
    """1,3,5-triazine in R-3c: three atoms, each on a two-fold axis.

    Small, and every atom is on a special position of a non-orthogonal cell,
    which is the combination that the diagonal rotations elsewhere in this
    file cannot produce.
    """
    cell = UnitCell.from_lengths_and_angles(
        (9.647, 9.647, 7.281), np.radians((90.0, 90.0, 120.0))
    )
    return Crystal(
        cell,
        SpaceGroup(167),
        AsymmetricUnit(
            [Element["C"], Element["H"], Element["N"]],
            np.array([[0.0, 0.1317, 0.25], [0.0, 0.2334, 0.25], [-0.1408, 0.0, 0.25]]),
        ),
    )


def test_a_site_basis_is_invariant_in_a_non_orthogonal_cell():
    """`R d = d` is the defining property, so check it where it is not free.

    In fractional coordinates a rotation is an integer matrix, orthogonal only
    when the cell is. For a trigonal cell the averaged site-symmetry operator
    is idempotent but not symmetric, and then its range -- the space it
    projects onto -- is not what symmetrising it, or taking eigenvectors of
    `P^T P`, returns. Both give a direction that satisfies `R d = d` nowhere
    near, and the atom walks off its special position.

    Every rotation in `test_site_basis_dimensions` is diagonal, which is the
    one case where the distinction cannot show.
    """
    from chmpy.opt.symmetry import _fixes

    group = SpaceGroup(167)
    site = np.array([0.0, 0.1317, 0.25])  # Wyckoff 18e, on a two-fold axis
    stabiliser = [
        np.asarray(operation.rotation, dtype=float)
        for operation in group.symmetry_operations
        if _fixes(operation, site)
    ]
    assert len(stabiliser) == 2

    averaged = np.mean(stabiliser, axis=0)
    assert np.allclose(averaged @ averaged, averaged)  # a projector
    assert not np.allclose(averaged, averaged.T)  # but not a symmetric one

    basis = site_displacement_basis(stabiliser)
    assert basis.shape[1] == 1
    for rotation in stabiliser:
        np.testing.assert_allclose(rotation @ basis, basis, atol=1e-12)


def test_every_freedom_keeps_the_atom_on_its_special_position():
    """The consequence: a bad basis changes what is in the cell.

    Moving off a two-fold axis doubles the site multiplicity from 18 to 36, so
    the cell comes back with 72 atoms rather than 54 and some of them land
    0.37 A apart. The optimiser would then report a structure that is not the
    one it relaxed, at an energy belonging to neither.
    """
    crystal = trigonal_triazine()
    coordinates = SymmetryAdapted(crystal, cell=True)
    atoms = len(coordinates.system)
    assert atoms == 54

    start = np.asarray(coordinates.get()).copy()
    for freedom in range(coordinates.n_dof):
        moved = start.copy()
        moved[freedom] += 0.05
        coordinates.set(moved)
        rebuilt = coordinates.to_crystal()
        assert len(rebuilt.unit_cell_atoms()["element"]) == atoms, (
            f"freedom {freedom} left the special position"
        )
        coordinates.set(start)


def test_a_moved_structure_still_rebuilds_exactly():
    """What `relax` hands back has to be what it evaluated.

    Compared as sets modulo lattice translations, since rebuilding wraps
    coordinates into [0, 1) and may reorder them.
    """
    crystal = trigonal_triazine()
    coordinates = SymmetryAdapted(crystal, cell=True)

    def canonical(fractional):
        wrapped = np.round(np.mod(np.asarray(fractional), 1.0), 6) % 1.0
        return wrapped[np.lexsort(wrapped.T[::-1])]

    start = np.asarray(coordinates.get()).copy()
    for freedom in range(coordinates.n_dof):
        moved = start.copy()
        moved[freedom] += 0.04
        coordinates.set(moved)
        evaluated = canonical(
            np.asarray(coordinates.system.positions)
            @ np.linalg.inv(np.asarray(coordinates.system.cell))
        )
        rebuilt = canonical(coordinates.to_crystal().unit_cell_atoms()["frac_pos"])
        np.testing.assert_allclose(evaluated, rebuilt, atol=1e-9)
        coordinates.set(start)

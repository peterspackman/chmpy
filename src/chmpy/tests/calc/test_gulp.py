"""GULP as a calculator, and as an external reference.

GULP computes elastic constants and second derivatives analytically, which
makes it something none of the other calculators here are: a check on our
finite differences rather than another estimate of the same kind.
"""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import System
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell

gulp = pytest.importorskip("chmpy.calc.adapters.gulp")
pytestmark = pytest.mark.skipif(
    not gulp.available(), reason="the gulp executable is not on PATH"
)

# rocksalt MgO with a rigid-ion Buckingham model: small, cubic, and with both
# atoms on special positions, so relaxed-ion and clamped-ion constants agree
POTENTIALS = """species
Mg core  2.0
O  core -2.0
buck
Mg core O  core  1428.5   0.2945  0.0      0.0 12.0
O  core O  core 22764.3   0.1490 27.88     0.0 12.0
"""


def magnesium_oxide(a=4.212):
    cell = UnitCell.from_lengths_and_angles((a, a, a), np.radians((90, 90, 90)))
    return Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit(
            [Element["Mg"], Element["O"]],
            np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        ),
    )


def analytic_elastic_constants(crystal):
    """What GULP itself says the elastic constants are."""
    import tempfile

    from chmpy.exe.gulp import Gulp
    from chmpy.fmt.gulp import crystal_to_gulp_input, parse_elastic_constants

    body = crystal_to_gulp_input(crystal, keywords=["conp", "property"])
    with tempfile.TemporaryDirectory() as scratch:
        job = Gulp(body + "\n" + POTENTIALS, working_directory=scratch)
        job.run()
        return parse_elastic_constants(job.output_contents)


@pytest.fixture(scope="module")
def relaxed():
    from chmpy.opt import relax

    calc = gulp.GulpCalculator(potentials=POTENTIALS)
    structure = relax(magnesium_oxide(), calc, fmax=1e-4, smax=1e-3, steps=60).structure
    return structure, calc


def test_a_potential_is_required():
    with pytest.raises(ValueError, match="needs a potential"):
        gulp.GulpCalculator(potentials="")


def test_energy_forces_and_stress_come_back(relaxed):
    structure, calc = relaxed
    result = calc(System.from_crystal(structure), ("energy", "forces", "stress"))
    assert result.energy < 0
    assert result.forces.shape == (8, 3)
    assert result.stress.shape == (3, 3)


def test_the_relaxed_structure_is_at_equilibrium(relaxed):
    structure, calc = relaxed
    result = calc(System.from_crystal(structure), ("energy", "forces", "stress"))
    assert result.fmax < 1e-4
    assert result.smax < 1e-3


def test_gradients_agree_with_finite_differences(relaxed):
    """GULP's analytic derivatives against differences of its own energies.

    Checked away from equilibrium, and with one atom moved rather than all of
    them: a rigid translation of a periodic cell changes nothing, so it tests
    only that zero equals zero, and at the minimum the stress has no signal to
    compare against either.
    """
    structure, calc = relaxed
    system = System.from_crystal(structure)
    system.set_cell(np.asarray(system.cell) * 0.98, scale_atoms=True)
    positions = np.array(system.positions)
    positions[1] += [0.08, -0.05, 0.03]
    system.set_positions(positions)

    # The step is chosen, not defaulted. Measured on this structure, the force
    # check degrades at small steps (round-off in a difference of two numbers
    # near -165 eV) while the stress check degrades at large ones (O(step^2)
    # truncation against constants of ~400 GPa): 1e-3 gives 1.1e-3 and 1.3e-2,
    # 1e-4 gives 1.3e-2 and 1.5e-4, and 3e-4 sits near the bottom of both.
    check = calc.check_gradients(system, step=3e-4, tolerance=1e-2)
    assert check.ok, str(check)


def test_the_elastic_tensor_converges_to_gulps_analytic_one(relaxed):
    """Quadratically, as a central difference should.

    This is the external check the rest of the elastic code does not have:
    every other validation compares our finite differences against our own
    energies, or against an experiment that measures something slightly
    different. Here the same quantity is available in closed form.
    """
    from chmpy.opt import elastic_tensor

    structure, calc = relaxed
    reference = analytic_elastic_constants(structure)
    assert reference is not None

    errors = {}
    for strain in (0.01, 0.005, 0.002):
        ours = elastic_tensor(structure, calc, strain=strain).c_voigt
        errors[strain] = float(np.abs(ours - reference).max())

    assert errors[0.002] < 0.1  # GPa, against constants of ~390
    # halving the strain should quarter the error
    assert errors[0.005] / errors[0.01] == pytest.approx(0.25, abs=0.15)


def test_the_cauchy_relation_holds_for_a_rigid_ion_model(relaxed):
    """C12 = C44 for central forces at equilibrium, and this model is central."""
    from chmpy.opt import elastic_tensor

    structure, calc = relaxed
    c = elastic_tensor(structure, calc, strain=0.002).c_voigt
    assert c[0, 1] == pytest.approx(c[3, 3], rel=1e-3)


def test_phonons_reproduce_gulps_elastic_tensor(relaxed):
    """The Christoffel check, against a tensor from outside this codebase."""
    from chmpy.vib import compare_with_elastic, force_constants

    structure, calc = relaxed
    reference = analytic_elastic_constants(structure)
    constants = force_constants(structure, calc, cutoff=6.0)

    assert constants.acoustic_error("cm-1") < 1e-3
    comparison = compare_with_elastic(constants, reference)
    # the residual is dominated by truncating the force constants at 6 A when
    # the potential itself runs to 12 A
    assert comparison["max_relative_error"] < 5e-3


def test_parse_elastic_constants_returns_none_without_the_property_keyword():
    from chmpy.fmt.gulp import parse_elastic_constants

    assert parse_elastic_constants("no elastic constants in here") is None


# a pair of neutral atoms with a plain Lennard-Jones between them, which needs
# no cell and no charges, so the same potential works isolated and periodic
DIATOMIC_POTENTIALS = """species
Ar core 0.0
lennard epsilon sigma 12 6
Ar core Ar core 0.0103 3.40 0.0 12.0
"""


def argon_pair(separation):
    return System(
        [18, 18],
        np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]),
        None,
        False,
    )


def test_an_isolated_molecule_gets_an_energy_and_gradients():
    """The molecule path needs `conv`, and the reason is easy to miss.

    Dropping `conp` makes GULP look for per-atom optimisation flags that the
    input does not carry. Replacing it with `noflag` stops that but leaves no
    atom variable, so GULP reports an energy, writes a `.drv` with no gradient
    section at all, and mentions it only in a warning.
    """
    calculator = gulp.GulpCalculator(potentials=DIATOMIC_POTENTIALS)
    result = calculator(argon_pair(4.0), ("energy", "forces"))
    assert result.energy < 0.0
    assert result.forces.shape == (2, 3)
    # the pair is past the minimum at 2^(1/6) sigma, so it is still attracting
    assert result.forces[0, 0] > 0.0
    assert np.allclose(result.forces[0], -result.forces[1])


def test_the_isolated_minimum_is_where_gulp_puts_it():
    """The well is exactly epsilon deep, at sigma.

    Not at 2^(1/6) sigma: under `lennard epsilon sigma` the second parameter is
    the position of the minimum rather than the zero crossing, so the same two
    numbers mean a different curve here than they do in
    `chmpy.calc.LennardJones`, by a factor of 1.12 in range.
    """
    calculator = gulp.GulpCalculator(potentials=DIATOMIC_POTENTIALS)
    sigma, epsilon = 3.40, 0.0103
    assert calculator.energy(argon_pair(sigma)) == pytest.approx(-epsilon, abs=1e-8)
    for separation in (sigma - 0.2, sigma + 0.2, 2 ** (1 / 6) * sigma):
        assert calculator.energy(argon_pair(separation)) > -epsilon


def test_a_potential_that_matches_nothing_is_an_error():
    """Zero is what GULP reports when no species matched, not a result."""
    calculator = gulp.GulpCalculator(
        potentials="species\nXe core 0.0\nbuck\nXe core Xe core 1.0 0.3 0.0 0.0 12.0\n"
    )
    with pytest.raises(gulp.PropertyNotAvailable, match="exactly zero"):
        calculator.energy(argon_pair(4.0))

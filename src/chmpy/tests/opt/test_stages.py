"""Staged relaxation protocols."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.calc import Calculator, LennardJones, Result, System
from chmpy.opt import Stage, coordinates_for, relax, two_stage
from chmpy.opt.relax import _plan

from .. import TEST_FILES


def argon(a=5.0):
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    positions = positions + np.array(
        [[0.08, 0, 0], [0, 0.05, 0], [0, 0, -0.06], [0.02, 0.02, 0.02]]
    )
    return System([18] * 4, positions, np.eye(3) * a, True)


def test_the_default_is_a_single_stage():
    plan = _plan(None, argon(), True, 0.01, 0.05, 200, "model")
    assert len(plan) == 1
    assert plan[0].cell is True


def test_two_stage_relaxes_the_cell_only_in_the_second():
    first, second = two_stage(fmax=0.01, smax=0.05)
    assert first.cell is False
    assert second.cell is True
    assert first.fmax > second.fmax  # the first pass is deliberately loose
    assert second.hessian == "carry"


def test_a_two_stage_run_reports_both_stages():
    result = relax(
        argon(), LennardJones(), stages="two-stage", fmax=1e-4, smax=1e-3, steps=200
    )
    assert result.converged
    assert len(result.stages) == 2
    assert result.evaluations == sum(stage.evaluations for stage in result.stages)
    assert result.steps == sum(stage.steps for stage in result.stages)
    # the first stage leaves the cell alone
    assert result.stages[0].measures.keys() == {"fmax"}
    assert result.stages[1].measures.keys() == {"fmax", "smax"}


def test_both_protocols_find_the_same_minimum():
    single = relax(argon(), LennardJones(), fmax=1e-4, smax=1e-3, steps=200)
    staged = relax(
        argon(), LennardJones(), stages="two-stage", fmax=1e-4, smax=1e-3, steps=200
    )
    assert single.converged and staged.converged
    assert single.energy == pytest.approx(staged.energy, abs=1e-7)


def test_a_custom_stage_list_is_honoured():
    result = relax(
        argon(),
        LennardJones(),
        stages=[
            Stage(cell=False, fmax=0.05, name="settle"),
            Stage(cell=True, fmax=1e-4, smax=1e-3, hessian="carry", name="cell"),
        ],
        steps=200,
    )
    assert result.converged
    assert len(result.stages) == 2


def test_the_carried_curvature_is_the_previous_stage_block():
    result = relax(
        argon(), LennardJones(), stages="two-stage", fmax=1e-3, smax=1e-2, steps=200
    )
    learned = result.stages[0].model
    assert learned.n == 12
    assert result.stages[1].model.n == 18

    # the second stage started from the block the first learned, not from the
    # model Hessian it would otherwise have used
    from chmpy.opt.relax import _extend_hessian

    coordinates = coordinates_for(argon(), cell=True)
    extended = _extend_hessian(learned, coordinates)
    np.testing.assert_allclose(extended[:12, :12], learned.to_dense(), atol=1e-10)


def test_a_stage_with_no_freedoms_is_skipped():
    """fcc argon's atom is on an m-3m site: a fixed-cell stage has nothing to do."""
    from chmpy.core import Element
    from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell

    cell = UnitCell.from_lengths_and_angles((5.0, 5.0, 5.0), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    calc = LennardJones()
    result = relax(crystal, calc, stages="two-stage", fmax=1e-4, smax=1e-3, steps=50)
    assert result.converged
    assert len(result.stages) == 1  # only the variable-cell stage ran


def test_a_structure_with_nothing_to_relax_is_refused():
    from chmpy.core import Element
    from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell

    cell = UnitCell.from_lengths_and_angles((5.0, 5.0, 5.0), np.radians((90, 90, 90)))
    crystal = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element[18]], np.array([[0.0, 0.0, 0.0]])),
    )
    with pytest.raises(ValueError, match="no degrees of freedom"):
        relax(crystal, LennardJones(), cell=False, steps=10)


def test_an_unknown_stages_argument_is_refused():
    with pytest.raises(ValueError, match="stages must be"):
        relax(argon(), LennardJones(), stages="three-stage", steps=1)


def test_a_noisy_energy_does_not_stall_the_optimiser():
    """A float32 model cannot resolve the energy changes near a minimum."""

    class Quantised(LennardJones):
        """A float32 model: its energies ARE float32 values, near -190 eV.

        The offset is not added back afterwards -- doing that in float64 makes
        the returned number a full-precision one again, which is exactly the
        thing being emulated away.
        """

        def compute(self, system, want):
            result = super().compute(system, want)
            return Result(
                energy=float(np.float32(result.energy - 190.0)),
                forces=result.forces,
                stress=result.stress,
                energies=result.energies,
                volume=result.volume,
            )

    calc = Quantised()
    result = relax(argon(), calc, fmax=1e-3, smax=1e-2, steps=300)
    assert result.converged, result
    assert calc.energy_noise(-190.0) > 1e-5


def test_the_energy_precision_of_a_float32_model_is_detected():
    class Quantised(LennardJones):
        def compute(self, system, want):
            result = super().compute(system, want)
            return Result(
                energy=float(np.float32(result.energy - 190.0)),
                forces=result.forces,
                stress=result.stress,
                energies=result.energies,
                volume=result.volume,
            )

    # The inference needs several *distinct* energies. A rigid translation of
    # a periodic cell is not one: it leaves the energy exactly unchanged.
    def feed(calculator):
        rng = np.random.default_rng(0)
        system = argon()
        for _ in range(4):
            calculator.energy(system)
            system.set_positions(system.positions + rng.normal(scale=0.02, size=(4, 3)))

    calc = Quantised()
    feed(calc)
    assert calc.energy_noise(-190.0) == pytest.approx(190.0 * np.finfo(np.float32).eps)

    exact = LennardJones()
    feed(exact)
    assert exact.energy_noise(-190.0) < 1e-12


def test_acetic_acid_two_stage_runs_end_to_end():
    crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
    calc = LennardJones(sigma=1.2, epsilon=1e-3, cutoff=6.0)
    result = relax(crystal, calc, stages="two-stage", fmax=0.5, smax=5.0, steps=40)
    assert len(result.stages) == 2
    assert isinstance(result.structure, Crystal)
    assert result.structure.space_group == crystal.space_group


def test_a_calculator_can_declare_its_own_precision():
    class Coarse(Calculator):
        provides = {"energy"}
        energy_precision = 1e-4

        def compute(self, system, want):
            return Result(energy=-100.0)

    assert Coarse().energy_noise(-100.0) == pytest.approx(1e-2)

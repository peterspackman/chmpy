"""One `progress` argument: silent by default, printed with True, or a hook."""

import numpy as np
import pytest

from chmpy.calc import LennardJones
from chmpy.opt import elastic_tensor, lattice_energy, relax
from chmpy.opt.progress import Progress, reporter
from chmpy.vib import force_constants

from .test_elastic import fcc_argon
from .test_lattice import BondedDiatomics, co_crystal


@pytest.fixture(scope="module")
def argon():
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    return relax(fcc_argon(5.4), calc, fmax=1e-6, smax=1e-5).structure, calc


def test_silence_is_the_default(argon, capsys):
    crystal, calc = argon
    relax(fcc_argon(5.5), calc)
    elastic_tensor(crystal, calc, strain=0.002)
    assert capsys.readouterr().out == ""


def test_true_prints_readable_lines(argon, capsys):
    crystal, calc = argon
    elastic_tensor(crystal, calc, strain=0.002, progress=True)
    out = capsys.readouterr().out
    assert "2 of 6 strains needed" in out
    assert "[1/4] voigt 0 +0.002" in out
    assert "[4/4] voigt 3 -0.002" in out


def test_a_hook_gets_counts_out_of_a_known_total(argon):
    crystal, calc = argon
    events = []
    elastic_tensor(crystal, calc, strain=0.002, progress=events.append)
    points = [event for event in events if event.total is not None]
    assert {event.total for event in points} == {4}
    assert sorted({event.index for event in points}) == [0, 1, 2, 3]
    assert all(isinstance(event, Progress) for event in events)


def test_force_constants_count_their_displacements(argon):
    crystal, calc = argon
    events = []
    force_constants(crystal, calc, cutoff=8.0, progress=events.append)
    displacements = [event for event in events if event.total is not None]
    # fcc: one atom, one direction, by symmetry
    assert [(event.index, event.total) for event in displacements] == [(0, 1)]


def test_lattice_energy_nests_the_relaxations_inside_its_stages():
    events = []
    lattice_energy(
        co_crystal(), BondedDiatomics(), fmax=1e-4, smax=1e-3, progress=events.append
    )
    outline = [event for event in events if event.depth == 0]
    assert [event.stage for event in outline if not event.done] == [
        "crystal",
        "molecule 0",
    ]
    assert {event.total for event in outline} == {2}

    steps = [event for event in events if event.step is not None]
    assert steps, "the relaxations inside should report their steps"
    assert all(event.depth == 2 for event in steps)
    assert {event.parents[0] for event in steps} == {
        ("lattice energy", "crystal"),
        ("lattice energy", "molecule 0"),
    }


def test_anything_else_is_refused():
    with pytest.raises(TypeError, match="progress should be"):
        reporter("loud")
    assert not reporter(None).active
    assert not reporter(False).active
    assert reporter(np.array).active

"""Relax a molecular crystal with PET-MAD: chmpy against the ASE stack.

The comparison is at *equal achieved accuracy*. ASE's cell filters converge on
`max|F|` over an array that mixes eV/A forces with a virial divided by
`cell_factor`, which is not the same criterion as ours and not a physical
quantity, so comparing "steps to converge" between the two would be comparing
different targets. Instead every optimiser is stopped the moment the structure
actually satisfies the same two conditions: a largest force below `FMAX` eV/A
and a largest stress component below `SMAX` GPa.

Do not read the energy column as "this optimiser finds a deeper minimum".
Every structure here is inside the same tolerance box, and the energy above a
minimum is about `g^T H^-1 g / 2` -- for these tolerances a few meV, which is
the spread you see. Driven to their noise floors, all of these agree to under
0.1 meV. What the column is good for is spotting an optimiser that stopped
somewhere else entirely.

    python benchmarks/relax_benchmark.py [structure.cif ...]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from chmpy import Crystal
from chmpy.calc import System
from chmpy.calc.adapters.metatomic import MetatomicCalculator
from chmpy.opt import relax

FMAX = 0.02  # eV/A
SMAX = 0.05  # GPa
MAX_STEPS = 300


def load_pet_mad():
    from pet_mad.calculator import PETMADCalculator

    return PETMADCalculator(version="latest")


def symmetry_error(space_group, cell, fractional, numbers):
    """How far a P1 geometry is from being invariant under a space group, in Angstroms.

    Measured against the ORIGINAL group's operations on the relaxed unit cell.
    Rebuilding the relaxed structure as a P1 crystal and asking whether it is
    invariant under P1 would answer yes for anything.
    """
    worst = 0.0
    for operation in space_group.symmetry_operations:
        image = (fractional @ np.asarray(operation.rotation).T) + operation.translation
        for position, number in zip(image % 1.0, numbers, strict=True):
            difference = np.abs((fractional % 1.0) - position)
            difference = np.minimum(difference, 1.0 - difference)
            distance = np.linalg.norm(difference @ cell, axis=1)
            distance = np.where(numbers == number, distance, np.inf)
            worst = max(worst, float(distance.min()))
    return worst


def geometry_of(system):
    return np.asarray(system.cell), system.scaled_positions, np.asarray(system.numbers)


def run_chmpy(crystal, model, symmetry=True, hessian="model"):
    calc = MetatomicCalculator(model)
    started = time.perf_counter()
    result = relax(
        crystal,
        calc,
        symmetry=symmetry,
        hessian=hessian,
        fmax=FMAX,
        smax=SMAX,
        steps=MAX_STEPS,
    )
    seconds = time.perf_counter() - started
    rebuilds = sum(nl.rebuilds for _, nl in calc._neighbor_lists)
    reuses = sum(nl.reuses for _, nl in calc._neighbor_lists)
    cell, fractional, numbers = geometry_of(
        result.structure
        if isinstance(result.structure, System)
        else System.from_crystal(result.structure)
    )
    return {
        "name": "chmpy TR"
        + ("" if symmetry else " P1")
        + ("" if hessian == "model" else " (identity H)"),
        "converged": result.converged,
        "calls": calc.stats.calls,
        "seconds": seconds,
        "energy": result.energy,
        "fmax": result.measures.get("fmax", 0.0),
        "smax": result.measures.get("smax", 0.0),
        "symmetry": symmetry_error(crystal.space_group, cell, fractional, numbers),
        "neighbours": f"{rebuilds} built, {reuses} reused",
    }


class CountingCalculator:
    """Counts how many times a calculator actually evaluates a geometry."""

    def __init__(self, calculator):
        self.calculator = calculator
        self.calls = 0
        self._calculate = calculator.calculate
        calculator.calculate = self._counted

    def _counted(self, *args, **kwargs):
        self.calls += 1
        return self._calculate(*args, **kwargs)

    def release(self):
        self.calculator.calculate = self._calculate


def run_ase(crystal, pet, optimizer_name="LBFGS", fix_symmetry=True):
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS, FIRE, LBFGS
    from ase.stress import voigt_6_to_full_3x3_stress

    from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

    optimizer = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}[optimizer_name]

    atoms = crystal.to_ase_atoms()
    atoms.calc = pet
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    counter = CountingCalculator(pet)
    started = time.perf_counter()
    opt = optimizer(FrechetCellFilter(atoms), logfile=None)
    converged = False
    for _ in opt.irun(fmax=1e-12, steps=MAX_STEPS):
        # read the criteria off the calculator's own cached results, so the
        # convergence check costs this run nothing it would not otherwise pay
        fmax = float(np.abs(atoms.get_forces()).max())
        stress = voigt_6_to_full_3x3_stress(atoms.get_stress())
        smax = float(np.abs(stress).max()) * EV_PER_ANGSTROM3_TO_GPA
        if fmax <= FMAX and smax <= SMAX:
            converged = True
            break
    seconds = time.perf_counter() - started
    calls = counter.calls
    counter.release()

    energy = float(atoms.get_potential_energy())
    fmax = float(np.abs(atoms.get_forces()).max())
    stress = voigt_6_to_full_3x3_stress(atoms.get_stress())
    smax = float(np.abs(stress).max()) * EV_PER_ANGSTROM3_TO_GPA
    return {
        "name": f"ASE {optimizer_name}{' + FixSymmetry' if fix_symmetry else ''}",
        "converged": converged,
        "calls": calls,
        "seconds": seconds,
        "energy": energy,
        "fmax": fmax,
        "smax": smax,
        "symmetry": symmetry_error(
            crystal.space_group,
            np.asarray(atoms.cell),
            atoms.get_scaled_positions(),
            atoms.numbers,
        ),
        "neighbours": f"{calls} built, 0 reused",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structures", nargs="*", help="CIF files to relax")
    args = parser.parse_args()

    paths = args.structures
    if not paths:
        from chmpy.tests import TEST_FILES

        paths = [str(TEST_FILES["acetic_acid.cif"])]

    pet = load_pet_mad()
    for path in paths:
        crystal = Crystal.load(path)
        print(
            f"\n{Path(path).name}: {crystal.space_group.symbol}, "
            f"{len(crystal.unit_cell_atoms()['element'])} atoms per cell, "
            f"{len(crystal.asymmetric_unit)} in the asymmetric unit"
        )
        print(f"converged when fmax <= {FMAX} eV/A and smax <= {SMAX} GPa\n")
        header = (
            f"{'method':28s} {'conv':5s} {'calls':>6s} {'time (s)':>9s} "
            f"{'energy (eV)':>14s} {'fmax':>8s} {'smax':>8s} {'sym err/A':>10s}"
            f"  {'neighbour lists':s}"
        )
        print(header)
        print("-" * len(header))
        for row in (
            run_chmpy(crystal, pet._model, symmetry=True),
            run_chmpy(crystal, pet._model, symmetry=True, hessian="identity"),
            run_chmpy(crystal, pet._model, symmetry=False),
            run_ase(crystal, pet, "LBFGS", fix_symmetry=True),
            run_ase(crystal, pet, "LBFGS", fix_symmetry=False),
            run_ase(crystal, pet, "FIRE", fix_symmetry=True),
        ):
            print(
                f"{row['name']:28s} {str(row['converged']):5s} {row['calls']:6d} "
                f"{row['seconds']:9.2f} {row['energy']:14.6f} {row['fmax']:8.4f} "
                f"{row['smax']:8.4f} {row['symmetry']:10.2e}  {row['neighbours']}"
            )


if __name__ == "__main__":
    main()

"""Lattice energies of the X23 set, with two optimisers driving the same model.

The lattice energy is a difference between two separately relaxed things, so it
inherits the error of both. That makes it a sharper test of an optimiser than a
single relaxation: a crystal left 5 meV above its minimum and a molecule left
5 meV above its own move the answer by up to `2 * 5 / Z` meV, and the two do
not cancel.

The comparison is at *equal achieved accuracy*, as in `relax_benchmark.py`.
ASE's cell filters converge on `max|F|` over an array mixing eV/A forces with a
virial divided by `cell_factor`, which is neither our criterion nor a physical
quantity, so "steps to converge" would be comparing different targets. Both
optimisers are instead stopped the moment the structure satisfies the same two
conditions -- largest force below `fmax` eV/A, largest stress component below
`smax` GPa -- and the molecules likewise on force alone.

Both paths are given the same model object, so the call counts are counted at
the model and mean the same thing.

    python benchmarks/lattice_benchmark.py --model mace-polar-m
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np

from chmpy import Crystal
from chmpy.opt import lattice_energy
from chmpy.util.unit import EV_TO_KJ_PER_MOL

DATA = Path(__file__).parent / "data"
MAX_STEPS = 500


def reference_energies(path):
    """The X23b lattice energies, in kJ/mol, keyed by structure name."""
    values = {}
    with open(path) as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            values[row["name"]] = (
                float(row["lattice_energy_ref_kJ_mol"]),
                float(row["monomer_correction_kJ_mol"]),
            )
    return values


class CountingCalculator:
    """Counts how many times the model actually evaluates a geometry."""

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


def _converged(atoms, fmax, smax, cell):
    """Our two criteria, read off results the optimiser has already paid for."""
    from ase.stress import voigt_6_to_full_3x3_stress

    from chmpy.calc.result import EV_PER_ANGSTROM3_TO_GPA

    force = float(np.abs(atoms.get_forces()).max())
    if not cell:
        return force <= fmax, force, 0.0
    stress = voigt_6_to_full_3x3_stress(atoms.get_stress())
    pressure = float(np.abs(stress).max()) * EV_PER_ANGSTROM3_TO_GPA
    return (force <= fmax and pressure <= smax), force, pressure


def _relax_with_ase(atoms, fmax, smax, cell, steps):
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    target = FrechetCellFilter(atoms) if cell else atoms
    optimizer = LBFGS(target, logfile=None)
    done, force, pressure = _converged(atoms, fmax, smax, cell)
    if not done:
        for _ in optimizer.irun(fmax=1e-12, steps=steps):
            done, force, pressure = _converged(atoms, fmax, smax, cell)
            if done:
                break
    return done, force, pressure


def ase_lattice_energy(crystal, model, info, fmax, smax, steps, fix_symmetry=True):
    """The recipe of `chmpy.opt.lattice_energy`, driven by ASE's optimisers.

    The molecules are taken from the *relaxed* crystal, as `lattice_energy`
    takes them, so that the frozen energies measure the conformation the
    crystal actually settled into. Taking them from the experimental structure
    instead leaves the X-ray hydrogen positions in the reference and reports
    hundreds of kJ/mol of strain that is really just a bad starting geometry.

    A relaxed cell rebuilt through ASE has no symmetry left on it, so the
    molecules are matched back to the original cell by position in the list --
    the atom order survives the round trip, so the connected components come
    out in the same order -- and checked by composition.
    """
    from ase.constraints import FixSymmetry

    in_cell = crystal.unit_cell_molecules()
    unique = crystal.symmetry_unique_molecules()
    counts = np.bincount(
        [m.properties["asym_mol_idx"] for m in in_cell], minlength=len(unique)
    )
    z = len(in_cell)

    def single_point(molecule):
        gas = molecule.to_ase_atoms()
        gas.calc = model
        gas.info.update(info or {})
        return gas, float(gas.get_potential_energy())

    atoms = crystal.to_ase_atoms()
    atoms.calc = model
    atoms.info.update(info or {})
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))
    crystal_ok, force, pressure = _relax_with_ase(atoms, fmax, smax, True, steps)
    crystal_energy = float(atoms.get_potential_energy())

    settled = Crystal.from_ase_atoms(atoms).unit_cell_molecules()
    if len(settled) != z:
        raise ValueError(
            f"the relaxed cell separates into {len(settled)} molecules where the "
            f"experimental one gave {z}; the relaxation changed the bonding"
        )
    representatives = {}
    for original, current in zip(in_cell, settled, strict=True):
        if sorted(original.atomic_numbers) != sorted(current.atomic_numbers):
            raise ValueError("molecules did not survive the round trip in order")
        representatives.setdefault(original.properties["asym_mol_idx"], current)

    relaxed, frozen, molecules_ok = [], [], True
    for index in range(len(unique)):
        gas, energy = single_point(representatives[index])
        frozen.append(energy)
        ok, _, _ = _relax_with_ase(gas, fmax, smax, False, steps)
        molecules_ok = molecules_ok and ok
        relaxed.append(float(gas.get_potential_energy()))

    total_relaxed = float(np.dot(counts, relaxed))
    total_frozen = float(np.dot(counts, frozen))
    return {
        "energy": (crystal_energy - total_relaxed) / z,
        "interaction": (crystal_energy - total_frozen) / z,
        "strain": (total_frozen - total_relaxed) / z,
        "crystal_energy": crystal_energy,
        "molecule_energies": relaxed,
        "converged": crystal_ok and molecules_ok,
        "z": z,
        "fmax": force,
        "smax": pressure,
    }


def run_one(crystal, model, calculator, info, args):
    """Both optimisers on one structure, each counted at the model."""
    row = {}
    for label in ("chmpy", "ase"):
        counter = CountingCalculator(model)
        started = time.perf_counter()
        try:
            if label == "chmpy":
                result = lattice_energy(
                    crystal,
                    calculator,
                    info=info,
                    fmax=args.fmax,
                    smax=args.smax,
                    steps=args.steps,
                )
                values = {
                    "energy": result.energy,
                    "interaction": result.interaction,
                    "strain": result.strain,
                    "molecule_energies": result.molecule_energies,
                    "converged": result.converged,
                    "z": result.z,
                }
            else:
                values = ase_lattice_energy(
                    crystal, model, info, args.fmax, args.smax, args.steps
                )
        except Exception as error:  # a failure on one structure is data, not a stop
            values = {"energy": float("nan"), "error": f"{type(error).__name__}: {error}"}
        elapsed = time.perf_counter() - started
        calls = counter.calls
        counter.release()
        row[label] = {**values, "calls": calls, "seconds": elapsed}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mace-polar-m")
    parser.add_argument("--fmax", type=float, default=0.01, help="eV/A")
    parser.add_argument("--smax", type=float, default=0.05, help="GPa")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--structures", type=Path, default=DATA / "x23")
    parser.add_argument("--reference", type=Path, default=DATA / "x23_reference.csv")
    parser.add_argument("--out", type=Path, default=Path("lattice_benchmark.csv"))
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from elastic_benchmark import load_model

    reference = reference_energies(args.reference)
    info = (
        {"charge": 0, "spin": 1, "external_field": [0.0, 0.0, 0.0]}
        if args.model.startswith("mace-polar")
        else None
    )
    calculator = load_model(args.model)
    model = calculator.ase_calculator

    fields = [
        "name", "z", "reference", "chmpy", "ase", "difference",
        "chmpy_interaction", "ase_interaction", "chmpy_strain", "ase_strain",
        "chmpy_calls", "ase_calls", "chmpy_seconds", "ase_seconds",
        "chmpy_converged", "ase_converged",
    ]
    handle = open(args.out, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()

    header = f"{'structure':20s} {'ref':>8s} {'chmpy':>8s} {'ase':>8s} {'diff':>7s} {'calls c/a':>12s}"
    print(header)
    print("-" * len(header))
    for path in sorted(args.structures.glob("*.cif")):
        name = path.stem
        if args.only and name not in args.only:
            continue
        crystal = Crystal.load(str(path))
        row = run_one(crystal, model, calculator, info, args)
        ours = row["chmpy"]["energy"] * EV_TO_KJ_PER_MOL
        theirs = row["ase"]["energy"] * EV_TO_KJ_PER_MOL
        ref = reference.get(name, (float("nan"), 0.0))[0]
        record = {
            "name": name,
            "z": row["chmpy"].get("z", 0),
            "reference": ref,
            "chmpy": ours,
            "ase": theirs,
            "difference": ours - theirs,
            "chmpy_interaction": row["chmpy"].get("interaction", float("nan")) * EV_TO_KJ_PER_MOL,
            "ase_interaction": row["ase"].get("interaction", float("nan")) * EV_TO_KJ_PER_MOL,
            "chmpy_strain": row["chmpy"].get("strain", float("nan")) * EV_TO_KJ_PER_MOL,
            "ase_strain": row["ase"].get("strain", float("nan")) * EV_TO_KJ_PER_MOL,
            "chmpy_calls": row["chmpy"]["calls"],
            "ase_calls": row["ase"]["calls"],
            "chmpy_seconds": round(row["chmpy"]["seconds"], 2),
            "ase_seconds": round(row["ase"]["seconds"], 2),
            "chmpy_converged": row["chmpy"].get("converged", False),
            "ase_converged": row["ase"].get("converged", False),
        }
        writer.writerow(record)
        handle.flush()
        print(
            f"{name:20s} {ref:8.1f} {ours:8.1f} {theirs:8.1f} {ours - theirs:7.2f} "
            f"{row['chmpy']['calls']:5d}/{row['ase']['calls']:<6d}"
        )
        for label in ("chmpy", "ase"):
            if "error" in row[label]:
                print(f"    {label} failed: {row[label]['error']}")
    handle.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

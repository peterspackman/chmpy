"""Elastic tensor from FHI-aims, with the strained cells run as independent jobs.

Two steps, with the DFT run however you like in between:

    python aims_elastic.py setup relaxed.cif -d work --species ~/aims/species_defaults \\
        --xc pbe --keyword "xdm 0.69125110 1.57470830"
    # ... run aims in every directory listed in work/jobs.txt ...
    python aims_elastic.py collect work --json

`setup` writes one directory per strained cell (`voigt0_plus`, `voigt0_minus`,
...), plus `reference/` for a single point of the unstrained cell, which
measures the residual stress. Only the strains the point group needs are
written: 4 cells for cubic, 6 for hexagonal or trigonal, 12 for triclinic.
Every directory shares one control.in, so the basis and k-grid are identical
across strains. The k-grid gives at least `--k-length` Angstrom (default 10)
of cell per k-point along each direction of the reference cell; `--k-grid`
sets it outright. The default strain is the one `elastic_tensor` uses with
MLIPs on molecular crystals. `plan.json` keeps the strain and the point-group rotations for
`collect`.

Each strained cell relaxes its atoms at fixed cell (`relax_geometry`, no
`relax_unit_cell`) and reports its final analytical stress. Pass `--clamped`
for single points instead, which gives clamped-ion constants.

The structure must already be relaxed with the same settings, cell included;
elastic constants are second derivatives at a minimum. A `geometry.in` (or
`geometry.in.next_step`) from that relaxation can be given directly; its
symmetry is detected.

With SLURM, one array task per directory:

    #SBATCH --array=1-N            # N = lines in jobs.txt
    cd "$(sed -n "${SLURM_ARRAY_TASK_ID}p" work/jobs.txt)"
    srun aims.x > aims.out
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chmpy import Crystal
from chmpy.calc import System
from chmpy.fmt.aims import (
    AimsOutput,
    generate_control_in,
    k_grid_for,
    system_to_geometry_string,
)
from chmpy.opt import cartesian_rotations, elastic_from_stresses, strain_points
from chmpy.opt.elastic import DEFAULT_STRAIN

PLAN = "plan.json"
REFERENCE = "reference"


def load_structure(path, tolerance):
    path = Path(path)
    if path.name.startswith("geometry.in"):
        crystal = Crystal.from_aims_file(str(path))
    else:
        crystal = Crystal.load(str(path))
    return crystal.detect_symmetry(tolerance=tolerance)


def control_text(crystal, args, k_grid, relax_atoms):
    keywords = {"compute_analytical_stress": ".true."}
    if relax_atoms:
        # the stress is only as converged as the forces the atoms relaxed to
        keywords["sc_accuracy_forces"] = f"{args.fmax / 10:g}"
    for item in args.keyword:
        key, _, value = item.partition(" ")
        keywords[key] = value.strip() or True
    return generate_control_in(
        crystal,
        species_defaults_dir=args.species,
        basis=args.basis,
        xc=args.xc,
        k_grid=k_grid,
        relax_geometry={"method": "trm", "convergence": args.fmax}
        if relax_atoms
        else False,
        extra_keywords=keywords,
    )


def setup(args):
    crystal = load_structure(args.structure, args.symprec)
    print(f"Loaded {crystal}, space group {crystal.space_group.symbol}")
    k_grid = tuple(args.k_grid or k_grid_for(crystal.unit_cell.direct, args.k_length))
    relax_atoms = not args.clamped
    rotations = cartesian_rotations(crystal) if args.symmetry else np.zeros((0, 3, 3))
    points = strain_points(crystal, strain=args.strain, rotations=rotations)

    work = Path(args.destination)
    work.mkdir(parents=True, exist_ok=True)
    jobs = {REFERENCE: (System.from_crystal(crystal), False)}
    jobs.update({p.name: (p.system, relax_atoms) for p in points})
    for name, (system, relaxed) in jobs.items():
        directory = work / name
        directory.mkdir(exist_ok=True)
        (directory / "geometry.in").write_text(system_to_geometry_string(system))
        (directory / "control.in").write_text(
            control_text(crystal, args, k_grid, relaxed)
        )

    (work / "jobs.txt").write_text(
        "".join(f"{(work / name).resolve()}\n" for name in jobs)
    )
    plan = {
        "structure": str(Path(args.structure).resolve()),
        "space_group": crystal.space_group.symbol,
        "strain": args.strain,
        "k_grid": list(k_grid),
        "basis": args.basis,
        "xc": args.xc,
        "relaxed_atoms": relax_atoms,
        "rotations": rotations.tolist(),
        "points": [{"name": p.name, "index": p.index, "sign": p.sign} for p in points],
    }
    (work / PLAN).write_text(json.dumps(plan, indent=2))
    indices = sorted({p.index for p in points})
    print(
        f"Wrote {len(jobs)} aims inputs to {work}/ (Voigt strains {indices} at "
        f"+/-{args.strain:g}, plus the reference), k-grid "
        f"{' '.join(map(str, k_grid))}; directories in {work}/jobs.txt"
    )


def read_stress(directory, output, relaxed):
    "The final stress of one job, or the reason there isn't a usable one"
    path = directory / output
    if not path.exists():
        return None, "no output"
    result = AimsOutput.from_file(path)
    if not result.finished_normally:
        return None, "did not finish"
    if result.scf_converged is False:
        return None, "SCF did not converge"
    if relaxed and not result.geometry_converged:
        return None, "atomic relaxation did not converge"
    if result.stress is None:
        return None, "no stress printed (compute_analytical_stress?)"
    return result.stress, None


def collect(args):
    work = Path(args.work)
    plan = json.loads((work / PLAN).read_text())
    relaxed = plan["relaxed_atoms"]

    stresses, problems = {}, []
    for point in plan["points"]:
        stress, problem = read_stress(work / point["name"], args.output, relaxed)
        if problem:
            problems.append(f"  {point['name']}: {problem}")
        else:
            stresses[(point["index"], point["sign"])] = stress
    if problems:
        print("Not every strained cell has a usable stress:", *problems, sep="\n")
        sys.exit(1)

    reference, problem = read_stress(work / REFERENCE, args.output, False)
    if problem:
        print(f"reference: {problem}; the residual stress is not reported")

    result = elastic_from_stresses(
        stresses,
        strain=plan["strain"],
        rotations=np.array(plan["rotations"]),
        reference_stress=reference,
        relaxed_atoms=relaxed,
    )
    print(result)
    if reference is not None:
        print(f"residual stress of the reference: {result.residual_stress:.3f} GPa")
    print("\nC (GPa, Voigt order xx yy zz yz xz xy):")
    for row in result.c_voigt:
        print("  " + " ".join(f"{value:9.2f}" for value in row))

    np.savetxt(work / "elastic_GPa.txt", result.c_voigt, fmt="%10.3f")
    if args.json:
        averages = result.tensor.averages()
        summary = {
            "structure": plan["structure"],
            "space_group": plan["space_group"],
            "c_voigt_GPa": result.c_voigt.tolist(),
            "independent_constants": result.n_independent,
            "strains_applied": list(result.strains),
            "strain": result.strain,
            "relaxed_atoms": relaxed,
            "stable": bool(result.tensor.is_stable()),
            "noise_fraction": result.noise_fraction,
            "residual_stress_GPa": result.residual_stress
            if reference is not None
            else None,
            "bulk_modulus_hill_GPa": averages["bulk_modulus_avg"]["hill"],
            "shear_modulus_hill_GPa": averages["shear_modulus_avg"]["hill"],
            "youngs_modulus_hill_GPa": averages["youngs_modulus_avg"]["hill"],
        }
        (work / "elastic.json").write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {work}/elastic.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("setup", help="write the aims inputs")
    make.add_argument("structure", help="relaxed structure: CIF or geometry.in")
    make.add_argument("-d", "--destination", default="elastic", help="work directory")
    make.add_argument("--species", required=True, help="aims species_defaults dir")
    make.add_argument("--basis", default="lightdense")
    make.add_argument("--xc", default="pbe")
    make.add_argument(
        "--k-length",
        type=float,
        default=10.0,
        help="Angstrom of cell per k-point along each axis (default 10)",
    )
    make.add_argument(
        "--k-grid", type=int, nargs=3, help="explicit k-grid, overriding --k-length"
    )
    make.add_argument(
        "--strain",
        type=float,
        default=DEFAULT_STRAIN,
        help=f"engineering strain (default {DEFAULT_STRAIN:g})",
    )
    make.add_argument(
        "--fmax", type=float, default=1e-3, help="relax_geometry tolerance, eV/A"
    )
    make.add_argument(
        "--keyword",
        action="append",
        default=[],
        help='extra control.in line, e.g. "xdm 0.69 1.57" (repeatable)',
    )
    make.add_argument(
        "--clamped", action="store_true", help="single points: clamped-ion constants"
    )
    make.add_argument(
        "--no-symmetry",
        dest="symmetry",
        action="store_false",
        help="all six strains, no projection onto the point group",
    )
    make.add_argument(
        "--symprec", type=float, default=0.01, help="symmetry tolerance, A"
    )
    make.set_defaults(run=setup)

    gather = commands.add_parser("collect", help="fit the tensor to the stresses")
    gather.add_argument("work", help="the work directory written by setup")
    gather.add_argument("--output", default="aims.out", help="aims output file name")
    gather.add_argument("--json", action="store_true", help="write elastic.json")
    gather.set_defaults(run=collect)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

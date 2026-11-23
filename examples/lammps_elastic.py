"""Finite-temperature elastic tensor from LAMMPS MD, one job per strained cell.

Three steps, with LAMMPS run however you like in between:

    python lammps_elastic.py setup benzene.cif -d work -T 94
    # ... run LAMMPS in work/npt ...
    python lammps_elastic.py strain work
    # ... run LAMMPS in every directory listed in work/jobs.txt ...
    python lammps_elastic.py collect work --json

`setup` writes `npt/`, an NPT run at zero pressure with all six cell
components free that ends by averaging the cell. `strain` applies the strains
the point group needs to that mean cell, or with `--input-cell` to the
structure's own cell (e.g. a measured low-temperature one), and writes an NVT
run for each strained cell and for the unstrained `reference/`. Each NVT run
equilibrates, then averages the pressure tensor; `collect` fits the tensor.

Data files hold one unit cell; LAMMPS replicates it to at least
`--min-length` Angstrom in every direction. `--pair` gives the potential
commands, one per use, with `{elements}` for the elements in type order. The
default is PET-MAD with D3(BJ) at PBEsol parameters, PET-MAD's training
functional.

The mean stresses are noisy. `collect` reports their standard error, from
1000-step blocks, and the resulting error on the constants; a bigger
supercell, longer runs or a larger strain reduce it.

With SLURM, one array task per directory:

    #SBATCH --array=1-N            # N = lines in jobs.txt
    cd "$(sed -n "${SLURM_ARRAY_TASK_ID}p" work/jobs.txt)"
    srun lmp -in in.lmp
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chmpy import Crystal
from chmpy.calc import System
from chmpy.fmt.lammps import (
    PRESSURE_UNITS,
    lammps_cell,
    parse_ave_time_string,
    stress_from_pressure,
    system_to_data_string,
    to_lammps_input,
    type_elements,
)
from chmpy.opt import cartesian_rotations, elastic_from_stresses, strain_points
from chmpy.opt.elastic import DEFAULT_STRAIN

PLAN = "plan.json"
NPT = "npt"
REFERENCE = "reference"
DATA = "cell.data"
INPUT = "in.lmp"
CELL_AVERAGE = "cell.avg"
PRESSURE_AVERAGE = "pressure.avg"
GPA = 160.21766208

DEFAULT_PAIR = ["pair_style pet pet-mad-s d3 pbesol", "pair_coeff * * {elements}"]

# samples every 10 steps, averaged in blocks of 1000 steps
AVERAGE_EVERY = "10 100 1000"


def load_structure(path, tolerance):
    path = Path(path)
    if path.name.startswith("geometry.in"):
        crystal = Crystal.from_aims_file(str(path))
    else:
        crystal = Crystal.load(str(path))
    return crystal.detect_symmetry(tolerance=tolerance)


def replication(cell, min_length):
    "Copies of the cell along each axis so every width is at least min_length"
    volume = abs(np.linalg.det(cell))
    a, b, c = cell
    widths = volume / np.linalg.norm(
        [np.cross(b, c), np.cross(c, a), np.cross(a, b)], axis=1
    )
    return [int(np.ceil(min_length / w)) for w in widths]


def steps(picoseconds, timestep_fs):
    return int(round(picoseconds * 1000 / timestep_fs))


def md_input(plan, sections):
    md = plan["md"]
    start = {
        "supercell": "replicate {} {} {}".format(*plan["replicate"]),
        "dynamics": (
            f"timestep {md['timestep_fs'] / 1000:g}\n"
            f"velocity all create {md['temperature']:g} {md['seed']} "
            "mom yes rot yes dist gaussian\n"
            f"thermo {steps(1.0, md['timestep_fs'])}"
        ),
    }
    return to_lammps_input(
        DATA,
        "\n".join(plan["pair"]),
        plan["elements"],
        sections={**start, **sections},
        preamble="\n".join(plan["preamble"]),
    )


def npt_sections(md):
    dt = md["timestep_fs"] / 1000  # ps
    return {
        "equilibrate": (
            f"fix md all npt temp {md['temperature']:g} {md['temperature']:g} "
            f"{100 * dt:g} tri 0.0 0.0 {1000 * dt:g}\n"
            f"run {steps(md['npt_equilibrate_ps'], md['timestep_fs'])}"
        ),
        "average the cell": (
            "variable lx equal lx\nvariable ly equal ly\nvariable lz equal lz\n"
            "variable xy equal xy\nvariable xz equal xz\nvariable yz equal yz\n"
            f"fix cell all ave/time {AVERAGE_EVERY} v_lx v_ly v_lz v_xy v_xz v_yz "
            f"file {CELL_AVERAGE}\n"
            f"run {steps(md['npt_average_ps'], md['timestep_fs'])}"
        ),
    }


def nvt_sections(md):
    dt = md["timestep_fs"] / 1000  # ps
    return {
        "equilibrate": (
            f"fix md all nvt temp {md['temperature']:g} {md['temperature']:g} "
            f"{100 * dt:g}\n"
            f"run {steps(md['nvt_equilibrate_ps'], md['timestep_fs'])}"
        ),
        "average the pressure": (
            "compute p all pressure thermo_temp\n"
            f"fix stress all ave/time {AVERAGE_EVERY} "
            "c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6] "
            f"file {PRESSURE_AVERAGE}\n"
            f"run {steps(md['nvt_average_ps'], md['timestep_fs'])}"
        ),
    }


def write_job(directory, system, plan, sections):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / DATA).write_text(
        system_to_data_string(system, species=plan["elements"])
    )
    (directory / INPUT).write_text(md_input(plan, sections))


def setup(args):
    crystal = load_structure(args.structure, args.symprec)
    print(f"Loaded {crystal}, space group {crystal.space_group.symbol}")
    system = System.from_crystal(crystal)
    work = Path(args.destination)
    plan = {
        "structure": str(Path(args.structure).resolve()),
        "space_group": crystal.space_group.symbol,
        "symprec": args.symprec,
        "symmetry": args.symmetry,
        "strain": args.strain,
        "replicate": replication(system.cell, args.min_length),
        "elements": type_elements(system.numbers),
        "pair": args.pair or DEFAULT_PAIR,
        "preamble": args.preamble,
        "md": {
            "temperature": args.temperature,
            "timestep_fs": args.timestep,
            "seed": args.seed,
            "npt_equilibrate_ps": args.npt_equilibrate,
            "npt_average_ps": args.npt_average,
            "nvt_equilibrate_ps": args.nvt_equilibrate,
            "nvt_average_ps": args.nvt_average,
        },
    }
    write_job(work / NPT, system, plan, npt_sections(plan["md"]))
    (work / PLAN).write_text(json.dumps(plan, indent=2))
    n = len(system) * int(np.prod(plan["replicate"]))
    print(
        f"Wrote {work / NPT}/ ({'x'.join(map(str, plan['replicate']))} supercell, "
        f"{n} atoms, {args.temperature:g} K); run it, then `strain {work}`"
    )


def mean_cell(work, plan, crystal):
    "The NPT run's mean unit cell, in the crystal's own orientation"
    path = work / NPT / CELL_AVERAGE
    if not path.exists():
        sys.exit(f"{path} not found: run the NPT job first, or pass --input-cell")
    columns = parse_ave_time_string(path.read_text())
    lx, ly, lz, xy, xz, yz = (
        columns[f"v_{k}"].mean() for k in ("lx", "ly", "lz", "xy", "xz", "yz")
    )
    supercell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    lower = supercell / np.array(plan["replicate"])[:, None]
    _, rotation = lammps_cell(crystal.unit_cell.direct)
    return lower @ rotation, len(columns["v_lx"])


def strain(args):
    work = Path(args.work)
    plan = json.loads((work / PLAN).read_text())
    crystal = load_structure(plan["structure"], plan["symprec"])
    system = System.from_crystal(crystal)
    if args.input_cell:
        plan["cell_source"] = "input"
    else:
        cell, blocks = mean_cell(work, plan, crystal)
        change = np.linalg.det(cell) / np.linalg.det(system.cell) - 1
        print(
            f"NPT mean cell from {blocks} blocks: volume {100 * change:+.2f}% vs input"
        )
        system.set_cell(cell, scale_atoms=True)
        plan["cell_source"] = "npt"
    plan["cell"] = np.array(system.cell).tolist()

    rotations = (
        cartesian_rotations(crystal) if plan["symmetry"] else np.zeros((0, 3, 3))
    )
    points = strain_points(system, strain=plan["strain"], rotations=rotations)
    jobs = {REFERENCE: system, **{p.name: p.system for p in points}}
    for name, job in jobs.items():
        write_job(work / name, job, plan, nvt_sections(plan["md"]))

    (work / "jobs.txt").write_text(
        "".join(f"{(work / name).resolve()}\n" for name in jobs)
    )
    plan["rotations"] = rotations.tolist()
    plan["points"] = [
        {"name": p.name, "index": p.index, "sign": p.sign} for p in points
    ]
    plan["frames"] = {
        name: lammps_cell(job.cell)[1].tolist() for name, job in jobs.items()
    }
    (work / PLAN).write_text(json.dumps(plan, indent=2))
    indices = sorted({p.index for p in points})
    print(
        f"Wrote {len(jobs)} NVT jobs to {work}/ (Voigt strains {indices} at "
        f"+/-{plan['strain']:g}, plus the reference); directories in {work}/jobs.txt"
    )


def read_stress(work, name, plan):
    "Mean stress (eV/A^3) and its standard error, or the reason there isn't one"
    path = work / name / PRESSURE_AVERAGE
    if not path.exists():
        return None, None, "no pressure.avg"
    columns = parse_ave_time_string(path.read_text())
    blocks = np.column_stack([columns[f"c_p[{k}]"] for k in range(1, 7)])
    if len(blocks) < 2:
        return None, None, f"{len(blocks)} averaging block(s)"
    rotation = np.array(plan["frames"][name])
    mean = stress_from_pressure(blocks.mean(axis=0), rotation=rotation)
    error = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    return mean, float(error.max() * PRESSURE_UNITS["metal"]), None


def collect(args):
    work = Path(args.work)
    plan = json.loads((work / PLAN).read_text())
    if "points" not in plan:
        sys.exit(f"no strained cells in {work}: run `strain` first")

    stresses, errors, problems = {}, [], []
    for point in plan["points"]:
        stress, error, problem = read_stress(work, point["name"], plan)
        if problem:
            problems.append(f"  {point['name']}: {problem}")
        else:
            stresses[(point["index"], point["sign"])] = stress
            errors.append(error)
    if problems:
        print("Not every strained cell has a usable stress:", *problems, sep="\n")
        sys.exit(1)
    reference, _, problem = read_stress(work, REFERENCE, plan)
    if problem:
        print(f"reference: {problem}; the residual stress is not reported")

    result = elastic_from_stresses(
        stresses,
        strain=plan["strain"],
        rotations=np.array(plan["rotations"]),
        reference_stress=reference,
        relaxed_atoms=True,
    )
    stress_error = max(errors) * GPA
    constant_error = stress_error / (np.sqrt(2) * plan["strain"])
    print(result)
    print(
        f"largest standard error of a mean stress: {stress_error:.4f} GPa, "
        f"about {constant_error:.2f} GPa on the constants"
    )
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
            "temperature_K": plan["md"]["temperature"],
            "cell_source": plan["cell_source"],
            "cell": plan["cell"],
            "pair": plan["pair"],
            "c_voigt_GPa": result.c_voigt.tolist(),
            "independent_constants": result.n_independent,
            "strains_applied": list(result.strains),
            "strain": result.strain,
            "stable": bool(result.tensor.is_stable()),
            "noise_fraction": result.noise_fraction,
            "stress_standard_error_GPa": stress_error,
            "constant_standard_error_GPa": constant_error,
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

    make = commands.add_parser("setup", help="write the NPT job")
    make.add_argument("structure", help="relaxed structure: CIF or geometry.in")
    make.add_argument("-d", "--destination", default="elastic", help="work directory")
    make.add_argument(
        "-T", "--temperature", type=float, default=94.0, help="K (default 94)"
    )
    make.add_argument(
        "--pair",
        action="append",
        help="potential command, repeatable; default: " + "; ".join(DEFAULT_PAIR),
    )
    make.add_argument(
        "--preamble",
        action="append",
        default=[],
        help="command before read_data, e.g. a package line (repeatable)",
    )
    make.add_argument(
        "--min-length", type=float, default=25.0, help="smallest supercell width, A"
    )
    make.add_argument(
        "--strain",
        type=float,
        default=DEFAULT_STRAIN,
        help=f"engineering strain (default {DEFAULT_STRAIN:g})",
    )
    make.add_argument("--timestep", type=float, default=0.5, help="fs (default 0.5)")
    make.add_argument("--seed", type=int, default=4928)
    make.add_argument("--npt-equilibrate", type=float, default=20.0, help="ps")
    make.add_argument("--npt-average", type=float, default=20.0, help="ps")
    make.add_argument("--nvt-equilibrate", type=float, default=5.0, help="ps")
    make.add_argument("--nvt-average", type=float, default=50.0, help="ps")
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

    deform = commands.add_parser("strain", help="write the strained NVT jobs")
    deform.add_argument("work", help="the work directory written by setup")
    deform.add_argument(
        "--input-cell",
        action="store_true",
        help="strain the structure's own cell instead of the NPT mean cell",
    )
    deform.set_defaults(run=strain)

    gather = commands.add_parser("collect", help="fit the tensor to the stresses")
    gather.add_argument("work", help="the work directory")
    gather.add_argument("--json", action="store_true", help="write elastic.json")
    gather.set_defaults(run=collect)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

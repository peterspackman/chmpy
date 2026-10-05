"""Compute the elastic tensor of a crystal from a CIF using MACE-Polar.

    python elastic_tensor.py structure.cif [--device cuda] [--strain auto] [--json]

The crystal is relaxed at variable cell within its space group (elastic
constants should be evaluated at a stress-free minimum), then `elastic_tensor`
applies the symmetry-independent strains, relaxes the atoms at each, and fits
the stress response.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from chmpy import Crystal
from chmpy.calc import Calculator, System
from chmpy.core import Element
from chmpy.opt import elastic_tensor, relax
from chmpy.vib import christoffel_velocities, density

#: MACE-Polar is a charge- and field-aware model; these go to it on every call
NEUTRAL_SINGLET = {"charge": 0, "spin": 1, "external_field": [0.0, 0.0, 0.0]}


def strain_size(text):
    "Accept a number or 'auto'"
    return text if text == "auto" else float(text)


def along_the_axes(crystal, result):
    """Young's modulus, linear compressibility and sound velocities along a, b, c."""
    cell = np.asarray(crystal.unit_cell.direct)
    axes = cell / np.linalg.norm(cell, axis=1)[:, None]
    tensor = result.tensor

    system = System.from_crystal(crystal)
    masses = [Element.from_atomic_number(int(z)).mass for z in system.numbers]
    rho = density(masses, system.cell)

    rows = []
    for name, axis, young, compressibility in zip(
        "abc",
        axes,
        tensor.youngs_modulus(axes),
        tensor.linear_compressibility(axes),
        strict=True,
    ):
        velocities = christoffel_velocities(result.c_voigt, rho, axis)
        rows.append(
            {
                "axis": name,
                "youngs_modulus_GPa": float(young),
                "linear_compressibility_per_TPa": float(compressibility),
                "sound_velocities_m_s": [float(v) for v in velocities],
            }
        )
    return rho, rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cif", help="crystal structure (CIF)")
    parser.add_argument("--model", default="polar-1-m", help="MACE-Polar model")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cuda")
    parser.add_argument(
        "--strain",
        type=strain_size,
        default=0.01,
        help="engineering strain, or 'auto' to grow it until the tensor's own "
        "noise estimate is acceptable",
    )
    parser.add_argument(
        "--fmax", type=float, default=1e-3, help="force tolerance, eV/A"
    )
    parser.add_argument(
        "--smax", type=float, default=0.01, help="stress tolerance, GPa"
    )
    parser.add_argument(
        "--no-relax", action="store_true", help="the CIF is already relaxed"
    )
    parser.add_argument("--destination", "-d", help="directory for the outputs")
    parser.add_argument("--json", action="store_true", help="write a JSON summary")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    from mace.calculators import mace_polar

    calc = Calculator.from_ase(
        mace_polar(model=args.model, device=args.device, default_dtype="float64")
    )
    progress = not args.quiet

    crystal = Crystal.load(args.cif)
    print(f"Loaded {crystal}")

    if not args.no_relax:
        relaxation = relax(
            crystal,
            calc,
            info=NEUTRAL_SINGLET,
            fmax=args.fmax,
            smax=args.smax,
            steps=500,
            progress=progress,
        )
        print(relaxation)
        if not relaxation.converged:
            print("WARNING: the reference structure did not relax fully")
        crystal = relaxation.structure

    result = elastic_tensor(
        crystal,
        calc,
        strain=args.strain,
        info=NEUTRAL_SINGLET,
        fmax=args.fmax,
        progress=progress,
    )
    print()
    print(result)
    print("\nC (GPa, Voigt order xx yy zz yz xz xy):")
    for row in result.c_voigt:
        print("  " + " ".join(f"{value:9.2f}" for value in row))

    rho, axes = along_the_axes(crystal, result)
    print(f"\ndensity {rho:.1f} kg/m^3")
    print(f"{'axis':>4} {'E (GPa)':>9} {'beta (1/TPa)':>13}  sound velocities (m/s)")
    for row in axes:
        velocities = " ".join(f"{v:7.0f}" for v in row["sound_velocities_m_s"])
        print(
            f"{row['axis']:>4} {row['youngs_modulus_GPa']:9.2f} "
            f"{row['linear_compressibility_per_TPa']:13.2f}  {velocities}"
        )

    stem = Path(args.cif).stem
    destination = Path(args.destination) if args.destination else Path(".")
    if args.destination:
        destination.mkdir(parents=True, exist_ok=True)
        crystal.save(str(destination / f"{stem}_opt.cif"))
        np.savetxt(
            destination / f"{stem}_elastic_GPa.txt", result.c_voigt, fmt="%10.3f"
        )
        print(f"\nWrote the relaxed structure and tensor to {destination}/")

    if args.json:
        averages = result.tensor.averages()
        summary = {
            "structure": str(Path(args.cif).resolve()),
            "model": args.model,
            "space_group": crystal.space_group.symbol,
            "c_voigt_GPa": result.c_voigt.tolist(),
            "independent_constants": result.n_independent,
            "strains_applied": list(result.strains),
            "strain": result.strain,
            "stable": bool(result.tensor.is_stable()),
            "noise_fraction": result.noise_fraction,
            "residual_stress_GPa": result.residual_stress,
            "bulk_modulus_hill_GPa": averages["bulk_modulus_avg"]["hill"],
            "shear_modulus_hill_GPa": averages["shear_modulus_avg"]["hill"],
            "youngs_modulus_hill_GPa": averages["youngs_modulus_avg"]["hill"],
            "density_kg_m3": rho,
            "along_axes": axes,
            "evaluations": result.evaluations,
        }
        path = destination / f"{stem}_elastic.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote results to {path}")


if __name__ == "__main__":
    main()

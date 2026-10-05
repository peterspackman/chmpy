"""Compute the lattice energy of a molecular crystal from a CIF using MACE-Polar.

    python lattice_energy.py structure.cif [--device cuda] [--destination out/] [--json]

Lattice energy per molecule = (E_crystal - sum of gas-phase monomer energies) / Z.
The crystal is relaxed at variable cell within its space group and each
symmetry-unique molecule in the gas phase, all via `Calculator.lattice_energy`.
"""

import argparse
import json
from pathlib import Path

from chmpy import Crystal
from chmpy.calc import Calculator

#: MACE-Polar is a charge- and field-aware model; these go to it on every call
NEUTRAL_SINGLET = {"charge": 0, "spin": 1, "external_field": [0.0, 0.0, 0.0]}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cif", help="crystal structure (CIF)")
    parser.add_argument("--model", default="polar-1-m", help="MACE-Polar model")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cuda")
    parser.add_argument(
        "--fmax", type=float, default=1e-3, help="force tolerance, eV/A"
    )
    parser.add_argument(
        "--smax", type=float, default=0.05, help="stress tolerance, GPa"
    )
    parser.add_argument("--destination", "-d", help="directory for relaxed geometries")
    parser.add_argument("--json", action="store_true", help="write a JSON summary")
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args()

    from mace.calculators import mace_polar

    calc = Calculator.from_ase(
        mace_polar(model=args.model, device=args.device, default_dtype="float64")
    )

    crystal = Crystal.load(args.cif)
    print(f"Loaded {crystal}")

    result = calc.lattice_energy(
        crystal,
        info=NEUTRAL_SINGLET,
        fmax=args.fmax,
        smax=args.smax,
        progress=not args.quiet,
    )
    print()
    print(result)
    if not result.converged:
        print(f"NOT CONVERGED: {', '.join(result.unconverged)}")

    stem = Path(args.cif).stem
    destination = Path(args.destination) if args.destination else Path(".")
    if args.destination:
        destination.mkdir(parents=True, exist_ok=True)
        result.structure.save(str(destination / f"{stem}_opt.cif"))
        for index, molecule in enumerate(result.molecules):
            molecule.save(str(destination / f"{stem}_monomer_{index}.xyz"))
        print(f"\nWrote relaxed geometries to {destination}/")

    if args.json:
        summary = {
            "structure": str(Path(args.cif).resolve()),
            "model": args.model,
            "converged": result.converged,
            "unconverged": result.unconverged,
            "z": result.z,
            "z_prime": result.z_prime,
            "crystal_energy_eV": result.crystal_energy,
            "monomers": [
                {
                    "index": index,
                    "formula": molecule.molecular_formula,
                    "multiplicity": multiplicity,
                    "energy_eV": energy,
                }
                for index, (molecule, multiplicity, energy) in enumerate(
                    zip(
                        result.molecules,
                        result.multiplicities,
                        result.molecule_energies,
                        strict=True,
                    )
                )
            ],
            "lattice_energy_eV": result.energy,
            "lattice_energy_kJ_mol": result.kj_per_mol,
            "interaction_kJ_mol": result.interaction_kj_per_mol,
            "strain_kJ_mol": result.strain_kj_per_mol,
            "evaluations": result.evaluations,
        }
        path = destination / f"{stem}_lattice_energy.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote results to {path}")


if __name__ == "__main__":
    main()

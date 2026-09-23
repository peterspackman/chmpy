"""Elastic tensors of molecular crystals, against experiment and a reference set.

Runs `chmpy.opt.elastic_tensor` over the compounds of the elastic-tensors2025
study, which carries 81 experimental tensors for 45 molecular crystals along
with tensors computed by several other methods.

    python benchmarks/elastic_benchmark.py --data ~/git/elastic-tensors2025 \
        --model pet-mad --max-atoms 100

Frames: `Crystal.load` puts a crystal in chmpy's standard setting, and the
reference tensors are reoriented into the same one, so the comparison is
direct. For orthorhombic and higher that setting is the crystallographic one;
monoclinic and triclinic entries are reported but flagged, since a published
tensor's frame is not always recoverable.
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np

from chmpy import Crystal
from chmpy.opt import elastic_tensor, relax

UNAMBIGUOUS = {"orthorhombic", "tetragonal", "hexagonal", "cubic"}


def load_model(name):
    """Return a chmpy Calculator for a named model."""
    warnings.filterwarnings("ignore")
    if name == "pet-mad":
        from pet_mad.calculator import PETMADCalculator

        from chmpy.calc.adapters.metatomic import MetatomicCalculator

        return MetatomicCalculator(PETMADCalculator(version="latest")._model)
    if name.startswith("mace-polar"):
        from mace.calculators import mace_polar

        from chmpy.calc import Calculator

        size = name.rsplit("-", 1)[-1] if name != "mace-polar" else "s"
        return Calculator.from_ase(
            mace_polar(
                model=f"polar-1-{size}", device="cpu", default_dtype="float64"
            )
        )
    if name.startswith("mace-off"):
        from mace.calculators import mace_off

        from chmpy.calc import Calculator

        size = name.split("-")[-1] if name.count("-") > 1 else "small"
        return Calculator.from_ase(
            mace_off(model=size, device="cpu", default_dtype="float64")
        )
    raise SystemExit(f"unknown model {name!r}")


def reference_tensors(data: Path):
    """(refcode -> {label: 6x6}) for experiment, and for the study's PET-MAD."""
    experiment, petmad = {}, {}
    for path in sorted((data / "data/elastic-tensors/experiment").glob("*.txt")):
        refcode, _, label = path.stem.partition("_")
        experiment.setdefault(refcode, {})[label] = np.loadtxt(path)
    for directory in sorted((data / "data/elastic-tensors/petmad/results").iterdir()):
        path = directory / "pet-mad_fd" / "elastic_tensor.txt"
        if path.exists():
            petmad[directory.name] = np.loadtxt(path)
    return experiment, petmad


def moduli(c_voigt):
    from chmpy.ext.elastic_tensor import ElasticTensor

    averages = ElasticTensor(c_voigt).averages()
    return (
        averages["bulk_modulus_avg"]["hill"],
        averages["shear_modulus_avg"]["hill"],
    )


def forbidden_norm(c_voigt, rotations):
    """How much of a tensor lies outside what its point group allows, in GPa."""
    from chmpy.opt.strain import (
        elastic_from_voigt,
        elastic_to_voigt,
        invariant_elastic_basis,
        project_elastic,
    )

    basis = invariant_elastic_basis(rotations)
    projected = elastic_to_voigt(project_elastic(elastic_from_voigt(c_voigt), basis))
    return float(np.abs(projected - c_voigt).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="pet-mad")
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--strain", type=float, default=0.02)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--ion-fmax", type=float, default=0.005,
                        help="ionic relaxation tolerance at each strain; must sit\n                              well below the calculator's force noise")
    parser.add_argument("--smax", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("elastic_benchmark.csv"))
    parser.add_argument("--only", nargs="*", help="limit to these refcodes")
    parser.add_argument("--info", action="store_true",
                        help="pass charge/spin/external_field, which polarisable "
                             "models require")
    args = parser.parse_args()

    model_inputs = (
        {"charge": 0, "spin": 1, "external_field": [0.0, 0.0, 0.0]}
        if args.info or args.model.startswith("mace-polar")
        else None
    )
    experiment, reference = reference_tensors(args.data)
    cifs = args.data / "data/cifs/original"
    calc = load_model(args.model)

    from chmpy.opt.strain import cartesian_rotations

    jobs = []
    for refcode in sorted(experiment):
        if args.only and refcode not in args.only:
            continue
        path = cifs / f"{refcode}.cif"
        if not path.exists():
            continue
        crystal = Crystal.load(str(path))
        n_atoms = len(crystal.unit_cell_atoms()["element"])
        if n_atoms > args.max_atoms:
            continue
        jobs.append((refcode, crystal, n_atoms))
    jobs.sort(key=lambda j: j[2])

    print(f"{len(jobs)} compounds with {args.model}, up to {args.max_atoms} atoms\n")
    header = (
        f"{'refcode':12s} {'lattice':13s} {'N':>4s} {'calls':>6s} {'time':>7s} "
        f"{'K_ours':>7s} {'G_ours':>7s} {'K_exp':>7s} {'G_exp':>7s} "
        f"{'K_ref':>7s} {'noise%':>7s}"
    )
    print(header)
    print("-" * len(header))

    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["refcode", "lattice", "n_atoms", "calls", "seconds", "converged",
             "K_ours", "G_ours", "K_exp", "G_exp", "K_ref", "G_ref",
             "rmse_vs_exp", "rmse_vs_ref", "forbidden_ref", "symmetry_residual",
             "frame_unambiguous"]
        )
        for refcode, crystal, n_atoms in jobs:
            lattice = crystal.space_group.lattice_type
            try:
                started = time.perf_counter()
                relaxation = relax(
                    crystal, calc, info=model_inputs,
                    fmax=args.fmax, smax=args.smax, steps=300,
                )
                result = elastic_tensor(
                    relaxation.structure, calc, strain=args.strain,
                    info=model_inputs, fmax=args.ion_fmax, steps=150,
                )
                elapsed = time.perf_counter() - started
            except Exception as exc:
                print(f"{refcode:12s} {lattice:13s} {n_atoms:4d}  failed: "
                      f"{type(exc).__name__}: {str(exc)[:50]}")
                continue

            ours = result.c_voigt
            k_ours, g_ours = moduli(ours)
            rotations = cartesian_rotations(relaxation.structure)

            exp = experiment.get(refcode, {})
            k_exp = g_exp = rmse_exp = None
            if exp:
                best = min(exp.values(), key=lambda t: np.abs(t - ours).max())
                k_exp, g_exp = moduli(best)
                rmse_exp = float(np.sqrt(np.mean((best - ours) ** 2)))

            ref = reference.get(refcode)
            k_ref = g_ref = rmse_ref = forbidden = None
            if ref is not None:
                k_ref, g_ref = moduli(ref)
                rmse_ref = float(np.sqrt(np.mean((ref - ours) ** 2)))
                forbidden = forbidden_norm(ref, rotations)

            def fmt(x, width=7):
                return f"{x:{width}.2f}" if x is not None else " " * (width - 1) + "-"

            print(f"{refcode:12s} {lattice:13s} {n_atoms:4d} "
                  f"{result.evaluations + relaxation.evaluations:6d} {elapsed:6.1f}s "
                  f"{fmt(k_ours)} {fmt(g_ours)} {fmt(k_exp)} {fmt(g_exp)} "
                  f"{fmt(k_ref)} {100 * result.noise_fraction:6.1f}%")

            writer.writerow([
                refcode, lattice, n_atoms,
                result.evaluations + relaxation.evaluations, round(elapsed, 2),
                relaxation.converged, k_ours, g_ours, k_exp, g_exp, k_ref, g_ref,
                rmse_exp, rmse_ref, forbidden, result.symmetry_residual,
                lattice in UNAMBIGUOUS,
            ])
            handle.flush()

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

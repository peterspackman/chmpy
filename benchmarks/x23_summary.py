"""Summarise a lattice_benchmark.py run over the X23 set.

Two questions, which are not the same one:

* Do the optimisers agree? That is about the machinery, and the answer should
  be "to well under the tolerance", because both are being driven to the same
  fmax and smax on the same model.
* Does the model agree with X23b? That is about MACE-Polar, and the answer is
  whatever it is.

    python benchmarks/x23_summary.py lattice_benchmark.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np


def load(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def statistics(predicted, reference):
    error = predicted - reference
    return {
        "n": len(error),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "r": float(np.corrcoef(predicted, reference)[0, 1]),
    }


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "lattice_benchmark.csv")
    rows = load(path)
    names = [r["name"] for r in rows]
    ours = np.array([float(r["chmpy"]) for r in rows])
    theirs = np.array([float(r["ase"]) for r in rows])
    reference = np.array([float(r["reference"]) for r in rows])
    our_calls = np.array([int(r["chmpy_calls"]) for r in rows])
    their_calls = np.array([int(r["ase_calls"]) for r in rows])
    our_time = np.array([float(r["chmpy_seconds"]) for r in rows])
    their_time = np.array([float(r["ase_seconds"]) for r in rows])

    print(f"{len(rows)} structures from {path}\n")

    print("== the two optimisers against each other ==")
    difference = ours - theirs
    print(f"  max |difference|   {np.abs(difference).max():8.4f} kJ/mol  ({names[int(np.abs(difference).argmax())]})")
    print(f"  mean |difference|  {np.abs(difference).mean():8.4f} kJ/mol")
    print(f"  rms difference     {np.sqrt((difference**2).mean()):8.4f} kJ/mol")

    print("\n== each against the X23b reference ==")
    print(f"  {'':10s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'r':>7s}")
    for label, values in (("chmpy", ours), ("ase", theirs)):
        s = statistics(values, reference)
        print(f"  {label:10s} {s['mae']:7.2f} {s['rmse']:7.2f} {s['bias']:7.2f} {s['r']:7.3f}")

    print("\n== cost ==")
    print(f"  {'':10s} {'calls':>8s} {'median':>8s} {'seconds':>9s}")
    print(f"  {'chmpy':10s} {our_calls.sum():8d} {np.median(our_calls):8.0f} {our_time.sum():9.0f}")
    print(f"  {'ase':10s} {their_calls.sum():8d} {np.median(their_calls):8.0f} {their_time.sum():9.0f}")
    ratio = their_calls / np.maximum(our_calls, 1)
    print(f"  ASE needs {their_calls.sum() / max(our_calls.sum(), 1):.2f}x the calls in total, "
          f"median {np.median(ratio):.2f}x per structure")
    wins = int((our_calls < their_calls).sum())
    print(f"  chmpy cheaper on {wins}/{len(rows)}, equal on {int((our_calls == their_calls).sum())}")

    unconverged = [
        (r["name"], label)
        for r in rows
        for label in ("chmpy", "ase")
        if r[f"{label}_converged"] != "True"
    ]
    if unconverged:
        print("\n== did not reach tolerance ==")
        for name, label in unconverged:
            print(f"  {name} ({label})")

    print("\n== worst disagreements with the reference ==")
    order = np.argsort(-np.abs(ours - reference))
    print(f"  {'structure':20s} {'ref':>8s} {'chmpy':>8s} {'error':>8s}")
    for index in order[:6]:
        print(f"  {names[index]:20s} {reference[index]:8.1f} {ours[index]:8.1f} "
              f"{ours[index] - reference[index]:+8.1f}")


if __name__ == "__main__":
    main()

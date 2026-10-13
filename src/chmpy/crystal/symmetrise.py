"""Restore space group symmetry to a relaxed copy of a crystal's unit cell.

Codes that relax in P1 return the unit cell with symmetry slightly broken.
Given the original crystal and the relaxed cell with its atoms in
`Crystal.unit_cell_atoms()` order, the metric is averaged over the point
group and each asymmetric unit atom is averaged over its images mapped back
through their generating operations.
"""

from __future__ import annotations

import numpy as np

from .asymmetric_unit import AsymmetricUnit
from .symmetry_operation import SymmetryOperation
from .unit_cell import UnitCell


def symmetrise_unit_cell(
    reference, unit_cell: UnitCell, frac_positions, max_deviation: float = 0.1
):
    """A crystal with the symmetry of `reference` and geometry of a relaxed cell.

    Args:
        reference: Crystal with the space group and asymmetric unit
        unit_cell: the relaxed unit cell
        frac_positions: (N, 3) relaxed fractional positions of the unit cell
            atoms, in the order of `reference.unit_cell_atoms()`
        max_deviation: largest allowed displacement (Angstroms) of any atom
            on symmetrising

    Returns:
        Crystal in the reference space group

    Raises:
        ValueError: when the atoms do not match the reference unit cell or
            the relaxed structure departs from the symmetry by more than
            `max_deviation`
    """
    from .crystal import Crystal

    atoms = reference.unit_cell_atoms()
    frac_positions = np.asarray(frac_positions, dtype=float)
    if frac_positions.shape != atoms["frac_pos"].shape:
        raise ValueError(
            f"Expected {len(atoms['frac_pos'])} unit cell atoms, "
            f"got {len(frac_positions)}"
        )

    ops = reference.space_group.symmetry_operations
    rotations = [np.round(op.rotation).astype(int) for op in ops]
    metric = np.mean([r.T @ unit_cell.metric_tensor @ r for r in rotations], axis=0)
    lengths = np.sqrt(np.diag(metric))
    angles = np.arccos(
        [
            metric[1, 2] / (lengths[1] * lengths[2]),
            metric[0, 2] / (lengths[0] * lengths[2]),
            metric[0, 1] / (lengths[0] * lengths[1]),
        ]
    )
    cell = UnitCell.from_lengths_and_angles(lengths, angles)

    asym = reference.asymmetric_unit
    totals = np.zeros_like(asym.positions)
    counts = np.zeros(len(asym.positions))
    for x, a, code in zip(
        frac_positions, atoms["asym_atom"], atoms["symop"], strict=True
    ):
        candidate = SymmetryOperation.from_integer_code(int(code)).inverse()(
            x[None, :]
        )[0]
        candidate += np.round(asym.positions[a] - candidate)
        totals[a] += candidate
        counts[a] += 1
    if np.any(counts == 0):
        raise ValueError("Some asymmetric unit atoms have no unit cell image")
    positions = totals / counts[:, None]

    result = Crystal(
        cell,
        reference.space_group,
        AsymmetricUnit(asym.elements, positions, labels=asym.labels),
        titl=reference.titl,
    )
    new = result.unit_cell_atoms()["frac_pos"]
    diff = new - frac_positions
    diff -= np.round(diff)
    deviation = np.linalg.norm(diff @ cell.direct, axis=1).max()
    if deviation > max_deviation:
        raise ValueError(
            f"Relaxed structure departs from {reference.space_group.symbol} "
            f"by {deviation:.3f} A (max {max_deviation} A)"
        )
    return result

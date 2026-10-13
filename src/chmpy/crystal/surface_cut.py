"""Molecular pairs broken by a surface cut, for surface and attachment energies.

For a cut at height c (in lattice-plane periods) on (hkl), a pair whose lower
molecule has centroid height z_i in [c, c + 1) and upper molecule z_j crosses
the infinite set of planes c + n exactly floor(z_j - c) times per surface
cell, so summing over one layer of molecules counts each broken pair once.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from chmpy.core.molecule import Molecule

from .slab import Termination, _surface_units, terminations
from .surface_symmetry import SurfaceSymmetry


@dataclass
class DimerCutCounts:
    """Pairs broken per surface cell by a cut on (hkl).

    Attributes:
        hkl: Miller indices
        offset: cut height in lattice-plane periods
        area: surface cell area (Angstroms^2)
        molecules_per_layer: molecules in one lattice-plane period of the
            surface cell
        unique_dimers: the crystal's symmetry unique dimers
        counts: (n_unique,) number of broken pairs of each unique dimer
    """

    hkl: tuple[int, int, int]
    offset: float
    area: float
    molecules_per_layer: int
    unique_dimers: list
    counts: np.ndarray

    def cleavage_energy(self, energies) -> float:
        "Energy to split the crystal along the cut, per surface cell."
        return -float(self.counts @ np.asarray(energies))

    def surface_energy(self, energies) -> float:
        "Unrelaxed surface energy (energy units per Angstrom^2)."
        return self.cleavage_energy(energies) / (2 * self.area)

    def attachment_energy(self, energies) -> float:
        "Energy per molecule of attaching a slice one lattice-plane period thick."
        return float(self.counts @ np.asarray(energies)) / self.molecules_per_layer


def dimer_cut_counts(
    crystal, hkl, termination: int | Termination = 0, radius: float = 3.8
) -> DimerCutCounts:
    """Count the symmetry unique dimers broken by a cut on (hkl).

    Args:
        crystal: a molecular crystal
        hkl: Miller indices
        termination: index into `terminations(crystal, hkl, True)` or a
            Termination
        radius: dimer cutoff, as in `Crystal.symmetry_unique_dimers`

    Returns:
        DimerCutCounts indexed like `crystal.symmetry_unique_dimers(radius)`
    """
    from chmpy.core.dimer import Dimer

    if isinstance(termination, int):
        termination = terminations(crystal, hkl, keep_molecules=True)[termination]
    offset = termination.offset
    symmetry = SurfaceSymmetry.from_crystal(crystal, hkl)
    vectors = np.array(symmetry.surface_vectors, dtype=float) @ crystal.unit_cell.direct
    unique_dimers, _ = crystal.symmetry_unique_dimers(radius=radius)

    units = _surface_units(crystal, symmetry, keep_molecules=True)
    molecules, heights = [], []
    for unit in units:
        positions = unit.positions.copy()
        positions[:, 2] -= np.floor(unit.reference[2] - offset)
        molecules.append(Molecule.from_arrays(unit.elements, positions @ vectors))
        heights.append(positions[:, 2].mean())
    centroids = np.array([m.centroid for m in molecules])
    extent = max(
        np.linalg.norm(m.positions - m.centroid, axis=1).max() for m in molecules
    )

    volume = abs(np.linalg.det(vectors))
    reach = radius + 2 * extent
    n = [
        int(
            np.ceil(
                reach
                * np.linalg.norm(np.cross(vectors[(k + 1) % 3], vectors[(k + 2) % 3]))
                / volume
            )
        )
        for k in range(3)
    ]
    shifts = np.array(list(product(*(range(-m, m + 1) for m in n))))

    counts = np.zeros(len(unique_dimers), dtype=int)
    for i, mol_i in enumerate(molecules):
        for j, mol_j in enumerate(molecules):
            for shift in shifts:
                z_j = heights[j] + shift[2]
                if z_j <= heights[i]:
                    continue
                cart_shift = shift @ vectors
                if np.linalg.norm(centroids[j] + cart_shift - centroids[i]) > reach:
                    continue
                crossings = int(np.floor(z_j - offset))
                if crossings == 0:
                    continue
                partner = mol_j.translated(cart_shift)
                r = mol_i.distance_to(partner, method="nearest_atom")
                if not 1e-1 < r < radius:
                    continue
                dimer = Dimer(mol_i, partner, separation=r)
                for u, reference in enumerate(unique_dimers):
                    if abs(reference.separation - r) < 1e-3 and reference == dimer:
                        counts[u] += crossings
                        break
                else:
                    raise ValueError(f"Pair at {r:.3f} matches no unique dimer")

    return DimerCutCounts(
        hkl=symmetry.hkl,
        offset=offset,
        area=symmetry.plane.area,
        molecules_per_layer=len(molecules),
        unique_dimers=unique_dimers,
        counts=counts,
    )

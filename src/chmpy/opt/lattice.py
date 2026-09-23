"""Lattice energy: what it costs to take a crystal apart.

The energy of the crystal, less the energy of its molecules relaxed on their
own, per molecule. Negative for anything that holds together.

    result = lattice_energy(crystal, calculator)
    print(result)            # kJ/mol, with the decomposition
    result.structure         # the relaxed crystal

Both sides are relaxed by default, which is what makes the number comparable
with a sublimation enthalpy and with other calculations. Relaxing neither gives
the interaction energy of the structure as supplied, which is a different and
also useful quantity -- see `relax`.

The result separates two contributions that are often conflated:

    E_lattice = E_interaction + E_strain

`E_interaction` is what the molecules gain by being packed together, measured
with each molecule held in the geometry it has in the crystal.
`E_strain` is what they pay to adopt that geometry rather than their relaxed
one, and it is positive by construction. A rigid molecule has none; a flexible
one can have tens of kJ/mol, and a lattice energy quoted without saying which
of the two it is can be out by that much.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from chmpy.util.unit import EV_TO_KJ_PER_MOL

from .relax import relax

LOG = logging.getLogger(__name__)

#: A molecule this far from being a closed shell probably was not separated
#: correctly, so say so rather than return a confident number.
SUSPICIOUS_MOLECULE_ATOMS = 2


@dataclass
class LatticeEnergy:
    """A lattice energy and the parts it is made of.

    Attributes:
        energy: lattice energy per molecule, eV. Negative for a bound crystal.
        interaction: the part from packing, with molecules held at their
            crystal geometry, eV per molecule
        strain: the part paid to adopt the crystal conformation, eV per
            molecule, positive by construction
        crystal_energy: total energy of the relaxed unit cell, eV
        molecule_energies: relaxed gas-phase energy of each unique molecule, eV
        frozen_energies: energy of each unique molecule at its crystal
            geometry, eV
        multiplicities: how many of each unique molecule the cell holds
        z: molecules per unit cell
        z_prime: symmetry-unique molecules per cell
        structure: the relaxed crystal
        molecules: the relaxed gas-phase molecules
        converged: whether every relaxation reached its tolerance. False means
            the energies are those of structures still on their way downhill,
            and the difference between two of those is not a lattice energy.
        unconverged: which relaxations fell short, as readable names
        evaluations: calculator evaluations used
    """

    energy: float
    interaction: float
    strain: float
    crystal_energy: float
    molecule_energies: list
    frozen_energies: list
    multiplicities: list
    z: int
    z_prime: int
    structure: object = None
    molecules: list = field(default_factory=list)
    converged: bool = True
    unconverged: list = field(default_factory=list)
    evaluations: int = 0

    @property
    def kj_per_mol(self) -> float:
        "Lattice energy per molecule in kJ/mol, the usual unit for it"
        return self.energy * EV_TO_KJ_PER_MOL

    @property
    def interaction_kj_per_mol(self) -> float:
        return self.interaction * EV_TO_KJ_PER_MOL

    @property
    def strain_kj_per_mol(self) -> float:
        return self.strain * EV_TO_KJ_PER_MOL

    def __repr__(self) -> str:
        warning = "" if self.converged else " NOT CONVERGED"
        return (
            f"<LatticeEnergy{warning} {self.kj_per_mol:.2f} kJ/mol per molecule, "
            f"Z={self.z} Z'={self.z_prime}, {self.evaluations} evaluations>"
        )

    def __str__(self) -> str:
        return "\n".join(
            [
                repr(self),
                f"  interaction {self.interaction_kj_per_mol:9.2f} kJ/mol",
                f"  strain      {self.strain_kj_per_mol:9.2f} kJ/mol",
                f"  lattice     {self.kj_per_mol:9.2f} kJ/mol",
            ]
        )


def lattice_energy(
    crystal,
    calculator,
    *,
    relax_crystal: bool = True,
    relax_molecules: bool = True,
    box: float | None = None,
    info=None,
    fmax: float = 0.01,
    smax: float = 0.05,
    steps: int = 300,
    logger=None,
    **kwargs,
) -> LatticeEnergy:
    """The lattice energy of a molecular crystal.

    Args:
        crystal: a `Crystal` whose molecules can be separated
        calculator: a `chmpy.calc.Calculator`
        relax_crystal: relax the crystal before taking its energy. With this
            off the crystal is used as supplied, which is what you want when
            comparing a set of structures at fixed geometry.
        relax_molecules: relax each unique molecule in the gas phase. With this
            off the result is the interaction energy and the strain is zero.
        box: put each isolated molecule in a cubic cell of this size in
            Angstroms, for models that require a periodic system. The default
            leaves them genuinely isolated.
        info: model inputs that are not geometry, e.g. a charge or a spin
        fmax: force convergence, eV/A
        smax: stress convergence for the crystal, GPa
        steps: iteration cap for each relaxation
        logger: called with a line per stage
        **kwargs: passed to `relax`

    Returns:
        LatticeEnergy
    """
    started = calculator.stats.calls

    unique = crystal.symmetry_unique_molecules()
    in_cell = crystal.unit_cell_molecules()
    _check_separation(crystal, unique, in_cell)

    counts = np.bincount(
        [m.properties["asym_mol_idx"] for m in in_cell], minlength=len(unique)
    )
    z = len(in_cell)

    unconverged = []
    if relax_crystal:
        outcome = relax(
            crystal, calculator, info=info, fmax=fmax, smax=smax, steps=steps, **kwargs
        )
        if not outcome.converged:
            unconverged.append("crystal")
            LOG.warning(
                "the crystal relaxation did not converge (fmax %.4g eV/A, "
                "smax %.4g GPa); the lattice energy is that of an unrelaxed "
                "structure",
                outcome.measures.get("fmax", float("nan")),
                outcome.measures.get("smax", float("nan")),
            )
        crystal = outcome.structure
        crystal_energy = outcome.energy
        # the relaxation can change the bonding, and a cell that collapsed is
        # not caught by anything downstream: the arithmetic still works, and
        # the molecules cut out of the wreckage give a confident nonsense
        # number rather than an error
        _check_still_molecular(crystal, z)
        unique = crystal.symmetry_unique_molecules()
        if logger is not None:
            logger(f"crystal: {outcome}")
    else:
        from chmpy.calc.system import System

        crystal_energy = calculator.energy(System.from_crystal(crystal, **(info or {})))

    frozen, relaxed, molecules = [], [], []
    for index, molecule in enumerate(unique):
        frozen.append(_molecule_energy(molecule, calculator, box, info))
        if relax_molecules:
            outcome = relax(
                molecule, calculator, info=info, fmax=fmax, steps=steps, **kwargs
            )
            if not outcome.converged:
                unconverged.append(f"molecule {index}")
                LOG.warning(
                    "molecule %d did not relax to fmax %g eV/A; its strain "
                    "energy is a lower bound",
                    index,
                    fmax,
                )
            relaxed.append(outcome.energy)
            molecules.append(outcome.structure)
            if logger is not None:
                logger(f"molecule {index}: {outcome}")
        else:
            relaxed.append(frozen[-1])
            molecules.append(molecule)

    total_frozen = float(np.dot(counts, frozen))
    total_relaxed = float(np.dot(counts, relaxed))
    interaction = (crystal_energy - total_frozen) / z
    strain = (total_frozen - total_relaxed) / z

    return LatticeEnergy(
        energy=(crystal_energy - total_relaxed) / z,
        interaction=interaction,
        strain=strain,
        crystal_energy=crystal_energy,
        molecule_energies=relaxed,
        frozen_energies=frozen,
        multiplicities=[int(c) for c in counts],
        z=z,
        z_prime=len(unique),
        structure=crystal,
        molecules=molecules,
        converged=not unconverged,
        unconverged=unconverged,
        evaluations=calculator.stats.calls - started,
    )


def _molecule_energy(molecule, calculator, box, info):
    """The energy of one molecule on its own, optionally in a large box."""
    from chmpy.calc.system import System

    system = System.from_molecule(molecule, **(info or {}))
    if box is not None:
        positions = np.asarray(system.positions)
        centred = positions - positions.mean(axis=0) + 0.5 * box
        system = System(
            system.numbers, centred, np.eye(3) * box, True, dict(system.info)
        )
    return calculator.energy(system)


def _check_separation(crystal, unique, in_cell) -> None:
    """Complain when the crystal did not come apart into sensible molecules.

    A lattice energy is only meaningful if the thing being separated is what
    the name says. An extended solid has no molecules to take away, and a
    bond-perception failure leaves atoms in no molecule at all or breaks one
    into a shower of fragments. Each is caught here rather than allowed to
    become a confident number.
    """
    if not in_cell:
        raise ValueError(
            f"{crystal} did not separate into molecules, so it has no lattice "
            "energy in this sense; a framework or ionic solid needs a different "
            "reference"
        )
    cell_atoms = len(crystal.unit_cell_atoms()["element"])
    covered = sum(len(molecule) for molecule in in_cell)
    if covered != cell_atoms:
        raise ValueError(
            f"the molecules found cover {covered} of {cell_atoms} atoms in the "
            "cell, so the separation is incomplete; check the bonding"
        )
    cut = _cut_bonds(crystal)
    if cut > 0:
        raise ValueError(
            f"separating {crystal} into molecules would break {cut:.0f} covalent "
            "bond(s), so it is a chain, sheet or framework rather than a molecular "
            "crystal; a lattice energy measured against isolated molecules would "
            "mean nothing here"
        )
    sizes = [len(molecule) for molecule in unique]
    # An atomic solid really is one atom per "molecule", and its lattice energy
    # is the cohesive energy, so say nothing. A mixture of a large molecule and
    # a stray atom or two is the shape a bond-perception failure takes.
    if (
        min(sizes) <= SUSPICIOUS_MOLECULE_ATOMS
        and max(sizes) > 2 * SUSPICIOUS_MOLECULE_ATOMS
    ):
        LOG.warning(
            "the smallest molecule has %d atom(s) beside one of %d; if that is "
            "not intended, the bond perception has fragmented the structure",
            min(sizes),
            max(sizes),
        )


def _cut_bonds(crystal, bond_tolerance: float = 0.4) -> float:
    """How many covalent bonds separating the cell into molecules would break.

    Every bond in the periodic structure is counted, then every bond the
    extracted molecules account for. A molecular crystal balances: each bond
    lies inside one molecule. A chain, a sheet or a framework does not, because
    the bonds that carry on into the next cell belong to no molecule.

    Counting atoms cannot do this. Four parallel covalent chains look exactly
    like four molecules, and the bonded neighbour of an atom in a chain may be
    that same atom one cell over, which the unit-cell bond graph does not
    record at all.
    """
    from scipy.spatial import KDTree

    from chmpy.core import Element

    slab = crystal.slab(bounds=((-1, -1, -1), (1, 1, 1)))
    n_uc = slab["n_uc"]
    positions = crystal.to_cartesian(slab["frac_pos"])
    radii = np.array([Element.from_atomic_number(x).cov for x in slab["element"]])

    cell = KDTree(positions[:n_uc])
    neighbourhood = KDTree(positions)
    reach = 2 * radii.max() + bond_tolerance
    candidates = cell.sparse_distance_matrix(
        neighbourhood, max_distance=reach, output_type="ndarray"
    )

    i, j, distance = candidates["i"], candidates["j"], candidates["v"]
    bonded = (distance > 1e-3) & (distance < radii[i] + radii[j] + bond_tolerance)
    # every bond is found from both of its ends, the far one through the image
    # of its atom in the surrounding cells
    periodic = np.count_nonzero(bonded) / 2

    molecular = sum(
        len(molecule.unique_bonds) for molecule in crystal.unit_cell_molecules()
    )
    return periodic - molecular


def _check_still_molecular(crystal, z: int) -> None:
    """Complain when relaxation destroyed the molecules it was meant to keep.

    A machine-learned potential has no repulsive wall unless it was given one,
    so a cell that collapses does not blow up -- it runs off to a large
    negative energy that looks like a very stable crystal. The molecules cut
    out of it are then whatever the bonding says, and the lattice energy comes
    back enormous and negative with nothing else out of place.
    """
    settled = crystal.unit_cell_molecules()
    if len(settled) != z:
        raise ValueError(
            f"the relaxed cell separates into {len(settled)} molecules where the "
            f"input gave {z}; the relaxation changed the bonding, so whatever it "
            "converged to is not the same crystal"
        )
    cut = _cut_bonds(crystal)
    if cut > 0:
        raise ValueError(
            f"the relaxed cell would need {cut:.0f} covalent bond(s) broken to "
            "separate into molecules; it has collapsed into an extended "
            "structure rather than relaxed"
        )

"""Symmetry of the vibrations at Gamma, and which are infrared or Raman active.

At Gamma every lattice translation acts trivially, so the 3N displacements of
the primitive cell carry a representation of the point group. Its character on
an operation is the number of atoms the operation leaves in place, up to a
lattice (or centring) translation, times the trace of its rotation; and the
character is the same on any operation of a class, so one of each is enough.

The three acoustic modes span the vector representation (x, y, z). An optical
mode is infrared active if its representation occurs in the vector
representation, as the dipole does, and Raman active if it occurs in the
symmetric square of it, as the polarisability does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chmpy.crystal.character_table import CharacterTable

from .force_constants import _omega, _unit_factor

#: two fractional positions are the same site within this
SITE_TOLERANCE = 1e-3


@dataclass(frozen=True, eq=False)
class GammaModes:
    """How the vibrations at Gamma divide among irreducible representations.

    Attributes:
        table: character table of the crystal's point group
        total: multiplicity of each representation among all 3N modes
        acoustic: multiplicity of each among the three acoustic modes
        infrared: the representations that are infrared active
        raman: the representations that are Raman active
    """

    table: CharacterTable
    total: dict[str, int]
    acoustic: dict[str, int]
    infrared: frozenset[str]
    raman: frozenset[str]

    @property
    def optical(self) -> dict[str, int]:
        "Multiplicity of each representation among the 3N - 3 optical modes"
        return _nonzero({k: n - self.acoustic.get(k, 0) for k, n in self.total.items()})

    @property
    def silent(self) -> dict[str, int]:
        "The optical modes that are neither infrared nor Raman active"
        return {
            k: n
            for k, n in self.optical.items()
            if k not in self.infrared and k not in self.raman
        }

    def dimension(self, counts: dict[str, int]) -> int:
        "The number of modes in a decomposition, counting degeneracy"
        dims = dict(zip(self.table.labels, self.table.characters[:, 0], strict=True))
        return int(sum(n * dims[k] for k, n in counts.items()))

    def __str__(self) -> str:
        optical = self.optical
        lines = [("optical", optical)]
        for name, active in (("infrared", self.infrared), ("raman", self.raman)):
            lines.append((name, {k: n for k, n in optical.items() if k in active}))
        lines.append(("silent", self.silent))
        return "\n".join(f"{name}: {_terms(counts)}" for name, counts in lines)


def gamma_modes(crystal) -> GammaModes:
    """The symmetry of a crystal's vibrations at Gamma.

    Args:
        crystal: a `Crystal`; its space group decides the symmetry

    Returns:
        GammaModes
    """
    space_group = crystal.space_group
    table = space_group.point_group.character_table
    positions = crystal.unit_cell_atoms()["frac_pos"]
    operations = space_group.symmetry_operations
    rotations = np.rint([s.rotation for s in operations]).astype(np.int64)
    translations = np.array([s.translation for s in operations])
    pure = np.all(rotations == np.eye(3, dtype=np.int64), axis=(1, 2))
    centring = translations[pure]

    classes = table.classes_of(rotations)
    first = np.array(
        [np.flatnonzero(classes == k)[0] for k in range(len(table.classes))]
    )
    r, t = rotations[first], translations[first]

    # (class, atom, centring, 3): image of each atom less itself and a centring
    images = np.einsum("kij,nj->kni", r, positions) + t[:, None, :]
    offset = images[:, :, None, :] - positions[None, :, None, :] - centring
    on_site = np.all(np.abs(offset - np.rint(offset)) < SITE_TOLERANCE, axis=-1)
    fixed = on_site.any(axis=-1).sum(axis=1) / len(centring)

    traces = np.trace(r, axis1=1, axis2=2)
    squares = np.einsum("kij,kji->k", r, r)
    vector = _counts(table, table.reduce(traces))
    polarisability = _counts(table, table.reduce((traces**2 + squares) / 2))
    return GammaModes(
        table=table,
        total=_counts(table, table.reduce(fixed * traces)),
        acoustic=vector,
        infrared=frozenset(vector),
        raman=frozenset(polarisability),
    )


@dataclass(frozen=True, eq=False)
class GammaMode:
    """One set of vibrations at Gamma, degenerate by symmetry.

    Attributes:
        frequency: in the units asked for; negative for an imaginary mode
        label: its representation, e.g. "E2g"; a sum for the acoustic set
            (e.g. "A2u + E1u"); "" for modes of a centred cell that belong to
            the zone boundary of the primitive cell and are folded onto Gamma
        degeneracy: how many modes the set holds
        acoustic: whether the set is the three uniform translations
        infrared: whether it is infrared active
        raman: whether it is Raman active
        vectors: (3N, degeneracy) orthonormal mass-weighted eigenvectors
    """

    frequency: float
    label: str
    degeneracy: int
    acoustic: bool
    infrared: bool
    raman: bool
    vectors: np.ndarray

    def __str__(self) -> str:
        active = [x for x, on in (("IR", self.infrared), ("Raman", self.raman)) if on]
        return f"{self.frequency:10.2f}  {self.label or '-':<10} {', '.join(active)}"


def label_gamma_modes(
    crystal, constants, units: str = "cm-1", tolerance: float = 0.1
) -> list[GammaMode]:
    """Label computed vibrations at Gamma by their symmetry.

    The uniform translations are projected out first, as in
    `ForceConstants.optical_frequencies`, and reported as one acoustic set.
    Optical frequencies within `tolerance` of each other form one set, whose
    characters are the traces of each operation on its eigenvectors, averaged
    over the operations of each class (centring translations included).

    Args:
        crystal: the `Crystal` the force constants were measured on
        constants: its `ForceConstants`
        units: "cm-1", "thz" or "rad/s", for the frequencies reported
        tolerance: largest splitting within a degenerate set, in cm-1

    Returns:
        list of GammaMode: the acoustic set, then optical sets low to high

    Raises:
        ValueError: if the force constants are for a different structure, or a
            set does not carry a representation (try another `tolerance`)
    """
    from chmpy.calc.system import System
    from chmpy.opt.symmetry import crystal_atom_map

    symmetry = gamma_modes(crystal)
    table = symmetry.table
    system = System.from_crystal(crystal)
    if len(system) != constants.n_atoms or not np.allclose(
        system.positions, constants.positions, atol=1e-6
    ):
        raise ValueError("the force constants are for a different structure")

    atom_map = crystal_atom_map(crystal, system)
    rotations = atom_map.cartesian(constants.cell)
    fractional = np.rint(atom_map.rotations)
    classes = table.classes_of(fractional)
    centring = np.all(fractional == np.eye(3), axis=(1, 2)).sum()
    # averages a character over each class's operations
    average = np.zeros((len(table.classes), len(classes)))
    average[classes, np.arange(len(classes))] = 1 / (
        centring * table.class_sizes[classes]
    )
    n = constants.n_atoms

    def mode(frequency, vectors, acoustic) -> GammaMode:
        x = vectors.reshape(n, 3, -1)
        traces = np.einsum("gnam,gab,nbm->g", x[atom_map.permutation], rotations, x)
        multiplicities = table.reduce(average @ traces)
        counts = _integral(table, multiplicities)
        if counts is None:
            raise ValueError(
                f"the modes near {frequency:.3f} {units} do not carry a "
                f"representation (multiplicities {np.round(multiplicities, 2)}); "
                "try a different tolerance"
            )
        return GammaMode(
            frequency=frequency,
            label=_terms(counts) if counts else "",
            degeneracy=vectors.shape[1],
            acoustic=acoustic,
            infrared=not acoustic and not symmetry.infrared.isdisjoint(counts),
            raman=not acoustic and not symmetry.raman.isdisjoint(counts),
            vectors=vectors,
        )

    root_mass = np.sqrt(constants.masses)
    translations = np.linalg.qr(np.kron(root_mass[:, None], np.eye(3)))[0]
    complement = constants._translation_complement()
    matrix = constants.dynamical_matrix(np.zeros(3)).real
    projected = complement.T @ matrix @ complement
    values, vectors = np.linalg.eigh(0.5 * (projected + projected.T))
    optical = complement @ vectors
    per_cm = _omega(values) * _unit_factor("cm-1")
    scale = _unit_factor(units) / _unit_factor("cm-1")

    sets = np.split(
        np.arange(len(values)), np.flatnonzero(np.diff(per_cm) > tolerance) + 1
    )
    return [mode(0.0, translations, True)] + [
        mode(float(per_cm[s].mean() * scale), optical[:, s], False) for s in sets
    ]


def _integral(table: CharacterTable, multiplicities) -> dict[str, int] | None:
    "The non-zero multiplicities by label, or None if they are not integers"
    rounded = np.rint(multiplicities)
    if not np.allclose(rounded, multiplicities, atol=1e-6):
        return None
    return _nonzero(dict(zip(table.labels, rounded.astype(int).tolist(), strict=True)))


def _counts(table: CharacterTable, multiplicities) -> dict[str, int]:
    "The non-zero multiplicities, by label, in the table's order"
    counts = _integral(table, multiplicities)
    if counts is None:
        raise ValueError(
            "the structure does not have its space group's symmetry: "
            f"non-integer multiplicities {np.round(multiplicities, 3).tolist()}"
        )
    return counts


def _nonzero(counts: dict[str, int]) -> dict[str, int]:
    return {k: n for k, n in counts.items() if n}


def _terms(counts: dict[str, int]) -> str:
    "e.g. 'A1g + 2B1u + 3Eu'"
    return " + ".join(f"{n if n > 1 else ''}{k}" for k, n in counts.items()) or "none"

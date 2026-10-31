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
    table = crystal.space_group.point_group.character_table
    r, t, centring = _class_operations(crystal.space_group, table)
    fixed = _fixed(crystal.unit_cell_atoms()["frac_pos"], r, t, centring)

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


def lattice_modes(crystal) -> dict[str, dict[str, int]]:
    """The symmetry of a molecular crystal's optical lattice modes at Gamma.

    By the correlation method: whole-molecule translations carry the vector
    representation and rotations the axial one, on the molecules each
    operation maps onto themselves. The acoustic modes are taken from the
    translations.

    Returns:
        {"translation": ..., "libration": ...}, each a multiplicity by label
    """
    table = crystal.space_group.point_group.character_table
    r, t, centring = _class_operations(crystal.space_group, table)
    centres = np.array(
        [crystal.to_fractional(m.center_of_mass) for m in crystal.unit_cell_molecules()]
    )
    fixed = _fixed(centres, r, t, centring)
    traces = np.trace(r, axis1=1, axis2=2)
    axial = np.rint(np.linalg.det(r)) * traces
    acoustic = gamma_modes(crystal).acoustic
    translation = _counts(table, table.reduce(fixed * traces))
    return {
        "translation": _nonzero(
            {k: n - acoustic.get(k, 0) for k, n in translation.items()}
        ),
        "libration": _counts(table, table.reduce(fixed * axial)),
    }


def rigid_body_fractions(crystal, modes) -> np.ndarray:
    """How much of each set of modes moves whole molecules.

    Each molecule's rigid translations, and rotations about its centre of
    mass, are mass-weighted displacements orthogonal to each other and to every
    other molecule's; the squared projection of a set's eigenvectors onto them
    is the share of its kinetic energy in translation and in libration.

    Args:
        crystal: the `Crystal` the modes are of, with its molecules
        modes: `GammaMode`s, e.g. from `label_gamma_modes`

    Returns:
        (len(modes), 2) fraction of each set in translation and in libration,
        averaged over the set's modes
    """
    bases = _rigid_body_bases(crystal)
    return np.array(
        [
            [np.sum((basis.T @ m.vectors) ** 2) / m.degeneracy for basis in bases]
            for m in modes
        ]
    )


def _rigid_body_bases(crystal):
    "Orthonormal mass-weighted (3N, 3Z) translations and (3N, <= 3Z) rotations"
    n = len(crystal.unit_cell_atoms()["element"])
    translations, rotations = [], []
    for molecule in crystal.unit_cell_molecules():
        atoms = np.asarray(molecule.properties["unit_cell_atoms"])
        root_mass = np.sqrt([e.mass for e in molecule.elements])[:, None]
        arm = molecule.positions - molecule.center_of_mass
        columns = np.zeros((2, 3, n, 3))
        for a, axis in enumerate(np.eye(3)):
            columns[0, a, atoms] = root_mass * axis
            columns[1, a, atoms] = root_mass * np.cross(axis, arm)
        translations.append(columns[0].reshape(3, -1).T / np.sqrt(molecule.mass))
        # a rotation about an axis a linear molecule lies on moves nothing
        u, sizes, _ = np.linalg.svd(columns[1].reshape(3, -1).T, full_matrices=False)
        rotations.append(u[:, sizes > 1e-8 * sizes.max()])
    return np.hstack(translations), np.hstack(rotations)


def _class_operations(space_group, table: CharacterTable):
    """One operation of each class, as rotations and translations, and the
    centring translations"""
    operations = space_group.symmetry_operations
    rotations = np.rint([s.rotation for s in operations]).astype(np.int64)
    translations = np.array([s.translation for s in operations])
    pure = np.all(rotations == np.eye(3, dtype=np.int64), axis=(1, 2))
    classes = table.classes_of(rotations)
    first = [np.flatnonzero(classes == k)[0] for k in range(len(table.classes))]
    return rotations[first], translations[first], translations[pure]


def _fixed(points, rotations, translations, centring) -> np.ndarray:
    """How many of the primitive cell's points each operation leaves in place,
    up to a lattice or centring translation"""
    # (operation, point, centring, 3): image of each point less itself
    images = np.einsum("kij,nj->kni", rotations, points) + translations[:, None, :]
    offset = images[:, :, None, :] - points[None, :, None, :] - centring
    on_site = np.all(np.abs(offset - np.rint(offset)) < SITE_TOLERANCE, axis=-1)
    return on_site.any(axis=-1).sum(axis=1) / len(centring)


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


#: largest acoustic sum rule residue, in cm-1, that labelling accepts
SUM_RULE_TOLERANCE = 1.0


def label_gamma_modes(
    crystal, constants, units: str = "cm-1", tolerance: float = 0.1
) -> list[GammaMode]:
    """Label computed vibrations at Gamma by their symmetry.

    The uniform translations are projected out first, as in
    `ForceConstants.optical_frequencies`, and reported as one acoustic set.
    Optical frequencies within `tolerance` of each other form one set, whose
    characters are the traces of each operation on its eigenvectors, averaged
    over the operations of each class (centring translations included). A set
    holding more than one representation, from an accidental degeneracy, is
    split with the projector onto each, so each mode gets a single label.

    Args:
        crystal: the `Crystal` the force constants were measured on
        constants: its `ForceConstants`
        units: "cm-1", "thz" or "rad/s", for the frequencies reported
        tolerance: largest splitting within a degenerate set, in cm-1; too
            large only costs time, too small can break a degenerate set

    Returns:
        list of GammaMode: the acoustic set, then one per copy of each
        representation among the optical modes, low to high

    Raises:
        ValueError: if the force constants are for a different structure,
            break the acoustic sum rule by more than `SUM_RULE_TOLERANCE`
            (projecting the translations out would hide that), or a set does
            not carry a representation (try another `tolerance`)
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
    residue = constants.acoustic_error()
    if residue > SUM_RULE_TOLERANCE:
        raise ValueError(
            f"the force constants break the acoustic sum rule by {residue:.2f} cm-1; "
            "impose it first with chmpy.vib.symmetrise"
        )

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
    n, g = constants.n_atoms, len(classes)
    # the projector onto representation i is scale_i * sum_g chi_i(g) D(g)
    per_operation = table.characters[:, classes] / centring
    scale = table.characters[:, 0] / (table.norms * table.order)
    to_units = _unit_factor(units)
    matrix = constants.dynamical_matrix(np.zeros(3)).real

    def images(vectors) -> np.ndarray:
        "(G, 3N, M): each operation applied to each column of `vectors`"
        x = vectors.reshape(n, 3, -1)
        out = np.empty((g, *x.shape))
        out[np.arange(g)[:, None], atom_map.permutation] = np.einsum(
            "gab,nbm->gnam", rotations, x
        )
        return out.reshape(g, 3 * n, -1)

    def make(vectors, label, acoustic=False) -> GammaMode:
        curvature = np.linalg.eigvalsh(vectors.T @ matrix @ vectors).mean()
        return GammaMode(
            frequency=0.0 if acoustic else float(_omega(curvature) * to_units),
            label=label,
            degeneracy=vectors.shape[1],
            acoustic=acoustic,
            infrared=not acoustic and label in symmetry.infrared,
            raman=not acoustic and label in symmetry.raman,
            vectors=vectors,
        )

    def split(vectors) -> list[GammaMode]:
        "One GammaMode per copy of each representation in a set of modes"
        moved = images(vectors)
        traces = np.einsum("gim,im->g", moved, vectors)
        multiplicities = table.reduce(average @ traces)
        counts = _integral(table, multiplicities)
        if counts is None:
            raise ValueError(
                f"modes near {make(vectors, '').frequency:.3f} {units} do not carry "
                f"a representation (multiplicities {np.round(multiplicities, 2)}); "
                "try a different tolerance"
            )
        modes, gamma = [], np.zeros_like(vectors)
        for label, count in counts.items():
            i = table.labels.index(label)
            projected = scale[i] * np.einsum("g,gim->im", per_operation[i], moved)
            gamma += projected
            basis = _range(projected)
            # within one representation the eigenvalues come in exact multiplets
            _, rotation = np.linalg.eigh(basis.T @ matrix @ basis)
            basis = basis @ rotation
            dim = int(table.characters[i, 0])
            modes += [
                make(basis[:, k : k + dim], label) for k in range(0, count * dim, dim)
            ]
        folded = _range(vectors - gamma)
        if folded.shape[1]:
            modes.append(make(folded, ""))
        return modes

    basis = constants._translation_basis()
    translations = basis[:, :3]
    acoustic = _integral(
        table,
        table.reduce(
            average @ np.einsum("gim,im->g", images(translations), translations)
        ),
    )
    complement = basis[:, 3:]
    projected = complement.T @ matrix @ complement
    values, vectors = np.linalg.eigh(0.5 * (projected + projected.T))
    optical = complement @ vectors
    per_cm = _omega(values) * _unit_factor("cm-1")
    sets = np.split(
        np.arange(len(values)), np.flatnonzero(np.diff(per_cm) > tolerance) + 1
    )
    modes = [m for s in sets for m in split(optical[:, s])]
    return [make(translations, _terms(acoustic), acoustic=True)] + sorted(
        modes, key=lambda m: m.frequency
    )


def _range(projected) -> np.ndarray:
    "Orthonormal basis for the range of a projector applied to orthonormal columns"
    u, sizes, _ = np.linalg.svd(projected, full_matrices=False)
    return u[:, sizes > 0.5]


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

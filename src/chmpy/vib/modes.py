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


def _counts(table: CharacterTable, multiplicities) -> dict[str, int]:
    "The non-zero multiplicities, by label, in the table's order"
    rounded = np.rint(multiplicities)
    if not np.allclose(rounded, multiplicities, atol=1e-6):
        raise ValueError(
            "the structure does not have its space group's symmetry: "
            f"non-integer multiplicities {np.round(multiplicities, 3).tolist()}"
        )
    return _nonzero(dict(zip(table.labels, rounded.astype(int).tolist(), strict=True)))


def _nonzero(counts: dict[str, int]) -> dict[str, int]:
    return {k: n for k, n in counts.items() if n}


def _terms(counts: dict[str, int]) -> str:
    "e.g. 'A1g + 2B1u + 3Eu'"
    return " + ".join(f"{n if n > 1 else ''}{k}" for k, n in counts.items()) or "none"

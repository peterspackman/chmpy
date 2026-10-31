"""Gamma-point mode symmetry, against the textbook decompositions."""

import numpy as np
import pytest

from chmpy import Crystal
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.vib import (
    gamma_modes,
    label_gamma_modes,
    lattice_modes,
    rigid_body_fractions,
)


def crystal(number, lengths, angles, sites):
    cell = UnitCell.from_lengths_and_angles(lengths, angles, unit="degrees")
    asym = AsymmetricUnit.from_records(
        [
            {"label": f"{e}{i}", "element": e, "position": p}
            for i, (e, p) in enumerate(sites)
        ]
    )
    return Crystal(cell, SpaceGroup(number), asym)


@pytest.fixture(scope="module")
def diamond():
    return crystal(227, [3.567] * 3, [90] * 3, [("C", (1 / 8, 1 / 8, 1 / 8))])


@pytest.fixture(scope="module")
def rock_salt():
    return crystal(
        225, [5.64] * 3, [90] * 3, [("Na", (0, 0, 0)), ("Cl", (0.5, 0.5, 0.5))]
    )


@pytest.fixture(scope="module")
def hcp():
    return crystal(194, [3.2, 3.2, 5.2], [90, 90, 120], [("Mg", (1 / 3, 2 / 3, 1 / 4))])


@pytest.fixture(scope="module")
def rutile():
    return crystal(
        136,
        [4.594, 4.594, 2.959],
        [90] * 3,
        [("Ti", (0, 0, 0)), ("O", (0.305, 0.305, 0))],
    )


@pytest.fixture(scope="module")
def quartz():
    return crystal(
        152,
        [4.913, 4.913, 5.405],
        [90, 90, 120],
        [("Si", (0.4697, 0, 1 / 3)), ("O", (0.4135, 0.2669, 0.1191))],
    )


def test_diamond_has_one_raman_active_optical_mode(diamond):
    modes = gamma_modes(diamond)
    assert modes.optical == {"T2g": 1}
    assert "T2g" in modes.raman and "T2g" not in modes.infrared


def test_rock_salt_has_one_infrared_active_optical_mode(rock_salt):
    modes = gamma_modes(rock_salt)
    assert modes.optical == {"T1u": 1}
    assert "T1u" in modes.infrared and "T1u" not in modes.raman


def test_the_acoustic_modes_span_the_vector_representation(rock_salt):
    assert gamma_modes(rock_salt).acoustic == {"T1u": 1}


def test_hcp(hcp):
    modes = gamma_modes(hcp)
    assert modes.optical == {"B1g": 1, "E2g": 1}
    assert modes.silent == {"B1g": 1}


def test_rutile(rutile):
    modes = gamma_modes(rutile)
    assert modes.optical == {
        "A1g": 1,
        "A2g": 1,
        "B1g": 1,
        "B2g": 1,
        "Eg": 1,
        "A2u": 1,
        "B1u": 2,
        "Eu": 3,
    }
    assert modes.acoustic == {"A2u": 1, "Eu": 1}
    assert {x for x in modes.optical if x in modes.raman} == {"A1g", "B1g", "B2g", "Eg"}
    assert {x for x in modes.optical if x in modes.infrared} == {"A2u", "Eu"}


def test_alpha_quartz(quartz):
    modes = gamma_modes(quartz)
    assert modes.optical == {"A1": 4, "A2": 4, "E": 8}
    assert modes.infrared == {"A2", "E"}
    assert modes.raman == {"A1", "E"}


def test_every_degree_of_freedom_is_counted(rutile, diamond):
    for structure, atoms in ((rutile, 6), (diamond, 2)):
        modes = gamma_modes(structure)
        assert modes.dimension(modes.total) == 3 * atoms


def test_the_summary_reads_like_a_paper(rutile):
    assert str(gamma_modes(rutile)).splitlines()[0] == (
        "optical: A1g + A2g + B1g + B2g + Eg + A2u + 2B1u + 3Eu"
    )


@pytest.fixture(scope="module")
def hcp_argon():
    from chmpy.calc import LennardJones
    from chmpy.vib import force_constants

    structure = crystal(
        194, [3.75, 3.75, 6.12], [90, 90, 120], [("Ar", (1 / 3, 2 / 3, 1 / 4))]
    )
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    return structure, force_constants(structure, calc, cutoff=8.0)


def test_computed_hcp_modes_are_labelled(hcp_argon):
    modes = label_gamma_modes(*hcp_argon)
    optical = [m for m in modes if not m.acoustic]
    assert len(optical) == 2
    assert {(m.label, m.degeneracy) for m in optical} == {("B1g", 1), ("E2g", 2)}
    assert [m.label for m in modes if m.acoustic] == ["A2u + E1u"]
    assert next(m for m in optical if m.label == "E2g").raman


def test_labelled_frequencies_are_the_optical_frequencies(hcp_argon):
    structure, constants = hcp_argon
    modes = label_gamma_modes(structure, constants)
    frequencies = [
        m.frequency for m in modes if not m.acoustic for _ in range(m.degeneracy)
    ]
    np.testing.assert_allclose(
        sorted(frequencies), constants.optical_frequencies("cm-1"), atol=1e-6
    )


def test_modes_folded_into_a_centred_cell_are_not_labelled():
    from chmpy.calc import LennardJones
    from chmpy.vib import force_constants

    fcc = crystal(225, [5.3] * 3, [90] * 3, [("Ar", (0, 0, 0))])
    calc = LennardJones(epsilon=0.0103, sigma=3.4, cutoff=8.0)
    modes = label_gamma_modes(fcc, force_constants(fcc, calc, cutoff=8.0))
    assert [(m.label, m.infrared) for m in modes if m.acoustic] == [("T1u", False)]
    assert {m.label for m in modes if not m.acoustic} == {""}
    assert sum(m.degeneracy for m in modes if not m.acoustic) == 9


def test_force_constants_of_another_structure_are_refused(hcp_argon, rock_salt):
    with pytest.raises(ValueError, match="different structure"):
        label_gamma_modes(rock_salt, hcp_argon[1])


@pytest.fixture(scope="module")
def acetic_acid():
    from pathlib import Path

    return Crystal.load(Path(__file__).parents[1] / "test_files" / "acetic_acid.res")


@pytest.fixture(scope="module")
def centrosymmetric():
    "Two planar four-atom molecules on inversion centres in P2_1/c"
    cell = UnitCell.from_lengths_and_angles([8, 6, 8], [90, 100, 90], unit="degrees")
    cartesian = np.array([[0.6, 0.0, 0.0], [1.0, 1.1, 0.3]])
    asym = AsymmetricUnit.from_records(
        [
            {"label": label, "element": e, "position": p}
            for label, e, p in zip(
                ("C1", "O1"), ("C", "O"), cell.to_fractional(cartesian), strict=True
            )
        ]
    )
    return Crystal(cell, SpaceGroup(14), asym)


def test_lattice_modes_on_general_positions(acetic_acid):
    modes = lattice_modes(acetic_acid)
    assert modes["translation"] == {"A1": 2, "A2": 3, "B1": 2, "B2": 2}
    assert modes["libration"] == {"A1": 3, "A2": 3, "B1": 3, "B2": 3}


def test_lattice_modes_of_molecules_on_inversion_centres(centrosymmetric):
    modes = lattice_modes(centrosymmetric)
    assert modes["libration"] == {"Ag": 3, "Bg": 3}
    assert modes["translation"] == {"Au": 2, "Bu": 1}


def _modes(vectors, sizes):
    "GammaModes holding consecutive columns of `vectors`"
    from chmpy.vib import GammaMode

    edges = np.cumsum([0, *sizes])
    return [
        GammaMode(0.0, "", int(b - a), False, False, False, vectors[:, a:b])
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def test_rigid_body_fractions_of_any_complete_set_count_the_molecules(acetic_acid):
    n = 3 * len(acetic_acid.unit_cell_atoms()["element"])
    basis = np.linalg.qr(np.random.default_rng(1).normal(size=(n, n)))[0]
    sizes = [1, 2, 3] * (n // 6) + [1] * (n % 6)
    modes = _modes(basis, sizes)
    fractions = rigid_body_fractions(acetic_acid, modes)
    totals = (fractions * [[m.degeneracy] for m in modes]).sum(axis=0)
    np.testing.assert_allclose(totals, [12, 12], atol=1e-9)


def test_a_rigid_translation_is_all_translation(centrosymmetric):
    from chmpy.core import Element

    elements = centrosymmetric.unit_cell_atoms()["element"]
    root_mass = np.sqrt([Element[z].mass for z in elements])
    vector = np.kron(root_mass, [0.0, 1.0, 0.0])
    modes = _modes((vector / np.linalg.norm(vector))[:, None], [1])
    np.testing.assert_allclose(
        rigid_body_fractions(centrosymmetric, modes), [[1, 0]], atol=1e-12
    )


def test_accidentally_degenerate_sets_are_split_by_symmetry(hcp_argon):
    structure, constants = hcp_argon
    lumped = label_gamma_modes(structure, constants, tolerance=1e4)
    separate = label_gamma_modes(structure, constants)
    assert [(m.label, m.degeneracy) for m in lumped] == [
        (m.label, m.degeneracy) for m in separate
    ]
    np.testing.assert_allclose(
        [m.frequency for m in lumped], [m.frequency for m in separate], atol=1e-6
    )

import numpy as np
import pytest

from chmpy import Molecule
from chmpy.fmt.orca import (
    Angle,
    Bond,
    Dihedral,
    Fixed,
    GeomSettings,
    Scan,
    counterpoise_inputs,
    parse_final_energy,
    parse_input_geometry,
    to_orca_input,
)

from .. import TEST_FILES

DLPNO = "DLPNO-CCSD(T) TightPNO aug-cc-pVTZ aug-cc-pVTZ/C def2/J RIJCOSX TightSCF"
F12 = "CCSD(T)-F12/RI cc-pVTZ-F12 cc-pVTZ-F12-CABS cc-pVTZ/C TightSCF"

OUTPUT = TEST_FILES["acetylene_dimer_cp_A.orca.out"]
INPUT = TEST_FILES["acetylene_dimer_cp_A.orca.inp"]


@pytest.fixture
def water():
    return Molecule.from_xyz_string(
        "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692"
    )


def test_final_energy_of_a_finished_run():
    assert parse_final_energy(OUTPUT) == pytest.approx(-77.187009801390, abs=1e-12)


def test_final_energy_from_text():
    assert parse_final_energy(OUTPUT.read_text()) == pytest.approx(-77.187009801390)


def test_a_run_cut_short_has_no_final_energy():
    text = OUTPUT.read_text()
    truncated = text[: text.index("ORCA TERMINATED NORMALLY")]
    assert "FINAL SINGLE POINT ENERGY" in truncated
    assert parse_final_energy(truncated) is None


def test_a_missing_file_has_no_final_energy(tmp_path):
    assert parse_final_energy(tmp_path / "missing.out") is None


def test_the_last_final_energy_wins():
    text = (
        "FINAL SINGLE POINT ENERGY      -1.0\n"
        "FINAL SINGLE POINT ENERGY      -2.5\n"
        "****ORCA TERMINATED NORMALLY****\n"
    )
    assert parse_final_energy(text) == -2.5


def test_input_geometry_with_ghosts():
    elements, positions, ghost = parse_input_geometry(INPUT)
    assert elements == ["C", "C", "H", "H"] * 2
    assert ghost == [False] * 4 + [True] * 4
    assert positions.shape == (8, 3)
    np.testing.assert_allclose(positions[4], [-3.16954163, -0.34248280, -2.48457603])


def test_writing_reproduces_the_real_geometry_block():
    elements, positions, ghost = parse_input_geometry(INPUT)
    ghosts = [i for i, g in enumerate(ghost) if g]
    text = to_orca_input(elements, positions, DLPNO, maxcore=2500, ghost=ghosts)
    original = INPUT.read_text()
    block = original[original.index("* xyz") :]
    assert block in text


def test_input_layout_and_blocks(water):
    text = to_orca_input(
        [str(e) for e in water.elements],
        water.positions,
        F12,
        blocks="%mdci STol 5e-5 end",
        nprocs=4,
        maxcore=2000,
    )
    lines = text.splitlines()
    assert lines[:4] == [
        f"! {F12}",
        "%maxcore 2000",
        "%pal nprocs 4 end",
        "%mdci STol 5e-5 end",
    ]
    assert lines[4] == ""
    assert lines[5] == "* xyz 0 1"
    assert lines[-1] == "*"


def test_written_input_round_trips(water):
    text = water.to_orca_input(DLPNO, ghost=[2])
    elements, positions, ghost = parse_input_geometry(text)
    assert elements == ["O", "H", "H"]
    assert ghost == [False, False, True]
    np.testing.assert_allclose(positions, water.positions, atol=1e-8)


def test_molecule_charge_and_multiplicity_are_used(water):
    water.charge, water.multiplicity = 1, 2
    assert "* xyz 1 2" in water.to_orca_input(DLPNO)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        to_orca_input(["C", "H"], np.zeros((1, 3)), DLPNO)


def test_counterpoise_ghosts_the_partner(water):
    other = Molecule.from_arrays(
        water.atomic_numbers, water.positions + [3.0, 0.0, 0.0]
    )
    inputs = counterpoise_inputs(water, other, DLPNO, nprocs=2)

    assert set(inputs) == {"AB", "A", "B"}
    for name, expected in (
        ("AB", [False] * 6),
        ("A", [False] * 3 + [True] * 3),
        ("B", [True] * 3 + [False] * 3),
    ):
        _, positions, ghost = parse_input_geometry(inputs[name])
        assert ghost == expected, name
        assert positions.shape == (6, 3)
        assert "%pal nprocs 2 end" in inputs[name]


def test_counterpoise_charges(water):
    cation = Molecule.from_arrays(
        water.atomic_numbers,
        water.positions + [3.0, 0.0, 0.0],
        charge=1,
        multiplicity=2,
    )
    inputs = counterpoise_inputs(water, cation, DLPNO, multiplicity=2)
    assert "* xyz 1 2" in inputs["AB"]
    assert "* xyz 0 1" in inputs["A"]
    assert "* xyz 1 2" in inputs["B"]


def test_counterpoise_needs_a_multiplicity_for_open_shells(water):
    radical = Molecule.from_arrays(
        water.atomic_numbers, water.positions, multiplicity=2
    )
    with pytest.raises(ValueError, match="multiplicity"):
        counterpoise_inputs(water, radical, DLPNO)


# -- constraints, scans and fragments -------------------------------------------


def chain(dihedral=60.0):
    "Four atoms, bonds of 1.5 A, angles of 110 degrees, given torsion"
    theta, phi = np.radians(110.0), np.radians(dihedral)
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.5, 0.0, 0.0])
    a = b + 1.5 * np.array([-np.cos(np.pi - theta), np.sin(np.pi - theta), 0.0])
    d = c + 1.5 * np.array(
        [-np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)]
    )
    return np.array([a, b, c, d])


def test_internal_coordinates_are_measured():
    positions = chain(60.0)
    assert Bond(1, 2).measure(positions) == pytest.approx(1.5)
    assert Angle(0, 1, 2).measure(positions) == pytest.approx(110.0)
    assert Dihedral(0, 1, 2, 3).measure(positions) == pytest.approx(60.0)
    assert Dihedral(0, 1, 2, 3).measure(chain(-100.0)) == pytest.approx(-100.0)


def test_constraints_block():
    geom = GeomSettings(
        constraints=[Bond(0, 1), Angle(0, 1, 2, 109.5), Dihedral(0, 1, 2, 3, 65)],
        options={"MaxIter": 200, "invertConstraints": True},
    )
    block, frag = geom.render(chain())
    assert frag == ""
    assert block == "\n".join(
        [
            "%geom",
            "  Constraints",
            "    { B 0 1 C }",
            "    { A 0 1 2 109.5 C }",
            "    { D 0 1 2 3 65 C }",
            "  end",
            "  MaxIter 200",
            "  invertConstraints true",
            "end",
        ]
    )


def test_fixed_atoms_are_written_as_ranges():
    geom = GeomSettings(constraints=[Fixed([8, 0, 2, 1, 5, 7])])
    block, _ = geom.render(np.zeros((9, 3)))
    assert "{ C 0:2 C }\n    { C 5 C }\n    { C 7:8 C }" in block


def test_fixed_components_get_one_line_each():
    geom = GeomSettings(constraints=[Fixed([3, 4], axes="zx")])
    block, _ = geom.render(np.zeros((5, 3)))
    assert "{ X 3:4 C }\n    { Z 3:4 C }" in block


def test_unknown_axes_are_refused():
    with pytest.raises(ValueError, match="axes"):
        Fixed([0], axes="xw")


def test_scans():
    geom = GeomSettings(
        scans=[
            Scan(Bond(0, 1), 1.35, 1.10, 12),
            Scan(Dihedral(0, 1, 2, 3), values=[180, 200, 240]),
        ]
    )
    block, _ = geom.render(chain())
    assert "  Scan B 0 1 = 1.35, 1.1, 12 end" in block
    assert "  Scan D 0 1 2 3 [180 200 240] end" in block
    assert "Simul_Scan" not in block


def test_a_simultaneous_scan():
    geom = GeomSettings(
        scans=[Scan(Bond(0, 1), 3, 1, 15), Scan(Bond(1, 2), 1, 3, 15)],
        simultaneous=True,
    )
    assert "  Simul_Scan true" in geom.render(chain())[0]


@pytest.mark.parametrize(
    "settings, message",
    [
        (GeomSettings(constraints=[Bond(0, 4)]), "names atoms \\[4\\]"),
        (GeomSettings(constraints=[Angle(0, 1, 0)]), "repeats an atom"),
        (
            GeomSettings(
                constraints=[Bond(0, 1)], scans=[Scan(Bond(1, 0), 1.0, 2.0, 5)]
            ),
            "both constrained and scanned",
        ),
        (
            GeomSettings(scans=[Scan(Bond(0, 1), 1, 2, 3), Scan(Bond(1, 0), 1, 2, 3)]),
            "scanned twice",
        ),
        (
            GeomSettings(
                scans=[
                    Scan(Bond(0, 1), 1, 2, 3),
                    Scan(Bond(1, 2), 1, 2, 3),
                    Scan(Bond(2, 3), 1, 2, 3),
                    Scan(Angle(0, 1, 2), 100, 120, 3),
                ]
            ),
            "at most 3",
        ),
        (
            GeomSettings(
                scans=[Scan(Bond(0, 1), 1, 2, 3), Scan(Bond(1, 2), 1, 2, 4)],
                simultaneous=True,
            ),
            "equal numbers",
        ),
        (GeomSettings(constrain_fragments=[1]), "need `fragments`"),
        (GeomSettings(fragments=[[0, 1], [1, 2]]), "missing \\[3\\], repeated \\[1\\]"),
        (
            GeomSettings(fragments=[[0, 1], [2, 3]], constrain_fragments=[3]),
            "fragments \\[3\\]",
        ),
        (
            GeomSettings(fragments=[[0, 1], [2, 3]], rigid_fragments=[1]),
            "RIGIDBODYOPT",
        ),
    ],
)
def test_inconsistent_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        settings.render(chain(), keywords="B3LYP def2-SVP Opt")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"start": 1.0, "stop": 2.0}, "start, stop and steps"),
        ({"start": 1.0, "stop": 2.0, "steps": 5, "values": [1, 2]}, "not both"),
        ({"values": [1.0]}, "at least two values"),
    ],
)
def test_a_scan_needs_one_complete_range(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Scan(Bond(0, 1), **kwargs)


def test_a_scanned_coordinate_has_no_target():
    with pytest.raises(ValueError, match="no target"):
        Scan(Bond(0, 1, 1.5), 1.0, 2.0, 5)


def test_a_distant_target_is_warned_about(caplog):
    GeomSettings(constraints=[Bond(0, 1, 2.5)]).render(chain())
    assert "starts at 1.500" in caplog.text


def test_a_dihedral_target_across_180_is_close(caplog):
    GeomSettings(constraints=[Dihedral(0, 1, 2, 3, -179.0)]).render(chain(179.0))
    assert caplog.text == ""


def test_rigid_monomers_in_a_dimer(water):
    other = Molecule.from_arrays(
        water.atomic_numbers, water.positions + [3.0, 0.0, 0.0]
    )
    elements = [str(e) for e in water.elements] * 2
    positions = np.vstack([water.positions, other.positions])
    geom = GeomSettings(fragments=[[0, 1, 2], [3, 4, 5]], constrain_fragments=[1, 2])
    text = to_orca_input(elements, positions, "B3LYP def2-SVP Opt", geom=geom)

    assert "  ConstrainFragments { 1 2 } end" in text
    # ORCA reads %frag Definition only after the coordinates
    tail = text[text.index("* xyz") :]
    assert tail.endswith(
        "*\n%frag\n  Definition\n    1 {0:2} end\n    2 {3:5} end\n  end\nend\n"
    )
    _, read, _ = parse_input_geometry(text)
    np.testing.assert_allclose(read, positions, atol=1e-8)


def test_rigid_bodies_need_and_use_the_keyword(water):
    geom = GeomSettings(fragments=[[0], [1, 2]], rigid_fragments=[2])
    text = water.to_orca_input("B3LYP def2-SVP L-Opt RIGIDBODYOPT", geom=geom)
    assert "  RigidFrags { 2 } end" in text


def test_a_hand_written_geom_block_is_not_doubled(water):
    with pytest.raises(ValueError, match="%geom"):
        water.to_orca_input(
            DLPNO,
            blocks="%geom MaxIter 10 end",
            geom=GeomSettings(constraints=[Bond(0, 1)]),
        )

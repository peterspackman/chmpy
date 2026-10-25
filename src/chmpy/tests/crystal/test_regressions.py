"""Regressions for crystal bugs that each gave a wrong structure without error."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, Crystal, SpaceGroup, UnitCell

from .. import TEST_FILES

_CUBIC_CIF = """data_x
_symmetry_space_group_name_H-M '{symbol}'
_symmetry_Int_Tables_number {number}
_cell_length_a {a}
_cell_length_b {b}
_cell_length_c {c}
_cell_angle_alpha {alpha}
_cell_angle_beta {beta}
_cell_angle_gamma {gamma}
loop_
_symmetry_equiv_pos_as_xyz
{ops}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{sites}
"""


def _cif(space_group, symbol, sites, lengths, angles):
    ops = "\n".join(f"'{s}'" for s in space_group.symmetry_operations)
    a, b, c = lengths
    alpha, beta, gamma = angles
    return _CUBIC_CIF.format(
        symbol=symbol,
        number=space_group.international_tables_number,
        ops=ops,
        sites=sites,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def test_cif_symops_win_over_a_shared_hm_symbol():
    # Fd-3m origin choice 1 and 2 share their symbol; the symops say which
    si = Crystal.from_cif_string(
        _cif(
            SpaceGroup(227, choice="2"),
            "F d -3 m",
            "Si1 Si 0.125 0.125 0.125",
            (5.431,) * 3,
            (90,) * 3,
        )
    )
    assert si.space_group.choice == "2"
    assert len(si.unit_cell_atoms()["element"]) == 8

    # R-3 on rhombohedral axes, not hexagonal ones
    r = Crystal.from_cif_string(
        _cif(
            SpaceGroup(148, choice="R"),
            "R -3",
            "C1 C 0.1 0.2 0.3",
            (5.0,) * 3,
            (80,) * 3,
        )
    )
    assert r.space_group.choice == "R"
    assert len(r.unit_cell_atoms()["element"]) == 6


@pytest.mark.parametrize("x", [0.001, -0.001, 1e-5])
def test_sites_merge_across_the_cell_boundary(x):
    cell = UnitCell.cubic(5.0)
    p1bar = Crystal(cell, SpaceGroup(2), AsymmetricUnit([Element["Ar"]], [[x, 0, 0]]))
    assert len(p1bar.unit_cell_atoms()["element"]) == 1
    pm3m = Crystal(cell, SpaceGroup(221), AsymmetricUnit([Element["Ar"]], [[x, 0, 0]]))
    assert len(pm3m.unit_cell_atoms()["element"]) == 1


def test_atoms_in_radius_finds_everything_in_an_oblique_cell():
    cell = UnitCell.from_lengths_and_angles(
        [5.0, 5.0, 5.0], [90, 90, 20], unit="degrees"
    )
    crystal = Crystal(cell, SpaceGroup(1), AsymmetricUnit([Element["Ar"]], [[0, 0, 0]]))
    big = crystal.slab(bounds=((-10, -10, -10), (10, 10, 10)))
    tree = cKDTree(big["cart_pos"])
    for radius in (3.0, 4.0, 5.0, 6.0):
        expected = len(tree.query_ball_point(np.zeros(3), radius))
        got = crystal.atoms_in_radius(radius, origin=np.zeros(3))
        assert len(got["element"]) == expected


def test_p1_supercell_keeps_a_rotated_cell_intact():
    rotation = np.array(
        [
            [0.8137977, -0.4409696, 0.3785223],
            [0.4698463, 0.8825641, 0.0180283],
            [-0.3420201, 0.1631759, 0.9254166],
        ]
    )
    a = 5.64
    cell = UnitCell(np.eye(3) * a @ rotation.T)
    nacl = Crystal(
        cell,
        SpaceGroup(225),
        AsymmetricUnit([Element["Na"], Element["Cl"]], [[0, 0, 0], [0.5, 0, 0]]),
    )

    def shortest(crystal):
        slab = crystal.slab()
        distances, _ = cKDTree(slab["cart_pos"]).query(
            crystal.unit_cell_atoms()["cart_pos"], k=2
        )
        return distances[:, 1].min()

    assert shortest(nacl) == pytest.approx(a / 2)
    assert shortest(nacl.as_P1()) == pytest.approx(a / 2)
    assert shortest(nacl.as_P1_supercell((2, 1, 1))) == pytest.approx(a / 2)


def test_gen_cartesian_coordinates_are_converted():
    gen = """2 S
Na Cl
1 1 0.0 0.0 0.0
2 2 2.82 0.0 0.0
0.0 0.0 0.0
5.64 0.0 0.0
0.0 5.64 0.0
0.0 0.0 5.64
"""
    crystal = Crystal.from_gen_string(gen)
    np.testing.assert_allclose(crystal.asymmetric_unit.positions[1], [0.5, 0, 0])


@pytest.mark.parametrize(
    "symbol,number,choice",
    [
        ("P 21/c", 14, "b1"),
        ("P 21/n", 14, "b2"),
        ("P 1 21/n 1", 14, "b2"),
        ("P 21 21 21", 19, ""),
        ("P n a 21", 33, ""),
        ("C 2/c", 15, "b1"),
        ("I 41/a", 88, "2"),
        ("R -3", 148, "H"),
        ("F m -3 m", 225, ""),
    ],
)
def test_space_group_from_short_symbols(symbol, number, choice):
    sg = SpaceGroup.from_symbol(symbol)
    assert (sg.international_tables_number, sg.choice) == (number, choice)


def test_dimers_include_the_positive_neighbouring_cell():
    # a chain of molecules along a, the asymmetric unit written mostly at
    # negative x: each molecule has a neighbour at -a and at +a
    cell = UnitCell.orthorhombic(8.0, 10.0, 10.0)
    x = np.array([-4.4, -2.88, -1.36, 0.16]) / 8.0
    positions = np.c_[x, np.full(4, 0.5), np.full(4, 0.5)]
    crystal = Crystal(
        cell, SpaceGroup(1), AsymmetricUnit([Element["C"]] * 4, positions)
    )
    _, dimers = crystal.symmetry_unique_dimers(radius=3.8)
    assert len(dimers[0]) == len(crystal.molecular_shell(0, radius=3.8)) == 2


def test_hydrogen_normalisation_follows_bonds_across_cells():
    crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
    crystal.unit_cell_molecules()  # populate the caches it must invalidate
    crystal.normalize_hydrogen_bondlengths()
    for molecule in crystal.unit_cell_molecules():
        numbers, positions = molecule.atomic_numbers, molecule.positions
        for h in np.where(numbers == 1)[0]:
            d = np.linalg.norm(positions - positions[h], axis=1)
            d[numbers == 1] = np.inf
            expected = {6: 1.083, 8: 0.983}[int(numbers[d.argmin()])]
            assert d.min() == pytest.approx(expected, abs=1e-6)


def test_monoclinic_unique_parameters_include_c():
    cell = UnitCell.monoclinic(5.0, 6.0, 7.0, np.radians(110))
    assert cell.unique_parameters_deg == pytest.approx((5.0, 6.0, 7.0, 110.0))


def test_translation_just_below_one_is_the_identity():
    from chmpy.crystal import SymmetryOperation

    assert SymmetryOperation.from_string_code(
        "x,y,z-0.0000001"
    ) == SymmetryOperation.from_string_code("x,y,z")

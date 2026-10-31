import numpy as np
import pytest

from chmpy.crystal import SymmetryOperation
from chmpy.crystal.point_group import POINT_GROUP_DATA, POINT_GROUP_FROM_NUMBER


def rotation(code):
    return np.rint(SymmetryOperation.from_string_code(code).rotation)


def group_rotations(pg):
    return np.array([np.rint(s.rotation) for s in pg.symmetry_operations])


def character(table, label, code):
    "The character of representation `label` on the operation `code`"
    k = table.classes_of(rotation(code))
    return table.characters[table.labels.index(label), k]


@pytest.mark.parametrize("pg", POINT_GROUP_DATA, ids=lambda pg: pg.symbol)
def test_rows_are_orthogonal(pg):
    t = pg.character_table
    gram = (t.characters * t.class_sizes) @ t.characters.T
    np.testing.assert_array_equal(gram, np.diag(t.order * t.norms))


@pytest.mark.parametrize("pg", POINT_GROUP_DATA, ids=lambda pg: pg.symbol)
def test_dimensions_account_for_the_group(pg):
    t = pg.character_table
    assert np.sum(t.characters[:, 0] ** 2 / t.norms) == t.order
    assert t.norms.sum() == len(t.classes)


@pytest.mark.parametrize("number", range(1, 33))
def test_labels_do_not_depend_on_the_setting(number):
    labels = {pg.character_table.labels for pg in POINT_GROUP_FROM_NUMBER[number]}
    assert len(labels) == 1


# characters on named operations, from the standard (Mulliken) tables
REFERENCE = {
    ("mm2", ""): {
        "A2": {"-x,-y,z": 1, "x,-y,z": -1, "-x,y,z": -1},
        "B1": {"-x,-y,z": -1, "x,-y,z": 1, "-x,y,z": -1},
        "B2": {"-x,-y,z": -1, "x,-y,z": -1, "-x,y,z": 1},
    },
    ("mmm", ""): {
        "B1g": {"-x,-y,z": 1, "-x,y,-z": -1, "-x,-y,-z": 1},
        "B3u": {"x,-y,-z": 1, "-x,-y,z": -1, "-x,-y,-z": -1},
    },
    ("4", ""): {"B": {"-y,x,z": -1, "-x,-y,z": 1}, "E": {"-y,x,z": 0, "-x,-y,z": -2}},
    ("4mm", ""): {
        "B1": {"-y,x,z": -1, "x,-y,z": 1, "y,x,z": -1},
        "B2": {"-y,x,z": -1, "x,-y,z": -1, "y,x,z": 1},
    },
    ("-42m", "-42m"): {
        "B1": {"y,-x,-z": -1, "x,-y,-z": 1, "y,x,z": -1},
        "B2": {"y,-x,-z": -1, "x,-y,-z": -1, "y,x,z": 1},
    },
    ("-4m2", "-4m2"): {
        "B1": {"y,-x,-z": -1, "y,x,-z": 1, "x,-y,z": -1},
        "B2": {"y,-x,-z": -1, "y,x,-z": -1, "x,-y,z": 1},
    },
    ("3", "H"): {"E": {"-y,x-y,z": -1}},
    ("3m1", "3m1"): {"A2": {"-y,-x,z": -1}, "E": {"-y,x-y,z": -1, "-y,-x,z": 0}},
    ("-3m", "R"): {
        "A2g": {"z,x,y": 1, "-y,-x,-z": -1, "-x,-y,-z": 1},
        "Eu": {"z,x,y": -1, "-y,-x,-z": 0, "-x,-y,-z": -2},
    },
    ("-6m2", "-6m2"): {
        "A2''": {"x,y,-z": -1, "-y,-x,-z": -1},
        "E'": {"x,y,-z": 2, "-y,x-y,z": -1},
    },
    ("-62m", "-62m"): {"A2''": {"x,y,-z": -1, "x-y,-y,-z": -1}},
    ("6", ""): {"E1": {"-x,-y,z": -2}, "E2": {"-x,-y,z": 2}},
    ("6/mmm", ""): {
        "B1g": {"x-y,-y,-z": 1, "-y,-x,-z": -1, "-x,-y,z": -1},
        "E2u": {"-x,-y,z": 2, "-x,-y,-z": -2},
    },
    ("-43m", ""): {"T2": {"y,-x,-z": -1, "y,x,z": 1, "z,x,y": 0}},
    ("m-3m", ""): {
        "A2g": {"-y,x,z": -1, "y,x,-z": -1},
        "T1u": {"-y,x,z": 1, "-x,-y,-z": -3},
        "T2g": {"-y,x,z": -1, "z,x,y": 0},
    },
}


def _point_group(symbol, choice):
    return next(
        pg
        for pg in POINT_GROUP_DATA
        if pg.symbol == symbol and (not choice or pg.choice == choice)
    )


@pytest.mark.parametrize("key", REFERENCE, ids=lambda k: k[0])
def test_characters_match_the_standard_tables(key):
    table = _point_group(*key).character_table
    for label, values in REFERENCE[key].items():
        for code, expected in values.items():
            assert character(table, label, code) == expected, (label, code)


# the irreps that (x, y, z) span: these fix the remaining conventions
VECTOR = {
    "2": ["A", "B"],
    "222": ["B1", "B2", "B3"],
    "mm2": ["A1", "B1", "B2"],
    "4/mmm": ["A2u", "Eu"],
    "-42m": ["B2", "E"],
    "-4m2": ["B2", "E"],
    "-3m": ["A2u", "Eu"],
    "-6m2": ["A2''", "E'"],
    "6": ["A", "E1"],
    "6/mmm": ["A2u", "E1u"],
    "-43m": ["T2"],
    "m-3m": ["T1u"],
}


@pytest.mark.parametrize("symbol", VECTOR)
def test_the_vector_representation(symbol):
    pg = _point_group(symbol, "")
    table = pg.character_table
    rotations = group_rotations(pg)
    traces = np.zeros(len(table.classes))
    np.add.at(
        traces, table.classes_of(rotations), np.trace(rotations, axis1=1, axis2=2)
    )
    counts = table.reduce(traces / table.class_sizes)
    present = {label for label, n in zip(table.labels, counts, strict=True) if n}
    assert present == set(VECTOR[symbol])

from chmpy.templates import load_template
from chmpy.core import Element, Molecule
from chmpy.crystal import AsymmetricUnit, UnitCell, SpaceGroup
import numpy as np
import logging
import re
from collections import namedtuple

LOG = logging.getLogger(__name__)
GULP_TEMPLATE = load_template("gulp")
NUMBER_REGEX = re.compile(r"([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)\s*([\w/\^\s]+)?")

Cell = namedtuple("Cell", "a b c alpha beta gamma")

def parse_value(string, with_units=False):
    """parse a value from a GULP output file to its appropriate type
    e.g. int, float, str etc. Will handle units with a space.

    Parameters
    ----------
    string: str
        the string containing the value to parse

    with_uncertainty: bool, optional
        return a tuple including uncertainty if a numeric type is expected

    Returns
    -------
    value
        the value coerced into the appropriate type

    >>> parse_value("2.3 kj/mol", with_units=True)
    (2.3, 'kj/mol')
    >>> parse_value("5 kgm^2", with_units=True)
    (5, 'kgm^2')
    >>> parse_value("string help")
    'string help'
    >>> parse_value("3.1415") * 4
    12.566
    """
    match = NUMBER_REGEX.match(string)
    try:
        if match and match:
            groups = match.groups()
            number = groups[0]
            number = float(number)
            if number.is_integer():
                number = int(number)
            if with_units and len(groups) > 1:
                return number, groups[1]
            return number
        else:
            s = string.strip()
            return s
    except Exception as e:
        print(e)
    return string


def crystal_to_gulp_input(crystal, keywords=[], additional_keywords={}):
    pos = crystal.asymmetric_unit.positions
    el = crystal.asymmetric_unit.elements
    return GULP_TEMPLATE.render(
        keywords=keywords,
        frac=True,
        cell=" ".join(f"{x:10.6f}" for x in crystal.unit_cell.parameters),
        atoms=zip(el, pos),
        spacegroup=crystal.space_group.crystal17_spacegroup_symbol(),
        additional_keywords=additional_keywords,
    )


def molecule_to_gulp_input(molecule, keywords=[], additional_keywords={}):
    pos = molecule.positions
    el = molecule.elements
    return GULP_TEMPLATE.render(
        keywords=keywords,
        frac=False,
        atoms=zip(el, pos),
        additional_keywords=additional_keywords,
    )


def parse_single_line(line):
    toks = re.split(r"\s*[=:]\s*", line)
    if toks is not None and len(toks) == 2:
        return toks[0].strip(), toks[1].strip()
    return None

def parse_gulp_output(contents):
    lines = contents.splitlines()
    lines_iter = iter(lines)
    outputs = {}
    for line in lines_iter:
        result = parse_single_line(line)
        if result is not None:
            k, v = result
            outputs[k] = v
    return outputs

from chmpy.templates import load_template
from chmpy.core import Element, Molecule
from chmpy.crystal import AsymmetricUnit, UnitCell, SpaceGroup
import numpy as np
import logging
from collections import namedtuple

LOG = logging.getLogger(__name__)
GULP_TEMPLATE = load_template("gulp")

Cell = namedtuple("Cell", "a b c alpha beta gamma")


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



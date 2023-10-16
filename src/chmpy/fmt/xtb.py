from chmpy.templates import load_template
from chmpy.core import Element, Molecule
from chmpy.crystal import AsymmetricUnit, UnitCell, SpaceGroup
import numpy as np
import logging
from collections import namedtuple

LOG = logging.getLogger(__name__)
TMOL_TEMPLATE = load_template("turbomole")

Cell = namedtuple("Cell", "a b c alpha beta gamma")


def crystal_to_turbomole_string(crystal, **kwargs):
    uc_atoms = crystal.unit_cell_atoms()
    pos = uc_atoms["frac_pos"]
    el = [Element[x] for x in uc_atoms["element"]]
    return TMOL_TEMPLATE.render(
        periodic=True,
        lattice=crystal.unit_cell.lattice.T,
        lattice_units="angs",
        atoms=zip(pos, el),
        units="frac",
        blocks=kwargs,
    )


def molecule_to_turbomole_string(molecule, **kwargs):
    return TMOL_TEMPLATE.render(
        atoms=zip(molecule.positions, molecule.elements), units="angs", blocks=kwargs,
    )


def turbomole_string(obj, **kwargs):
    if isinstance(obj, Molecule):
        return molecule_to_turbomole_string(obj, **kwargs)
    else:
        return crystal_to_turbomole_string(obj, **kwargs)


def load_turbomole_string(tmol_string):
    from chmpy.util.unit import units

    "Initialize from an xtb coord string resulting from optimization"
    data = {}
    sections = tmol_string.split("$")
    for section in sections:
        if not section or section.startswith("end"):
            continue
        lines = section.strip().splitlines()
        label = lines[0].strip()
        data[label] = [x.strip() for x in lines[1:]]
    lattice = [] if "lattice bohr" in data else None
    elements = []
    positions = []
    for line in data.pop("coord"):
        x, y, z, el = line.split()
        positions.append((float(x), float(y), float(z)))
        elements.append(Element[el])
    ANGS = units.angstrom(1.0)
    pos_cart = np.array(positions) * ANGS
    result = {
        "positions": pos_cart,
        "elements": elements,
    }
    if lattice is not None:
        for line in data.pop("lattice bohr"):
            lattice.append([float(x) for x in line.split()])
        direct = np.array(lattice) * ANGS
        uc = UnitCell(direct)
        pos_frac = uc.to_fractional(pos_cart)
        asym = AsymmetricUnit(elements, pos_frac)
        result["unit_cell"] = uc
        result["asymmetric_unit"] = asym
        result["space_group"] = SpaceGroup(1)

    result.update(**data)
    return result

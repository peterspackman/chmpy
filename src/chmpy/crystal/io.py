"""File I/O functions for Crystal structures.

All loaders and savers are implemented as standalone functions.
Loaders return Crystal instances; savers accept a crystal as first argument.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np

from chmpy.core.element import Element
from chmpy.fmt.cif import Cif

from .asymmetric_unit import AsymmetricUnit
from .space_group import SpaceGroup, SymmetryOperation
from .unit_cell import UnitCell

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch maps
# ---------------------------------------------------------------------------

def _ext_load_map():
    return {
        ".cif": from_cif_file,
        ".res": from_shelx_file,
        ".vasp": from_vasp_file,
        ".pdb": from_pdb_file,
        ".gen": from_gen_file,
        ".in": from_aims_file,
    }


def _ext_save_map(crystal):
    return {".cif": lambda f, **kw: to_cif_file(crystal, f, **kw),
            ".res": lambda f, **kw: to_shelx_file(crystal, f)}


def _fname_load_map():
    return {
        "POSCAR": from_vasp_file,
        "CONTCAR": from_vasp_file,
        "geometry.in": from_aims_file,
    }


def _fname_save_map(crystal):
    return {
        "POSCAR": lambda f, **kw: to_poscar_file(crystal, f, **kw),
        "CONTCAR": lambda f, **kw: to_poscar_file(crystal, f, **kw),
    }


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def load(filename, **kwargs) -> Union["Crystal", dict]:
    """
    Load a crystal structure from file (.res, .cif)

    Args:
        filename (str): the path to the crystal structure file

    Returns:
        the resulting crystal structure or dictionary of crystal structures
    """
    fpath = Path(filename)
    n = fpath.name
    fname_map = _fname_load_map()
    if n in fname_map:
        return fname_map[n](filename)
    extension_map = _ext_load_map()
    extension = kwargs.pop("fmt", fpath.suffix.lower())
    if not extension.startswith("."):
        extension = "." + extension
    return extension_map[extension](filename, **kwargs)


def save(crystal, filename, **kwargs):
    """Save a crystal structure to file (.cif, .res, POSCAR)"""
    fpath = Path(filename)
    n = fpath.name
    fname_map = _fname_save_map(crystal)
    if n in fname_map:
        return fname_map[n](filename, **kwargs)
    extension_map = _ext_save_map(crystal)
    extension = kwargs.pop("fmt", fpath.suffix.lower())
    if not extension.startswith("."):
        extension = "." + extension
    return extension_map[extension](filename, **kwargs)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def from_vasp_string(string, **kwargs):
    "Initialize a crystal structure from a VASP POSCAR string"
    from .crystal import Crystal

    from chmpy.fmt.vasp import parse_poscar

    vasp_data = parse_poscar(string)
    uc = UnitCell(vasp_data["direct"])
    sg = SpaceGroup(1)
    coords = vasp_data["positions"]
    if not vasp_data["coord_type"].startswith("d"):
        coords = uc.to_fractional(coords)
    asym = AsymmetricUnit(vasp_data["elements"], coords)
    return Crystal(uc, sg, asym, titl=vasp_data["name"])


def from_vasp_file(filename, **kwargs):
    "Initialize a crystal structure from a VASP POSCAR file"
    return from_vasp_string(Path(filename).read_text(), **kwargs)


def from_aims_string(string, **kwargs):
    "Initialize a crystal structure from an FHI-aims geometry.in string"
    from .crystal import Crystal

    from chmpy.fmt.aims import parse_geometry_string

    aims_data = parse_geometry_string(string)
    if "lattice" not in aims_data:
        raise ValueError("FHI-aims geometry.in file must contain lattice vectors for Crystal")

    uc = UnitCell(aims_data["lattice"])
    sg = SpaceGroup(1)

    # Convert to fractional if necessary
    coords = aims_data["positions"]
    if not aims_data["fractional"]:
        coords = uc.to_fractional(coords)

    asym = AsymmetricUnit(aims_data["elements"], coords)
    return Crystal(uc, sg, asym)


def from_aims_file(filename, **kwargs):
    "Initialize a crystal structure from an FHI-aims geometry.in file"
    return from_aims_string(Path(filename).read_text(), **kwargs)


def from_ase_atoms(atoms, **kwargs):
    from chmpy.ext.ase import ase_to_crystal

    return ase_to_crystal(atoms, **kwargs)


def from_cif_data(cif_data, titl=None):
    """Initialize a crystal structure from a dictionary
    of CIF data"""
    from .crystal import Crystal

    labels = cif_data.get("atom_site_label", None)
    symbols = cif_data.get("atom_site_type_symbol", None)
    if symbols is None:
        if labels is None:
            raise ValueError(
                "Unable to determine elements in CIF, "
                "need one of _atom_site_label or "
                "_atom_site_type_symbol present"
            )
        elements = [Element[x] for x in labels]
    else:
        elements = [Element[x] for x in symbols]
    x = np.asarray(cif_data.get("atom_site_fract_x", []))
    y = np.asarray(cif_data.get("atom_site_fract_y", []))
    z = np.asarray(cif_data.get("atom_site_fract_z", []))
    occupation = np.asarray(cif_data.get("atom_site_occupancy", [1] * len(x)))
    frac_pos = np.array([x, y, z]).T
    asym = AsymmetricUnit(
        elements=elements, positions=frac_pos, labels=labels, occupation=occupation
    )
    lengths = [cif_data[f"cell_length_{x}"] for x in ("a", "b", "c")]
    angles = [cif_data[f"cell_angle_{x}"] for x in ("alpha", "beta", "gamma")]
    unit_cell = UnitCell.from_lengths_and_angles(lengths, angles, unit="degrees")

    space_group = SpaceGroup(1)
    symop_data_names = (
        "symmetry_equiv_pos_as_xyz",
        "space_group_symop_operation_xyz",
    )
    number = space_group.international_tables_number
    for k in ("space_group_IT_number", "symmetry_Int_Tables_number"):
        if k in cif_data:
            number = cif_data[k]
            break

    # Try to parse the Hermann-Mauguin symbol first
    hm_parsed = False
    hm_symbol = cif_data.get("symmetry_space_group_name_H-M", "").strip()
    if hm_symbol:
        try:
            # Convert CIF Hermann-Mauguin notation to correct SpaceGroup
            space_group = _parse_hermann_mauguin_symbol(hm_symbol, number)
            hm_parsed = True
        except (ValueError, KeyError):
            # Fall back to symmetry operations if HM symbol parsing fails
            pass

    # Only try symmetry operations if HM parsing failed
    if not hm_parsed:
        for symop_data_block in symop_data_names:
            if symop_data_block in cif_data:
                symops = [
                    SymmetryOperation.from_string_code(x)
                    for x in cif_data[symop_data_block]
                ]
                try:
                    new_sg = SpaceGroup.from_symmetry_operations(symops)
                    space_group = new_sg
                except ValueError:
                    space_group.symmetry_operations = symops
                    symbol = cif_data.get(
                        "symmetry_space_group_name_H-M", "Unknown"
                    )
                    space_group.international_tables_number = number
                    space_group.symbol = symbol
                    space_group.full_symbol = symbol
                    LOG.warn(
                        "Initializing non-standard spacegroup setting %s, "
                        "some SG data may be missing",
                        symbol,
                    )
                break
        else:
            # fall back to international tables number
            space_group = SpaceGroup(number)

    return Crystal(unit_cell, space_group, asym, cif_data=cif_data, titl=titl)


def _parse_hermann_mauguin_symbol(hm_symbol, sg_number):
    """
    Parse Hermann-Mauguin symbol from CIF and find matching SpaceGroup.

    Args:
        hm_symbol (str): Hermann-Mauguin symbol from CIF (e.g. 'P C M 21')
        sg_number (int): Space group number from CIF

    Returns:
        SpaceGroup: Matching space group object
    """
    from .space_group import SG_FROM_NUMBER

    # Clean up the symbol - remove extra spaces, normalize
    clean_symbol = " ".join(hm_symbol.upper().split())

    # Get all possible settings for this space group number
    if str(sg_number) not in SG_FROM_NUMBER:
        raise ValueError(f"Space group number {sg_number} not found")

    sg_settings = SG_FROM_NUMBER[str(sg_number)]

    # Try each setting and check if crystal17_spacegroup_symbol matches
    for sg_data in sg_settings:
        try:
            sg = SpaceGroup(sg_number, choice=sg_data.choice)
            crystal17_symbol = sg.crystal17_spacegroup_symbol().upper()

            if clean_symbol == crystal17_symbol:
                return sg
        except Exception as e:
            LOG.debug(
                "Exception encountered when determining space group setting: %s", e
            )
            continue

    # If no match found, raise error
    raise ValueError(
        f"Could not match Hermann-Mauguin symbol '{hm_symbol}' "
        f"to any setting of space group #{sg_number}"
    )


def from_cif_file(filename, data_block_name=None):
    """Initialize a crystal structure from a CIF file"""
    cif = Cif.from_file(filename)
    if data_block_name is not None:
        return from_cif_data(cif.data[data_block_name], titl=data_block_name)

    crystals = {
        name: from_cif_data(data, titl=name) for name, data in cif.data.items()
    }
    keys = list(crystals.keys())
    if len(keys) == 1:
        return crystals[keys[0]]
    return crystals


def from_pdb_file(filename):
    from .crystal import Crystal

    from chmpy.fmt.pdb import Pdb

    pdb = Pdb.from_file(filename)
    uc = UnitCell.from_lengths_and_angles(
        [pdb.unit_cell["a"], pdb.unit_cell["b"], pdb.unit_cell["c"]],
        [pdb.unit_cell["alpha"], pdb.unit_cell["beta"], pdb.unit_cell["gamma"]],
        unit="degrees",
    )
    pos_cart = np.c_[pdb.atoms["x"], pdb.atoms["y"], pdb.atoms["z"]]
    pos_frac = uc.to_fractional(pos_cart)
    elements = [Element.from_string(x) for x in pdb.atoms["element"]]
    labels = pdb.atoms["name"]
    asym = AsymmetricUnit(elements, pos_frac, labels=labels)
    sg = SpaceGroup.from_symbol(pdb.space_group)
    return Crystal(uc, sg, asym)


def from_cif_string(file_content, **kwargs):
    data_block_name = kwargs.get("data_block_name", None)
    cif = Cif.from_string(file_content)
    if data_block_name is not None:
        return from_cif_data(cif.data[data_block_name], titl=data_block_name)

    crystals = {
        name: from_cif_data(data, titl=name) for name, data in cif.data.items()
    }
    keys = list(crystals.keys())
    if len(keys) == 1:
        return crystals[keys[0]]
    return crystals


def from_shelx_file(filename, **kwargs):
    """Initialize a crystal structure from a shelx .res file"""
    p = Path(filename)
    titl = p.stem
    return from_shelx_string(p.read_text(), titl=titl, **kwargs)


def from_shelx_string(file_content, **kwargs):
    """Initialize a crystal structure from a shelx .res string"""
    from .crystal import Crystal

    from chmpy.fmt.shelx import parse_shelx_file_content

    shelx_dict = parse_shelx_file_content(file_content)
    asymmetric_unit = AsymmetricUnit.from_records(shelx_dict["ATOM"])
    space_group = SpaceGroup.from_symmetry_operations(
        shelx_dict["SYMM"], expand_latt=shelx_dict["LATT"]
    )
    unit_cell = UnitCell.from_lengths_and_angles(
        shelx_dict["CELL"]["lengths"], shelx_dict["CELL"]["angles"], unit="degrees"
    )
    return Crystal(unit_cell, space_group, asymmetric_unit, **kwargs)


def from_crystal17_opt_string(string, **kwargs):
    from .crystal import Crystal

    from chmpy.fmt.crystal17 import load_crystal17_geometry_string

    data = load_crystal17_geometry_string(string)
    unit_cell = UnitCell(data["direct"])
    space_group = SpaceGroup.from_symmetry_operations(data["symmetry_operations"])
    asym = AsymmetricUnit(data["elements"], unit_cell.to_fractional(data["xyz"]))
    return Crystal(unit_cell, space_group, asym)


def from_crystal17_opt_file(filename, **kwargs):
    p = Path(filename)
    titl = p.stem
    return from_crystal17_opt_string(p.read_text(), titl=titl, **kwargs)


def from_molecule(molecule, **kwargs):
    from .crystal import Crystal

    unit_cell = UnitCell.cubic(1000)

    asym = AsymmetricUnit(
        elements=molecule.elements,
        positions=unit_cell.to_fractional(molecule.positions),
        labels=molecule.labels,
    )
    space_group = SpaceGroup(1)
    x = Crystal(unit_cell, space_group, asym)
    _ = x.unit_cell_atoms(
        tolerance=1e-12
    )  # need to workaround default tolerance as we have a massive cell
    return x


def from_gen_string(contents, **kwargs):
    from .crystal import Crystal

    from chmpy.fmt.gen import parse_gen_string

    elements, positions, cell, fractional = parse_gen_string(contents)
    unit_cell = UnitCell(cell[1:4, :])

    asym = AsymmetricUnit(
        elements=elements,
        positions=positions,
    )
    space_group = SpaceGroup(1)
    return Crystal(unit_cell, space_group, asym, **kwargs)


def from_gen_file(filename, **kwargs):
    p = Path(filename)
    titl = p.stem
    return from_gen_string(p.read_text(), titl=titl, **kwargs)


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------

def to_ase_atoms(crystal, **kwargs):
    from chmpy.ext.ase import crystal_to_ase

    return crystal_to_ase(crystal)


def to_cif_data(crystal, data_block_name=None) -> dict:
    "Convert a crystal structure to cif data dict"
    version = "1.0a1"
    if data_block_name is None:
        data_block_name = crystal.titl
    if "cif_data" in crystal.properties:
        cif_data = crystal.properties["cif_data"]
        cif_data["audit_creation_method"] = (
            f"chmpy python library version {version}"
        )
        cif_data["atom_site_fract_x"] = crystal.asymmetric_unit.positions[:, 0]
        cif_data["atom_site_fract_y"] = crystal.asymmetric_unit.positions[:, 1]
        cif_data["atom_site_fract_z"] = crystal.asymmetric_unit.positions[:, 2]
    else:
        cif_data = {
            "audit_creation_method": f"chmpy python library version {version}",
            "symmetry_equiv_pos_site_id": list(
                range(1, len(crystal.symmetry_operations) + 1)
            ),
            "symmetry_equiv_pos_as_xyz": [str(x) for x in crystal.symmetry_operations],
            "cell_length_a": crystal.unit_cell.a,
            "cell_length_b": crystal.unit_cell.b,
            "cell_length_c": crystal.unit_cell.c,
            "cell_angle_alpha": crystal.unit_cell.alpha_deg,
            "cell_angle_beta": crystal.unit_cell.beta_deg,
            "cell_angle_gamma": crystal.unit_cell.gamma_deg,
            "atom_site_label": crystal.asymmetric_unit.labels,
            "atom_site_type_symbol": [
                x.symbol for x in crystal.asymmetric_unit.elements
            ],
            "atom_site_fract_x": crystal.asymmetric_unit.positions[:, 0],
            "atom_site_fract_y": crystal.asymmetric_unit.positions[:, 1],
            "atom_site_fract_z": crystal.asymmetric_unit.positions[:, 2],
            "atom_site_occupancy": crystal.asymmetric_unit.properties.get(
                "occupation", np.ones(len(crystal.asymmetric_unit))
            ),
        }
    return {data_block_name: cif_data}


def to_cif_file(crystal, filename, **kwargs):
    "save a crystal to a CIF formatted file"
    cif_data = to_cif_data(crystal, **kwargs)
    return Cif(cif_data).to_file(filename)


def to_cif_string(crystal, **kwargs):
    "save a crystal to a CIF formatted string"
    cif_data = to_cif_data(crystal, **kwargs)
    return Cif(cif_data).to_string()


def to_poscar_string(crystal, **kwargs):
    "save a crystal to a VASP POSCAR formatted string"
    from chmpy.ext.vasp import poscar_string

    return poscar_string(crystal, name=crystal.titl)


def to_poscar_file(crystal, filename, **kwargs):
    "save a crystal to a VASP POSCAR formatted file"
    Path(filename).write_text(to_poscar_string(crystal, **kwargs))


def to_shelx_file(crystal, filename):
    """Write a crystal structure as a shelx .res formatted file"""
    Path(filename).write_text(to_shelx_string(crystal))


def to_shelx_string(crystal, titl=None):
    """Represent a crystal structure as a shelx .res formatted string"""
    from chmpy.fmt.shelx import to_res_contents

    sfac = list(np.unique(crystal.site_atoms))
    atom_sfac = [sfac.index(x) + 1 for x in crystal.site_atoms]
    shelx_data = {
        "TITL": crystal.titl if titl is None else titl,
        "CELL": crystal.unit_cell.parameters,
        "SFAC": [Element[x].symbol for x in sfac],
        "SYMM": [
            str(s)
            for s in crystal.space_group.reduced_symmetry_operations()
            if not s.is_identity()
        ],
        "LATT": crystal.space_group.latt,
        "ATOM": [
            "{:3} {:3} {: 20.12f} {: 20.12f} {: 20.12f}".format(l, s, *pos)
            for l, s, pos in zip(
                crystal.asymmetric_unit.labels,
                atom_sfac,
                crystal.site_positions,
                strict=False,
            )
        ],
    }
    return to_res_contents(shelx_data)


def to_pdb_string(crystal, header=None):
    """Represent a crystal structure as a PDB formatted string."""
    from chmpy.fmt.pdb import Pdb

    pdb = Pdb.from_crystal(crystal, header=header)
    return pdb.to_string()


def to_pdb_file(crystal, filename, header=None):
    """Write a crystal structure as a PDB formatted file."""
    Path(filename).write_text(to_pdb_string(crystal, header=header))

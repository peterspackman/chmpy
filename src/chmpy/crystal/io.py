"""File I/O functions for Crystal structures.

All loaders and savers are implemented as standalone functions.
Loaders return Crystal instances; savers accept a crystal as first argument.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from chmpy.core.element import Element
from chmpy.fmt.cif import Cif, CifBlock, canonical_data_name, is_scalar

from .asymmetric_unit import AsymmetricUnit
from .space_group import SpaceGroup, SymmetryOperation
from .unit_cell import UnitCell

if TYPE_CHECKING:
    from .crystal import Crystal

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

def load(filename, **kwargs) -> Crystal | dict:
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
    from chmpy.fmt.vasp import parse_poscar

    from .crystal import Crystal

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
    from chmpy.fmt.aims import parse_geometry_string

    from .crystal import Crystal

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
    lengths = [cif_data[f"cell_length_{k}"] for k in ("a", "b", "c")]
    angles = [cif_data[f"cell_angle_{k}"] for k in ("alpha", "beta", "gamma")]
    unit_cell = UnitCell.from_lengths_and_angles(lengths, angles, unit="degrees")

    if "atom_site_fract_x" in cif_data:
        frac_pos = np.array(
            [np.asarray(cif_data[f"atom_site_fract_{k}"]) for k in "xyz"]
        ).T
    elif "atom_site_Cartn_x" in cif_data:
        # mmCIF stores orthogonal coordinates rather than fractional ones
        cart_pos = np.array(
            [np.asarray(cif_data[f"atom_site_Cartn_{k}"], dtype=float) for k in "xyz"]
        ).T
        frac_pos = unit_cell.to_fractional(cart_pos)
    else:
        raise ValueError(
            "Unable to determine atomic positions in CIF, need either "
            "_atom_site_fract_{x,y,z} or _atom_site_Cartn_{x,y,z}"
        )

    occupation = np.asarray(cif_data.get("atom_site_occupancy", [1] * len(frac_pos)))
    asym = AsymmetricUnit(
        elements=elements, positions=frac_pos, labels=labels, occupation=occupation
    )

    space_group = SpaceGroup(1)
    symop_data_names = (
        "symmetry_equiv_pos_as_xyz",
        "space_group_symop_operation_xyz",
    )
    number = space_group.international_tables_number
    for k in ("space_group_IT_number", "symmetry_Int_Tables_number"):
        if k in cif_data:
            number = int(cif_data[k])
            break

    # Try to parse the Hermann-Mauguin symbol first
    hm_parsed = False
    hm_symbol = str(cif_data.get("symmetry_space_group_name_H-M") or "").strip()
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


def from_cif_file(filename, data_block_name=None, options=None):
    """Initialize a crystal structure from a CIF file

    ``options`` is a :class:`chmpy.fmt.cif.CifOptions`, for the rare file that
    needs reading in a way other than plain CIF 1.1.
    """
    cif = Cif.from_file(filename, options=options)
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
    from chmpy.fmt.pdb import Pdb

    from .crystal import Crystal

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


def from_cif_string(file_content, data_block_name=None, options=None, **kwargs):
    """Initialize a crystal structure from the contents of a CIF

    ``options`` is a :class:`chmpy.fmt.cif.CifOptions`, for the rare file that
    needs reading in a way other than plain CIF 1.1.
    """
    cif = Cif.from_string(file_content, options=options)
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
    from chmpy.fmt.shelx import parse_shelx_file_content

    from .crystal import Crystal

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
    from chmpy.fmt.crystal17 import load_crystal17_geometry_string

    from .crystal import Crystal

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
    from chmpy.fmt.gen import parse_gen_string

    from .crystal import Crystal

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


#: Data names a crystal structure cannot vouch for once it has been built.
#: The space group is the crystal's own, and bond lengths and anisotropic
#: displacements describe sites that may since have moved or gone.
REPLACED_CIF_CATEGORIES = (
    "symmetry_",
    "space_group_",
    "geom_",
    "atom_site_aniso",
)


def _is_per_site(name, value):
    "whether a carried over data name is a column with one value per atom site"
    return canonical_data_name(name).startswith("atom_site") and not is_scalar(value)


def _carried_over(source, structural, n_sites):
    """The parts of a source CIF that are still true of the crystal.

    Anything the crystal describes itself is dropped in favour of its own
    state; the rest -- who published it, what it is, how it was measured --
    is carried through untouched.
    """
    regenerated = {canonical_data_name(k) for k in structural}
    kept = {}
    for name, value in source.items():
        canonical = canonical_data_name(name)
        if canonical in regenerated:
            continue  # written from the crystal instead
        if canonical.startswith(REPLACED_CIF_CATEGORIES):
            continue
        if _is_per_site(name, value) and len(value) != n_sites:
            continue  # no longer one value per site
        kept[name] = value
    return kept


#: the atom site columns a crystal structure always writes, in order
CORE_ATOM_SITE_NAMES = (
    "atom_site_label",
    "atom_site_type_symbol",
    "atom_site_fract_x",
    "atom_site_fract_y",
    "atom_site_fract_z",
    "atom_site_occupancy",
)
SYMMETRY_LOOP_NAMES = ("symmetry_equiv_pos_site_id", "symmetry_equiv_pos_as_xyz")


def to_cif_data(crystal, data_block_name=None, source_data=True) -> CifBlock:
    """Convert a crystal structure to CIF data.

    The crystal is the source of truth for the cell, the symmetry and the
    atom sites: these are always written from its current state, so a
    structure altered since it was read is written as it is now, not as it
    was on disk.

    When the crystal was read from a CIF, the parts of that file which are
    not about the structure -- bibliography, chemical identity, experimental
    and refinement details -- are carried through unchanged, along with any
    extra per-site column that still has one value per site.  Pass
    ``source_data=False`` for just the structure.
    """
    version = "1.0a1"
    if data_block_name is None:
        data_block_name = crystal.titl
    asym = crystal.asymmetric_unit
    cell = crystal.unit_cell
    space_group = crystal.space_group
    symmetry_operations = crystal.symmetry_operations
    positions = asym.positions

    structural = {
        "audit_creation_method": f"chmpy python library version {version}",
        "cell_length_a": cell.a,
        "cell_length_b": cell.b,
        "cell_length_c": cell.c,
        "cell_angle_alpha": cell.alpha_deg,
        "cell_angle_beta": cell.beta_deg,
        "cell_angle_gamma": cell.gamma_deg,
        "cell_volume": cell.volume(),
        "symmetry_space_group_name_H-M": space_group.symbol,
        "symmetry_Int_Tables_number": space_group.international_tables_number,
        "symmetry_equiv_pos_site_id": list(range(1, len(symmetry_operations) + 1)),
        "symmetry_equiv_pos_as_xyz": [str(x) for x in symmetry_operations],
        "atom_site_label": list(asym.labels),
        "atom_site_type_symbol": [x.symbol for x in asym.elements],
        "atom_site_fract_x": positions[:, 0],
        "atom_site_fract_y": positions[:, 1],
        "atom_site_fract_z": positions[:, 2],
        "atom_site_occupancy": asym.properties.get("occupation", np.ones(len(asym))),
    }

    source = crystal.properties.get("cif_data") if source_data else None
    carried = _carried_over(source, structural, len(asym)) if source else {}
    # a per-site column belongs in the atom site loop, the rest is metadata
    # and reads more naturally before the structure
    per_site = {k: v for k, v in carried.items() if _is_per_site(k, v)}
    metadata = {k: v for k, v in carried.items() if k not in per_site}

    block = CifBlock(name=data_block_name)
    block.update(metadata)
    block.update(structural)
    block.update(per_site)

    # say outright which columns form which loop, rather than leaving the
    # writer to guess from the data names
    block.loops = [
        list(SYMMETRY_LOOP_NAMES),
        list(CORE_ATOM_SITE_NAMES) + list(per_site),
    ]
    placed = {n for loop in block.loops for n in loop}
    for loop in getattr(source, "loops", ()):
        group = [n for n in loop if n in metadata and n not in placed]
        if group:
            block.loops.append(group)
            placed.update(group)
    return {data_block_name: block}


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

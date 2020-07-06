import logging
import numpy as np
from spglib import standardize_cell, get_symmetry_dataset
from chmpy import Element
from chmpy.crystal.space_group import SpaceGroup
from chmpy.crystal import Crystal, AsymmetricUnit, UnitCell

LOG = logging.getLogger(__name__)


def standardize_crystal(crystal, method="spglib", **kwargs):
    if method != "spglib":
        raise NotImplementedError("Only spglib is currently supported")

    lattice = crystal.unit_cell.direct
    uc_dict = crystal.unit_cell_atoms()
    positions = uc_dict["frac_pos"]
    elements = uc_dict["element"]
    asym_atoms = uc_dict["asym_atom"]
    asym_labels = uc_dict["label"]
    cell = lattice, positions, elements

    reduced_cell = standardize_cell(cell, **kwargs)

    if reduced_cell is None:
        LOG.warn("Could not find reduced cell for crystal %s", crystal)
        return None
    dataset = get_symmetry_dataset(reduced_cell)
    asym_idx = np.unique(dataset["equivalent_atoms"])
    asym_idx = asym_idx[np.argsort(asym_atoms[asym_idx])]
    sg = SpaceGroup(dataset["number"], choice=dataset["choice"])

    reduced_lattice, positions, elements = reduced_cell
    unit_cell = UnitCell(reduced_lattice)
    asym = AsymmetricUnit(
        [Element[x] for x in elements[asym_idx]],
        positions[asym_idx],
        labels=asym_labels[asym_idx],
    )
    return Crystal(unit_cell, sg, asym)


def detect_symmetry(crystal, method="spglib", **kwargs):
    if method != "spglib":
        raise NotImplementedError("Only spglib is currently supported")

    lattice = crystal.unit_cell.direct
    uc_dict = crystal.unit_cell_atoms()
    positions = uc_dict["frac_pos"]
    elements = uc_dict["element"]
    asym_atoms = uc_dict["asym_atom"]
    cell = lattice, positions, elements
    dataset = get_symmetry_dataset(cell, **kwargs)
    if dataset["number"] == crystal.space_group.international_tables_number:
        LOG.warn("Could not find additional symmetry for crystal %s", crystal)
        return None
    asym_idx = np.unique(dataset["equivalent_atoms"])
    asym_idx = asym_idx[np.argsort(asym_atoms[asym_idx])]
    sg = SpaceGroup(dataset["number"], choice=dataset["choice"])
    asym = AsymmetricUnit(
        [Element[x] for x in dataset["std_types"][asym_idx]],
        dataset["std_positions"][asym_idx],
    )
    unit_cell = UnitCell(dataset["std_lattice"])
    return Crystal(unit_cell, sg, asym)

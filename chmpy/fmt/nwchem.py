from chmpy.templates import load_template
from chmpy import Element
from pathlib import Path
import numpy as np
import logging

LOG = logging.getLogger(__name__)
NWCHEM_INPUT_TEMPLATE = load_template("nwchem_input")


def to_nwchem_input(molecule, **kwargs):
    blocks = {}
    tasks = ("scf",)

    method = kwargs.get("method", "scf")
    if method.lower() == "hf":
        method = "scf"
    basis_set = kwargs.get("basis_set", "3-21G")
    geometry_keywords = kwargs.get("geometry_keywords", "noautoz nocenter")

    blocks[f"geometry {geometry_keywords}"] = molecule.to_xyz_string(header=False)

    return NWCHEM_INPUT_TEMPLATE.render(
        title=kwargs.get("title", molecule.molecular_formula),
        method=method,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
        basis_set=basis_set,
        cartesian_basis=True,
        blocks=blocks,
        tasks=tasks,
    )

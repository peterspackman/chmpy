from chmpy import Crystal
from chmpy import Element
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist


def kpoints_string():
    return ""


def incar_string():
    return ""


def poscar_string(crystal, name="vasp_input"):
    uc = crystal.unit_cell_atoms(tolerance=1e-2)
    elements = uc["element"]
    pos = uc["frac_pos"]
    ordering = np.argsort(elements)
    coord = pos[ordering]
    elements = elements[ordering]
    element_counts = Counter(elements)
    direct = "\n".join(
        f"{x:12.8f} {y:12.8f} {z:12.8f}" for x, y, z in crystal.unit_cell.direct
    )
    els = " ".join(f"{Element[x].symbol:>3s}" for x in element_counts.keys())
    counts = " ".join(f"{x:>3d}" for x in element_counts.values())
    coords = "\n".join(f"{x:12.8f} {y:12.8f} {z:12.8f}" for x, y, z in coord)
    return f"{name}\n1.0\n{direct}\n{els}\n{counts}\nDirect\n{coords}"


def generate_vasp_inputs(crystal, dest="."):
    from pathlib import Path

    dest = Path(dest)
    if not dest.exists():
        dest.mkdir()
    Path(dest, "POSCAR").write_text(poscar_string(crystal, name=dest.name))
    Path(dest, "INCAR").write_text(incar_string())
    Path(dest, "KPOINTS").write_text(kpoints_string())


def load_vasprun(filename):
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from chmpy.crystal import UnitCell, Crystal, AsymmetricUnit, SpaceGroup

    xml = ET.parse(filename)
    root = xml.getroot()
    structures = {}
    atominfo = root.find("atominfo")
    elements = []
    for child in atominfo.findall("array"):
        if child.get("name") == "atoms":
            atoms = child.find("set")
            for atom in atoms.findall("rc"):
                elements.append(Element[atom.find("c").text])

    for structure in root.findall("structure"):
        name = structure.get("name")
        crystal = structure.find("crystal")
        positions = structure.find("varray")
        direct = None
        for child in crystal:
            if child.get("name") == "basis":
                basis = []
                for row in child:
                    basis.append([float(x) for x in row.text.split()])
                direct = np.array(basis)
        pos = []
        for row in positions:
            pos.append([float(x) for x in row.text.split()])
        pos = np.array(pos)
        structures[name] = Crystal(
            UnitCell(direct), SpaceGroup(1), AsymmetricUnit(elements, pos)
        )
    return structures

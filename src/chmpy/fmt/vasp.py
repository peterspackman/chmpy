import logging

import numpy as np

from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def parse_poscar(poscar_string):
    "Read in a VASP POSCAR or CONTCAR file"
    result = {
        "name": None,
        "direct": None,
        "elements": None,
        "positions": None,
    }
    lines = poscar_string.splitlines()
    result["name"] = lines[0].strip()
    scale_factor = float(lines[1])
    result["direct"] = scale_factor * np.fromstring(
        " ".join(lines[2:5]), sep=" "
    ).reshape((3, 3))
    elements = []
    for el, x in zip(lines[5].split(), lines[6].split(), strict=False):
        elements += [Element[el]] * int(x)
    result["elements"] = elements
    start = 7
    if lines[start].strip().lower().startswith("s"):
        start += 1
    result["coord_type"] = lines[start].strip().lower()
    N = len(elements)
    rows = [line.split() for line in lines[start + 1 : start + 1 + N]]
    result["positions"] = np.array([[float(x) for x in r[:3]] for r in rows])
    if start == 8:
        result["selective_dynamics"] = np.array(
            [[x == "T" for x in r[3:6]] for r in rows]
        )
    return result


def slab_to_poscar_string(slab, fixed=None):
    """VASP POSCAR for a periodic slab model.

    Args:
        slab: Slab
        fixed: (N,) boolean mask of atoms to fix with selective dynamics;
            defaults to `slab.fixed`, and selective dynamics is written only
            when some atom is fixed

    Returns:
        str: POSCAR contents
    """
    fixed = slab.fixed if fixed is None else np.asarray(fixed, dtype=bool)
    order = np.argsort(slab.elements, kind="stable")
    elements = slab.elements[order]
    frac = slab.frac_positions[order]
    fixed = fixed[order]
    symbols, counts = np.unique(elements, return_counts=True)
    lines = [slab.title, "1.0"]
    lines += [f"{x:16.10f} {y:16.10f} {z:16.10f}" for x, y, z in slab.cell]
    lines.append(" ".join(f"{Element[x].symbol:>3s}" for x in symbols))
    lines.append(" ".join(f"{x:>3d}" for x in counts))
    selective = bool(fixed.any())
    if selective:
        lines.append("Selective dynamics")
    lines.append("Direct")
    for (x, y, z), fix in zip(frac, fixed, strict=True):
        line = f"{x:16.10f} {y:16.10f} {z:16.10f}"
        if selective:
            flag = "F" if fix else "T"
            line += f" {flag} {flag} {flag}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def poscar_string_to_slab(contents):
    """Read a Slab from POSCAR contents.

    The third lattice vector is taken as the vacuum direction; atoms with
    all selective dynamics flags F are marked fixed.
    """
    from chmpy.crystal.slab import Slab, parse_slab_title

    data = parse_poscar(contents)
    positions = data["positions"]
    if data["coord_type"][0] not in ("c", "k"):
        positions = positions @ data["direct"]
    fixed = None
    if "selective_dynamics" in data:
        fixed = ~data["selective_dynamics"].any(axis=1)
    return Slab(
        data["direct"],
        data["elements"],
        positions,
        fixed=fixed,
        **parse_slab_title(data["name"]),
    )

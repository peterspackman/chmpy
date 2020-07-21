from chmpy.crystal import Crystal, AsymmetricUnit, SpaceGroup
from chmpy.fmt.xyz_file import parse_traj_file
from pathlib import Path


def to_xyz_string(elements, positions, comment=""):
    lines = [f"{len(elements)}", comment]
    for el, (x, y, z) in zip(elements, positions):
        lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")
    return "\n".join(lines)


def expand_periodic_images(cell, filename, dest=None, supercell=(1, 1, 1)):
    frames = parse_traj_file(filename)
    sg = SpaceGroup(1)
    xyz_strings = []
    for elements, comment, positions in frames:
        asym = AsymmetricUnit(elements, cell.to_fractional(positions))
        c = Crystal(cell, sg, asym).as_P1_supercell(supercell)
        pos = c.to_cartesian(c.asymmetric_unit.positions)
        el = c.asymmetric_unit.elements
        xyz_strings.append(to_xyz_string(el, pos, comment=comment))
    if dest is not None:
        Path(dest).write_text("\n".join(xyz_strings))
    else:
        return "\n".join(xyz_strings)

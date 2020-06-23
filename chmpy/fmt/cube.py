from pathlib import Path
import numpy as np
from io import StringIO


def parse_title_lines(lines):
    return [line.strip() for line in lines]


def parse_atom_lines(lines):
    elements = []
    masses = []
    positions = []
    for line in lines:
        tokens = line.split()
        elements.append(int(tokens[0]))
        masses.append(float(tokens[1]))
        positions.append(np.fromstring(" ".join(tokens[2:]), sep=" "))
    return np.array(elements), np.array(masses), np.vstack(positions)


def parse_cube_buf(buf):
    content = {}
    n_atom = -1
    content["title"], content["subtitle"] = parse_title_lines(
        (buf.readline(), buf.readline())
    )
    tokens = buf.readline().split()
    content["natom"] = int(tokens[0])
    content["volume_origin"] = np.fromstring(" ".join(tokens[1:]), sep=" ")
    atoms = []
    for ax in ("x", "y", "z"):
        tokens = buf.readline().split()
        content[f"n{ax}"] = int(tokens[0])
        content[f"{ax}_basis"] = np.fromstring(" ".join(tokens[1:]), sep=" ")

    elements, masses, positions = parse_atom_lines(
        buf.readline() for i in range(content["natom"])
    )
    content["atomic_number"] = elements
    content["mass"] = masses
    content["position"] = positions
    content["data"] = np.fromstring(buf.read(), sep=" ")
    return content


def parse_cube_string(string):
    return parse_cube_buf(StringIO(string))


def parse_cube_file(filename):
    with Path(filename).open() as f:
        return parse_cube_buf(f)

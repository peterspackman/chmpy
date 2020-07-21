import logging
from pathlib import Path
import numpy as np
from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def parse_xyz_string(contents, filename=None):
    """Convert provided xmol .xyz file contents into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    contents: str
        text contents of the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        array of (N) atomic numbers and (N, 3) Cartesian positions
        read from the given file
    """
    lines = contents.splitlines()
    natom = int(lines[0].strip())
    LOG.debug("Expecting %d atoms %s", natom, "in " + filename if filename else "")
    elements = []
    positions = []
    for line in lines[2:]:
        if not line.strip():
            break
        tokens = line.strip().split()
        xyz = tuple(float(x) for x in tokens[1:4])
        positions.append(xyz)
        elements.append(Element[tokens[0]])
    LOG.debug(
        "Found %d atoms lines in %s",
        len(elements),
        "in " + filename if filename else "",
    )
    return elements, np.asarray(positions)


def parse_traj_string(contents, filename=None):
    """Convert provided xmol .xyz file contents into list of arrays of
    atomic numbers and cartesian positions

    Parameters
    ----------
    contents: str
        text contents of the .xyz file to read

    Returns
    -------
    list of tuple of :obj:`np.ndarray`
        list of (N) :obj:`Element` and (N, 3) Cartesian positions
        read from the given file
    """
    lines = contents.splitlines()
    i = 0
    frames = []
    while i < len(lines):
        natom = int(lines[i].strip())
        elements = []
        positions = []
        i += 1
        comment = lines[i]
        i += 1
        for line in lines[i : i + natom]:
            if not line.strip():
                continue
            tokens = line.strip().split()
            xyz = tuple(float(x) for x in tokens[1:4])
            positions.append(xyz)
            elements.append(Element[tokens[0]])
        frames.append((elements, comment, np.asarray(positions)))
        i += natom
    return frames


def parse_xyz_file(filename):
    """Convert a provided xmol .xyz file into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    filename: str
        path to the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        array of (N) atomic numbers and (N, 3) Cartesian positions
        read from the given file
    """
    path = Path(filename)
    return parse_xyz_string(path.read_text(), filename=str(path.absolute()))


def parse_traj_file(filename):
    """Convert a provided xmol .xyz file into an array of
    atomic numbers and cartesian positions

    Parameters
    ----------
    filename: str
        path to the .xyz file to read

    Returns
    -------
    tuple of :obj:`np.ndarray`
        array of (N) atomic numbers and (N, 3) Cartesian positions
        read from the given file
    """
    path = Path(filename)
    return parse_traj_string(path.read_text(), filename=str(path.absolute()))

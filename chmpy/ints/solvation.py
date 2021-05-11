import logging
import numpy as np
from .lebedev import load_grid_num_points
from scipy.spatial.kdtree import cKDTree as KDTree

LOG = logging.getLogger(__name__)

SOLVENT_RADII = [
    1.300,
    1.638,
    1.404,
    1.053,
    2.0475,
    2.00,
    1.830,
    1.720,
    1.720,
    1.8018,
    1.755,
    1.638,
    1.404,
    2.457,
    2.106,
    2.160,
    2.05,
]

DEFAULT_RADIUS = 2.223


def get_solvent_radii(atomic_numbers):
    radii = np.empty_like(atomic_numbers, dtype=np.float32)
    for i in range(len(atomic_numbers)):
        n = atomic_numbers[i]
        if n <= 17 and n > 0:
            radii[i] = SOLVENT_RADII[n - 1]
        else:
            radii[i] = DEFAULT_RADIUS
    return radii


def solvent_surface(molecule, num_points_per_atom=140, delta=0.1):
    radii = get_solvent_radii(molecule.atomic_numbers)
    N = len(molecule)
    grid = load_grid_num_points(num_points_per_atom)
    num_points_per_atom = grid.shape[0]
    axes = molecule.axes()
    positions = molecule.positions_in_molecular_axis_frame()
    surface = np.empty((N * num_points_per_atom, 4))
    atom_idx = np.empty(N * num_points_per_atom, dtype=np.int32)

    for i in range(N):
        r = radii[i] + delta
        l, u = num_points_per_atom * i, num_points_per_atom * (i + 1)
        surface[l:u, 3] = grid[:, 3] * 4 * np.pi * radii[i] * radii[i]
        surface[l:u, :3] = grid[:, :3] * r
        surface[l:u, :3] += positions[i, :]
        atom_idx[l:u] = i

    mask = np.ones_like(atom_idx, dtype=bool)

    tree = KDTree(surface[:, :3])
    for i in range(N):
        p = positions[i]
        radius = radii[i] + delta
        idxs = tree.query_ball_point(p, radius - 1e-12)
        mask[idxs] = False

    surface = surface[mask, :]
    surface[:, :3] = (
        np.dot(surface[:, :3], axes) + molecule.center_of_mass[np.newaxis, :]
    )
    atom_idx = atom_idx[mask]
    return surface

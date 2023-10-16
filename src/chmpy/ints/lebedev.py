import numpy as np
from pathlib import Path
from chmpy.util.num import spherical_to_cartesian

_GRIDS = {k: v for k, v in np.load(Path(__file__).parent / "lebedev_grids.npz").items()}
_GRID_LMAX_LIST = tuple(sorted(int(k.split("_")[-1]) for k in _GRIDS.keys()))
_GRIDS_NUM_POINTS = {k: v.shape[0] for k, v in _GRIDS.items()}


def load_grid(l_max):
    for g in _GRID_LMAX_LIST:
        if g > l_max:
            lowest_grid = g
            break
    else:
        raise ValueError(f"No available Lebedev grid for l_max = {l_max}")
    grid = _GRIDS[f"l_max_{g}"]

    return grid


def load_grid_num_points(num_points):
    ordered_grids = sorted(
        ((k, v) for k, v in _GRIDS_NUM_POINTS.items()), key=lambda x: x[1]
    )
    for g, k in ordered_grids:
        if k > num_points:
            break
    else:
        raise ValueError(f"No available Lebedev grid for num_points = {num_points}")
    grid = _GRIDS[g]
    return grid

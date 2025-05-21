from pathlib import Path

import numpy as np

_GRIDS = dict(np.load(Path(__file__).parent / "lebedev_grids.npz").items())
_GRID_LMAX_LIST = tuple(sorted(int(k.split("_")[-1]) for k in _GRIDS.keys()))
_GRIDS_NUM_POINTS = {k: v.shape[0] for k, v in _GRIDS.items()}


def load_grid(l_max):
    for g in _GRID_LMAX_LIST:
        if g > l_max:
            break
    else:
        raise ValueError(f"No available Lebedev grid for l_max = {l_max}")
    grid = _GRIDS[f"l_max_{g}"]

    return grid


def load_grid_num_points(num_points):
    ordered_grids = sorted(
        ((k, v) for k, v in _GRIDS_NUM_POINTS.items()), key=lambda x: x[1]
    )
    grid = None
    for g, k in ordered_grids:
        grid = _GRIDS[g]
        if k > num_points:
            break
    else:
        raise ValueError(f"No available Lebedev grid for num_points = {num_points}")
    return grid

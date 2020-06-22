import numpy as np
from pathlib import Path
from chmpy.util.num import spherical_to_cartesian

_GRIDS = {k: v for k, v in np.load(Path(__file__).parent / "lebedev_grids.npz").items()}
_GRID_LMAX_LIST = tuple(sorted(int(k.split("_")[-1]) for k in _GRIDS.keys()))


def load_grid(l_max, cartesian=False):
    for g in _GRID_LMAX_LIST:
        if g > l_max:
            lowest_grid = g
            break
    else:
        raise ValueError(f"No available Lebedev grid for l_max = {l_max}")
    grid = _GRIDS[f"l_max_{g}"]

    if not cartesian:
        return grid
    tp = grid[:, :2]
    w = grid[:, 2]
    r = np.ones(tp.shape[0])
    xyz = spherical_to_cartesian(np.c_[r, tp[:, 0], tp[:, 1]])
    return np.c_[xyz[:, 0], xyz[:, 1], xyz[:, 2], w]

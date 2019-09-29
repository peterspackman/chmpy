import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from os.path import join, dirname

_DATA_DIR = dirname(__file__)
_INTERPOLATOR_DATA = np.load(join(_DATA_DIR, "thakkar_interp.npz"))

class PromoleculeDensity:
    def __init__(self, mol):
        n, pos = mol
        self.elements = np.asarray(n)
        self.positions = np.asarray(pos)

        if np.any(self.elements < 1) or np.any(self.elements > 103):
            raise ValueError("All elements must be atomic numbers between [1,103]")

        self.interpolators = {
            el_number: interp1d(
                _INTERPOLATOR_DATA["x"],
                _INTERPOLATOR_DATA["y"][el_number - 1, :],
                assume_sorted=True,
                fill_value=0.0
            )
            for el_number in np.unique(self.elements)
        }

    def rho(self, positions):
        r = cdist(self.positions, positions)
        density = np.zeros(positions.shape[0], dtype=np.float64)
        for i, n in enumerate(self.elements):
            density[:] += self.interpolators[n](r[i, :])
        return density

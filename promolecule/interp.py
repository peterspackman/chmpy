import numpy as np
from .linterp import log_interp

class InterpolatorLog1D:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __call__(self, pts):
        q = np.array(pts.ravel(), dtype=np.float32)
        results = log_interp(q, self.xs, self.ys)
        return results.reshape(pts.shape)

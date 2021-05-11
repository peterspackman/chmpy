import numpy as np
from ._linterp import log_interp


class InterpolatorLog1D:
    """Wrapper class around the cython code for interpolating
    1D functions on a log separated domain.

    No checking is performed for sorted domain or xs and ys
    being the same length. This may result in errors further
    in usage if these assumptions are not met.

    Parameters
    ----------
    xs: array_like
        (N) array of domain for interpolation: x, assumed to be log separated
        and sorted.
    ys: array_like
        (N) array of range for interpolation: f(x), because this is a linear
        interpolator smooth functions will be significantly
        more performant in terms of accuracy.
    """

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __call__(self, pts):
        """Evaluate the interpolated function on the given set of
        points.

        Parameters
        ----------
        pts: array_like
            set of points where we wish to evaluate the function

        Returns
        ________
        :obj:`np.ndarray`
            values of the interpolated function at the given set of points.
        """
        q = np.array(pts.ravel(), dtype=np.float32)
        results = log_interp(q, self.xs, self.ys)
        return results.reshape(pts.shape)

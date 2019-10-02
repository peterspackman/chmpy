from numpy.core.multiarray import interp

class Interpolator1D:
    def __init__(self, xs, ys, left=None, right=None, assume_sorted=True):
        if not assume_sorted:
            sort_idx_xs = np.argsort(xs)
            self.xs = xs[sort_idx_xs]
            self.ys = ys[sort_idx_xs]
        self.xs = xs
        self.ys = ys
        self.left = left
        self.right = right

    def __call__(self, pts):
        # this could be sped up by writing a custom function to 
        # improve the guess as we have log spaced values
        return interp(pts, self.xs, self.ys, self.left, self.right)


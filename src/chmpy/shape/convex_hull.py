from scipy.spatial import ConvexHull
import numpy as np


def _ray_hull_intersection(direction, hull):
    eq = hull.equations
    X, y = eq[:, :-1], eq[:, -1]
    d_dot_X = np.dot(X, direction)
    d_dot_X[d_dot_X == 0.0] = 1e-6
    alpha = -y / d_dot_X
    d = np.min(alpha[alpha > 0])
    return np.linalg.norm(d * direction)

def _ray_hull_intersections_batch(directions, hull):
    eq = hull.equations
    X, y = eq[:, :-1], eq[:, -1]
    d_dot_X= directions @ X.T
    d_dot_X[d_dot_X<= 0] = 1e-6
    alpha = -y / d_dot_X
    d = np.min(alpha, axis=1)
    return np.linalg.norm(d[:, np.newaxis] * directions, axis=1)


def ray_hull_intersections(directions, hull, method="fast"):
    """
    Find the distance from the origin to the intersection with the 
    given ConvexHull for a list of directions. Assumes `directions`
    is a (N, 3) array of unit vectors representing directions, and
    `hull` is a `ConvexHull` object centered about the origin.


    Args:
        directions (np.ndarray): (N, 3) array of unit vectors
        hull (ConvexHull): A ConvexHull for which to find intersections
        

    Returns:
        np.ndarray: (N,) array of the distances for each intersection
    """
    if method == "fast":
        return _ray_hull_intersections_batch(directions, hull)
    else:
        return np.array([_ray_hull_intersection(p, hull) for p in directions])


def transform_hull(sht, hull, **kwargs):
    """
    Calculate the spherical harmonic transform of the shape of the
    provided convex hull

    Args:
        sht (SHT): the spherical harmonic transform object handle
        n_i (ConvexHull): the convex hull (or shape to describe)
        kwargs (dict): keyword arguments for optional settings.
            Options include:
            ```
            origin (np.ndarray): specify the center of the surface
                (default is the geometric centroid of the interior atoms)
            ```
            distances (bool): also return the distances of intersection

    Returns:
        np.ndarray: the coefficients from the analysis step of the SHT
    """

    x, y, z = sht.grid_cartesian
    directions = np.c_[x.flatten(), y.flatten(), z.flatten()]

    r = ray_hull_intersections(directions, hull).reshape(x.shape)
    coeffs = sht.analysis(r)
    if kwargs.get("distances", False):
        return coeffs, r
    else:
        return coeffs

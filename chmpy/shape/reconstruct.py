import logging
import numpy as np
from .sht import SHT

LOG = logging.getLogger(__name__)


def reconstruct(coefficients, real=True):
    if real:
        n = len(coefficients)
        l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
    else:
        l_max = int(np.sqrt(len(coefficients))) - 1
    LOG.debug("Reconstructing deduced l_max = %d", l_max)
    sht = SHT(l_max)
    x, y, z = sht.grid_cartesian
    pts = np.c_[x.flatten(), y.flatten(), z.flatten()]
    pts = pts * sht.synthesis(coefficients).flatten()[:, np.newaxis].real
    return pts


def reconstructed_surface_convex(coefficients, real=True):
    from trimesh import Trimesh
    from scipy.spatial import ConvexHull
    pts = reconstruct(coefficients, real=real)
    cvx = ConvexHull(pts)
    return Trimesh(vertices=pts, faces=cvx.simplices)


def reconstructed_surface_icosphere(coefficients, real=True, subdivisions=3):
    if real:
        n = len(coefficients)
        l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
    else:
        raise NotImplementedError(
            "Complex reconstructed surface case not yet implemented"
        )
    LOG.debug("Reconstructing deduced l_max = %d", l_max)
    sht = SHT(l_max)

    from trimesh.creation import icosphere

    sphere = icosphere(subdivisions=subdivisions)
    theta = np.arccos(sphere.vertices[:, 2])
    phi = np.arctan2(sphere.vertices[:, 1], sphere.vertices[:, 0])
    r = np.empty_like(phi)
    for i in range(phi.shape[0]):
        r[i] = sht.evaluate_at_points(coefficients, theta[i], phi[i])
    sphere.vertices *= r[:, np.newaxis]
    return sphere

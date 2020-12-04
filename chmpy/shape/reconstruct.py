import logging
import numpy as np
from .sht import SHT
from trimesh import PointCloud
from chmpy.util.num import cartesian_to_spherical
LOG = logging.getLogger(__name__)


def reconstruct(coefficients, real=True):
    if real:
        n = len(coefficients)
        l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
        func = "synth_real"
    else:
        l_max = int(np.sqrt(len(coefficients))) - 1
        func = "synth_cplx"
    LOG.debug("Reconstructing deduced l_max = %d", l_max)
    sht = SHT(l_max=l_max)
    pts = sht.grid_cartesian * getattr(sht, func)(coefficients)[:, np.newaxis]
    return pts


def reconstructed_surface(coefficients, real=True):
    if real:
        n = len(coefficients)
        l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
        func = "synth_real"
    else:
        raise NotImplementedError("Complex reconstructed surface case not yet implemented")
    LOG.debug("Reconstructing deduced l_max = %d", l_max)
    sht = SHT(l_max=l_max)

    from trimesh.creation import icosphere
    sphere = icosphere(subdivisions=3)
    rtp = cartesian_to_spherical(sphere.vertices)
    for i in range(rtp.shape[0]):
        cost = np.cos(rtp[i, 2])
        phi = rtp[i, 1]
        rtp[i, 0] = sht._shtns.SH_to_point(coefficients, cost, phi)
    sphere.vertices *= rtp[:, 0, np.newaxis]
    return sphere

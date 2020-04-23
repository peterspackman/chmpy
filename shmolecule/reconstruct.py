import logging
import numpy as np
from .sht import SHT
from trimesh import PointCloud

LOG = logging.getLogger(__name__)


def reconstruct(coefficients, kind="real"):
    if kind == "complex":
        l_max = int(np.sqrt(len(coefficients))) - 1
        func = "synth_cplx"
    else:
        n = len(coefficients)
        l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
        func = "synth_real"
    LOG.debug("Reconstructing deduced l_max = %d", l_max)
    sht = SHT(l_max=l_max)
    pts = sht.grid_cartesian * getattr(sht, func)(coefficients)[:, np.newaxis]
    return pts

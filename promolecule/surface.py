import numpy as np
from collections import namedtuple
import time
import logging

IsosurfaceMesh = namedtuple("IsosurfaceMesh", "vertices faces normals vertex_prop")
LOG = logging.getLogger(__name__)


def promolecule_density_isosurface(promol, isovalue=0.005, sep=0.2):
    # from mcubes import marching_cubes
    from skimage.measure import marching_cubes_lewiner as marching_cubes
    t1 = time.time()

    l, u = promol.bb()
    x, y, z = np.meshgrid(
        np.arange(l[0], u[0], sep),
        np.arange(l[1], u[1], sep),
        np.arange(l[2], u[2], sep),
    )
    shape = x.shape
    pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
    d = promol.rho(pts).reshape(shape)
    verts, faces, normals, _ = marching_cubes(d, isovalue, gradient_direction="ascent")
    verts = verts * sep + pts[0, :]
    props = {}
    t2 = time.time()
    LOG.info("PromoleculeDensity isosurface took %.3fms", 1000*(t2 - t1))
    return IsosurfaceMesh(verts, faces, normals, props)


def stockholder_weight_isosurface(s, isovalue=0.5, sep=0.2, props=True):
    # from mcubes import marching_cubes
    from skimage.measure import marching_cubes_lewiner as marching_cubes
    t1 = time.time()

    l, u = s.bb()
    x, y, z = np.meshgrid(
        np.arange(l[0], u[0], sep, dtype=np.float32),
        np.arange(l[1], u[1], sep, dtype=np.float32),
        np.arange(l[2], u[2], sep, dtype=np.float32),
    )
    shape = x.shape
    pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
    pts = np.array(pts, dtype=np.float32)
    weights = s.weights(pts).reshape(shape)
    verts, faces, normals, _ = marching_cubes(
        weights, isovalue, spacing=(sep,sep,sep), gradient_direction="ascent"
    )
    verts = verts + l
    vertex_props = {}
    if props:
        d_i, d_e, d_norm_i, d_norm_e = s.d_norm(verts)
        vertex_props["d_i"] = d_i
        vertex_props["d_e"] = d_e
        vertex_props["d_norm_i"] = d_norm_i
        vertex_props["d_norm_e"] = d_norm_e
        vertex_props["d_norm"] = d_norm_i + d_norm_e
    t2 = time.time()
    LOG.info("StockholderWeight isosurface took %.3fs, %d pts", t2 - t1, len(pts))
    return IsosurfaceMesh(verts, faces, normals, vertex_props)

import numpy as np
from collections import namedtuple

IsosurfaceMesh = namedtuple("IsosurfaceMesh", "vertices faces normals vertex_prop")


def promolecular_isosurface(promol, isovalue=0.005, sep=0.2, orientation="xyz"):
    # from mcubes import marching_cubes
    from skimage.measure import marching_cubes_lewiner as marching_cubes

    l, u = promol.aabb()
    x, y, z = np.meshgrid(
        np.arange(l[0], u[0], sep),
        np.arange(l[1], u[1], sep),
        np.arange(l[2], u[2], sep),
    )
    shape = x.shape
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    d = promol.rho(pts, frame="molecule").reshape(shape)
    verts, faces, normals, _ = marching_cubes(d, isovalue, gradient_direction="ascent")
    verts = verts * sep + pts[0, :]
    props = {}
    if orientation == "xyz":
        verts = np.dot(verts, np.linalg.inv(promol.principal_axes.T)) + promol.centroid
    return IsosurfaceMesh(verts, faces, normals, props)


def stockholder_weight_isosurface(s, isovalue=0.5, sep=0.2, props=True):
    # from mcubes import marching_cubes
    from skimage.measure import marching_cubes_lewiner as marching_cubes

    l, u = s.bb()
    x, y, z = np.meshgrid(
        np.arange(l[0], u[0], sep),
        np.arange(l[1], u[1], sep),
        np.arange(l[2], u[2], sep),
    )
    shape = x.shape
    pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
    import time
    t1 = time.time()
    weights = s.weight(pts).reshape(shape)
    t2 = time.time()
#    print("Weights took", t2 - t1)
    t1 = time.time()
    verts, faces, normals, _ = marching_cubes(
        weights, isovalue, spacing=(sep,sep,sep), gradient_direction="ascent"
    )
    t2 = time.time()
#    print("Marching cubes took", t2 - t1)
    verts = verts + l
#    print(np.mean(verts, axis=0), np.mean(pts, axis=0))
    vertex_props = {}
    t1 = time.time()
    if props:
        d_i, d_e, d_norm_i, d_norm_e = s.d_norm(verts)
        vertex_props["d_i"] = d_i
        vertex_props["d_e"] = d_e
        vertex_props["d_norm_i"] = d_norm_i
        vertex_props["d_norm_e"] = d_norm_e
        vertex_props["d_norm"] = d_norm_i + d_norm_e
    t2 = time.time()
#    print("Properties took", t2 - t1)
    return IsosurfaceMesh(verts, faces, normals, vertex_props)

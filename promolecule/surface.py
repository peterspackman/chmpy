import numpy as np
from collections import namedtuple

IsosurfaceMesh = namedtuple("IsosurfaceMesh", "vertices faces normals vertex_prop")

def promolecular_isosurface(promol, isovalue=0.005, sep=0.2, orientation="xyz"):
    #from mcubes import marching_cubes
    from skimage.measure import marching_cubes_lewiner as marching_cubes
    l, u = promol.aabb()
    x,y,z = np.meshgrid(
        np.arange(l[0], u[0], sep),
        np.arange(l[1], u[1], sep),
        np.arange(l[2], u[2], sep),
    )
    shape = x.shape
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    d = promol.rho(pts, frame="molecule").reshape(shape)
    verts, faces, normals, _ =  marching_cubes(
        d, isovalue, gradient_direction="ascent"
    )
    verts = verts * sep + pts[0, :]
    props = {}
    if orientation == "xyz":
        verts = np.dot(verts, np.linalg.inv(promol.principal_axes.T)) + promol.centroid
    return IsosurfaceMesh(verts, faces, normals, props)

import logging

import numpy as np

from chmpy.util.num import cartesian_to_spherical_mgrid

from .sht import SHT

LOG = logging.getLogger(__name__)


def reconstruct(coefficients, real=True, pole_extension_factor=2):
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
    # Add centroid points for the poles
    north_pole_indices = np.arange((sht.ntheta - 1) * sht.nphi, sht.ntheta * sht.nphi)
    south_pole_indices = np.arange(0, sht.nphi)
    north_pole_centroid = np.mean(pts[north_pole_indices], axis=0)
    south_pole_centroid = np.mean(pts[south_pole_indices], axis=0)

    # Compute the average distance of the surrounding points from the origin
    north_pole_radius = np.mean(np.linalg.norm(pts[north_pole_indices], axis=1))
    south_pole_radius = np.mean(np.linalg.norm(pts[south_pole_indices], axis=1))

    # Compute the extension factor based on the radius ratio
    north_pole_extension_factor = north_pole_radius / np.linalg.norm(
        north_pole_centroid
    )
    south_pole_extension_factor = south_pole_radius / np.linalg.norm(
        south_pole_centroid
    )

    # Extend the poles based on the extension factor
    north_pole_extension = north_pole_centroid * north_pole_extension_factor
    south_pole_extension = south_pole_centroid * south_pole_extension_factor

    pts = np.vstack((pts, north_pole_extension, south_pole_extension))

    # Update the faces to include the centroid points
    faces = sht.faces()
    north_pole_index = len(pts) - 2
    south_pole_index = len(pts) - 1
    for i in range(sht.nphi):
        faces.append(
            [
                north_pole_index,
                north_pole_indices[(i + 1) % sht.nphi],
                north_pole_indices[i],
            ]
        )
        faces.append(
            [
                south_pole_index,
                south_pole_indices[(i + 1) % sht.nphi],
                south_pole_indices[i],
            ]
        )
    return pts, faces


def reconstructed_surface_convex(coefficients, real=True):
    from scipy.spatial import ConvexHull
    from trimesh import Trimesh

    pts, _ = reconstruct(coefficients, real=real)
    cvx = ConvexHull(pts)
    return Trimesh(vertices=pts, faces=cvx.simplices)


def reconstructed_surface(coefficients, real=True):
    from trimesh import Trimesh

    pts, faces = reconstruct(coefficients, real=real)
    mesh = Trimesh(vertices=pts, faces=faces)
    return mesh


def reconstructed_surface_icosphere(coefficients, real=True, subdivisions=3):
    if real:
        n = len(coefficients)
        l_max = int((-3 + np.sqrt(8 * n + 1)) // 2)
    else:
        n = len(coefficients)
        l_max = int(np.sqrt(n)) - 1
    LOG.debug("Reconstructing deduced l_max = %d", l_max)
    sht = SHT(l_max)

    from trimesh.creation import icosphere

    datatype = np.float64 if real else np.complex128

    sphere = icosphere(subdivisions=subdivisions)
    r, theta, phi = cartesian_to_spherical_mgrid(
        sphere.vertices[:, 0], sphere.vertices[:, 1], sphere.vertices[:, 2]
    )
    r = np.empty_like(phi, datatype)
    for i in range(phi.shape[0]):
        r[i] = sht.evaluate_at_points(coefficients, theta[i], phi[i])
    sphere.vertices *= np.real(r[:, np.newaxis])

    if not real:
        sphere.vertex_attributes["property"] = np.imag(r)
    return sphere

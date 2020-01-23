import numpy as np
import logging
import matplotlib.colors as colors

LOG = logging.getLogger(__name__)


def save_mesh(mesh, filename):
    """Save the given Trimesh to a file.

    Parameters
    ----------
    mesh: :obj:`trimesh.Trimesh`
        The mesh to save.
    filename: str
        The path to the destination file.
    """
    ext = filename.split(".")[-1]
    with open(filename, "wb") as f:
        mesh.export(f, ext)

    LOG.debug("Saved mesh %s to %s", mesh, filename)


def cartesian_product(*arrays):
    """Efficiently calculate the Cartesian product of the
    provided vectors A X B X C ... etc. This will maintain
    order in loops from the right most array.

    Parameters
    ----------
    *arrays: array_like
        1D arrays to use for the Cartesian product

    Returns
    -------
    :obj:`np.ndarray`
        The Cartesian product of the given vectors.
    """
    arrays = [np.asarray(a) for a in arrays]
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def spherical_to_cartesian(rtp, dtype=np.float64):
    """Given an N by 3 array of (r, theta, phi) spherical coordinates
    return an N by 3 array of Cartesian(x, y, z) coordinates.

    Uses the following convention::

        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cos(theta)

    Parameters
    ----------
    rtp: array_like
        (N,3) array of of r, theta, phi coordinates in spherical coordinate
        system.

    Returns
    -------
    :obj:`np.ndarray`
        (N,3) array of x,y,z Cartesian coordinates
    """
    xyz = np.zeros(rtp.shape, dtype=dtype)

    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])

    return xyz

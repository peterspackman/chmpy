import numpy as np
from numbers import Number
from typing import Tuple


def is_perfect_square(value: Number) -> bool:
    """
    Check if a number is perfect square.

    Args:
        value (Number): the number in question

    Returns:
        bool: `True` if the number is a perfect square, otherwise `False`
    """
    import math

    root = math.sqrt(value)
    if int(root + 0.5) ** 2 == value:
        return True
    else:
        return False


def cartesian_product(*arrays) -> np.ndarray:
    """
    Efficiently calculate the Cartesian product of the
    provided vectors A x B x C ... etc. This will maintain
    order in loops from the right most array.

    Args:
        *arrays (array_like): 1D arrays to use for the Cartesian product

    Returns:
        np.ndarray: The Cartesian product of the provided vectors.
    """
    arrays = [np.asarray(a) for a in arrays]
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def spherical_to_cartesian(rtp: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Given an N by 3 array of (r, theta, phi) spherical coordinates
    return an N by 3 array of Cartesian(x, y, z) coordinates.

    Uses the following convention::

        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cos(theta)

    Args:
        rtp (array_like): (N,3) array of of r, theta, phi coordinates
            in the above spherical coordinate system.
        dtype: numpy datatype or string

    Returns:
        np.ndarray: (N,3) array of x,y,z Cartesian coordinates
    """
    xyz = np.empty(rtp.shape, dtype=dtype)

    xyz[:, 0] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
    xyz[:, 1] = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])

    return xyz

def spherical_to_cartesian_mgrid(r: np.ndarray, t: np.ndarray, p: np.ndarray):
    """
    Given 3 arrays of (r, theta, phi) spherical coordinates
    return 3 arrays of Cartesian(x, y, z) coordinates.

    Uses the following convention::

        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cos(theta)

    Args:
        r (array_like): array of of r coordinates
            in the above spherical coordinate system.

        t (array_like): array of of theta coordinates
            in the above spherical coordinate system.

        p (array_like): array of phi coordinates
            in the above spherical coordinate system.
        dtype: numpy datatype or string

    Returns:
        Tuple[np.ndarray]: 3 arrays of x,y,z Cartesian coordinates
    """
    return r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)

def cartesian_to_spherical_mgrid(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Given 3 arrays of of (x, y, z) Cartesian coordinates
    return 3 arrays of of spherical (r, theta, phi) coordinates.

    Uses the following convention::

        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cos(theta)

    Args:
        x (array_like): (N) array of of z coordinates
            in the above spherical coordinate system.
        y (array_like): (N) array of of y coordinates
            in the above spherical coordinate system.
        z (array_like): (N) array of of z coordinates
            in the above spherical coordinate system.
    Returns:
        Tuple[np.ndarray]: 3 arrays of r, theta, phi spherical coordinates
    """
    r = np.sqrt(x*x + y*y + z*z)
    return r, np.arccos(z / r), np.arctan2(y, x)


def cartesian_to_spherical(xyz: np.ndarray, dtype=np.float64) -> np.ndarray:
    """
    Given an N by 3 array of (x, y, z) spherical coordinates
    return an N by 3 array of Cartesian(r, theta, phi) coordinates.

    Uses the following convention::

        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cos(theta)

    Args:
        rtp (array_like): (N,3) array of of r, theta, phi coordinates
            in the above spherical coordinate system.
        dtype: numpy datatype or string

    Returns:
        np.ndarray: (N,3) array of x,y,z Cartesian coordinates
    """
    rtp = np.empty(xyz.shape, dtype=dtype)
    rtp[:, 0] = np.sqrt(
        xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1] + xyz[:, 2] * xyz[:, 2]
    )
    rtp[:, 1] = np.fmod(np.arctan2(xyz[:, 1], xyz[:, 0]) + 4 * np.pi, 2 * np.pi)
    rtp[:, 2] = np.arccos(xyz[:, 2] / rtp[:, 0])
    return rtp


def rmsd_points(A, B, reorient="kabsch"):
    """
    Rotate the points in `A` onto `B` and calculate
    their RMSD

    Args:
        A (np.ndarray): (N,D) matrix where N is the number of vectors and D
            is the dimension of each vector
        B (np.ndarray): (N,D) matrix where N is the number of
            vectors and D is the dimension of each vector

    Returns:
        float: root mean squared deviation
    """
    if reorient:
        A = reorient_points(A, B, method=reorient)
    diff = B - A
    return np.sqrt(np.vdot(diff, diff) / diff.shape[0])


def reorient_points(A, B, method="kabsch"):
    """
    Rotate the points in `A` onto `B`

    Args:
        A (np.ndarray): (N,D) matrix where N is the number of vectors and D
            is the dimension of each vector
        B (np.ndarray): (N,D) matrix where N is the number of
            vectors and D is the dimension of each vector

    Returns:
        np.ndarray: (N,D) matrix where N is the number of vectors and D
            is the dimension of each vector, now rotated to align with B
    """
    if method != "kabsch":
        raise NotImplementedError("Only kabsch algorithm is currently implemented")
    R = kabsch_rotation_matrix(A, B)
    A = np.dot(A, R)
    return A


def kabsch_rotation_matrix(A, B):
    """
    Calculate the optimal rotation matrix `R` to rotate
    `A` onto `B`, minimising root-mean-square deviation so that
    this may be then calculated.

    See https://en.wikipedia.org/wiki/Kabsch_algorithm

    References:
        Kabsch, W. Acta Cryst. A, 32, 922-923, (1976)
            DOI: http://dx.doi.org/10.1107/S0567739476001873

    Args:
        A (np.ndarray): (N,D) matrix where N is the number of vectors and D
            is the dimension of each vector
        B (np.ndarray): (N,D) matrix where N is the number of
            vectors and D is the dimension of each vector
    Returns:
        np.ndarray (D,D) rotation matrix where D is the dimension of each vector
    """

    # Calculate the covariance matrix
    cov = np.dot(np.transpose(A), B)

    # Use singular value decomposition to calculate
    # the optimal rotation matrix
    v, s, w = np.linalg.svd(cov)

    # check the determinant to ensure a right-handed
    # coordinate system
    if (np.linalg.det(v) * np.linalg.det(w)) < 0.0:
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    R = np.dot(v, w)
    return R

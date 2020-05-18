import numpy as np
import logging
import matplotlib.colors as colors

LOG = logging.getLogger(__name__)

BOHR_PER_ANGSTROM = 1.8897259886
ANGSTROM_PER_BOHR = 0.529177249

SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "a": "ₐ",
    "b": "♭",
    "c": "꜀",
    "d": "ᑯ",
    "e": "ₑ",
    "f": "բ",
    "g": "₉",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "q": "૧",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "w": "w",
    "x": "ₓ",
    "y": "ᵧ",
    "z": "₂",
    "A": "ₐ",
    "B": "₈",
    "C": "C",
    "D": "D",
    "E": "ₑ",
    "F": "բ",
    "G": "G",
    "H": "ₕ",
    "I": "ᵢ",
    "J": "ⱼ",
    "K": "ₖ",
    "L": "ₗ",
    "M": "ₘ",
    "N": "ₙ",
    "O": "ₒ",
    "P": "ₚ",
    "Q": "Q",
    "R": "ᵣ",
    "S": "ₛ",
    "T": "ₜ",
    "U": "ᵤ",
    "V": "ᵥ",
    "W": "w",
    "X": "ₓ",
    "Y": "ᵧ",
    "Z": "Z",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
}


def subscript(x):
    return SUBSCRIPT_MAP.get(x, x)


def overline(x):
    return f"\u0305{x}"


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


# TODO add LinearSegmentedColormap objects for other
# CrystalExplorer default colors
DEFAULT_COLORMAPS = {
    "d_norm": "bwr_r",
    "d_e": "viridis_r",
    "d_i": "viridis_r",
    "d_norm_i": "bwr",
    "d_norm_e": "bwr_r",
    "esp": "coolwarm_r",
}


def property_to_color(prop, cmap="viridis", **kwargs):
    from matplotlib.cm import get_cmap

    midpoint = kwargs.get("midpoint", 0.0 if cmap in ("d_norm", "esp") else None)
    colormap = get_cmap(kwargs.get("colormap", DEFAULT_COLORMAPS.get(cmap, cmap)))
    norm = None
    if midpoint is not None:
        try:
            from matplotlib.colors import TwoSlopeNorm
        except ImportError:
            from matplotlib.colors import DivergingNorm as TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=prop.min(), vcenter=midpoint, vmax=prop.max())
        prop = norm(prop)
    return colormap(prop)

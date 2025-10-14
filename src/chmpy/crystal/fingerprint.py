import matplotlib.pyplot as plt
import numpy as np

from chmpy.core.element import Element


def sample_face_points(vertices, faces, samples_per_edge=4):
    """Generate sample points within triangle faces using barycentric coordinates."""
    points = []
    weights = []

    for i in range(samples_per_edge + 1):
        for j in range(samples_per_edge + 1 - i):
            a = i / samples_per_edge
            b = j / samples_per_edge
            c = 1.0 - a - b
            points.append([a, b, c])
            weights.append(1.0 / ((samples_per_edge + 1) * (samples_per_edge + 2) / 2))

    points = np.array(points)
    weights = np.array(weights)

    face_vertices = vertices[faces]

    points = points[:, None, :]

    interpolated = np.sum(points[..., None] * face_vertices[None, ...], axis=2)

    return interpolated, weights


def filtered_histogram(
    mesh, internal, external, bins=200, xrange=None, yrange=None, samples_per_edge=4, include_inverse=False
):
    """Create histogram with multiple samples per face.

    Args:
        mesh: mesh with vertex attributes
        internal: atomic number for internal atom
        external: atomic number for external atom
        bins (int, optional): number of bins for histogram (default: 200)
        xrange (tuple, optional): range for x-axis (d_i)
        yrange (tuple, optional): range for y-axis (d_e)
        samples_per_edge (int, optional): number of samples per triangle edge (default: 4)
        include_inverse (bool, optional): if True, also include contacts where internal and
            external are swapped (e.g., C...H and H...C). Default: False

    Returns:
        tuple: (histogram, xedges, yedges) as returned by np.histogram2d
    """
    di = mesh.vertex_attributes["d_i"]
    de = mesh.vertex_attributes["d_e"]
    if xrange is None:
        xrange = np.min(di), np.max(di)
    if yrange is None:
        yrange = np.min(de), np.max(de)

    di_atom = mesh.vertex_attributes["nearest_atom_internal"]
    de_atom = mesh.vertex_attributes["nearest_atom_external"]
    vertex_mask = (de_atom == external) & (di_atom == internal)

    if include_inverse:
        vertex_mask_inverse = (de_atom == internal) & (di_atom == external)
        vertex_mask = vertex_mask | vertex_mask_inverse

    face_mask = np.any(vertex_mask[mesh.faces], axis=1)
    filtered_faces = mesh.faces[face_mask]

    if len(filtered_faces) == 0:
        return np.histogram2d([], [], bins=bins, range=(xrange, yrange))

    vertices = np.stack([di, de], axis=1)

    interpolated, weights = sample_face_points(
        vertices, filtered_faces, samples_per_edge
    )

    di_samples = interpolated[..., 0].flatten()
    de_samples = interpolated[..., 1].flatten()

    weights_tiled = np.tile(weights, len(filtered_faces))

    return np.histogram2d(
        di_samples, de_samples, bins=bins, range=(xrange, yrange), weights=weights_tiled
    )


def filtered_histogram_by_elements(
    mesh, internal_element, external_element, bins=200, xrange=None, yrange=None, samples_per_edge=4, include_inverse=False
):
    """Create filtered histogram by specifying element symbols.

    This is a convenience method that allows filtering by element symbols
    instead of atomic numbers.

    Args:
        mesh: mesh with vertex attributes
        internal_element (str or int): element symbol (e.g., "C") or atomic number for internal atom
        external_element (str or int): element symbol (e.g., "H") or atomic number for external atom
        bins (int, optional): number of bins for histogram (default: 200)
        xrange (tuple, optional): range for x-axis (d_i)
        yrange (tuple, optional): range for y-axis (d_e)
        samples_per_edge (int, optional): number of samples per triangle edge (default: 4)
        include_inverse (bool, optional): if True, also include contacts where internal and
            external are swapped (e.g., C...H and H...C). Default: False

    Returns:
        tuple: (histogram, xedges, yedges) as returned by np.histogram2d

    Examples:
        >>> # Filter for C...H contacts only
        >>> hist = filtered_histogram_by_elements(mesh, "C", "H")
        >>> # Filter for both C...H and H...C contacts
        >>> hist = filtered_histogram_by_elements(mesh, "C", "H", include_inverse=True)
        >>> # Or equivalently using atomic numbers
        >>> hist = filtered_histogram_by_elements(mesh, 6, 1, include_inverse=True)
    """
    if isinstance(internal_element, str):
        internal_element = Element.from_string(internal_element).atomic_number
    if isinstance(external_element, str):
        external_element = Element.from_string(external_element).atomic_number

    return filtered_histogram(
        mesh, internal_element, external_element, bins, xrange, yrange, samples_per_edge, include_inverse
    )


def fingerprint_histogram(mesh, bins=200, xrange=None, yrange=None, samples_per_edge=4):
    """Create histogram for all faces with multiple samples per face."""
    di = mesh.vertex_attributes["d_i"]
    de = mesh.vertex_attributes["d_e"]
    if xrange is None:
        xrange = np.min(di), np.max(di)
    if yrange is None:
        yrange = np.min(de), np.max(de)

    vertices = np.stack([di, de], axis=1)

    interpolated, weights = sample_face_points(vertices, mesh.faces, samples_per_edge)

    di_samples = interpolated[..., 0].flatten()
    de_samples = interpolated[..., 1].flatten()

    weights_tiled = np.tile(weights, len(mesh.faces))

    return np.histogram2d(
        di_samples, de_samples, bins=bins, range=(xrange, yrange), weights=weights_tiled
    )


def plot_fingerprint_histogram(
    hist, ax=None, filename=None, cmap="coolwarm", xlim=(0.5, 2.5), ylim=(0.5, 2.5)
):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    H1, xedges, yedges = hist
    X, Y = np.meshgrid(xedges, yedges)
    H1[H1 == 0] = np.nan
    ax.pcolormesh(X, Y, H1, cmap=cmap, shading="auto")
    ax.set_xlabel(r"$d_i$ ($\AA$)", fontsize=11)
    ax.set_ylabel(r"$d_e$ ($\AA$)", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")


def plot_filtered_histogram(
    hist_filtered,
    hist,
    ax=None,
    filename=None,
    cmap="coolwarm",
    xlim=(0.5, 2.5),
    ylim=(0.5, 2.5),
):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    H1, xedges1, yedges1 = hist
    H2, xedges2, yedges2 = hist_filtered
    X1, Y1 = np.meshgrid(xedges1, yedges1)
    H1_binary = np.where(H1 > 0, 1, np.nan)
    H2[H2 == 0] = np.nan
    ax.pcolormesh(X1, Y1, H1_binary, cmap="Greys_r", alpha=0.15, shading="auto")
    ax.pcolormesh(X1, Y1, H2, cmap=cmap, shading="auto")
    ax.set_xlabel(r"$d_i$ ($\AA$)", fontsize=11)
    ax.set_ylabel(r"$d_e$ ($\AA$)", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

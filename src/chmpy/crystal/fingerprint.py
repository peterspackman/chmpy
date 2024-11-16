import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np

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

def filtered_histogram(mesh, internal, external, bins=200, xrange=None, yrange=None, samples_per_edge=4):
    """Create histogram with multiple samples per face."""
    di = mesh.vertex_attributes["d_i"]
    de = mesh.vertex_attributes["d_e"]
    if xrange is None:
        xrange = np.min(di), np.max(di)
    if yrange is None:
        yrange = np.min(de), np.max(de)

    di_atom = mesh.vertex_attributes["nearest_atom_internal"]
    de_atom = mesh.vertex_attributes["nearest_atom_external"]
    vertex_mask = (de_atom == external) & (di_atom == internal)

    face_mask = np.any(vertex_mask[mesh.faces], axis=1)
    filtered_faces = mesh.faces[face_mask]

    if len(filtered_faces) == 0:
        return np.histogram2d([], [], bins=bins, range=(xrange, yrange))

    vertices = np.stack([di, de], axis=1)

    interpolated, weights = sample_face_points(vertices, filtered_faces, samples_per_edge)

    di_samples = interpolated[..., 0].flatten()
    de_samples = interpolated[..., 1].flatten()

    weights_tiled = np.tile(weights, len(filtered_faces))

    return np.histogram2d(
        di_samples, de_samples,
        bins=bins, range=(xrange, yrange),
        weights=weights_tiled
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
        di_samples, de_samples,
        bins=bins, range=(xrange, yrange),
        weights=weights_tiled
    )

def plot_fingerprint_histogram(hist, ax=None, filename=None, cmap="coolwarm",
                             xlim=(0.5, 2.5), ylim=(0.5, 2.5)):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    H1, xedges, yedges = hist
    X, Y = np.meshgrid(xedges, yedges)
    H1[H1 == 0] = np.nan
    ax.pcolormesh(X, Y, H1, cmap=cmap)
    ax.set_xlabel(r'$d_i$')
    ax.set_ylabel(r'$d_e$')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

def plot_filtered_histogram(hist_filtered, hist, ax=None, filename=None, cmap="coolwarm",
                          xlim=(0.5, 2.5), ylim=(0.5, 2.5)):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)
    H1, xedges1, yedges1 = hist
    H2, xedges2, yedges2 = hist_filtered
    X1, Y1 = np.meshgrid(xedges1, yedges1)
    H1_binary = np.where(H1 > 0, 1, np.nan)
    H2[H2 == 0] = np.nan
    ax.pcolormesh(X1, Y1, H1_binary, cmap='Greys_r', alpha=0.15)
    ax.pcolormesh(X1, Y1, H2, cmap=cmap)
    ax.set_xlabel(r'$d_i$')
    ax.set_ylabel(r'$d_e$')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

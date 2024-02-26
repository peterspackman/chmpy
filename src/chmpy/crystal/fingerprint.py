import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def filtered_histogram(mesh, internal, external, bins=200, xrange=None, yrange=None):
    di = mesh.vertex_attributes["d_i"]
    de = mesh.vertex_attributes["d_e"]
    if xrange is None:
        xrange = np.min(di), np.max(di)
    if yrange is None:
        yrange = np.min(de), np.max(de)
    di_atom = mesh.vertex_attributes["nearest_atom_internal"]
    de_atom = mesh.vertex_attributes["nearest_atom_external"]
    mask = (de_atom == external) & (di_atom == internal)
    return np.histogram2d(di[mask], de[mask], bins=bins, range=(xrange, yrange))

def fingerprint_histogram(mesh, bins=200, xrange=None, yrange=None):
    di = mesh.vertex_attributes["d_i"]
    de = mesh.vertex_attributes["d_e"]
    if xrange is None:
        xrange = np.min(di), np.max(di)
    if yrange is None:
        yrange = np.min(de), np.max(de)
    return np.histogram2d(di, de, bins=bins, range=(xrange, yrange))


def plot_fingerprint_histogram(hist, ax=None, filename=None, cmap="coolwarm",
                               xlim=(0.5, 2.5), ylim=(0.5, 2.5)):
    if ax is None:
        fig, ax = plt.subplots()
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


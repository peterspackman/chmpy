import matplotlib.pyplot as plt
import numpy as np


def fingerprint_histogram(mesh, bins=200, xrange=(0.5, 2.5), yrange=(0.5, 2.5)):
    di = mesh.vertex_attributes["d_i"]
    de = mesh.vertex_attributes["d_e"]
    return np.histogram2d(di, de, bins=bins, range=(xrange, yrange))


def plot_fingerprint_histogram(hist, ax=None, filename=None):
    if ax is None:
        fig, ax = plt.subplots()
    H1, xedges, yedges = hist
    X, Y = np.meshgrid(xedges, yedges)
    H1[H1 == 0] = np.nan
    ax.pcolormesh(X, Y, H1, cmap='coolwarm')
    ax.set_xlabel(r'$d_i$')
    ax.set_ylabel(r'$d_e$')
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

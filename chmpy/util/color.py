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
        vmin = prop.min()
        vmax = prop.max()
        if vmin >= 0.0:
            vmin = -1.0
        if vmax <= 0.0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=midpoint, vmax=vmax)
        prop = norm(prop)
    return colormap(prop)

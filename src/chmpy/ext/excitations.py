import logging
import numpy as np

LOG = logging.getLogger(__name__)

_FAC = 1.3062974e8


def add_gaussian_curve_contribution(x, band, strength, std):
    std_i = 1 / std
    x_i = 1 / x
    band_i = 1 / band
    exponent = (x_i - band_i) / std_i
    return _FAC * (strength / (1e7 * std_i)) * np.exp(-exponent * exponent)


def add_lorentz_curve_contribution(x, band, strength, std, gamma):
    std_i = 1 / std
    x_band = x - band
    x_band2 = x_band * x_band
    gamma2 = gamma * gamma
    return _FAC * (strength / (1e7 * std_i)) * (gamma2 / (x_band2 + gamma2))


def plot_spectra(
    energies,
    osc,
    bounds=(1, 1500),
    bins=1000,
    std=12398.4,
    kind="gaussian",
    gamma=12.5,
    label=None,
    **kwargs
):
    """Plot the (UV-Vis) spectra.

    Args:
        energies (np.ndarray): excitation energies/bands in nm.
        osc (np.ndarray): oscillator strengths (dimensionless).

    """
    import matplotlib.pyplot as plt

    x = np.linspace(bounds[0], bounds[1], bins)
    total = 0
    for e, f in zip(energies, osc):
        if kind == "gaussian":
            peak = add_gaussian_curve_contribution(x, e, f, std)
        else:
            peak = add_lorentz_curve_contribution(x, e, f, std, gamma)
        total += peak

    ax = plt.gca()
    total = total / np.max(total)
    ax.plot(x, total, label=label, **kwargs)
    ax.set_xlabel(r"$\lambda$ (nm)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r"Intensity")
    return ax

import matplotlib.pyplot as plt
from chmpy.crystal.sfac import LAMBDA_Cu
import numpy as np


class PowderPattern:
    def __init__(self, two_theta, f2, **kwargs):
        self.two_theta = two_theta
        self.f2 = f2
        self.two_theta_range = kwargs.get("two_theta_range", (5, 50))
        self.wavelength = kwargs.get("wavelength", LAMBDA_Cu)
        self.source = kwargs.get(
            "source", "unknown" if self.wavelength != LAMBDA_Cu else "Cu"
        )
        self.bins = kwargs.get(
            "bins", (self.two_theta_range[1] - self.two_theta_range[0]) * 10
        )
        self.bin_edges, self.bin_heights = np.histogram(
            self.two_theta, bins=self.bins, weights=self.f2, range=self.two_theta_range
        )

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.hist(
            self.two_theta, bins=self.bins, weights=self.f2, range=self.two_theta_range
        )
        ax.set_xlabel(kwargs.get("xlabel", r"2$\theta$"))
        ax.set_ylabel(kwargs.get("ylabel", r"Intensity"))
        ax.set_title(
            kwargs.get("title", f"Powder pattern in range {self.two_theta_range}")
        )
        return ax

    def binned(self):
        return np.histogram(
            self.two_theta, bins=self.bins, weights=self.f2, range=self.two_theta_range
        )

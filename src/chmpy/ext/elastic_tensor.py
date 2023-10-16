"""
Heavily inspired by fxcoudert's ELATE, and since
it is a modified version of that, this is subject
to the same MIT license.

See the page, and the source here:
```
    http://progs.coudert.name/elate
    https://github.com/fxcoudert/elate
```
"""

import numpy as np
from scipy import optimize
import random


def angles_to_cartesian(theta, phi):
    sint = np.sin(theta)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    cosp = np.cos(phi)
    return np.c_[sint * cosp, sint * sinp, cost]


def angles_to_cartesian_2(theta, phi, chi):
    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)
    sinc = np.sin(chi)
    cosc = np.cos(chi)
    return np.c_[
        cost * cosp * cosp - sinp * sinc, cost * sinp * cosc + cosp * sinc, -sint * cosc
    ]


def _minimize(func, dim):
    if dim == 2:
        r = ((0, np.pi), (0, np.pi))
        n = 25
    elif dim == 3:
        r = ((0, np.pi), (0, np.pi), (0, np.pi))
        n = 10
    return optimize.brute(func, r, Ns=n, full_output=True, finish=optimize.fmin)[0:2]


def _maximize(func, dim):
    res = _minimize(lambda x: -func(x), dim)
    return (res[0], -res[1])


class ElasticTensor:
    """
    Class to represent an elastic tensor, along with methods to access it
    """

    def __init__(self, mat):
        mat = np.asarray(mat, dtype=np.float64)
        if mat.shape != (6, 6):
            # Is it upper triangular?
            if list(map(len, mat)) == [6, 5, 4, 3, 2, 1]:
                mat = [[0] * i + mat[i] for i in range(6)]
                mat = np.array(mat)

            # Is it lower triangular?
            if list(map(len, mat)) == [1, 2, 3, 4, 5, 6]:
                mat = [mat[i] + [0] * (5 - i) for i in range(6)]
                mat = np.array(mat)

        if mat.shape != (6, 6):
            raise ValueError("should be a square matrix")

        # Check that is is symmetric, or make it symmetric
        if np.linalg.norm(np.tril(mat, -1)) == 0:
            mat = mat + np.triu(mat, 1).transpose()
        if np.linalg.norm(np.triu(mat, 1)) == 0:
            mat = mat + np.tril(mat, -1).transpose()
        if np.linalg.norm(mat - mat.transpose()) > 1e-3:
            raise ValueError("should be symmetric, or triangular")
        elif np.linalg.norm(mat - mat.transpose()) > 0:
            mat = 0.5 * (mat + mat.transpose())

        # Store it
        self.c_voigt = np.array(mat)

        # Put it in a more useful representation
        try:
            self.s_voigt = np.linalg.inv(self.c_voigt)
        except:
            raise ValueError("matrix is singular")

        vm = np.array(((0, 5, 4), (5, 1, 3), (4, 3, 2)))

        def sv_coeff(p, q):
            return 1.0 / ((1 + p // 3) * (1 + q // 3))

        smat = [
            [
                [
                    [
                        sv_coeff(vm[i, j], vm[k, l]) * self.s_voigt[vm[i, j], vm[k, l]]
                        for i in range(3)
                    ]
                    for j in range(3)
                ]
                for k in range(3)
            ]
            for l in range(3)
        ]
        self.elasticity_tensor = np.array(smat)

    @classmethod
    def from_string(cls, s):
        """Initialize the elastic tensor from a string"""
        if not s:
            raise ValueError("no matrix was provided")

        if isinstance(s, str):
            # Remove braces and pipes
            s = s.replace("|", " ").replace("(", " ").replace(")", " ")

            # Remove empty lines
            lines = [line for line in s.split("\n") if line.strip()]
            if len(lines) != 6:
                raise ValueError("should have six rows")

            # Convert to float
            try:
                mat = [[float(x) for x in line.split()] for line in lines]
            except:
                raise ValueError("not all entries are numbers")
        return cls(mat)

    def youngs_modulus_angular(self, theta, phi):
        a = angles_to_cartesian(theta, phi)
        return self.youngs_modulus(a)

    def youngs_modulus(self, a):
        return 1 / np.einsum("ai,aj,ak,al,ijkl->a", a, a, a, a, self.elasticity_tensor)

    def linear_compressibility_angular(self, theta, phi):
        a = angles_to_cartesian(theta, phi)
        return self.linear_compressibility(a)

    def linear_compressibility(self, a):
        return 1000 * np.einsum("ai,aj,ijkk->a", a, a, self.elasticity_tensor)

    def shear_modulus(self, a, b):
        return 0.25 / np.einsum(
            "ai,aj,ak,al,ijkl->a", a, b, a, b, self.elasticity_tensor
        )

    def shear_modulus_angular(self, theta, phi, chi):
        a = angles_to_cartesian(theta, phi)
        b = angles_to_cartesian_2(theta, phi, chi)
        return self.shear_modulus(a, b)

    def poisson_ratio(self, a, b):
        r1 = np.einsum("ai,aj,ak,al,ijkl->a", a, a, b, b, self.elasticity_tensor)
        r2 = np.einsum("ai,aj,ak,al,ijkl->a", a, a, a, a, self.elasticity_tensor)
        return -r1 / r2

    def poisson_ratio_angular(self, theta, phi, chi):
        a = angles_to_cartesian(theta, phi)
        b = angles_to_cartesian_2(theta, phi, chi)
        return self.poisson_ratio(a, b)

    def averages(self):
        A = (self.c_voigt[0, 0] + self.c_voigt[1, 1] + self.c_voigt[2, 2]) / 3
        B = (self.c_voigt[1, 2] + self.c_voigt[0, 2] + self.c_voigt[0, 1]) / 3
        C = (self.c_voigt[3, 3] + self.c_voigt[4, 4] + self.c_voigt[5, 5]) / 3
        a = (self.s_voigt[0, 0] + self.s_voigt[1, 1] + self.s_voigt[2, 2]) / 3
        b = (self.s_voigt[1, 2] + self.s_voigt[0, 2] + self.s_voigt[0, 1]) / 3
        c = (self.s_voigt[3, 3] + self.s_voigt[4, 4] + self.s_voigt[5, 5]) / 3

        KV = (A + 2 * B) / 3
        GV = (A - B + 3 * C) / 5

        KR = 1 / (3 * a + 6 * b)
        GR = 5 / (4 * a - 4 * b + 3 * c)

        KH = (KV + KR) / 2
        GH = (GV + GR) / 2

        return {
            "bulk_modulus_avg": {"voigt": KV, "reuss": KR, "hill": KH,},
            "shear_modulus_avg": {"voigt": GV, "reuss": GR, "hill": GH},
            "youngs_modulus_avg": {
                "voigt": 1 / (1 / (3 * GV) + 1 / (9 * KV)),
                "reuss": 1 / (1 / (3 * GR) + 1 / (9 * KR)),
                "hill": 1 / (1 / (3 * GH) + 1 / (9 * KH)),
                "spackman": self.spackman_average(kind="youngs_modulus"),
            },
            "poissons_ratio_avg": {
                "voigt": (1 - 3 * GV / (3 * KV + GV)) / 2,
                "reuss": (1 - 3 * GR / (3 * KR + GR)) / 2,
                "hill": (1 - 3 * GH / (3 * KH + GH)) / 2,
            },
        }

    def plot2d(self, kind="youngs_modulus", axis="xy", npoints=100, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        u = np.linspace(0, np.pi * 2, npoints)
        v = np.zeros_like(u)
        f = getattr(self, kind + "_angular")
        fig, ax = plt.subplots()
        fig.set_size_inches(kwargs.pop("figsize", (3, 3)))
        font = "Arial"
        xlims = kwargs.pop("xlim", None)
        ylims = kwargs.pop("ylim", None)
        #        ax.set_title(f"${axis}$-plane", fontname=font, fontsize=12)
        ax.set_xlabel(f"{axis[0]}", fontsize=12)
        ax.set_ylabel(f"{axis[1]}", fontsize=12, rotation=0)
        if axis == "xy":
            v[:] = np.pi / 2
            r = f(v, u)
            x = r * np.cos(u)
            y = r * np.sin(u)
        elif axis == "xz":
            r = f(u, v)
            y = r * np.cos(u)
            x = r * np.sin(u)
        elif axis == "yz":
            v[:] = np.pi / 2
            r = f(u, v)
            y = r * np.cos(u)
            x = r * np.sin(u)
        if xlims:
            ax.set_xlim(*xlims)
        if ylims:
            ax.set_ylim(*ylims)
        ax.xaxis.set_major_locator(plt.MaxNLocator(9))
        ax.yaxis.set_major_locator(plt.MaxNLocator(9))
        ax.plot(x, y, c="k", **kwargs)
        sns.despine(ax=ax, offset=0)
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        if kwargs.get("grid", False):
            ax.grid("minor", color="#BBBBBB", linewidth=0.5)
        zero_tick = (len(ax.get_xticks()) - 1) // 2
        for tick in ax.get_xticklabels():
            tick.set_fontname(font)
            tick.set_fontsize(4)
        for tick in ax.get_yticklabels():
            tick.set_fontname(font)
            tick.set_fontsize(4)
        ax.get_xticklabels()[zero_tick].set_visible(False)
        ax.get_yticklabels()[zero_tick].set_visible(False)
        ax.xaxis.set_label_coords(1.05, 0.53)
        ax.yaxis.set_label_coords(0.49, 1.02)
        return ax

    def mesh(self, kind="youngs_modulus", subdivisions=3):
        import trimesh

        f = getattr(self, kind)
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        r = f(sphere.vertices)
        sphere.vertices *= r[:, np.newaxis]
        return sphere

    def shape_descriptors(self, kind="youngs_modulus", l_max=5, **kwargs):
        from chmpy.shape.shape_descriptors import make_invariants
        from chmpy.shape.sht import SHT

        sht = SHT(l_max=l_max)
        f = getattr(self, kind)
        points = sht.grid_cartesian
        vals = f(points)
        if kwargs.get("normalize", False):
            vals = vals / self.spackman_average(kind=kind)
        coeffs = sht.analyse(vals)
        invariants = make_invariants(l_max, coeffs)
        if kwargs.get("coefficients", False):
            return coeffs, invariants
        return invariants

    def spackman_average(self, kind="youngs_modulus"):
        mesh = self.mesh(kind=kind)
        return np.mean(np.linalg.norm(mesh.vertices, axis=1), axis=0)

    def __repr__(self):
        s = np.array2string(
            self.c_voigt, precision=4, suppress_small=True, separator="  "
        )
        s = s.replace("[", " ")
        s = s.replace("]", " ")
        return f"<ElasticTensor:\n{s}>"

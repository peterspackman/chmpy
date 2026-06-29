"""
Simulated powder X-ray diffraction (PXRD) patterns.

A powder pattern is computed from a crystal structure in a few steps:

1. enumerate reflections (h, k, l) up to a resolution limit (`generate_hkl`),
2. compute the structure factor F(hkl) = sum_j occ_j f_j(s) exp(2*pi*i h.r_j)
   from the unit-cell atoms and their X-ray form factors (`structure_factors`),
3. place each reflection at its Bragg angle (lambda = 2 d sin(theta)) with
   intensity |F|^2 times the Lorentz-polarization correction, and
4. (optionally) bin/convolve into a 1D profile I(2*theta).

Systematically absent reflections need no special handling: their structure
factor is identically zero.

Intensities use neutral-atom X-ray form factors and the Lorentz-polarization
factor only; no Debye-Waller (temperature) factor and no anomalous dispersion
(f', f'') are applied, so relative intensities are most accurate for light
atoms and may differ for heavy atoms near an absorption edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scattering_factors import atomic_form_factor

#: Common characteristic X-ray wavelengths (K-alpha, Angstroms)
LAMBDA_CU_KA = 1.5405980
LAMBDA_MO_KA = 0.7107300
LAMBDA_CO_KA = 1.7889650
LAMBDA_CR_KA = 2.2897000
LAMBDA_FE_KA = 1.9373500
LAMBDA_AG_KA = 0.5594180


def generate_hkl(unit_cell, d_min: float):
    """Enumerate Miller indices with d-spacing >= d_min.

    The bound on each index is exact: since the reciprocal vector q of (h,k,l)
    satisfies q . a = h (and similarly for k, l), |h| <= |a| / d_min for any
    reflection with d = 1/|q| >= d_min.

    Args:
        unit_cell: a UnitCell
        d_min: minimum d-spacing in Angstroms

    Returns:
        (hkl, d_spacings): an (N, 3) integer array and an (N,) float array,
        excluding (0, 0, 0), sorted by decreasing d-spacing.
    """
    lengths = np.linalg.norm(unit_cell.direct, axis=1)
    bounds = np.floor(lengths / d_min).astype(int)
    ranges = [np.arange(-b, b + 1) for b in bounds]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]  # drop (0, 0, 0)

    q = grid @ unit_cell.reciprocal_lattice
    q_norm = np.linalg.norm(q, axis=1)
    d = 1.0 / q_norm
    keep = d >= d_min
    hkl = grid[keep]
    d = d[keep]
    order = np.argsort(-d)
    return hkl[order], d[order]


def structure_factors(crystal, hkl, d_spacings, dtype=np.float64) -> np.ndarray:
    """Structure factors F(hkl) for the given reflections.

        F(hkl) = sum_j  occ_j * f_j(s) * exp(2*pi*i * (h.r_j))

    summed over all atoms in the unit cell, with s = sin(theta)/lambda = 1/(2d).

    Atoms are grouped by element so each form factor is evaluated once per
    distinct element rather than once per atom:

        F(hkl) = sum_e f_e(s) * sum_{j in e} occ_j * exp(2*pi*i * h.r_j)

    Args:
        crystal: a Crystal
        hkl: (N, 3) integer Miller indices
        d_spacings: (N,) d-spacings for those reflections
        dtype: floating precision for the trigonometric sum. The default
            (float64) is exact; float32 is ~4x faster with a relative error in
            |F|^2 of ~1e-6, which is negligible for relative powder intensities.

    Returns:
        (N,) complex array of structure factors.
    """
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    uc = crystal.unit_cell_atoms()
    frac = uc["frac_pos"].astype(dtype)
    atomic_numbers = uc["element"]
    occupation = uc.get("occupation", np.ones(len(frac))).astype(dtype)
    hkl = np.asarray(hkl).astype(dtype)

    s = 1.0 / (2.0 * np.asarray(d_spacings))
    f = np.zeros(len(hkl), dtype=complex_dtype)
    two_pi = np.asarray(2.0 * np.pi, dtype=dtype)
    for z in np.unique(atomic_numbers):
        mask = atomic_numbers == z
        phase = two_pi * (hkl @ frac[mask].T)  # (N, M_e)
        # cos + i sin is markedly faster than np.exp on a complex array
        oscillation = np.cos(phase) + 1j * np.sin(phase)
        geometric = oscillation @ occupation[mask]  # (N,)
        f += atomic_form_factor([z], s)[0] * geometric
    return f


def _next_fast_length(n: int) -> int:
    "Smallest 7-smooth integer >= n (an efficient FFT size)."
    while True:
        m = n
        for factor in (2, 3, 5, 7):
            while m % factor == 0:
                m //= factor
        if m == 1:
            return n
        n += 1


def structure_factors_fft(
    crystal,
    hkl,
    d_spacings,
    oversample: float = 2.0,
    half_width: int = 4,
    sigma: float = 1.0,
) -> np.ndarray:
    """Structure factors via gridding + FFT (a non-uniform FFT, type 1).

    Each element's atoms are spread onto a real-space grid with a Gaussian
    kernel; one FFT then yields the geometric structure factors at every Miller
    index simultaneously, and the smooth atomic form factors are applied
    analytically. The Gaussian spreading is deconvolved exactly in reciprocal
    space.

    This costs O(V log V + M) instead of the direct method's O(N_refl * M), so
    it is much faster for large, low-symmetry cells (thousands of atoms /
    reflections) but slower for small or high-symmetry ones. See `powder_pattern`
    with ``method="auto"``.

    Args:
        crystal: a Crystal
        hkl: (N, 3) integer Miller indices
        d_spacings: (N,) d-spacings
        oversample: reciprocal-grid oversampling factor (>=1.5 recommended)
        half_width: Gaussian spreading half-width in grid points
        sigma: Gaussian width in grid units

    Returns:
        (N,) complex array of structure factors (relative error ~1e-5 versus the
        direct method with the defaults).
    """
    hkl = np.asarray(hkl)
    uc = crystal.unit_cell_atoms()
    frac = uc["frac_pos"] % 1.0
    atomic_numbers = uc["element"]
    occupation = uc.get("occupation", np.ones(len(frac)))

    h_max = np.abs(hkl).max(axis=0)
    n = np.array(
        [
            _next_fast_length(max(int(oversample * (2 * h + 1)), 2 * half_width + 2))
            for h in h_max
        ]
    )
    ny, nz = int(n[1]), int(n[2])
    grid_size = int(n[0] * ny * nz)
    grid_index = hkl % n

    # exact deconvolution of the Gaussian spreading kernel
    deconvolution = np.ones(len(hkl))
    for axis in range(3):
        deconvolution *= np.exp(
            2 * np.pi**2 * sigma**2 * (hkl[:, axis] / n[axis]) ** 2
        ) / (np.sqrt(2 * np.pi) * sigma)

    offsets = np.arange(-half_width, half_width + 1)
    s = 1.0 / (2.0 * np.asarray(d_spacings))
    f = np.zeros(len(hkl), dtype=np.complex128)
    for z in np.unique(atomic_numbers):
        mask = atomic_numbers == z
        coords = frac[mask] * n  # continuous grid coordinates
        base = np.round(coords).astype(int)
        # separable Gaussian weights along each axis
        wx = np.exp(
            -(((base[:, 0, None] + offsets) - coords[:, 0, None]) ** 2) / (2 * sigma**2)
        )
        wy = np.exp(
            -(((base[:, 1, None] + offsets) - coords[:, 1, None]) ** 2) / (2 * sigma**2)
        )
        wz = np.exp(
            -(((base[:, 2, None] + offsets) - coords[:, 2, None]) ** 2) / (2 * sigma**2)
        )
        ix = (base[:, 0, None] + offsets) % n[0]
        iy = (base[:, 1, None] + offsets) % ny
        iz = (base[:, 2, None] + offsets) % nz
        flat = (
            ix[:, :, None, None] * (ny * nz)
            + iy[:, None, :, None] * nz
            + iz[:, None, None, :]
        )
        weights = (
            occupation[mask][:, None, None, None]
            * wx[:, :, None, None]
            * wy[:, None, :, None]
            * wz[:, None, None, :]
        )
        grid = np.bincount(flat.ravel(), weights.ravel(), minlength=grid_size).reshape(
            tuple(n)
        )
        geometric = np.conj(np.fft.fftn(grid))  # +2pi i convention
        gathered = geometric[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]]
        f += atomic_form_factor([z], s)[0] * gathered * deconvolution
    return f


def _laue_rotations(space_group) -> np.ndarray:
    """Distinct rotation matrices of the Laue group (point group + inversion).

    Friedel's law (|F(h)| == |F(-h)| without anomalous dispersion) means powder
    intensities have the symmetry of the Laue group, so reflections related by
    these rotations are equivalent.
    """
    rotations = {}
    for op in space_group.symmetry_operations:
        r = np.round(op.rotation).astype(int)
        rotations[r.tobytes()] = r
        neg = -r
        rotations[neg.tobytes()] = neg
    return np.array(list(rotations.values()))


def _symmetry_unique(hkl: np.ndarray, d: np.ndarray, laue_rotations: np.ndarray):
    """Reduce reflections to a Laue-unique set with multiplicities.

    Each reflection's orbit under the Laue group is collapsed to one
    representative; the multiplicity is the number of reflections in the input
    sphere mapping to it (the full orbit is present because the sphere is closed
    under the rotations).

    Args:
        hkl: (N, 3) integer reflections (a full resolution sphere)
        d: (N,) d-spacings
        laue_rotations: (G, 3, 3) Laue rotation matrices

    Returns:
        (unique_hkl, unique_d, multiplicity)
    """
    # canonical key = lexicographic-max integer encoding over the orbit
    offset = int(np.abs(hkl).max()) + 1
    base = 2 * offset + 1
    weights = np.array([base * base, base, 1], dtype=np.int64)
    keys = None
    for rotation in laue_rotations:
        k = ((hkl @ rotation) + offset) @ weights  # (N,)
        keys = k if keys is None else np.maximum(keys, k)

    _, rep_index, multiplicity = np.unique(keys, return_index=True, return_counts=True)
    return hkl[rep_index], d[rep_index], multiplicity


def lorentz_polarization(two_theta: np.ndarray) -> np.ndarray:
    """Lorentz-polarization correction for an unpolarised powder source.

        Lp = (1 + cos^2(2*theta)) / (sin^2(theta) * cos(theta))

    Args:
        two_theta: 2*theta values in radians

    Returns:
        the Lp factor at each angle.
    """
    two_theta = np.asarray(two_theta, dtype=np.float64)
    theta = two_theta / 2.0
    return (1.0 + np.cos(two_theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))


@dataclass
class PowderPattern:
    """A simulated powder diffraction pattern.

    Reflections are Laue-symmetry-unique; `multiplicity` is the number of
    equivalent reflections and is already folded into `intensity`.

    Attributes:
        wavelength: X-ray wavelength in Angstroms
        two_theta_range: (min, max) 2*theta in degrees this pattern spans
        hkl: (N, 3) Miller indices (one representative per Laue orbit)
        d_spacing: (N,) d-spacings in Angstroms
        two_theta: (N,) Bragg angles 2*theta in degrees
        multiplicity: (N,) number of equivalent reflections
        f2: (N,) |F(hkl)|^2
        intensity: (N,) multiplicity * |F|^2 * Lorentz-polarization
    """

    wavelength: float
    two_theta_range: tuple
    hkl: np.ndarray
    d_spacing: np.ndarray
    two_theta: np.ndarray
    multiplicity: np.ndarray
    f2: np.ndarray
    intensity: np.ndarray

    def __len__(self) -> int:
        return len(self.hkl)

    def __repr__(self) -> str:
        lo, hi = self.two_theta_range
        return (
            f"<PowderPattern: {len(self)} reflections, "
            f"lambda={self.wavelength:.4f} A, 2theta {lo:g}-{hi:g} deg>"
        )

    def peaks(self, n=None, normalize=True):
        """The reflections sorted by decreasing intensity.

        Args:
            n: keep only the n strongest reflections (default: all)
            normalize: scale intensities so the strongest is 100

        Returns:
            a structured array with fields ('h', 'k', 'l', 'd', 'two_theta',
            'multiplicity', 'intensity'), one row per reflection.
        """
        order = np.argsort(-self.intensity)
        if n is not None:
            order = order[:n]
        intensity = self.intensity[order]
        if normalize and len(intensity) and intensity.max() > 0:
            intensity = 100.0 * intensity / intensity.max()
        rows = np.empty(
            len(order),
            dtype=[
                ("h", int),
                ("k", int),
                ("l", int),
                ("d", float),
                ("two_theta", float),
                ("multiplicity", int),
                ("intensity", float),
            ],
        )
        rows["h"], rows["k"], rows["l"] = self.hkl[order].T
        rows["d"] = self.d_spacing[order]
        rows["two_theta"] = self.two_theta[order]
        rows["multiplicity"] = self.multiplicity[order]
        rows["intensity"] = intensity
        return rows

    def profile(self, two_theta_range=None, num_bins=4500, fwhm=None, normalize=False):
        """Bin (and optionally broaden) the reflections into a 1D profile.

        Args:
            two_theta_range: (min, max) 2*theta in degrees; defaults to the
                range this pattern was computed over.
            num_bins: number of bins across the range
            fwhm: if given, convolve with a Gaussian of this FWHM (degrees) to
                approximate peak shapes; otherwise a stick histogram is returned.
            normalize: scale the profile so its maximum is 100.

        Returns:
            (two_theta, intensity): bin-centre angles (degrees) and intensities.
        """
        if two_theta_range is None:
            two_theta_range = self.two_theta_range
        lo, hi = two_theta_range
        counts, edges = np.histogram(
            self.two_theta, bins=num_bins, range=two_theta_range, weights=self.intensity
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        if fwhm:
            step = (hi - lo) / num_bins
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            half = int(np.ceil(4 * sigma / step))
            x = np.arange(-half, half + 1) * step
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
            counts = np.convolve(counts, kernel, mode="same")
        if normalize and counts.max() > 0:
            counts = 100.0 * counts / counts.max()
        return centres, counts

    def plot(
        self,
        two_theta_range=None,
        num_bins=4500,
        fwhm=0.1,
        normalize=True,
        ax=None,
        **kwargs,
    ):
        """Plot the pattern as intensity versus 2*theta.

        Args:
            two_theta_range: (min, max) 2*theta in degrees; defaults to the
                range this pattern was computed over.
            num_bins: number of bins across the range
            fwhm: Gaussian peak width in degrees (set None for a stick pattern)
            normalize: scale the maximum intensity to 100
            ax: an existing matplotlib Axes to draw on (a new one is created if
                not given)
            **kwargs: forwarded to `Axes.plot`

        Returns:
            the matplotlib Axes the pattern was drawn on.
        """
        import matplotlib.pyplot as plt

        x, y = self.profile(
            two_theta_range=two_theta_range,
            num_bins=num_bins,
            fwhm=fwhm,
            normalize=normalize,
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(r"$2\theta$ (degrees)")
        ax.set_ylabel("Intensity" + (" (normalised)" if normalize else ""))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        return ax


def plot_powder_patterns(
    patterns,
    labels=None,
    offset=10.0,
    x_offset=0.0,
    two_theta_range=None,
    num_bins=4500,
    fwhm=0.1,
    normalize=True,
    ax=None,
    **kwargs,
):
    """Overlay several powder patterns, stacked with an offset.

    Each pattern is drawn in the next colour of the matplotlib cycle and shifted
    from the one below it (a waterfall plot). With a non-zero `x_offset` the
    stack is also sheared sideways for a pseudo-3D look.

    Args:
        patterns: an iterable of PowderPattern
        labels: optional labels, one per pattern, shown in a legend
        offset: vertical shift between successive patterns (intensity units;
            with normalize=True the patterns are scaled to a maximum of 100)
        x_offset: horizontal shift between successive patterns, in 2*theta
            degrees (default 0 for a purely vertical stack)
        two_theta_range: (min, max) 2*theta in degrees; defaults to the union
            of the patterns' ranges
        num_bins: number of bins across the range
        fwhm: Gaussian peak width in degrees (set None for stick patterns)
        normalize: scale each pattern's maximum to 100
        ax: an existing matplotlib Axes to draw on (created if not given)
        **kwargs: forwarded to `Axes.plot`

    Returns:
        the matplotlib Axes the patterns were drawn on.
    """
    import matplotlib.pyplot as plt

    patterns = list(patterns)
    if two_theta_range is None and patterns:
        lows = [p.two_theta_range[0] for p in patterns]
        highs = [p.two_theta_range[1] for p in patterns]
        two_theta_range = (min(lows), max(highs))
    if ax is None:
        _, ax = plt.subplots()

    # draw back to front so the baseline pattern sits in front of the stacked
    # ones above it
    n = len(patterns)
    base_zorder = kwargs.pop("zorder", None)
    for i, pattern in enumerate(patterns):
        x, y = pattern.profile(
            two_theta_range=two_theta_range,
            num_bins=num_bins,
            fwhm=fwhm,
            normalize=normalize,
        )
        label = labels[i] if labels is not None and i < len(labels) else None
        zorder = base_zorder if base_zorder is not None else n - i
        ax.plot(x + i * x_offset, y + i * offset, label=label, zorder=zorder, **kwargs)

    ax.set_xlabel(r"$2\theta$ (degrees)")
    ax.set_ylabel("Intensity" + (" (offset)" if offset else ""))
    if two_theta_range is not None:
        lo, hi = two_theta_range
        shift = (n - 1) * x_offset
        ax.set_xlim(min(lo, lo + shift), max(hi, hi + shift))
    ax.set_ylim(bottom=0)
    if labels is not None:
        ax.legend()
    return ax


# above this estimated cost (n_unique_reflections * n_atoms) the gridding+FFT
# method overtakes the direct summation
_FFT_COST_THRESHOLD = 5_000_000


def powder_pattern(
    crystal,
    wavelength: float = LAMBDA_CU_KA,
    two_theta_range=(5.0, 50.0),
    dtype=np.float32,
    method: str = "auto",
) -> PowderPattern:
    """Compute a powder diffraction pattern for a crystal.

    Reflections are reduced to the Laue-unique set with multiplicities, so only
    one structure factor is computed per symmetry-equivalent group.

    Args:
        crystal: a Crystal
        wavelength: X-ray wavelength in Angstroms (default Cu K-alpha)
        two_theta_range: (min, max) 2*theta in degrees to include
        dtype: precision for the direct structure-factor sum; float32 (default)
            is fastest and accurate to ~1e-6 in relative intensity. Pass float64
            for full double precision.
        method: "direct" (exact summation), "fft" (gridding + FFT, faster for
            large low-symmetry cells), or "auto" (default) to pick per structure.

    Returns:
        a PowderPattern with per-reflection data; call `.profile()` to bin it.
    """
    if method not in ("auto", "direct", "fft"):
        raise ValueError(f"Unknown method {method!r}; use 'direct', 'fft' or 'auto'")

    lo, hi = two_theta_range
    d_min = wavelength / (2.0 * np.sin(np.radians(hi) / 2.0))
    hkl, d = generate_hkl(crystal.unit_cell, d_min)

    empty = PowderPattern(wavelength, tuple(two_theta_range), *([np.empty(0)] * 6))
    if len(hkl) == 0:
        return empty
    hkl, d, multiplicity = _symmetry_unique(
        hkl, d, _laue_rotations(crystal.space_group)
    )

    if method == "auto":
        n_atoms = len(crystal.unit_cell_atoms()["frac_pos"])
        method = "fft" if len(hkl) * n_atoms > _FFT_COST_THRESHOLD else "direct"
    if method == "fft":
        f = structure_factors_fft(crystal, hkl, d)
    else:
        f = structure_factors(crystal, hkl, d, dtype=dtype)

    two_theta = 2.0 * np.arcsin(np.clip(wavelength / (2.0 * d), -1.0, 1.0))
    f2 = (np.abs(f) ** 2).astype(np.float64)
    intensity = multiplicity * f2 * lorentz_polarization(two_theta)

    two_theta_deg = np.degrees(two_theta)
    keep = (two_theta_deg >= lo) & (two_theta_deg <= hi)
    return PowderPattern(
        wavelength=wavelength,
        two_theta_range=tuple(two_theta_range),
        hkl=hkl[keep],
        d_spacing=d[keep],
        two_theta=two_theta_deg[keep],
        multiplicity=multiplicity[keep],
        f2=f2[keep],
        intensity=intensity[keep],
    )

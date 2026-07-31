"""Character-grid plots.

Unlike the structure views, a plot is mostly axes, ticks and labels - things a
character grid does well and a pixel buffer does badly. So this stays on the
Canvas and is drawn entirely in box-drawing characters - rules and axes, no
block fills, so it reads as a stick pattern rather than a bar chart.
"""

from __future__ import annotations

import numpy as np

from .canvas import Canvas, theme_for

STICK = "│"
STICK_FOOT = "┴"  # where a reflection meets the axis


def _miller(h, k, l):  # noqa: E741 - hkl is the standard name
    """Compact hkl label.

    Without anomalous dispersion |F(hkl)| = |F(-h-k-l)|, so the Friedel mate
    with a positive leading index is free to choose and keeps most labels
    sign-free. Overbars are combining characters and would break the grid.
    """
    hkl = [int(i) for i in (h, k, l)]
    for i in hkl:
        if i != 0:
            if i < 0:
                hkl = [-x for x in hkl]
            break
    return "".join(str(i) for i in hkl)


def _nice_ticks(lo, hi, target=6):
    """Round tick positions covering [lo, hi], at a 1/2/5 x 10^n spacing."""
    span = max(hi - lo, 1e-9)
    raw = span / max(target, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if span / step <= target:
            break
    first = np.ceil(lo / step) * step
    ticks = np.arange(first, hi + 0.5 * step, step)
    return [t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]


def powder_plot(
    pattern,
    ncols=86,
    height=16,
    label_peaks=8,
    style="sticks",
    fwhm=0.08,
    theme=None,
):
    """A powder XRD pattern as a bar chart with box-drawing axes."""
    theme = theme or theme_for(None)
    lo, hi = pattern.two_theta_range
    gutter = 5  # width of the y tick labels, before the axis column
    plot_w = max(8, ncols - gutter - 1)

    def column_of(two_theta):
        return int(round((two_theta - lo) / max(hi - lo, 1e-9) * (plot_w - 1)))

    heights = np.zeros(plot_w)
    if style == "profile":
        _, prof = pattern.profile(num_bins=plot_w, fwhm=fwhm, normalize=True)
        heights = prof / max(prof.max(), 1e-30)
    else:
        for p in pattern.peaks():
            heights[column_of(p["two_theta"])] = max(
                heights[column_of(p["two_theta"])], p["intensity"] / 100.0
            )

    label_rows = 2
    top = label_rows  # first row of the plot body
    bottom = top + height - 1  # last body row; the axis sits below it
    cv = Canvas(top + height + 3, ncols)

    # ---- one vertical rule per reflection, rising from the axis
    for c, v in enumerate(heights):
        if v <= 0:
            continue
        # no minimum height: forcing every negligible reflection to one row
        # turns the baseline into a picket fence. Below half a row it is not
        # resolvable at this scale, so it is honest to leave it out
        rows = int(round(np.clip(v, 0, 1) * height))
        if rows == 0:
            continue
        col = gutter + 1 + c
        for i in range(rows):
            cv.put(bottom - i, col, STICK, theme.accent)

    # ---- y axis: ticks every quarter of full scale
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        row = bottom - int(round(frac * (height - 1)))
        label = f"{frac * 100:.0f}"
        cv.text(row, gutter - len(label), label, theme.dim)
        cv.put(row, gutter, "┤", theme.dim)
    for row in range(top, bottom + 1):
        if cv.chars[row, gutter] == " ":
            cv.put(row, gutter, "│", theme.dim)

    # ---- x axis, with ticks at round 2-theta values
    axis = bottom + 1
    cv.put(axis, gutter, "└", theme.dim)
    for c in range(plot_w):
        # a reflection joins the axis rather than sitting on top of it
        drawn = int(round(np.clip(heights[c], 0, 1) * height)) > 0
        foot = STICK_FOOT if drawn else "─"
        cv.put(axis, gutter + 1 + c, foot, theme.dim)
    for value in _nice_ticks(lo, hi):
        col = gutter + 1 + column_of(value)
        cv.put(axis, col, "┬", theme.dim)
        label = f"{value:g}"
        cv.text(axis + 1, col - len(label) // 2, label, theme.dim)

    # ---- hkl labels above the strongest reflections
    occupied = []
    for p in pattern.peaks(n=label_peaks):
        col = gutter + 1 + column_of(p["two_theta"])
        text = _miller(p["h"], p["k"], p["l"])
        start = max(gutter + 1, min(ncols - len(text), col - len(text) // 2))
        span = range(start - 1, start + len(text) + 1)
        if any(s in occupied for s in span):
            continue
        occupied.extend(span)
        cv.text(0, start, text, theme.dim)
        cv.put(1, col, "╵", theme.dim)

    footer = f"2θ / °      λ = {pattern.wavelength:.4f} Å"
    cv.text(axis + 2, gutter + 1, footer, theme.dim)
    return cv

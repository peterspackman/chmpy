"""Terminal rendering for chmpy, at two RGB pixels per character cell.

A character cell is drawn as an upper-half block with a foreground and a
background colour, so one cell carries two independently coloured pixels and
the pixels come out square. That is enough resolution to rasterise van der
Waals spheres with a z-buffer, i.e. to actually render a structure rather
than spell it out in letters.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------- characters ----

UPPER, LOWER, FULL = "▀", "▄", "█"
RESET = "\x1b[0m"
DEFAULT_BG = "\x1b[49m"


def _fg(c):
    return f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"


def _bg(c):
    return f"\x1b[48;2;{c[0]};{c[1]};{c[2]}m"


class Canvas:
    """A character grid where each cell has a glyph, a fg and a bg colour."""

    def __init__(self, nrows, ncols):
        self.chars = np.full((nrows, ncols), " ", dtype="<U1")
        self.fg = np.full((nrows, ncols, 3), -1, dtype=np.int16)
        self.bg = np.full((nrows, ncols, 3), -1, dtype=np.int16)

    @property
    def shape(self):
        return self.chars.shape

    def put(self, r, c, ch, fg=None, bg=None):
        if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
            return
        self.chars[r, c] = ch
        if fg is not None:
            self.fg[r, c] = fg[:3]
        if bg is not None:
            self.bg[r, c] = bg[:3]

    def text(self, r, c, s, fg=None, bg=None):
        for i, ch in enumerate(s):
            self.put(r, c + i, ch, fg, bg)

    def render(self, colors="truecolor"):
        """Emit the grid as text. `colors` selects the escape-sequence depth."""
        if colors == "none":
            return "\n".join(
                "".join(self.chars[r]).rstrip() for r in range(self.shape[0])
            )
        wide = colors == "truecolor"
        fg_of = _fg if wide else (lambda c: f"\x1b[38;5;{to_256(c)}m")
        bg_of = _bg if wide else (lambda c: f"\x1b[48;5;{to_256(c)}m")

        lines = []
        for r in range(self.shape[0]):
            parts, cur_fg, cur_bg = [], None, None
            for c in range(self.shape[1]):
                f = tuple(self.fg[r, c])
                b = tuple(self.bg[r, c])
                f = None if f[0] < 0 else f
                b = None if b[0] < 0 else b
                if f != cur_fg:
                    parts.append(RESET if f is None else fg_of(f))
                    if f is None:
                        cur_bg = None  # RESET cleared the background too
                    cur_fg = f
                if b != cur_bg:
                    parts.append(DEFAULT_BG if b is None else bg_of(b))
                    cur_bg = b
                parts.append(self.chars[r, c])
            parts.append(RESET)
            lines.append("".join(parts))
        return "\n".join(lines)

    def __str__(self):
        return self.render(colors="none")


# ------------------------------------------------------------ pixel plane ----


class Pixels:
    """An RGB image with a coverage mask, rendered two rows per text line."""

    def __init__(self, height, width):
        self.rgb = np.zeros((height, width, 3), dtype=np.float64)
        self.alpha = np.zeros((height, width), dtype=bool)

    @property
    def shape(self):
        return self.alpha.shape

    def to_canvas(self):
        h, w = self.shape
        if h % 2:  # pad to an even number of rows
            self.rgb = np.vstack([self.rgb, np.zeros((1, w, 3))])
            self.alpha = np.vstack([self.alpha, np.zeros((1, w), bool)])
            h += 1
        rgb = np.clip(self.rgb, 0, 255).astype(np.int16)
        top, bot = rgb[0::2], rgb[1::2]
        atop, abot = self.alpha[0::2], self.alpha[1::2]

        cv = Canvas(h // 2, w)
        both = atop & abot
        cv.chars[both] = UPPER
        cv.fg[both] = top[both]
        cv.bg[both] = bot[both]

        only_top = atop & ~abot
        cv.chars[only_top] = UPPER
        cv.fg[only_top] = top[only_top]

        only_bot = abot & ~atop
        cv.chars[only_bot] = LOWER
        cv.fg[only_bot] = bot[only_bot]
        return cv


ASCII_RAMP = " .:-=+*#%@"


def ascii_art(px, ramp=ASCII_RAMP):
    """Render pixels as shaded text, for output that carries no colour.

    Half blocks say nothing without colour - the glyph is identical everywhere
    and the picture lives entirely in the escape sequences - so a colourless
    terminal, a pipe or a log file needs brightness mapped to characters
    instead. Vertical pairs are averaged so one character is one cell, keeping
    the aspect ratio the same as every other output mode.
    """
    rgb, alpha = px.rgb, px.alpha
    h, w = alpha.shape
    if h % 2:
        rgb = np.vstack([rgb, np.zeros((1, w, 3))])
        alpha = np.vstack([alpha, np.zeros((1, w), bool)])
        h += 1

    lum = rgb @ np.array([0.299, 0.587, 0.114])
    lum = np.where(alpha, lum, 0.0).reshape(h // 2, 2, w).mean(axis=1)
    covered = alpha.reshape(h // 2, 2, w).any(axis=1)

    top = lum[covered].max() if covered.any() else 1.0
    steps = np.clip(lum / max(top, 1e-9), 0, 1) * (len(ramp) - 1)
    index = np.rint(steps).astype(int)
    # anything covered gets at least the first visible step, so a dark atom is
    # never silently indistinguishable from empty space
    index[covered] = np.maximum(index[covered], 1)
    index[~covered] = 0
    return "\n".join("".join(ramp[i] for i in row).rstrip() for row in index)


# ------------------------------------------------------- rasterisation ----
#
# Software rasterisation in numpy, with no Python loop over primitives.
#
# Each primitive contributes a rectangle of candidate pixels. Those rectangles
# are expanded into one flat *fragment* list (one entry per primitive-pixel
# pair), the surface test and shading run over the whole list at once, and the
# depth test is a single sort-and-scatter. That is the standard shape of a
# vectorised software rasteriser: the per-primitive cost becomes a handful of
# array ops instead of an interpreted iteration.
#
# Fragments are produced in chunks so peak memory stays bounded no matter how
# large the scene is; chunks are resolved in sequence against the same buffer,
# which is what keeps the result identical to drawing one primitive at a time.

# light comes from the upper left, slightly toward the viewer
LIGHT = np.array([-0.45, 0.55, 0.70])
LIGHT /= np.linalg.norm(LIGHT)

#: soft cap on fragments held in memory at once (~50 MB of working arrays)
CHUNK_FRAGMENTS = 2_000_000


def _pixel_boxes(px_cx, px_cy, px_r, shape):
    """Clipped integer pixel bounds for a set of circular footprints."""
    h, w = shape
    c0 = np.clip(np.floor(px_cx - px_r).astype(np.int64), 0, w)
    c1 = np.clip(np.ceil(px_cx + px_r).astype(np.int64) + 1, 0, w)
    r0 = np.clip(np.floor(px_cy - px_r).astype(np.int64), 0, h)
    r1 = np.clip(np.ceil(px_cy + px_r).astype(np.int64) + 1, 0, h)
    return r0, r1, c0, c1


def _chunk_spans(counts, budget=CHUNK_FRAGMENTS):
    """Split primitives into runs whose fragment counts fit the budget."""
    if not len(counts):
        return []
    budget = max(int(budget), int(counts.max()), 1)
    group = np.cumsum(counts) // budget
    edges = np.flatnonzero(np.diff(group)) + 1
    return np.split(np.arange(len(counts)), edges)


def _fragments(r0, r1, c0, c1):
    """Expand pixel boxes into flat (primitive, row, col) fragment arrays."""
    heights, widths = r1 - r0, c1 - c0
    counts = heights * widths
    total = int(counts.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    prim = np.repeat(np.arange(len(counts)), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    offset = np.arange(total) - np.repeat(starts, counts)
    w = widths[prim]
    return prim, r0[prim] + offset // w, c0[prim] + offset % w


def _resolve(px, zbuffer, rows, cols, z, rgb):
    """Depth-test a fragment list and scatter the survivors into the image.

    `np.maximum.at` resolves every pixel collision in one unbuffered scatter.
    The old advice was that `ufunc.at` is slow and a sort-and-group is faster;
    numpy has since grown a fast path for it, and measured on a real frame it
    beats a lexsort by around seventy times, so the sort is not worth keeping.

    A fragment wins its pixel when its depth equals the post-scatter maximum,
    having also improved on what was already there.
    """
    if not len(rows):
        return
    index = (rows, cols)
    prior = zbuffer[index]
    np.maximum.at(zbuffer, index, z)
    win = np.flatnonzero((z > prior) & (z == zbuffer[index]))
    if not len(win):
        return
    wr, wc = rows[win], cols[win]
    px.rgb[wr, wc] = np.clip(rgb[win], 0, 255)
    px.alpha[wr, wc] = True


def shade_surface(normals, base, ambient, specular, shininess, shading="lit"):
    """Shade fragments given per-fragment normals (M, 3) and colours (M, 3)."""
    if shading == "flat":
        # constant colour with a darkened rim where the surface turns away,
        # which keeps touching atoms distinguishable without any lighting
        rim = np.clip((normals[:, 2] - 0.15) / 0.35, 0, 1)
        return base * (0.55 + 0.45 * rim)[:, None]
    lam = np.clip(normals @ LIGHT, 0, 1)
    half = LIGHT + np.array([0.0, 0.0, 1.0])
    half /= np.linalg.norm(half)
    spec = np.clip(normals @ half, 0, 1) ** shininess
    # damp the highlight on already-bright colours (hydrogen is pure white in
    # CPK) so they keep their shading instead of clipping to a flat blob
    lum = base @ np.array([0.299, 0.587, 0.114])
    strength = specular * (1.0 - 0.75 * lum / 255.0)
    return base * (ambient + (1 - ambient) * lam)[:, None] + 255 * (
        strength * spec
    )[:, None]


def _apply_depth_cue(rgb, z, depth_cue, fade_to):
    if depth_cue is None:
        return rgb
    near, far = depth_cue
    f = np.clip((z - far) / max(near - far, 1e-9), 0, 1)[:, None]
    # aerial perspective: distant surfaces recede toward the background,
    # which is only "toward black" if the background happens to be dark
    if fade_to is None:
        return rgb * (0.35 + 0.65 * f)
    f = 0.35 + 0.65 * f
    return rgb * f + np.asarray(fade_to, dtype=float) * (1 - f)


def draw_spheres(
    px,
    centers,
    radii,
    colors,
    scale,
    origin,
    zbuffer=None,
    ambient=0.30,
    specular=0.45,
    shininess=18.0,
    depth_cue=None,
    fade_to=None,
    shading="lit",
):
    """Rasterise shaded spheres into `px` with a z-buffer.

    centers are (N, 3) in world units with +z toward the viewer; `scale` is
    pixels per world unit and `origin` is the world point at pixel (0, 0) of
    the top-left, with screen y increasing downward.
    """
    h, w = px.shape
    if zbuffer is None:
        zbuffer = np.full((h, w), -np.inf)
    centers = np.asarray(centers, dtype=float)
    if not len(centers):
        return zbuffer
    radii = np.asarray(radii, dtype=float)
    colors = np.asarray(colors, dtype=float)

    pcx = (centers[:, 0] - origin[0]) * scale
    pcy = (origin[1] - centers[:, 1]) * scale
    pr = radii * scale
    r0, r1, c0, c1 = _pixel_boxes(pcx, pcy, pr, (h, w))
    live = np.flatnonzero((r1 > r0) & (c1 > c0) & (pr > 0))
    if not len(live):
        return zbuffer
    # nearest first, so early-Z has something to reject against straight away
    live = live[np.argsort(-(centers[live, 2] + radii[live]))]

    counts = (r1[live] - r0[live]) * (c1[live] - c0[live])
    for span in _chunk_spans(counts):
        sel = live[span]
        prim, rows, cols = _fragments(r0[sel], r1[sel], c0[sel], c1[sel])
        if not len(prim):
            continue
        idx = sel[prim]
        # unit-sphere coordinates of each fragment relative to its centre
        dx = (cols + 0.5 - pcx[idx]) / pr[idx]
        dy = (pcy[idx] - rows - 0.5) / pr[idx]
        d2 = dx * dx + dy * dy
        hit = d2 <= 1.0
        if not hit.any():
            continue
        idx, rows, cols = idx[hit], rows[hit], cols[hit]
        dx, dy = dx[hit], dy[hit]
        dz = np.sqrt(1.0 - d2[hit])
        z = centers[idx, 2] + dz * radii[idx]

        # early-Z: discard occluded fragments before paying for shading and,
        # more importantly, before they bloat the sort in _resolve
        alive = z > zbuffer[rows, cols]
        if not alive.any():
            continue
        idx, rows, cols = idx[alive], rows[alive], cols[alive]
        dx, dy, dz, z = dx[alive], dy[alive], dz[alive], z[alive]

        normals = np.stack([dx, dy, dz], axis=1)
        rgb = shade_surface(
            normals, colors[idx], ambient, specular, shininess, shading
        )
        rgb = _apply_depth_cue(rgb, z, depth_cue, fade_to)
        _resolve(px, zbuffer, rows, cols, z, rgb)
    return zbuffer


def draw_cylinders(
    px,
    starts,
    ends,
    radii,
    colors0,
    colors1,
    scale,
    origin,
    zbuffer=None,
    ambient=0.30,
    specular=0.45,
    shininess=18.0,
    depth_cue=None,
    fade_to=None,
    shading="lit",
):
    """Rasterise shaded cylinders by exact ray intersection, z-buffered.

    A chain of spheres is the easy way to fake a bond, but a sphere's normal
    points radially outward in every direction, so each bead picks up its own
    highlight and the bond reads as a string of beads. A cylinder's normal is
    perpendicular to its axis; solving the intersection properly is what makes
    a bond look like a rod.
    """
    h, w = px.shape
    if zbuffer is None:
        zbuffer = np.full((h, w), -np.inf)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if not len(starts):
        return zbuffer
    radii = np.asarray(radii, dtype=float)
    colors0 = np.asarray(colors0, dtype=float)
    colors1 = np.asarray(colors1, dtype=float)

    axis = ends - starts
    dd = np.einsum("ij,ij->i", axis, axis)
    # the ray runs along +z, so an axis parallel to it degenerates to a disc;
    # those bonds point at the viewer and their end atoms cover them anyway
    denom = 1.0 - np.divide(axis[:, 2] ** 2, dd, out=np.zeros_like(dd), where=dd > 0)

    sx = (starts[:, 0] - origin[0]) * scale
    sy = (origin[1] - starts[:, 1]) * scale
    ex = (ends[:, 0] - origin[0]) * scale
    ey = (origin[1] - ends[:, 1]) * scale
    pr = radii * scale
    c0 = np.clip(np.floor(np.minimum(sx, ex) - pr).astype(np.int64), 0, w)
    c1 = np.clip(np.ceil(np.maximum(sx, ex) + pr).astype(np.int64) + 1, 0, w)
    r0 = np.clip(np.floor(np.minimum(sy, ey) - pr).astype(np.int64), 0, h)
    r1 = np.clip(np.ceil(np.maximum(sy, ey) + pr).astype(np.int64) + 1, 0, h)

    live = np.flatnonzero((r1 > r0) & (c1 > c0) & (dd > 1e-12) & (denom > 1e-6))
    if not len(live):
        return zbuffer
    nearest = np.maximum(starts[:, 2], ends[:, 2]) + radii
    live = live[np.argsort(-nearest[live])]

    counts = (r1[live] - r0[live]) * (c1[live] - c0[live])
    for span in _chunk_spans(counts):
        sel = live[span]
        prim, rows, cols = _fragments(r0[sel], r1[sel], c0[sel], c1[sel])
        if not len(prim):
            continue
        idx = sel[prim]
        a = axis[idx]
        # vector from the cylinder start to the pixel's ray origin (z = 0)
        wx = origin[0] + (cols + 0.5) / scale - starts[idx, 0]
        wy = origin[1] - (rows + 0.5) / scale - starts[idx, 1]
        wz = -starts[idx, 2]
        wa = wx * a[:, 0] + wy * a[:, 1] + wz * a[:, 2]
        ww = wx * wx + wy * wy + wz * wz
        ddi = dd[idx]

        qb = 2.0 * (wz - a[:, 2] * wa / ddi)
        qc = ww - wa * wa / ddi - radii[idx] ** 2
        disc = qb * qb - 4.0 * denom[idx] * qc
        hit = disc >= 0
        if not hit.any():
            continue
        s = np.zeros_like(disc)
        s[hit] = (-qb[hit] + np.sqrt(disc[hit])) / (2.0 * denom[idx][hit])
        t = (wa + s * a[:, 2]) / ddi
        hit &= (t >= 0.0) & (t <= 1.0)
        if not hit.any():
            continue

        idx, rows, cols = idx[hit], rows[hit], cols[hit]
        a, t, s = axis[idx], t[hit], s[hit]
        wx, wy, wz = wx[hit], wy[hit], wz[hit]

        alive = s > zbuffer[rows, cols]  # early-Z, as for spheres
        if not alive.any():
            continue
        idx, rows, cols = idx[alive], rows[alive], cols[alive]
        a, t, s = a[alive], t[alive], s[alive]
        wx, wy, wz = wx[alive], wy[alive], wz[alive]

        nx = wx - t * a[:, 0]
        ny = wy - t * a[:, 1]
        nz = wz + s - t * a[:, 2]
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        norm[norm == 0] = 1.0
        normals = np.stack([nx / norm, ny / norm, nz / norm], axis=1)

        base = np.where((t < 0.5)[:, None], colors0[idx], colors1[idx])
        rgb = shade_surface(normals, base, ambient, specular, shininess, shading)
        rgb = _apply_depth_cue(rgb, s, depth_cue, fade_to)
        _resolve(px, zbuffer, rows, cols, s, rgb)
    return zbuffer


# ------------------------------------------------------ terminal support ----


@dataclass
class Terminal:
    """What the attached terminal can actually do."""

    colors: str = "truecolor"  # 'truecolor' | '256' | 'none'
    graphics: bool = False  # Kitty graphics protocol

    @property
    def best(self):
        if self.graphics:
            return "graphics"
        return "blocks" if self.colors != "none" else "text"


#: terminals known to implement the Kitty graphics protocol
_GRAPHICS_ENV = ("KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "WEZTERM_EXECUTABLE")
_GRAPHICS_PROGRAMS = ("ghostty", "wezterm", "kitty")
#: TERM values set by terminals that speak the graphics protocol. TERM is
#: forwarded over ssh where TERM_PROGRAM and the vendor variables are not, so
#: this is the only signal that survives a remote session.
_GRAPHICS_TERMS = ("kitty", "ghostty", "wezterm")
#: multiplexers rewrite escape sequences, and pass the graphics protocol
#: through only when explicitly configured, so assume they break it
_MULTIPLEXER_TERMS = ("screen", "tmux")


def detect(stream=None, allow_graphics=True):
    """Inspect the environment for colour and inline-image support.

    Detection is environment-based rather than by querying the terminal: a
    query needs raw tty mode and a read timeout, which misbehaves when output
    is piped or run under a test harness.
    """
    stream = stream or sys.stdout
    interactive = hasattr(stream, "isatty") and stream.isatty()

    if os.environ.get("NO_COLOR") is not None:
        return Terminal("none", graphics=False)
    if not interactive and not os.environ.get("CHMPY_FORCE_COLOR"):
        return Terminal("none", graphics=False)

    term = os.environ.get("TERM", "")
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if os.environ.get("COLORTERM", "") in ("truecolor", "24bit"):
        colors = "truecolor"
    elif "256color" in term or program in ("iterm.app", "ghostty", "wezterm"):
        colors = "256"
    elif term in ("dumb", ""):
        colors = "none"
    else:
        colors = "256"

    graphics = allow_graphics and (
        any(name in term for name in _GRAPHICS_TERMS)
        or program in _GRAPHICS_PROGRAMS
        or any(os.environ.get(v) for v in _GRAPHICS_ENV)
    )
    if graphics and _multiplexed(term):
        # the vendor variables leak through tmux from the outer terminal, so
        # without this a multiplexed session claims a protocol it will mangle
        graphics = bool(os.environ.get("CHMPY_FORCE_GRAPHICS"))
    return Terminal(colors, graphics)


def _multiplexed(term):
    """Whether we are inside tmux or screen, which rewrite escape sequences."""
    if os.environ.get("TMUX") or os.environ.get("STY"):
        return True
    return any(term.startswith(name) for name in _MULTIPLEXER_TERMS)


_CUBE = np.array([0, 95, 135, 175, 215, 255])


def to_256(rgb):
    """Nearest xterm-256 index for an RGB triple."""
    r, g, b = (int(x) for x in rgb)
    idx = [int(np.argmin(np.abs(_CUBE - v))) for v in (r, g, b)]
    cube = 16 + 36 * idx[0] + 6 * idx[1] + idx[2]
    cube_err = sum((_CUBE[i] - v) ** 2 for i, v in zip(idx, (r, g, b), strict=False))
    # the 24-step grey ramp often beats the cube for desaturated colours
    grey = int(round((r + g + b) / 3))
    step = int(np.clip(round((grey - 8) / 10), 0, 23))
    gv = 8 + 10 * step
    grey_err = (gv - r) ** 2 + (gv - g) ** 2 + (gv - b) ** 2
    return 232 + step if grey_err < cube_err else cube


def kitty_image(px, background=None, cols=None):
    """Encode a Pixels image as a Kitty graphics protocol escape sequence.

    This transmits real pixels rather than half-blocks, so the picture is not
    limited to two rows per character cell. Uncovered pixels are sent as
    transparent RGBA rather than filled with an assumed colour, so the image
    sits on whatever the terminal background actually is - matching how the
    half-block path behaves, where a clear pixel simply emits no background.

    Pass `background` only to force an opaque fill.
    """
    from PIL import Image

    rgb = np.clip(px.rgb, 0, 255).astype(np.uint8)
    if background is None:
        alpha = (px.alpha * 255).astype(np.uint8)
        im = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    else:
        flat = np.where(px.alpha[..., None], px.rgb, np.array(background))
        im = Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    h, w = px.shape
    cols = cols or w
    rows = max(1, int(round(cols * h / w / 2)))  # cells are twice as tall as wide

    out, chunk = [], 4096
    pieces = [data[i : i + chunk] for i in range(0, len(data), chunk)] or [""]
    for n, piece in enumerate(pieces):
        more = 1 if n < len(pieces) - 1 else 0
        if n == 0:
            head = f"a=T,f=100,c={cols},r={rows},m={more}"
        else:
            head = f"m={more}"
        out.append(f"\x1b_G{head};{piece}\x1b\\")
    return "".join(out) + "\n"


def display(px, terminal=None, cols=None, background=None):
    """Render a Pixels image using the best mode the terminal supports.

    Uncovered pixels stay transparent in both modes, so the terminal's own
    background shows through. Pass `background` only to force an opaque fill.
    """
    terminal = terminal or detect()
    if terminal.best == "graphics":
        return kitty_image(px, background=background, cols=cols)
    if terminal.colors == "none":
        return ascii_art(px)
    return px.to_canvas().render(colors=terminal.colors)


# ------------------------------------------------------------------ theme ----


@dataclass
class Theme:
    """Chrome colours chosen to read against the terminal's own background."""

    background: tuple | None  # None when the terminal did not tell us
    text: tuple
    dim: tuple
    accent: tuple
    edge: tuple
    warn: tuple

    @property
    def is_dark(self):
        return luminance(self.background or (13, 17, 23)) < 128

    @property
    def fill(self):
        """A concrete colour for the few places that must composite."""
        return self.background or (13, 17, 23)


def luminance(rgb):
    r, g, b = rgb[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def theme_for(background=None):
    """Derive readable chrome from a terminal background colour.

    `background` of None means the terminal did not answer the OSC 11 query,
    in which case assume dark, which is by far the more common default.
    """
    if background is not None and luminance(background) >= 128:
        return Theme(
            background=background,
            text=(35, 38, 48),
            dim=(120, 122, 135),
            accent=(30, 90, 170),
            edge=(70, 72, 90),
            warn=(150, 90, 10),
        )
    return Theme(
        background=background,
        text=(200, 200, 215),
        dim=(110, 110, 125),
        accent=(150, 200, 255),
        edge=(215, 220, 235),
        warn=(240, 180, 90),
    )


#: Pixels per character column when the terminal can display real images.
#: Half-block output is fixed at one pixel per column (two per cell, so the
#: pixels stay square); the graphics protocol has no such limit, so render
#: larger and let the terminal scale the image down.
GRAPHICS_OVERSAMPLE = 8

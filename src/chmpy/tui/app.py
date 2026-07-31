"""The interactive viewer: state, key handling and the redraw loop.

Renders at full resolution when idle and at draft resolution while a key is
held, so turning a structure stays responsive in graphics mode where a
full-quality frame costs longer than a keypress.

Argument parsing and loading live in `__main__`; this module starts from an
already-loaded list of frames, which is what makes the whole thing drivable
from a test without a terminal.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chmpy.crystal.crystal import Crystal

from . import tty
from .canvas import GRAPHICS_OVERSAMPLE, Canvas, display, theme_for
from .plots import powder_plot
from .scene import (
    crystal_scene,
    format_direction,
    molecule_scene,
    parse_direction,
    render_scene,
    rotation_matrix,
    view_rotation,
)

DRAFT_OVERSAMPLE = 2
FULL_OVERSAMPLE = GRAPHICS_OVERSAMPLE
#: how often to notice the window has been resized, in seconds
IDLE_TIMEOUT = 0.4
#: 15 degrees divides 90, so repeated presses land exactly on an axis view
STEP = np.radians(15)
YAW = rotation_matrix(STEP, 0.0)
PITCH = rotation_matrix(0.0, STEP)
#: the number keys, as (label, indices, kind) per structure type
QUICK_AXES = {
    True: [("a", (1, 0, 0), "uvw"), ("b", (0, 1, 0), "uvw"), ("c", (0, 0, 1), "uvw")],
    False: [
        ("x", (1, 0, 0), "cartesian"),
        ("y", (0, 1, 0), "cartesian"),
        ("z", (0, 0, 1), "cartesian"),
    ],
}


def default_view(structure):
    """Down c for a crystal, down z for a molecule already in its PCA frame."""
    if isinstance(structure, Crystal):
        return view_rotation((0, 0, 1), "uvw", structure), "[001]"
    return np.eye(3), "z"


# ------------------------------------------------------------------ state ----


@dataclass
class State:
    frames: list
    index: int = 0
    view: str = "structure"
    rotation: np.ndarray | None = None
    direction: str = ""  # label of the aimed view, blank once freely rotated
    prompt: str | None = None  # direction being typed, when the prompt is open
    zoom: float = 1.0
    style: str = "ball-and-stick"
    shading: str = "lit"
    cells: int = 1
    frame: tuple | None = None  # framing extent, expand-only
    message: str = ""
    theme: object = field(default_factory=lambda: theme_for(None))
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.rotation is None:
            self.rotation, self.direction = default_view(self.structure)

    @property
    def structure(self):
        return self.frames[self.index].structure

    def step(self, delta):
        """Move through the ensemble, stopping at the ends rather than wrapping.

        Rotation, zoom and framing deliberately survive the move: comparing
        frames is the whole point, and a view that reset itself each step
        would make that impossible.
        """
        target = int(np.clip(self.index + delta, 0, len(self.frames) - 1))
        if target == self.index:
            return
        was_crystal = self.is_crystal
        self.index = target
        if self.is_crystal != was_crystal:  # mixed ensemble; the view no longer applies
            self.rotation, self.direction = default_view(self.structure)
            self.frame = None

    def aim(self, indices, kind):
        self.rotation = view_rotation(indices, kind, self.structure
                                      if self.is_crystal else None)
        self.direction = format_direction(indices, kind)
        self.frame = None  # a new direction deserves a fresh tight fit

    @property
    def is_crystal(self):
        return isinstance(self.structure, Crystal)

    def views(self):
        return ["structure", "powder"] if self.is_crystal else ["structure"]

    def memo(self, key, build):
        if key not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[key] = build()
        return self.cache[key]


# ---------------------------------------------------------------- drawing ----


def build_scene(state):
    def make():
        if state.is_crystal:
            n = state.cells
            return crystal_scene(state.structure, cells=(n, n, n), style=state.style)
        return molecule_scene(state.structure, style=state.style)

    return state.memo(("scene", state.index, state.style, state.cells), make)


def render_body(state, terminal, cols, rows, oversample):
    """Produce (payload, kind) where kind is 'pixels' or 'canvas'."""
    if state.view not in state.views():
        state.view = "structure"

    if state.view == "powder":
        pattern = state.memo(
            ("powder", state.index),
            lambda: state.structure.powder_pattern(two_theta_range=(5, 55)),
        )
        return powder_plot(
            pattern, ncols=cols, height=max(6, rows - 5), theme=state.theme
        ), "canvas"

    factor = oversample if terminal.graphics else 1
    scene = build_scene(state)
    rotation = state.rotation

    # start tight, then only ever grow: a fit recomputed from scratch each
    # frame would make the structure swell and shrink as it turns, and a fixed
    # rotation-invariant bound wastes most of the frame
    needed = scene.view_extent(rotation)
    state.frame = (
        needed
        if state.frame is None
        else (max(state.frame[0], needed[0]), max(state.frame[1], needed[1]))
    )

    px = render_scene(
        scene,
        rotation,
        width=cols * factor,
        height=rows * 2 * factor,
        zoom=state.zoom,
        theme=state.theme,
        shading=state.shading,
        extent=state.frame,
    )
    return px, "pixels"


def status_bar(state, cols, timing):
    t = state.theme
    bar = Canvas(2, cols)
    x = 0
    for name in state.views():
        selected = name == state.view
        label = f" {name} "
        bar.text(0, x, label, t.fill if selected else t.dim,
                 t.accent if selected else None)
        x += len(label) + 1

    right = f"{timing:.0f} ms "
    bar.text(0, max(0, cols - len(right)), right, t.dim)
    # the colour key lives on the tab row, where there is room; the hint row
    # below is already full of key bindings on all but the widest terminals
    if state.is_crystal and state.view == "structure" and state.prompt is None:
        _axis_key(bar, cols - len(right) - 1, after=x)

    if state.prompt is not None:
        names = " ".join(a[0] for a in QUICK_AXES[state.is_crystal])
        names += " a* b* c*" if state.is_crystal else ""
        bar.text(1, 0, "view along: ", t.text)
        bar.text(1, 12, state.prompt + "\u2588", t.accent)
        hint = f"[uvw] (hkl) {names}   enter to aim, esc to cancel"
        bar.text(1, max(0, cols - len(hint)), hint, t.dim)
        return bar

    if state.view == "structure":
        axes = "".join(a[0] for a in QUICK_AXES[state.is_crystal])
        keys = (
            f"hjkl rotate  123 {axes}  d direction  +/- zoom  s {state.style}"
            f"  f {state.shading}"
            + (f"  c {state.cells}³" if state.is_crystal else "")
            + "  r reset  q quit"
        )
    else:
        keys = "tab view   q quit"
    if len(state.frames) > 1:
        keys = "n/p frame  " + keys

    if state.message:
        bar.text(1, 0, state.message, t.warn)
    else:
        bar.text(1, 0, keys, t.dim)
    return bar


def _axis_key(bar, right_edge, after=0):
    """Colour key for the a/b/c cell vectors, ending at `right_edge`."""
    from .scene import AXIS_COLORS

    label = "cell a b c"
    start = right_edge - len(label)
    if start < after + 2:
        return
    bar.text(0, start, "cell ", (110, 110, 125))
    for i, name in enumerate("abc"):
        bar.put(0, start + 5 + 2 * i, name, AXIS_COLORS[i])


def header_segments(state):
    """The identity of what is on screen, most important first.

    Leads with whatever the file calls the structure - a CIF data block name,
    an SDF title - rather than the file name, which is often a database code
    and tells you nothing about what you are looking at.
    """
    t = state.theme
    frame = state.frames[state.index]
    structure = state.structure

    segments = [(frame.title, t.text)]
    # a CIF block name usually becomes the structure title as well; showing it
    # twice pushes the genuinely different fields off the end of the line
    if frame.label and frame.label != frame.title:
        segments.append((frame.label, t.accent))
    formula = (
        structure.molecular_formula
        if hasattr(structure, "molecular_formula")
        else structure.asymmetric_unit.formula
    )
    # some formats (PDB) name a structure by its formula; do not say it twice
    if formula != frame.title:
        segments.append((formula, t.dim))
    if state.is_crystal:
        segments.append((structure.space_group.symbol, t.dim))
        uc = structure.unit_cell
        segments.append((f"{uc.a:.2f} {uc.b:.2f} {uc.c:.2f} Å", t.dim))
    # only worth showing when it says something the title does not
    if Path(frame.source).stem != frame.title:
        segments.append((frame.source, t.dim))
    return segments


def title_bar(state, terminal, cols):
    """Where you are, what you are looking at, and how it is being drawn.

    Laid out right to left so the fixed-width indicators keep their place,
    then filled left to right dropping whole segments that will not fit
    rather than truncating one mid-word.
    """
    t = state.theme
    bar = Canvas(1, cols)

    # most important first; the rightmost is dropped first when space runs out
    right = []
    if state.direction and state.view == "structure":
        right.append((f"along {state.direction}", t.accent))
    mode = "images" if terminal.graphics else f"{terminal.colors} blocks"
    right.append((mode, t.dim))
    # never let the indicators crowd out the name of the thing on screen
    budget = cols // 2
    while right and sum(len(text) + 2 for text, _ in right) > budget:
        right.pop()

    limit = cols
    for text, colour in reversed(right):
        limit -= len(text) + 2
        bar.text(0, limit + 1, text, colour)

    x = 1
    if len(state.frames) > 1:
        counter = f"{state.index + 1}/{len(state.frames)}"
        if x + len(counter) < limit:
            bar.text(0, x, counter, t.accent)
            x += len(counter) + 2

    for i, (text, colour) in enumerate(header_segments(state)):
        if not text:
            continue
        if i and x + 3 + len(text) > limit:
            break
        if i:
            bar.text(0, x, "·", t.dim)
            x += 2
        bar.text(0, x, text[: max(0, limit - x)], colour)
        x += len(text) + 1
    return bar


def draw(state, terminal, oversample, size=None):
    """Repaint the screen.

    Every row is placed with an absolute cursor move and nothing emits a
    trailing newline. Letting the rows flow with \\n scrolls the screen the
    moment the content is one line too tall - and a graphics-protocol image
    advances the cursor by its full height - which silently pushes the header
    off the top. Text-only views never showed the problem, which is what made
    it look like the header was missing only in the render view.

    `size` is the (columns, rows) to draw for; the caller passes the size it
    measured so a window resized mid-frame is noticed rather than missed.
    """
    cols, rows = size or tty.size()
    width = max(20, cols - 2)
    top, status_rows = 1, 2
    body_top = top + 1
    body_rows = max(4, rows - top - status_rows)

    t0 = time.monotonic()
    payload, kind = render_body(state, terminal, width, body_rows, oversample)
    elapsed = (time.monotonic() - t0) * 1000

    out = [tty.DELETE_IMAGES]
    out.append(tty.goto(top, 1) + tty.CLEAR_LINE)
    out.append(title_bar(state, terminal, width).render(colors=terminal.colors))

    # wipe the body before drawing, since an image leaves no text to overwrite
    for i in range(body_rows):
        out.append(tty.goto(body_top + i, 1) + tty.CLEAR_LINE)

    if kind == "canvas":
        lines = payload.render(colors=terminal.colors).split("\n")
    elif terminal.graphics:
        show_cols = min(width, max(8, payload.shape[1] // oversample))
        lines = [display(payload, terminal, cols=show_cols).rstrip("\n")]
    else:
        lines = payload.to_canvas().render(colors=terminal.colors).split("\n")

    for i, line in enumerate(lines[:body_rows]):
        out.append(tty.goto(body_top + i, 1) + line)

    status = status_bar(state, width, elapsed).render(colors=terminal.colors)
    for i, line in enumerate(status.split("\n")[:status_rows]):
        out.append(
            tty.goto(rows - status_rows + 1 + i, 1) + tty.CLEAR_LINE + line
        )

    sys.stdout.write("".join(out))
    sys.stdout.flush()


# ------------------------------------------------------------------- keys ----


def handle_prompt(state, key):
    """Keys while a direction is being typed. Never quits."""
    if key == "enter":
        text, state.prompt = state.prompt, None
        if text.strip():
            try:
                state.aim(*parse_direction(text))
            except ValueError as exc:
                state.message = str(exc)
    elif key == "escape":
        state.prompt = None
    elif key == "backspace":
        state.prompt = state.prompt[:-1]
    elif len(key) == 1 and key.isprintable():
        state.prompt += key
    return True


def handle(state, key):
    """Update state for a keypress. Returns False to quit."""
    if state.prompt is not None:
        return handle_prompt(state, key)

    state.message = ""
    views = state.views()

    if key in ("q", "ctrl-c", "escape"):
        return False
    if key == "tab":
        state.view = views[(views.index(state.view) + 1) % len(views)]
    elif key == "shift-tab":
        state.view = views[(views.index(state.view) - 1) % len(views)]
    elif key == "n":
        state.step(1)
    elif key == "p":
        state.step(-1)
    elif key == "N":
        state.step(10)
    elif key == "P":
        state.step(-10)
    elif key == "r":
        state.rotation, state.direction = default_view(state.structure)
        state.zoom = 1.0
        state.frame = None
    elif state.view == "structure":
        if key in ("left", "h"):
            state.rotation = YAW.T @ state.rotation
            state.direction = ""
        elif key in ("right", "l"):
            state.rotation = YAW @ state.rotation
            state.direction = ""
        elif key in ("up", "k"):
            state.rotation = PITCH.T @ state.rotation
            state.direction = ""
        elif key in ("down", "j"):
            state.rotation = PITCH @ state.rotation
            state.direction = ""
        elif key == "d":
            state.prompt = ""
        elif key in ("1", "2", "3"):
            state.aim(*QUICK_AXES[state.is_crystal][int(key) - 1][1:])
        elif key in ("+", "="):
            state.zoom = min(state.zoom * 1.2, 12.0)
        elif key in ("-", "_"):
            state.zoom = max(state.zoom / 1.2, 0.2)
        elif key == "s":
            state.style = (
                "ball-and-stick" if state.style == "space-filling" else "space-filling"
            )
            state.frame = None
        elif key == "f":
            state.shading = "flat" if state.shading == "lit" else "lit"
        elif key == "c" and state.is_crystal:
            state.cells = state.cells % 3 + 1
            state.frame = None
    return True


def run(frames, terminal, args=None):
    """Run the interactive loop over an already-loaded ensemble."""
    state = State(frames=frames, theme=theme_for(tty.query_background()))
    if args is not None:
        state.style = getattr(args, "style", state.style)
        state.shading = getattr(args, "shading", state.shading)
        state.cells = getattr(args, "cells", state.cells)
        if getattr(args, "direction", None):
            state.aim(*parse_direction(args.direction))

    with tty.fullscreen():
        oversample = FULL_OVERSAMPLE
        dirty, drafting = True, False
        size = tty.size()
        while True:
            if dirty:
                size = tty.size()
                draw(state, terminal, oversample, size)
                dirty = False
            key = tty.read_key(timeout=0.15 if drafting else IDLE_TIMEOUT)
            if key is None:
                if drafting:  # input stopped: redraw at full quality
                    drafting, oversample, dirty = False, FULL_OVERSAMPLE, True
                elif tty.size() != size:
                    # polled rather than driven by SIGWINCH: a signal handler
                    # does not interrupt the select() above, so it would not
                    # repaint until the next keypress anyway
                    dirty = True
                continue
            if not handle(state, key):
                break
            tty.drain()  # do not queue frames for a held-down key
            drafting, oversample, dirty = True, DRAFT_OVERSAMPLE, True
    return 0

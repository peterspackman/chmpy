"""Render molecules and crystals in a terminal.

Two ways in. As a library, `render` turns a structure into a string you can
print - no tty, no side effects, so it works in a script, a job log or a
notebook:

    from chmpy import Crystal
    from chmpy.tui import render

    print(render(Crystal.load("structure.cif"), cols=100))

As an application, `chmpy-view` is an interactive viewer over the same
machinery. It exists because checking a structure on a remote machine
otherwise means copying files back to something with a window.

Nothing here needs a graphical display. Where the terminal supports the Kitty
graphics protocol the output is a real image; otherwise it is half-block
characters, two RGB pixels per cell; and failing that, plain text.
"""

from __future__ import annotations

import numpy as np

from .canvas import (
    GRAPHICS_OVERSAMPLE,
    Canvas,
    Pixels,
    Terminal,
    detect,
    display,
    theme_for,
)
from .scene import (
    Scene,
    crystal_scene,
    look_along,
    molecule_scene,
    parse_direction,
    render_scene,
    rotation_matrix,
    view_rotation,
)

__all__ = [
    "Canvas",
    "Pixels",
    "Scene",
    "Terminal",
    "crystal_scene",
    "detect",
    "display",
    "look_along",
    "molecule_scene",
    "parse_direction",
    "render",
    "render_scene",
    "rotation_matrix",
    "theme_for",
    "view_rotation",
]

#: a sensible default when the terminal size is unknown
DEFAULT_COLS = 80


def render(
    structure: object,
    cols: int = DEFAULT_COLS,
    rows: int | None = None,
    direction: str | None = None,
    style: str = "ball-and-stick",
    shading: str = "lit",
    cells: int = 1,
    terminal: Terminal | None = None,
    theme: object | None = None,
) -> str:
    """Render a structure to a string ready to print.

    Args:
        structure: a `Crystal` or a `Molecule`
        cols: width in terminal columns
        rows: height in terminal rows; by default derived from the shape of
            the structure in this view, so a wide flat cell does not come out
            surrounded by blank lines
        direction: view direction, e.g. "a", "[111]", "(001)", "z". Defaults
            to down **c** for a crystal and down **z** for a molecule.
        style: "ball-and-stick" or "space-filling"
        shading: "lit" or "flat"
        cells: for a crystal, how many unit cells along each axis
        terminal: a `Terminal` describing what the output supports; detected
            from the environment when omitted
        theme: a `Theme` for the chrome colours

    Returns:
        str: the rendered frame, including escape sequences unless the
        terminal reports no colour support.
    """
    from chmpy.crystal.crystal import Crystal

    terminal = terminal or detect()
    theme = theme or theme_for(None)

    is_crystal = isinstance(structure, Crystal)
    if is_crystal:
        scene = crystal_scene(structure, cells=(cells,) * 3, style=style)
    else:
        scene = molecule_scene(structure, style=style)

    if direction is None:
        rotation = (
            view_rotation((0, 0, 1), "uvw", structure) if is_crystal else None
        )
    else:
        indices, kind = parse_direction(direction)
        rotation = view_rotation(indices, kind, structure if is_crystal else None)

    if rows is None:
        # a character cell is twice as tall as it is wide, so the row count
        # that makes the view square is half what the aspect ratio suggests
        extent_x, extent_y = scene.view_extent(
            rotation if rotation is not None else np.eye(3)
        )
        rows = max(2, int(round(cols * extent_y / extent_x / 2)))

    factor = GRAPHICS_OVERSAMPLE if terminal.graphics else 1
    pixels = render_scene(
        scene,
        rotation,
        width=cols * factor,
        height=rows * 2 * factor,
        theme=theme,
        shading=shading,
    )
    return display(pixels, terminal, cols=cols)

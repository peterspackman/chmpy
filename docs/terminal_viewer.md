# Terminal viewer

Render molecules and crystals as images in a terminal — no graphical display
needed, which makes it a practical way to check a structure over `ssh` on a
machine that has no window system.

## As a command

```bash
chmpy-view structure.cif              # interactive
chmpy-view trajectory.xyz             # n / p step through frames
chmpy-view relax/                     # a directory is one ensemble
chmpy-view a.cif b.cif geometry.in    # so is a list of files
cat structure.cif | chmpy-view -      # format guessed from the contents

chmpy-view structure.cif --once       # render one frame and exit
chmpy-view structure.cif --once -d '[111]' --cols 100
```

`--once` writes a single frame to stdout and returns, which is what you want
in a job script or when the link is slow enough that interaction is painful.

### Keys

| key | does |
| --- | --- |
| `hjkl` or arrows | rotate in 15° steps — 15 divides 90, so repeated presses land exactly on an axis view |
| `1` `2` `3` | look down **a**, **b**, **c** for a crystal, or **x**, **y**, **z** for a molecule |
| `d` | type a view direction (see below) |
| `n` `p` | next / previous frame; `N` `P` jump ten |
| `+` `-` | zoom |
| `s` | ball-and-stick / space-filling |
| `f` | lit / flat shading |
| `c` | 1–3 unit cells |
| `r` | reset the view |
| `tab` `q` | switch view, quit |

### View directions

`d` accepts ordinary crystallographic notation:

| input | means |
| --- | --- |
| `[uvw]`, or bare digits like `111` | a zone axis, `u`**a** + `v`**b** + `w`**c** |
| `(hkl)` | the normal to a plane, `h`**a**\* + `k`**b**\* + `l`**c**\* |
| `a` `b` `c`, `a*` `b*` `c*` | the cell vectors and their reciprocals |
| `x` `y` `z` | cartesian axes |

A minus binds to the digit after it, so `1-10` is [1,-1,0]. Indices needing
more than one digit are separated by spaces: `10 -2 1`.

For a cell that is not orthogonal `[uvw]` and `(hkl)` are genuinely different
directions, so both are supported rather than conflated.

## As a library

`render` returns a string. It touches no terminal and has no side effects, so
it works in a script, a notebook, or a log file:

``` py
from chmpy import Crystal
from chmpy.tui import render

crystal = Crystal.load("structure.cif")
print(render(crystal, cols=100, direction="[111]"))
```

::: chmpy.tui.render

## Output modes

The output adapts to what the terminal reports:

| mode | when | what it emits |
| --- | --- | --- |
| graphics | terminal speaks the Kitty graphics protocol | a real image, at full resolution |
| half blocks | 24-bit or 256 colour | `▀` with a foreground and background colour, so one cell carries two square pixels |
| text | `NO_COLOR`, a pipe, or `TERM=dumb` | shaded characters, since half blocks say nothing without colour |

Detection is environment-based. Two cases are worth knowing about:

- **Over `ssh`**, `TERM` is forwarded but `TERM_PROGRAM` and the vendor
  variables are not, so detection relies on `TERM`.
- **Inside `tmux` or `screen`** graphics are disabled, because a multiplexer
  rewrites escape sequences and passes the protocol through only when
  explicitly configured. Set `CHMPY_FORCE_GRAPHICS=1` if yours is.

## Reading structures

The viewer loads through [`chmpy.fmt.load`][chmpy.fmt.load], which dispatches
on its own rather than making you choose `Crystal` or `Molecule` in advance.
An FHI-aims geometry is periodic exactly when it declares `lattice_vector`,
decided from the contents rather than the file name, since the same format
arrives as `geometry.in`, `geometry.in.next_step`, and whatever a run
directory happened to call it.

A file may hold several structures — an XYZ trajectory, a multi-block CIF —
and several files may be viewed as one ensemble, so a single structure is
simply the one-frame case.

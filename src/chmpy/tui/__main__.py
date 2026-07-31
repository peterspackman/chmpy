"""Command line entry point for the terminal viewer."""

from __future__ import annotations

import argparse
import sys

from chmpy.fmt.load import load_frames, load_stdin, supported


def build_parser():
    parser = argparse.ArgumentParser(
        prog="chmpy-view",
        description="View molecules and crystals in a terminal.",
        epilog=f"supported formats:\n{supported()}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "structures",
        nargs="*",
        help="structure files, directories, or - to read from stdin",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="render one frame to stdout and exit, instead of running interactively",
    )
    parser.add_argument(
        "-d",
        "--direction",
        help="view direction: a, b, c, x, y, z, [uvw] or (hkl)",
    )
    parser.add_argument("--cols", type=int, help="width in columns (default: terminal)")
    parser.add_argument("--rows", type=int, help="height in rows, for --once")
    parser.add_argument(
        "--style",
        choices=("ball-and-stick", "space-filling"),
        default="ball-and-stick",
    )
    parser.add_argument("--shading", choices=("lit", "flat"), default="lit")
    parser.add_argument(
        "--cells", type=int, default=1, help="unit cells along each axis, for a crystal"
    )
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="disable inline images and draw with half blocks instead",
    )
    parser.add_argument("--256", dest="ansi256", action="store_true",
                        help="force 256-colour output")
    parser.add_argument("--no-color", action="store_true", help="disable colour")
    return parser


def terminal_for(args):
    from .canvas import Terminal, detect

    if args.no_color:
        return Terminal("none", graphics=False)
    if args.ansi256:
        return Terminal("256", graphics=False)
    return detect(allow_graphics=not args.blocks)


def render_once(args, frames, terminal):
    """One frame per structure, straight to stdout."""
    import shutil

    from . import render

    cols = args.cols or min(shutil.get_terminal_size((80, 24)).columns - 2, 100)
    for frame in frames:
        if len(frames) > 1:
            # the label is what tells one frame of a trajectory from the next;
            # without it every heading in the file is the same line of text
            name = f"{frame.title}  {frame.label}" if frame.label else frame.title
            print(f"{name}  ({frame.source})")
        print(
            render(
                frame.structure,
                cols=cols,
                rows=args.rows,
                direction=args.direction,
                style=args.style,
                shading=args.shading,
                cells=args.cells,
                terminal=terminal,
            )
        )
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.structures:
        parser.print_help()
        return 1

    # check the direction up front: both the interactive and the --once path
    # would otherwise fail with a traceback, one of them mid-redraw
    if args.direction:
        from .scene import parse_direction

        try:
            parse_direction(args.direction)
        except ValueError as exc:
            print(f"{exc}\ntry a, b, c, x, y, z, [111] or (001)", file=sys.stderr)
            return 1

    try:
        if args.structures == ["-"]:
            frames = load_stdin(sys.stdin.read())
        else:
            frames = load_frames(args.structures)
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    terminal = terminal_for(args)
    if args.once:
        return render_once(args, frames, terminal)

    # interactive mode needs a keyboard, which piped stdin cannot provide
    if args.structures == ["-"]:
        try:
            sys.stdin = open("/dev/tty")
        except OSError as exc:
            print(
                f"reading from stdin needs a terminal on /dev/tty for keys ({exc}); "
                "use --once to render without interacting",
                file=sys.stderr,
            )
            return 1
    if not sys.stdin.isatty():
        print(
            "chmpy-view needs a terminal for keyboard input; "
            "use --once to render a frame instead",
            file=sys.stderr,
        )
        return 1

    from .app import run

    return run(frames, terminal, args)


if __name__ == "__main__":
    sys.exit(main())

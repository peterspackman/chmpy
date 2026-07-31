"""Minimal raw-mode terminal driver.

Deliberately not curses: curses owns colour through its own limited pair table
and would fight both 24-bit colour and the Kitty graphics protocol, which are
the two things this whole renderer depends on. Raw mode plus direct escape
sequences is less machinery and gives full control.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from contextlib import contextmanager

ALT_SCREEN_ON = "\x1b[?1049h"
ALT_SCREEN_OFF = "\x1b[?1049l"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR = "\x1b[2J"
HOME = "\x1b[H"
CLEAR_LINE = "\x1b[K"
#: remove every image placement (Kitty graphics), so frames do not stack up
DELETE_IMAGES = "\x1b_Ga=d,d=A\x1b\\"

KEYS = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[Z": "shift-tab",
    "\x1b[5~": "pgup",
    "\x1b[6~": "pgdn",
    "\r": "enter",
    "\n": "enter",
    "\t": "tab",
    "\x7f": "backspace",
    "\x1b": "escape",
    "\x03": "ctrl-c",
}


@contextmanager
def fullscreen(stream=None):
    """Raw mode on the alternate screen, restored however we leave."""
    stream = stream or sys.stdout
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    stream.write(ALT_SCREEN_ON + HIDE_CURSOR + CLEAR)
    stream.flush()
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        stream.write(DELETE_IMAGES + SHOW_CURSOR + ALT_SCREEN_OFF)
        stream.flush()


def read_key(timeout=None):
    """One keypress, or None if `timeout` seconds pass with no input."""
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], timeout)[0]:
        return None
    ch = os.read(fd, 1).decode("utf-8", "replace")
    if ch != "\x1b":
        return KEYS.get(ch, ch)
    # an escape may start a sequence; drain what arrived with it
    buf = ch
    while select.select([fd], [], [], 0.02)[0]:
        buf += os.read(fd, 1).decode("utf-8", "replace")
        if buf in KEYS:
            return KEYS[buf]
        if len(buf) > 8:
            break
    return KEYS.get(buf, "escape" if buf == "\x1b" else buf)


def drain():
    """Discard pending input, so held-down keys do not queue up frames."""
    fd = sys.stdin.fileno()
    n = 0
    while select.select([fd], [], [], 0)[0]:
        os.read(fd, 1024)
        n += 1
    return n


def size(default=(96, 30)):
    try:
        c = os.get_terminal_size()
        return c.columns, c.lines
    except OSError:
        return default


def goto(row, col=1):
    return f"\x1b[{row};{col}H"


def query_background(timeout=0.15):
    """Ask the terminal for its background colour via OSC 11.

    Returns an (r, g, b) triple, or None if the terminal does not answer.
    Must not be called while another raw-mode reader is running, so query
    once before entering fullscreen.
    """
    import re
    import time

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b]11;?\x1b\\")
        sys.stdout.flush()
        buf, deadline = "", time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if not select.select([fd], [], [], max(remaining, 0))[0]:
                break
            buf += os.read(fd, 64).decode("utf-8", "replace")
            if buf.endswith("\x07") or buf.endswith("\x1b\\"):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    m = re.search(r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
    if not m:
        return None

    def channel(text):
        # components come back as 1-4 hex digits scaled to that width
        value = int(text, 16)
        return round(value * 255 / (16 ** len(text) - 1))

    return tuple(channel(m.group(i)) for i in (1, 2, 3))

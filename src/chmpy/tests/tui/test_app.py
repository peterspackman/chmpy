"""The interactive viewer's state machine and its drawing path.

Neither needs a terminal: `handle` is a pure state transition, and `draw`
takes the screen size as an argument, so the parts that used to be reachable
only by running the app under a tty are testable here.
"""

import contextlib
import io
import re
import unittest

import numpy as np

from chmpy import Crystal, Molecule
from chmpy.fmt.load import Frame
from chmpy.tui import Terminal
from chmpy.tui.app import State, draw, handle

from .. import TEST_FILES

BLOCKS = Terminal("truecolor", graphics=False)
GRAPHICS = Terminal("truecolor", graphics=True)


def strip_ansi(text):
    return re.sub(r"\x1b(\[[0-9;]*[A-Za-z]|_G[^\x1b]*\x1b\\|\]|.)", "", text)


class StateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        cls.molecule = Molecule.load(TEST_FILES["water.xyz"])

    def state(self, *structures):
        structures = structures or (self.crystal,)
        return State(frames=[Frame(s, "test.cif") for s in structures])

    def test_quits_on_q(self):
        for key in ("q", "escape", "ctrl-c"):
            with self.subTest(key=key):
                self.assertFalse(handle(self.state(), key))

    def test_rotation_keys_turn_and_clear_the_named_direction(self):
        for key in ("h", "j", "k", "l", "left", "right", "up", "down"):
            with self.subTest(key=key):
                state = self.state()
                before = state.rotation.copy()
                self.assertTrue(handle(state, key))
                self.assertFalse(np.allclose(state.rotation, before), "did not turn")
                self.assertEqual(state.direction, "", "still claims an axis view")

    def test_six_rotations_return_to_the_start(self):
        """15 degrees divides 90, which is what makes axis views reachable."""
        state = self.state()
        start = state.rotation.copy()
        for _ in range(24):
            handle(state, "l")
        np.testing.assert_allclose(state.rotation, start, atol=1e-9)

    def test_number_keys_aim_down_the_cell_axes(self):
        for key, expected in (("1", "[100]"), ("2", "[010]"), ("3", "[001]")):
            with self.subTest(key=key):
                state = self.state()
                handle(state, key)
                self.assertEqual(state.direction, expected)

    def test_number_keys_aim_down_cartesian_axes_for_a_molecule(self):
        state = self.state(self.molecule)
        handle(state, "2")
        self.assertEqual(state.direction, "y")

    def test_stepping_stops_at_the_ends(self):
        state = self.state(self.crystal, self.molecule)
        handle(state, "p")
        self.assertEqual(state.index, 0, "stepped before the first frame")
        handle(state, "n")
        handle(state, "n")
        self.assertEqual(state.index, 1, "stepped past the last frame")

    def test_the_view_survives_a_step_between_like_structures(self):
        """Comparing frames is the point; a view that reset would prevent it."""
        state = self.state(self.crystal, self.crystal)
        handle(state, "l")
        turned = state.rotation.copy()
        handle(state, "n")
        np.testing.assert_allclose(state.rotation, turned)

    def test_the_view_resets_when_the_kind_of_structure_changes(self):
        state = self.state(self.crystal, self.molecule)
        handle(state, "l")
        handle(state, "n")
        self.assertEqual(state.direction, "z", "kept a lattice view for a molecule")

    def test_reset_restores_the_default_view(self):
        state = self.state()
        handle(state, "l")
        handle(state, "+")
        handle(state, "r")
        self.assertEqual(state.direction, "[001]")
        self.assertEqual(state.zoom, 1.0)

    def test_toggles_cycle(self):
        state = self.state()
        handle(state, "s")
        self.assertEqual(state.style, "space-filling")
        handle(state, "f")
        self.assertEqual(state.shading, "flat")
        for expected in (2, 3, 1):
            handle(state, "c")
            self.assertEqual(state.cells, expected)

    def test_only_a_crystal_has_a_powder_view(self):
        self.assertEqual(self.state(self.molecule).views(), ["structure"])
        self.assertIn("powder", self.state().views())

    def test_tab_cycles_views(self):
        state = self.state()
        handle(state, "tab")
        self.assertEqual(state.view, "powder")
        handle(state, "tab")
        self.assertEqual(state.view, "structure")


class PromptTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def state(self):
        return State(frames=[Frame(self.crystal, "test.cif")])

    def typed(self, text):
        state = self.state()
        handle(state, "d")
        for ch in text:
            handle(state, ch)
        return state

    def test_typing_a_direction_aims_the_view(self):
        state = self.typed("(110)")
        handle(state, "enter")
        self.assertIsNone(state.prompt)
        self.assertEqual(state.direction, "(110)")

    def test_a_bad_direction_reports_rather_than_raises(self):
        state = self.typed("nonsense")
        handle(state, "enter")
        self.assertTrue(state.message, "said nothing about the bad input")
        self.assertEqual(state.direction, "[001]", "view moved anyway")

    def test_escape_cancels_without_quitting(self):
        state = self.typed("11")
        self.assertTrue(handle(state, "escape"))
        self.assertIsNone(state.prompt)

    def test_letters_that_are_commands_elsewhere_are_typed(self):
        """`q` while typing must not quit, or 'a*' style input is unreachable."""
        state = self.state()
        handle(state, "d")
        self.assertTrue(handle(state, "q"))
        self.assertEqual(state.prompt, "q")

    def test_backspace_deletes(self):
        state = self.typed("111")
        handle(state, "backspace")
        self.assertEqual(state.prompt, "11")


class DrawTestCase(unittest.TestCase):
    """What draw writes to the terminal, without one attached."""

    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def output(self, terminal=BLOCKS, size=(80, 24), **kwargs):
        state = State(frames=[Frame(self.crystal, "acetic_acid.cif")], **kwargs)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            draw(state, terminal, oversample=2, size=size)
        return buf.getvalue()

    def test_never_emits_a_newline(self):
        """A newline scrolls the screen, which pushes the header off the top.

        Every row is placed with an absolute cursor move instead, so the frame
        cannot scroll however tall its contents turn out to be.
        """
        for terminal in (BLOCKS, GRAPHICS):
            with self.subTest(terminal=terminal.best):
                self.assertNotIn("\n", self.output(terminal))

    def rows_written(self, size):
        out = self.output(size=size)
        return [int(r) for r, _ in re.findall(r"\x1b\[(\d+);(\d+)H", out)]

    def test_stays_within_the_screen(self):
        placed = self.rows_written((80, 24))
        self.assertTrue(placed, "placed nothing")
        self.assertLessEqual(max(placed), 24)
        self.assertGreaterEqual(min(placed), 1)

    def test_honours_the_size_it_is_given(self):
        """Resize is noticed by redrawing at a new size, so it must take one."""
        self.assertLessEqual(max(self.rows_written((60, 12))), 12)
        self.assertGreater(max(self.rows_written((60, 30))), 12)

    def test_names_the_structure_at_the_top(self):
        out = self.output()
        header = strip_ansi(out.split("\x1b[2;1H")[0])
        self.assertIn("acetic_acid", header)

    def test_the_powder_view_draws(self):
        out = strip_ansi(self.output(view="powder"))
        self.assertIn("2θ", out)


if __name__ == "__main__":
    unittest.main()

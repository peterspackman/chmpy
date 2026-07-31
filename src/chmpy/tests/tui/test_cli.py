"""The chmpy-view command line, exercised through main() without a terminal."""

import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from chmpy.tui.__main__ import main

from .. import TEST_FILES

TRAJECTORY = """3
step 0
O 0.00 0.00 0.00
H 0.96 0.00 0.00
H 0.00 0.96 0.00
3
step 1
O 0.00 0.00 0.00
H 0.97 0.00 0.00
H 0.00 0.97 0.00
"""


def run(argv):
    """(exit code, stdout, stderr) for one invocation."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class CommandLineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cif = str(TEST_FILES["acetic_acid.cif"])

    def test_no_arguments_prints_help(self):
        code, out, _ = run([])
        self.assertEqual(code, 1)
        self.assertIn("chmpy-view", out)

    def test_renders_one_frame_to_stdout(self):
        code, out, _ = run([self.cif, "--once", "--no-color", "--cols", "40"])
        self.assertEqual(code, 0)
        self.assertTrue(out.strip(), "rendered nothing")
        self.assertNotIn("\x1b", out, "emitted colour with --no-color")

    def test_help_lists_the_supported_formats(self):
        with self.assertRaises(SystemExit):
            run(["--help"])

    def test_a_bad_direction_is_a_message_not_a_traceback(self):
        code, _, err = run([self.cif, "--once", "-d", "garbage"])
        self.assertEqual(code, 1)
        self.assertIn("direction", err)

    def test_an_unreadable_file_reports_and_exits(self):
        code, _, err = run(["/nonexistent/structure.cif", "--once"])
        self.assertEqual(code, 1)
        self.assertTrue(err.strip())

    def test_every_frame_of_a_trajectory_gets_its_own_heading(self):
        """Without the frame label each heading is the same line of text."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.xyz"
            path.write_text(TRAJECTORY)
            _, out, _ = run([str(path), "--once", "--no-color", "--cols", "30"])
        self.assertIn("step 0", out)
        self.assertIn("step 1", out)

    def test_forcing_colour_overrides_the_redirected_output(self):
        """--256 and --no-color are explicit, so detection must not override."""
        _, colour, _ = run([self.cif, "--once", "--256", "--cols", "30"])
        _, plain, _ = run([self.cif, "--once", "--no-color", "--cols", "30"])
        self.assertIn("\x1b[38;5;", colour)
        self.assertNotIn("\x1b", plain)


if __name__ == "__main__":
    unittest.main()

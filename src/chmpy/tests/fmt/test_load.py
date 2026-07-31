"""Dispatching any structure file to the right reader."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from chmpy import Crystal, Molecule
from chmpy.fmt.load import (
    Frame,
    UnknownFormat,
    load_file,
    load_frames,
    load_stdin,
    sniff,
    supported,
)

from .. import TEST_FILES

TRAJECTORY = "\n".join(
    line
    for step in range(5)
    for line in (
        "3",
        f"step {step} energy -76.4{step}",
        "O 0.0 0.0 0.0",
        f"H {0.9 + 0.05 * step} 0.0 0.0",
        f"H 0.0 {0.9 + 0.05 * step} 0.0",
    )
)

AIMS_PERIODIC = """
lattice_vector 5.0 0.0 0.0
lattice_vector 0.0 5.0 0.0
lattice_vector 0.0 0.0 5.0
atom 0.0 0.0 0.0 O
atom 0.96 0.0 0.0 H
atom 0.0 0.96 0.0 H
"""

AIMS_MOLECULAR = """
atom 0.0 0.0 0.0 O
atom 0.96 0.0 0.0 H
atom 0.0 0.96 0.0 H
"""


class LoadFileTestCase(unittest.TestCase):
    def test_periodic_formats_give_crystals(self):
        for name in ("acetic_acid.cif", "acetic_acid.res", "example.gen"):
            with self.subTest(name=name):
                frames = load_file(TEST_FILES[name])
                self.assertEqual(len(frames), 1)
                self.assertIsInstance(frames[0].structure, Crystal)

    def test_molecular_formats_give_molecules(self):
        for name in ("water.xyz", "DB09563.sdf"):
            with self.subTest(name=name):
                frames = load_file(TEST_FILES[name])
                self.assertIsInstance(frames[0].structure, Molecule)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_file("/nonexistent/structure.cif")

    def test_unknown_format_lists_what_is_supported(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mystery.qqq"
            path.write_text("not a structure")
            with self.assertRaises(UnknownFormat) as caught:
                load_file(path)
        message = str(caught.exception)
        self.assertIn("supported formats", message)
        self.assertIn(".cif", message)


class TrajectoryTestCase(unittest.TestCase):
    def frames(self, text, name="traj.xyz"):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(text)
            return load_file(path)

    def test_every_frame_is_read(self):
        frames = self.frames(TRAJECTORY)
        self.assertEqual(len(frames), 5)
        self.assertTrue(all(len(f.structure) == 3 for f in frames))

    def test_frame_label_comes_from_the_comment(self):
        frames = self.frames(TRAJECTORY)
        self.assertEqual(frames[0].label, "step 0 energy -76.40")
        self.assertEqual(frames[4].label, "step 4 energy -76.44")

    def test_geometry_differs_between_frames(self):
        frames = self.frames(TRAJECTORY)
        first = frames[0].structure.distance_matrix
        last = frames[-1].structure.distance_matrix
        self.assertFalse((first == last).all())

    def test_single_frame_xyz_has_no_label(self):
        frames = self.frames("3\nwater\nO 0 0 0\nH 0.96 0 0\nH 0 0.96 0\n")
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].label, "")

    def test_trailing_blank_line_is_tolerated(self):
        """Most XYZ writers leave one, and it used to break the reader."""
        frames = self.frames(TRAJECTORY + "\n\n")
        self.assertEqual(len(frames), 5)


class AimsTestCase(unittest.TestCase):
    """Periodicity is decided by the contents, not the file name."""

    def load(self, text, name):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(text)
            return load_file(path)[0]

    def test_lattice_vectors_give_a_crystal(self):
        self.assertIsInstance(self.load(AIMS_PERIODIC, "geometry.in").structure, Crystal)

    def test_no_lattice_vectors_gives_a_molecule(self):
        frame = self.load(AIMS_MOLECULAR, "geometry.in")
        self.assertIsInstance(frame.structure, Molecule)

    def test_name_does_not_decide(self):
        for name in ("geometry.in", "geometry.in.next_step", "relaxed.in"):
            with self.subTest(name=name):
                frame = self.load(AIMS_PERIODIC, name)
                self.assertIsInstance(frame.structure, Crystal)


class EnsembleTestCase(unittest.TestCase):
    def test_several_files_become_one_ensemble(self):
        frames = load_frames(
            [TEST_FILES["acetic_acid.cif"], TEST_FILES["water.xyz"]]
        )
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].is_crystal)
        self.assertFalse(frames[1].is_crystal)

    def test_a_directory_is_an_ensemble(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.xyz").write_text("1\nfirst\nH 0 0 0\n")
            (root / "b.xyz").write_text("1\nsecond\nH 0 0 0\n")
            (root / "notes.txt").write_text("skip me")
            frames = load_frames([root])
        self.assertEqual(len(frames), 2, "unreadable files should be skipped")

    def test_a_directory_of_nothing_readable_raises(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text("skip me")
            with self.assertRaises(UnknownFormat):
                load_frames([Path(tmp)])

    def test_a_single_path_need_not_be_wrapped(self):
        self.assertEqual(len(load_frames(TEST_FILES["water.xyz"])), 1)


class StdinTestCase(unittest.TestCase):
    def test_format_is_guessed_from_content(self):
        cases = (
            (TEST_FILES["acetic_acid.cif"].read_text(), Crystal),
            (TEST_FILES["water.xyz"].read_text(), Molecule),
            (AIMS_PERIODIC, Crystal),
            (AIMS_MOLECULAR, Molecule),
        )
        for text, expected in cases:
            with self.subTest(expected=expected.__name__):
                frame = load_stdin(text)[0]
                self.assertIsInstance(frame.structure, expected)
                self.assertEqual(frame.source, "<stdin>")

    def test_trajectory_from_stdin(self):
        self.assertEqual(len(load_stdin(TRAJECTORY)), 5)

    def test_unrecognisable_input(self):
        with self.assertRaises(UnknownFormat):
            sniff("this is not a structure at all")


class FrameTitleTestCase(unittest.TestCase):
    """The title names the structure, not the file it arrived in."""

    def setUp(self):
        self.molecule = Molecule.load(TEST_FILES["water.xyz"])

    def test_structure_name_wins(self):
        frame = load_file(TEST_FILES["DB09563.sdf"])[0]
        self.assertEqual(frame.title, "Choline C-11")

    def test_block_names_an_unnamed_structure(self):
        self.assertEqual(Frame(self.molecule, "multi.cif", block="AABHTZ").title, "AABHTZ")

    def test_a_state_description_is_not_a_title(self):
        """A trajectory comment describes a state, so it must not become one."""
        frame = Frame(self.molecule, "traj.xyz", label="step 4")
        self.assertEqual(frame.title, "traj")

    def test_falls_back_to_the_file_stem(self):
        self.assertEqual(Frame(self.molecule, "run_07.xyz").title, "run_07")


class SupportedTestCase(unittest.TestCase):
    def test_lists_each_group(self):
        text = supported()
        for expected in ("periodic", "molecular", "either", "multi-frame", ".cif"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()

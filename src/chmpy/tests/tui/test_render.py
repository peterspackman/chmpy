"""The rendering API, which must work with no terminal attached."""

import re
import unittest

import numpy as np

from chmpy import Crystal, Molecule
from chmpy.tui import Terminal, render
from chmpy.tui.canvas import Pixels, ascii_art, display, theme_for, to_256
from chmpy.tui.scene import (
    crystal_scene,
    format_direction,
    lattice_direction,
    molecule_scene,
    parse_direction,
    render_scene,
    rotation_matrix,
    view_rotation,
)

from .. import TEST_FILES

BLOCKS = Terminal("truecolor", graphics=False)
PLAIN = Terminal("none", graphics=False)
GRAPHICS = Terminal("truecolor", graphics=True)


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class RenderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
        cls.molecule = Molecule.load(TEST_FILES["water.xyz"])

    def test_renders_a_crystal_and_a_molecule(self):
        for structure in (self.crystal, self.molecule):
            with self.subTest(structure=type(structure).__name__):
                out = render(structure, cols=40, terminal=BLOCKS)
                self.assertTrue(out.strip(), "produced nothing")

    def test_respects_the_requested_width(self):
        for cols in (20, 40, 80):
            with self.subTest(cols=cols):
                out = render(self.crystal, cols=cols, rows=10, terminal=BLOCKS)
                widest = max(len(strip_ansi(ln)) for ln in out.split("\n"))
                self.assertLessEqual(widest, cols)

    def test_respects_the_requested_height(self):
        out = render(self.crystal, cols=40, rows=12, terminal=BLOCKS)
        self.assertEqual(len(out.split("\n")), 12)

    def test_height_follows_the_structure_when_unset(self):
        """A wide flat cell should not come back surrounded by blank rows."""
        out = render(self.crystal, cols=60, terminal=PLAIN)
        lines = out.split("\n")
        leading = next(
            (i for i, line in enumerate(lines) if line.strip()), len(lines)
        )
        self.assertLess(leading, 3, f"{leading} blank rows before any content")

    def test_no_colour_gives_readable_text(self):
        out = render(self.crystal, cols=50, terminal=PLAIN)
        self.assertNotIn("\x1b", out)
        # half blocks carry no information without colour, so the fallback
        # must be shaded characters instead
        self.assertNotIn("▀", out)
        self.assertTrue(set(out) - set(" \n"), "no visible characters")

    def test_graphics_mode_emits_one_image(self):
        out = render(self.crystal, cols=40, terminal=GRAPHICS)
        self.assertIn("\x1b_G", out)
        self.assertIn("a=T", out)

    def test_styles_and_shading_all_render(self):
        for style in ("ball-and-stick", "space-filling"):
            for shading in ("lit", "flat"):
                with self.subTest(style=style, shading=shading):
                    out = render(
                        self.crystal,
                        cols=30,
                        rows=8,
                        style=style,
                        shading=shading,
                        terminal=BLOCKS,
                    )
                    self.assertTrue(out.strip())

    def test_directions_are_accepted(self):
        for direction in ("a", "b", "c", "[111]", "(001)", "1-10"):
            with self.subTest(direction=direction):
                out = render(
                    self.crystal, cols=30, rows=8, direction=direction, terminal=BLOCKS
                )
                self.assertTrue(out.strip())

    def test_cartesian_directions_work_for_molecules(self):
        for direction in ("x", "y", "z"):
            with self.subTest(direction=direction):
                out = render(
                    self.molecule, cols=30, rows=8, direction=direction, terminal=BLOCKS
                )
                self.assertTrue(out.strip())

    def test_bad_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            render(self.crystal, cols=30, direction="not a direction")


class DirectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])

    def test_parses_crystallographic_notation(self):
        cases = {
            "100": ((1, 0, 0), "uvw"),
            "[1-10]": ((1, -1, 0), "uvw"),
            "1-10": ((1, -1, 0), "uvw"),
            "(001)": ((0, 0, 1), "hkl"),
            "10 -2 1": ((10, -2, 1), "uvw"),
            "a": ((1, 0, 0), "uvw"),
            "c*": ((0, 0, 1), "hkl"),
            "z": ((0, 0, 1), "cartesian"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_direction(text), expected)

    def test_rejects_nonsense(self):
        for text in ("", "11", "1 2 3 4", "q", "000", "abc"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_direction(text)

    def test_format_round_trips(self):
        for text in ("100", "1-10", "(001)", "10 -2 1"):
            with self.subTest(text=text):
                indices, kind = parse_direction(text)
                again = parse_direction(format_direction(indices, kind))
                self.assertEqual(again, (indices, kind))

    def test_aiming_puts_the_direction_toward_the_viewer(self):
        for text in ("a", "b", "c", "111", "(001)", "1-10"):
            with self.subTest(text=text):
                indices, kind = parse_direction(text)
                rotation = view_rotation(indices, kind, self.crystal)
                world = lattice_direction(self.crystal, indices, kind)
                screen = rotation @ (world / np.linalg.norm(world))
                np.testing.assert_allclose(screen, [0, 0, 1], atol=1e-9)

    def test_rotations_are_orthonormal(self):
        rotation = view_rotation((1, 1, 1), "uvw", self.crystal)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)

    def test_zone_axis_and_plane_normal_differ_when_oblique(self):
        """[uvw] and (hkl) are only the same direction in an orthogonal cell."""
        oblique = Crystal.load(TEST_FILES["HXACAN01.pdb"])
        zone = lattice_direction(oblique, (1, 0, 0), "uvw")
        normal = lattice_direction(oblique, (1, 0, 0), "hkl")
        cosine = zone @ normal / (np.linalg.norm(zone) * np.linalg.norm(normal))
        self.assertLess(cosine, 0.999, "monoclinic cell should distinguish them")


class FramingTestCase(unittest.TestCase):
    """The view must fit whatever the aspect ratio, and never crop."""

    @classmethod
    def setUpClass(cls):
        cls.scene = crystal_scene(Crystal.load(TEST_FILES["acetic_acid.cif"]))

    def rendered(self, width, height, yaw=0.4, pitch=0.25):
        return render_scene(
            self.scene, rotation_matrix(yaw, pitch), width=width, height=height
        )

    def test_never_touches_the_frame_edge(self):
        for width, height in ((200, 200), (400, 120), (120, 400), (188, 60)):
            for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                with self.subTest(size=(width, height), yaw=round(yaw, 2)):
                    alpha = self.rendered(width, height, yaw=yaw).alpha
                    self.assertFalse(alpha[0].any() or alpha[-1].any(), "cropped rows")
                    self.assertFalse(
                        alpha[:, 0].any() or alpha[:, -1].any(), "cropped columns"
                    )

    def test_uses_a_reasonable_share_of_the_frame(self):
        alpha = self.rendered(188, 60).alpha
        rows = np.flatnonzero(alpha.any(axis=1))
        cols = np.flatnonzero(alpha.any(axis=0))
        used = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
        self.assertGreater(used / alpha.size, 0.25, "framing is too loose")

    def test_an_explicit_extent_is_honoured(self):
        wide = render_scene(self.scene, width=100, height=100, extent=(50.0, 50.0))
        tight = render_scene(self.scene, width=100, height=100, extent=(8.0, 8.0))
        self.assertLess(wide.alpha.sum(), tight.alpha.sum())


class DegenerateGeometryTestCase(unittest.TestCase):
    """Molecules with no well-defined principal axes must still render."""

    def scene_for(self, numbers, positions):
        return molecule_scene(
            Molecule.from_arrays(np.array(numbers), np.array(positions, dtype=float))
        )

    def test_single_atom(self):
        scene = self.scene_for([8], [[0, 0, 0]])
        self.assertTrue(render_scene(scene, width=32, height=32).alpha.any())

    def test_diatomic(self):
        scene = self.scene_for([1, 1], [[0, 0, 0], [0.74, 0, 0]])
        self.assertTrue(render_scene(scene, width=32, height=32).alpha.any())

    def test_linear_triatomic(self):
        scene = self.scene_for(
            [6, 8, 8], [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]
        )
        self.assertTrue(render_scene(scene, width=32, height=32).alpha.any())


class OutputFidelityTestCase(unittest.TestCase):
    """What we emit has to be what a terminal would show."""

    def sparse_pixels(self):
        px = Pixels(5, 7)
        px.rgb[1, 2] = (200, 30, 40)
        px.alpha[1, 2] = True
        px.rgb[2, 2] = (10, 220, 90)
        px.alpha[2, 2] = True
        return px

    def test_half_blocks_never_set_a_background_they_do_not_own(self):
        """Transparency is what lets the terminal background show through."""
        out = display(self.sparse_pixels(), BLOCKS)
        self.assertNotIn("48;2", out)

    def test_graphics_output_is_rgba_with_matching_coverage(self):
        import base64
        import io

        px = self.sparse_pixels()
        chunks = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\", display(px, GRAPHICS))
        payload = "".join(c[1] for c in chunks)
        from PIL import Image

        got = np.array(Image.open(io.BytesIO(base64.b64decode(payload))))
        self.assertEqual(got.shape[2], 4, "should be RGBA")
        np.testing.assert_array_equal(got[..., 3] > 0, px.alpha)

    def test_256_colour_indices_are_in_range(self):
        out = display(self.sparse_pixels(), Terminal("256", graphics=False))
        for _, value in re.findall(r"\x1b\[(38|48);5;(\d+)m", out):
            self.assertLessEqual(int(value), 255)

    def test_256_quantisation_is_close(self):
        for colour in ((255, 255, 255), (0, 0, 0), (255, 13, 13), (144, 144, 144)):
            with self.subTest(colour=colour):
                self.assertLessEqual(to_256(colour), 255)

    def test_ascii_art_marks_covered_pixels(self):
        px = self.sparse_pixels()
        art = ascii_art(px)
        self.assertTrue(set(art) - set(" \n"), "covered pixels vanished")


class ThemeTestCase(unittest.TestCase):
    def test_dark_and_light_backgrounds_invert_the_text(self):
        dark = theme_for((13, 17, 23))
        light = theme_for((253, 246, 227))
        self.assertTrue(dark.is_dark)
        self.assertFalse(light.is_dark)
        self.assertGreater(sum(dark.text), sum(light.text))

    def test_unknown_background_assumes_dark(self):
        self.assertTrue(theme_for(None).is_dark)


if __name__ == "__main__":
    unittest.main()

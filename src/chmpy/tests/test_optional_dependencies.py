"""matplotlib and trimesh are extras, and must stay that way.

The import-time checks run in a subprocess: by the time this test module is
collected the test session has almost certainly imported both already, so
asking `sys.modules` in-process would pass no matter what.
"""

import json
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from chmpy.util import optional


def run(source):
    """Run a snippet in a clean interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"subprocess failed:\n{result.stderr}")
    # chmpy.tests prints on import, so only the final line is ours
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


class ImportWeightTestCase(unittest.TestCase):
    """Importing chmpy must not drag in the heavy optional extras."""

    def test_import_chmpy_is_light(self):
        loaded = run("""
            import sys
            import chmpy  # noqa: F401
            heavy = [m for m in ("matplotlib", "trimesh", "seaborn", "pandas")
                     if m in sys.modules]
            print("HEAVY:" + ",".join(heavy))
        """)
        self.assertEqual(loaded, "HEAVY:", f"import chmpy pulled in {loaded}")

    def test_core_works_without_the_extras(self):
        out = run("""
            import builtins
            real = builtins.__import__

            def guard(name, *args, **kwargs):
                if name.split(".")[0] in {"matplotlib", "trimesh"}:
                    raise ImportError(f"no module named {name!r}")
                return real(name, *args, **kwargs)

            builtins.__import__ = guard
            from chmpy import Crystal
            from chmpy.tests import TEST_FILES
            c = Crystal.load(TEST_FILES["acetic_acid.cif"])
            print(len(c.unit_cell_molecules()), len(c.powder_pattern()))
        """)
        molecules, reflections = (int(x) for x in out.split())
        self.assertEqual(molecules, 4)
        self.assertGreater(reflections, 0)


class RequireTestCase(unittest.TestCase):
    """`require` returns the module, or explains which extra provides it.

    The missing case is simulated by patching the import inside the module
    under test. Patching `builtins.__import__` does not work here, because
    `importlib.import_module` goes through `_gcd_import` and never consults
    it - which made an earlier version of this test pass by accident.
    """

    @staticmethod
    def missing(module, purpose=None):
        with mock.patch.object(
            optional.importlib, "import_module", side_effect=ImportError("absent")
        ), unittest.TestCase().assertRaises(ImportError) as caught:
            optional.require(module, purpose)
        return str(caught.exception)

    def test_returns_the_module_when_present(self):
        self.assertIs(optional.require("json"), json)

    def test_names_the_extra_to_install(self):
        message = self.missing("matplotlib.pyplot")
        self.assertIn("pip install", message)
        self.assertIn("chmpy[plots]", message)

    def test_maps_known_modules_to_their_extra(self):
        for module, extra in (
            ("matplotlib", "plots"),
            ("matplotlib.pyplot", "plots"),
            ("trimesh", "mesh"),
            ("trimesh.creation", "mesh"),
            ("ase.io", "ase"),
        ):
            with self.subTest(module=module):
                self.assertIn(f"chmpy[{extra}]", self.missing(module))

    def test_mentions_the_purpose(self):
        self.assertIn("drawing a thing", self.missing("trimesh", "drawing a thing"))

    def test_unknown_module_falls_back_to_its_own_name(self):
        self.assertIn("chmpy[widget]", self.missing("widget.thing"))

    def test_have_does_not_raise(self):
        self.assertTrue(optional.have("json"))
        self.assertFalse(optional.have("definitely_not_installed_xyz"))


if __name__ == "__main__":
    unittest.main()

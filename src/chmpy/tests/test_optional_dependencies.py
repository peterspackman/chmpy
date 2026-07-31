"""matplotlib and trimesh are extras, and must stay that way.

The import-time checks run in a subprocess: by the time this test module is
collected the test session has almost certainly imported both already, so
asking `sys.modules` in-process would pass no matter what.
"""

import ast
import json
import pathlib
import re
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import chmpy
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


class DeclaredDependencyTestCase(unittest.TestCase):
    """Nothing may be imported at module level unless chmpy depends on it.

    A transitive dependency is not a dependency. pyparsing used to arrive with
    matplotlib and pandas with seaborn, so `chmpy.fmt.smiles` imported a
    package nothing declared and broke the moment those became extras. An
    import inside a function is fine - `require` turns it into a message -
    but a module-level one fails on import, which no amount of skipping saves.
    """

    #: not third-party: the package itself, and what setuptools puts on the path
    IGNORED = {"chmpy", "pytest", "occpy"}

    #: distributions that import under a different name than pip installs
    MODULE_NAME = {"pillow": "PIL", "scikit_learn": "sklearn"}

    @classmethod
    def declared(cls):
        """Every module name provided by a distribution chmpy requires.

        Read from chmpy's own metadata so the test cannot drift from
        pyproject. `packages_distributions()` would resolve the module names
        without a hand-written table, but it returns almost nothing inside an
        isolated uv environment, which is exactly where this needs to work.
        """
        from importlib.metadata import requires

        modules = set()
        for spec in requires("chmpy") or ():
            name = re.split(r"[<>=!~;\[\s]", spec, maxsplit=1)[0].strip()
            name = name.lower().replace("-", "_")
            if name and name != "chmpy":
                modules.add(cls.MODULE_NAME.get(name, name))
        return modules

    @staticmethod
    def module_level_imports(path):
        tree = ast.parse(path.read_text())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_no_undeclared_module_level_imports(self):
        declared = self.declared() | self.IGNORED | set(sys.stdlib_module_names)
        root = pathlib.Path(chmpy.__file__).parent
        offenders = {}
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts:
                continue
            extra = {
                name
                for name in self.module_level_imports(path)
                if name not in declared and not name.startswith("_")
            }
            if extra:
                offenders[str(path.relative_to(root))] = sorted(extra)
        self.assertEqual(offenders, {}, f"undeclared module-level imports: {offenders}")


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

    def test_a_module_no_extra_provides_is_installed_directly(self):
        """Naming an extra that does not exist would send the reader nowhere."""
        message = self.missing("widget.thing")
        self.assertIn("pip install widget", message)
        self.assertNotIn("chmpy[", message)

    def test_have_does_not_raise(self):
        self.assertTrue(optional.have("json"))
        self.assertFalse(optional.have("definitely_not_installed_xyz"))


if __name__ == "__main__":
    unittest.main()

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chmpy.exe import Gaussian, Gulp, Tonto, Xtb
from chmpy.exe.exe import AbstractExecutable


def _fake_executable(directory, name, script):
    path = Path(directory, name)
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class _Dummy(AbstractExecutable):
    def result(self):
        pass

    def resolve_dependencies(self):
        pass

    def post_process(self):
        pass

    def run(self):
        pass


class ExecutableTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def test_executable_setter(self):
        exe = _fake_executable(self.dir, "fake", "exit 0\n")
        job = _Dummy()
        job.executable = exe
        self.assertEqual(job.executable, exe)

    def test_default_output_file(self):
        job = _Dummy()
        self.assertIsInstance(job.output_file, str)
        self.assertTrue(os.path.exists(job.output_file))
        os.remove(job.output_file)

    def test_tonto_environment(self):
        exe = _fake_executable(
            self.dir, "tonto", 'echo "[$TONTO_BASIS_SET_DIRECTORY]"\n'
        )
        job = Tonto("input", working_directory=self.dir)
        job._executable_location = exe
        job.basis_set_directory = "/opt/basis"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TONTO_BASIS_SET_DIRECTORY", None)
            job.run()
            self.assertNotIn("TONTO_BASIS_SET_DIRECTORY", os.environ)
        self.assertEqual(job.stdout_contents.strip(), "[/opt/basis]")

    def test_xtb_relative_working_directory(self):
        exe = _fake_executable(
            self.dir, "xtb", '[ -f "$1" ] && echo "[$OMP_NUM_THREADS]" || exit 3\n'
        )
        cwd = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, cwd)
        os.makedirs("rel")
        job = Xtb("$coord\n0 0 0 h\n$end\n", working_directory="rel", opt=False)
        job._executable_location = exe
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMP_NUM_THREADS", None)
            job.run()
            self.assertNotIn("OMP_NUM_THREADS", os.environ)
        self.assertEqual(job.output_contents.strip(), "[1,1]")
        self.assertEqual(job.uhf_file, os.path.join("rel", ".UHF"))

    def test_input_paths_are_relative_to_working_directory(self):
        cwd = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, cwd)
        os.makedirs("rel")
        gulp = Gulp("", working_directory="rel")
        self.assertEqual(str(gulp.arg), "gulp_job")
        g = Gaussian("", name="job", working_directory="rel")
        with patch.object(Gaussian, "_run_raw") as run_raw:
            g.run()
        run_raw.assert_called_once_with("job.gjf")

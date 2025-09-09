import copy
import logging
from os import environ
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np

from chmpy.util.exe import which

from .exe import AbstractExecutable, ReturnCodeError

GULP_EXEC = which("gulp")
LOG = logging.getLogger("gulp")


class Gulp(AbstractExecutable):
    _input_file = "gulp_job.gin"
    _output_file = "gulp_job.gout"
    _drv_file = "gulp_job.drv"
    _res_file = "gulp_job.res"
    _executable_location = GULP_EXEC
    _timeout = 10086400.0

    def __init__(self, input_contents, *args, working_directory=".", **kwargs):
        self._timeout = kwargs.get("timeout", self._timeout)
        self.name = kwargs.get("name", "gulp_job")
        self.solvent = kwargs.get("solvent", None)
        self.threads = kwargs.get("threads", 1)
        self.input_contents = input_contents
        self.output_contents = None
        self.restart_contents = None
        self.kwargs = kwargs.copy()
        self.working_directory = working_directory
        self.arg = Path(self.input_file).with_suffix("")
        LOG.debug("Initializing gulp calculation, timeout = %s", self.timeout)
        self.error_contents = None

    @property
    def input_file(self):
        return Path(self.working_directory, self._input_file)

    @property
    def output_file(self):
        return Path(self.working_directory, self._output_file)

    @property
    def drv_file(self):
        return Path(self.working_directory, self._drv_file)

    def resolve_dependencies(self):
        """Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        LOG.debug("Writing GULP input file to %s", self.input_file)
        Path(self.input_file).write_text(
            self.input_contents + f"\noutput drv {self.drv_file}"
        )

    def result(self):
        return self.output_contents

    def post_process(self):
        self.output_contents = Path(self.output_file).read_text()
        if self.drv_file.exists():
            self.drv_contents = self.drv_file.read_text()
            # Parse the .drv file for structured data
            from chmpy.fmt.gulp import parse_drv_file

            try:
                self.drv_data = parse_drv_file(self.drv_file)
            except Exception as e:
                LOG.warning(f"Failed to parse .drv file: {e}")
                self.drv_data = None
        else:
            self.drv_contents = ""
            self.drv_data = None

    def run(self, *args, **kwargs):
        LOG.debug("Running %s %s", self._executable_location, self.arg)
        try:
            with TemporaryFile() as tmp:
                env = copy.deepcopy(environ)
                env.update(
                    {
                        "OMP_NUM_THREADS": str(self.threads) + ",1",
                        "OMP_MAX_ACTIVE_LEVELS": "1",
                        "MKL_NUM_THREADS": str(self.threads),
                    }
                )
                self._run_raw(self.arg, stderr=tmp, env=env)
                tmp.seek(0)
                self.error_contents = tmp.read().decode("utf-8")
        except ReturnCodeError as e:
            from shutil import copytree

            from chmpy.util.path import list_directory

            LOG.error("GULP execution failed: %s", e)
            self.post_process()
            LOG.error("output: %s", self.output_contents)
            LOG.error("Directory contents\n%s", list_directory(self.working_directory))
            copytree(self.working_directory, "failed_job")
            raise e

    def cleanup(self):
        """Clean up temporary files created during GULP calculation."""
        files_to_clean = [
            self.input_file,
            self.output_file,
            self.drv_file,
            Path(self.working_directory) / self._res_file,
        ]

        for file_path in files_to_clean:
            if file_path.exists():
                try:
                    file_path.unlink()
                    LOG.debug(f"Cleaned up: {file_path}")
                except Exception as e:
                    LOG.warning(f"Failed to clean up {file_path}: {e}")

    @property
    def energy(self) -> float | None:
        """Energy from parsed .drv data."""
        if self.drv_data:
            return self.drv_data.get("energy")
        return None

    @property
    def gradients(self) -> np.ndarray | None:
        """Gradients from parsed .drv data."""
        if self.drv_data:
            return self.drv_data.get("gradients")
        return None

    @property
    def stress_raw(self) -> np.ndarray | None:
        """Raw stress gradients from parsed .drv data."""
        if self.drv_data:
            return self.drv_data.get("stress_raw")
        return None

    def calculate_stress(self, volume: float) -> np.ndarray | None:
        """Calculate stress tensor from strain gradients and volume."""
        if self.stress_raw is not None and volume > 0:
            # Convert strain gradients to stress: stress = (1/V) * dE/d_strain
            return self.stress_raw / volume
        return None

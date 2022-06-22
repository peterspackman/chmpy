from .exe import AbstractExecutable, ReturnCodeError
import logging
from os import environ
from pathlib import Path
from chmpy.util.exe import which
import copy
from tempfile import TemporaryFile
import numpy as np

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
        LOG.debug(
            "Initializing gulp calculation, timeout = %s",
            self.timeout
        )
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
        Path(self.input_file).write_text(self.input_contents + f"\noutput drv {self.drv_file}")

    def result(self):
        return self.output_contents

    def post_process(self):
        self.output_contents = Path(self.output_file).read_text()
        if Path(self.drv_file).exists():
            self.drv_contents = Path(self.drv_file).read_text()
        else:
            self.drv_contents = ""

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
            from chmpy.util.path import list_directory
            from shutil import copytree

            LOG.error("GULP execution failed: %s", e)
            self.post_process()
            LOG.error("output: %s", self.output_contents)
            LOG.error("Directory contents\n%s", list_directory(self.working_directory))
            copytree(self.working_directory, "failed_job")
            raise e

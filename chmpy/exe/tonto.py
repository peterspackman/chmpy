from .exe import AbstractExecutable, run_subprocess, ReturnCodeError
from chmpy.util.exe import which
from pathlib import Path
import logging

LOG = logging.getLogger("tonto")

TONTO_EXEC = which("tonto")


class Tonto(AbstractExecutable):

    _executable_location = TONTO_EXEC
    _STDIN = "stdin"
    _STDOUT = "stdout"
    _STDERR = "stderr"

    def __init__(
        self,
        input_file,
        name="tonto_job",
        working_directory=".",
        output_file=None,
        extra_inputs=(),
        extra_outputs=(),
    ):
        """
        Parameters
        ----------
        input_file : str
            string of tonto input format
        output_file : str, optional
            output_file to store tonto output in,
            by default will be returned as the result
        """
        assert isinstance(input_file, str)

        self.name = name
        self.working_directory = working_directory
        self.input_file_contents = input_file
        self.output_file = str(self.stdout_file)
        if output_file:
            self.output_file = str(output_file)
        self.stdin_contents = None
        self.stdout_contents = None
        self.stderr_contents = None
        self.extra_inputs = extra_inputs
        self.extra_outputs = extra_outputs
        self.basis_set_directory = ""

    @property
    def stdin_file(self):
        return Path(self.working_directory, self._STDIN)

    @property
    def stdout_file(self):
        return Path(self.working_directory, self._STDOUT)

    @property
    def stderr_file(self):
        return Path(self.working_directory, self._STDERR)

    def read_stderr(self):
        if self.stderr_file.exists():
            return self.stderr_file.read_text()
        return ""

    def read_stdout(self):
        if self.stdout_file.exists():
            return self.stdout_file.read_text()
        return ""

    def write_inputs(self):
        self.stdin_file.write_text(self.input_file_contents)

    def resolve_dependencies(self):
        """Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.write_inputs()

    def result(self):
        return self.stdout_contents

    def post_process(self):
        self.stdout_contents = self.read_stdout()
        self.stderr_contents = self.read_stderr()

        if self.stdin_file.exists():
            self.stdin_file.unlink()
        if self.stderr_file.exists():
            self.stderr_file.unlink()
        if self.stdout_file.exists():
            self.stdout_file.unlink()

        output_file = Path(self.output_file)
        if not output_file.exists():
            output_file.write_text(self.stdout_contents)

    def run(self, *args, **kwargs):
        from os import environ
        from copy import deepcopy

        env = deepcopy(environ)
        env.update(
            {"TONTO_BASIS_SET_DIRECTORY": str(self.basis_set_directory),}
        )
        self._run_raw()

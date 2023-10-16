from .exe import AbstractExecutable, run_subprocess, ReturnCodeError
from chmpy.util.exe import which
from pathlib import Path
import logging

LOG = logging.getLogger("gaussian")

GAUSSIAN_EXEC = which("g09")
FORMCHK_EXEC = which("formchk")


class Gaussian(AbstractExecutable):

    _executable_location = GAUSSIAN_EXEC
    _JOB_FILE_FMT = "{}.gjf"
    _LOG_FILE_FMT = "{}.log"

    def __init__(
        self,
        input_file,
        name="job",
        run_formchk=None,
        working_directory=".",
        output_file=None,
    ):
        """
        Parameters
        ----------
        input_file : str
            string of gaussian input format
        output_file : str, optional
            output_file to store gaussian output in,
            by default will be returned as the result
        """
        assert isinstance(input_file, str)

        self.name = name
        self.working_directory = working_directory
        self._fchk_filename = None
        self.input_file_contents = input_file
        self.output_file = str(self.log_file)
        if output_file:
            self.output_file = str(output_file)
        self.run_formchk = run_formchk
        if self.run_formchk:
            self._fchk_filename = self.run_formchk.replace(".chk", ".fchk")
            self._chk_filename = self.run_formchk
        self.log_contents = None
        self.fchk_contents = None

    @property
    def job_file(self):
        return Path(self.working_directory, self._JOB_FILE_FMT.format(self.name))

    @property
    def log_file(self):
        return Path(self.working_directory, self._LOG_FILE_FMT.format(self.name))

    @property
    def fchk_file(self):
        if self._fchk_filename is not None:
            return Path(self.working_directory, self._fchk_filename)

    @property
    def chk_file(self):
        if self._chk_filename is not None:
            return Path(self.working_directory, self._chk_filename)

    def write_inputs(self):
        self.job_file.write_text(self.input_file_contents)

    def resolve_dependencies(self):
        """Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        self.write_inputs()

    def result(self):
        return self.log_contents

    def post_process(self):
        self.log_contents = self.log_file.read_text()

        output_file = Path(self.output_file)
        if not output_file.exists():
            output_file.write_text(self.log_contents)

        if self.run_formchk:
            self._run_formchk()
            assert self.fchk_file.exists(), f"{self.fchk_file} not found"
            self.fchk_contents = self.fchk_file.read_text()

    def _run_formchk(self):
        """Run formchk, may throw exceptions"""
        cmd_list = [FORMCHK_EXEC, str(self.chk_file), str(self.fchk_file)]
        with open("/dev/null", "w+") as of:
            command = run_subprocess(cmd_list, stdout=of, timeout=self.timeout)
            result = command.returncode
        if result != 0:
            raise ReturnCodeError(
                "Command '{}' exited with return code {}".format(
                    " ".join(cmd_list), result
                )
            )
        return result

    def run(self, *args, **kwargs):
        self._run_raw(self.job_file)

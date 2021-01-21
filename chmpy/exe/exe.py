import abc
import os
import tempfile
import logging
from chmpy.util.path import dir_exists_or_is_creatable, path_exists_or_is_creatable
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired, CalledProcessError, CompletedProcess
import signal
import sys


LOG = logging.getLogger(__name__)
ABC = abc.ABCMeta("ABC", (object,), {})

if sys.platform == "win32":
    _SIGNAL = None
else:
    _SIGNAL = signal.SIGKILL


def run_subprocess(
    *popenargs,
    input=None,
    capture_output=None,
    timeout=None,
    check=False,
    signal=_SIGNAL,
    **kwargs,
):
    if input is not None:
        if "stdin" in kwargs:
            raise ValueError("stdin and input arguments may not both be used.")
        kwargs["stdin"] = PIPE

    if capture_output:
        if ("stdout" in kwargs) or ("stderr" in kwargs):
            raise ValueError(
                "stdout and stderr arguments may not be used " "with capture_output."
            )
        kwargs["stdout"] = PIPE
        kwargs["stderr"] = PIPE

    with Popen(*popenargs, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired:
            if signal is None:
                os.kill(process.pid)
            else:
                process.send_signal(signal)
            stdout, stderr = process.communicate()
            raise TimeoutExpired(process.args, timeout, output=stdout, stderr=stderr)
        except:
            if signal is None:
                os.kill(process.pid)
            else:
                process.send_signal(signal)
            raise
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(
                retcode, process.args, output=stdout, stderr=stderr
            )
        return CompletedProcess(process.args, retcode, stdout, stderr)


class ReturnCodeError(Exception):
    pass


class AbstractExecutable(ABC):
    """ Abstract base class of an Executable"""

    _name = "job"
    _has_dependencies = False
    _working_directory = None
    _stdout = ""
    _stderr = ""
    _timeout = 86400.0  # timeout in seconds
    _result = None
    _output_file = None
    _executable_location = None

    @property
    def executable(self):
        return self._executable_location

    @executable.setter
    def executable(self, value):
        os.access(value)
        self._executable_location = os.path.abspath(value)

    @property
    def working_directory(self):
        """Return the current working directory for this job"""
        return self._working_directory

    @working_directory.setter
    def working_directory(self, dirname):
        """"Set the working directory for this job"""
        assert dir_exists_or_is_creatable(
            dirname
        ), f"{dirname} either cannot be found or is not createable"
        self._working_directory = dirname

    @abc.abstractmethod
    def result(self):
        """Return the result of this calculation"""
        raise NotImplementedError

    @property
    def stdout(self):
        """Return the output to stdout for this job"""
        return self._stdout

    @property
    def has_dependencies(self):
        """Does this job require some work before it
        can be run?"""
        return self._has_dependencies

    @abc.abstractmethod
    def resolve_dependencies(self):
        """Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        raise NotImplementedError

    @abc.abstractmethod
    def post_process(self):
        """ Do whatever needs to be done after the job."""
        raise NotImplementedError

    @property
    def name(self):
        """ The name of the job as a string."""
        return self._name

    @name.setter
    def name(self, name):
        """ Change the name of the job. """
        self._name = name

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = float(value)

    @property
    def output_file(self):
        if not self._output_file:
            self._output_file = tempfile.mkstemp(prefix="cspy-", suffix=".txt")
        return self._output_file

    @output_file.setter
    def output_file(self, value):
        if value:
            assert path_exists_or_is_creatable(value)
            self._output_file = value

    def _run_raw(self, *args, **kwargs):
        """Run the calculation, may throw exceptions"""
        self.resolve_dependencies()
        self.returncode = 130
        with Path(self.output_file).open("w+") as of:
            cmd_list = [self.executable] + [x for x in args]
            command = run_subprocess(
                cmd_list,
                stdout=of,
                timeout=self.timeout,
                cwd=self.working_directory,
                **kwargs,
            )
        self.returncode = command.returncode
        self._check_returncode(cmd_list)
        self.post_process()

    def _run_raw_stdin(self, *args, **kwargs):
        """Run the calculation, may throw exceptions"""
        self.resolve_dependencies()
        self.returncode = 130
        with Path(self.input_file).open() as inp:
            with Path(self.output_file).open("w+") as of:
                cmd_list = [self.executable] + [x for x in args]
                command = run_subprocess(
                    cmd_list,
                    stdout=of,
                    stdin=inp,
                    timeout=self.timeout,
                    cwd=self.working_directory,
                    **kwargs,
                )
        self.returncode = command.returncode
        self._check_returncode(cmd_list)
        self.post_process()

    def _check_returncode(self, cmd_list):
        if self.returncode != 0:
            raise ReturnCodeError(
                "Command '{}' exited with return code {}".format(
                    " ".join(str(x) for x in cmd_list), self.returncode
                )
            )

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.name)

    def __repr__(self):
        return str(self)

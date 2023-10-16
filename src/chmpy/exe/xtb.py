from .exe import AbstractExecutable, ReturnCodeError
import logging
from os import environ
from os.path import exists, join
from pathlib import Path
from chmpy.util.exe import which
import copy
from tempfile import TemporaryFile
import numpy as np

XTB_EXEC = which("xtb")
LOG = logging.getLogger("xtb")


class Xtb(AbstractExecutable):
    _input_file = "xtb.coord"
    _charge_file = ".CHRG"
    _partial_charge_file = "charges"
    _uhf_file = ".UHF"
    _output_file = "xtbopt.stdout"
    _executable_location = XTB_EXEC
    _timeout = 1800.0

    def __init__(self, input_contents, *args, working_directory=".", **kwargs):
        self._timeout = kwargs.get("timeout", self._timeout)
        self.name = kwargs.get("name", "coord")
        self.gfn = kwargs.get("gfn", 0)
        self.opt = kwargs.get("opt", True)
        self.solvent = kwargs.get("solvent", None)
        self.stacksize = kwargs.get("stacksize", "2GB")
        self.threads = kwargs.get("threads", 1)
        self.input_contents = input_contents
        self.output_contents = None
        self.opt_log_contents = None
        self.opt_coord_contents = None
        self.kwargs = kwargs.copy()
        self.esp = None
        self.working_directory = working_directory
        self.args = [self.input_file, f"--gfn{self.gfn}"]
        if self.solvent is not None:
            self.args += ["--gbsa", self.solvent]
        LOG.debug(
            "Initializing GFN%s-xTB calculation opt: %s, timeout: %ss",
            self.gfn,
            self.opt,
            self.timeout,
        )
        self.error_contents = None
        if self.opt:
            self.args.append("--opt")

    @property
    def input_file(self):
        return join(self.working_directory, self._input_file)

    @property
    def charge_file(self):
        return join(self.working_directory, self._charge_file)

    @property
    def partial_charge_file(self):
        return join(self.working_directory, self._partial_charge_file)

    @property
    def uhf_file(self):
        return join(self.working_directory, self._charge_file)

    @property
    def output_file(self):
        return join(self.working_directory, self._output_file)

    @property
    def esp_file(self):
        return join(self.working_directory, "xtb_esp.dat")

    def resolve_dependencies(self):
        """Do whatever needs to be done before running
        the job (e.g. write input file etc.)"""
        LOG.debug("Writing input file to %s", self.input_file)
        with open(self.input_file, "w") as f:
            f.write(self.input_contents)

    def result(self):
        return self.output_contents

    def post_process(self):
        with open(self.output_file) as f:
            self.output_contents = f.read()

        opt_files = {
            "opt_log": join(self.working_directory, "xtbopt.log"),
            "opt_coord": join(self.working_directory, "xtbopt.coord"),
            "trajectory": join(self.working_directory, "xtbopt.trj"),
            "chg": self.charge_file,
            "uhf": self.uhf_file,
        }
        for k, loc in opt_files.items():
            if exists(loc):
                LOG.debug("Reading %s: %s", k, loc)
                setattr(self, k + "_contents", Path(loc).read_text())
        if exists(self.esp_file):
            setattr(self, "esp", np.loadtxt(self.esp_file))

        if exists(self.partial_charge_file):
            setattr(self, "partial_charges", np.loadtxt(self.partial_charge_file))

    def run(self, *args, **kwargs):
        LOG.debug("Running `xtb %s`", " ".join(self.args))
        try:
            with TemporaryFile() as tmp:
                env = copy.deepcopy(environ)
                env.update(
                    {
                        "OMP_NUM_THREADS": str(self.threads) + ",1",
                        "OMP_STACKSIZE": str(self.stacksize),
                        "OMP_MAX_ACTIVE_LEVELS": "1",
                        "MKL_NUM_THREADS": str(self.threads),
                    }
                )
                self._run_raw(*self.args, stderr=tmp, env=env)
                tmp.seek(0)
                self.error_contents = tmp.read().decode("utf-8")
        except ReturnCodeError as e:
            from chmpy.util.path import list_directory
            from shutil import copytree

            LOG.error("XTB failed: %s", e)
            self.post_process()
            LOG.error("output: %s", self.output_contents)
            LOG.error("Directory contents\n%s", list_directory(self.working_directory))
            copytree(self.working_directory, "failed_job")
            raise e


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    input_contents = Path("coord").read_text()
    xtb = Xtb(input_contents, opt=True)
    xtb.run()
    print(xtb.output_contents)

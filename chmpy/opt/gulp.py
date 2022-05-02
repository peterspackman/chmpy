from chmpy import Crystal, Molecule
from collections.abc import Iterable
from chmpy.exe.gulp import Gulp
from chmpy.exe import ReturnCodeError, TimeoutExpired
from chmpy.fmt.gulp import crystal_to_gulp_input, molecule_to_gulp_input, parse_gulp_output
from pathlib import Path
import os
import logging
import time
from tempfile import TemporaryDirectory
import re
import numpy as np

LOG = logging.getLogger(__name__)
NUMERIC_CONST_PATTERN = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
single_line_value_regex = re.compile(r"\s*(.*)\s*[:=]\s*(.+)")


def find_outputs(stdout):
    matches = single_line_value_regex.findall(stdout)
    return {
        k.strip(): v.strip() for k, v in matches
    }


class GulpOptimizer:

    def __init__(self, name="molecule", **kwargs):
        self.ff = kwargs.pop("ff", "gfn-ff")
        LOG.debug("Initializing GulpOptimizer(ff=%s)", self.ff)
        self.maxcycle = kwargs.get("maxcycle", 1000)
        self.name = name
        self.kwargs = kwargs
        self.last_output_contents = None
        self.last_log_contents = None
        self.prefix = kwargs.get("prefix", None)
        self.keywords = ["conp", "gfnff", "prop", "phon", "noden", "gwolf"]
        self.additional_keywords = {}
        if "temperature" in kwargs:
            self.additional_keywords["temperature"] = kwargs["temperature"]
        if "supercell" in kwargs:
            self.additional_keywords["supercell"] = "{} {} {}".format(*kwargs["supercell"])

    def _run_in_tempdir(self, input_contents):
      with TemporaryDirectory(prefix=self.prefix) as tmpdirname:
            exe = Gulp(
                input_contents,
                name=self.name,
                working_directory=tmpdirname,
                **self.kwargs,
            )
            self.last_input_contents = input_contents
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in Gulp minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_drv_contents = exe.drv_contents
            self.last_parsed_output = parse_gulp_output(self.last_output_contents)
            return success

    def single_point_crystal(self, crystal, **kwargs):
        input_contents = crystal_to_gulp_input(
            crystal,
            keywords=self.keywords,
            additional_keywords=self.additional_keywords
        )
        LOG.debug("Input contents:\n%s", input_contents)
        return self._run_in_tempdir(input_contents)

    def minimize_crystal(self, crystal, **kwargs):
        input_contents = crystal_to_gulp_input(
            crystal,
            keywords=self.keywords + ["opti", "rfo"],
            additional_keywords=self.additional_keywords
        )
        return self._run_in_tempdir(input_contents)

    def minimize_molecule(self, molecule, **kwargs):
        mol_kws = self.additional_keywords.copy()
        mol_kws.pop("supercell")
        input_contents = molecule_to_gulp_input(
            molecule,
            keywords=self.keywords + ["opti"],
            additional_keywords=mol_kws,
        )
        return self._run_in_tempdir(input_contents)

    def single_point_molecule(self, molecule, **kwargs):
        mol_kws = self.additional_keywords.copy()
        mol_kws.pop("supercell")
        input_contents = molecule_to_gulp_input(
            molecule,
            keywords=self.keywords,
            additional_keywords=mol_kws,
        )
        return self._run_in_tempdir(input_contents)

    def minimize(self, obj, **kwargs):
        if isinstance(obj, Crystal):
            return self.minimize_crystal(obj, **kwargs)
        elif isinstance(obj, Molecule):
            return self.minimize_molecule(obj, **kwargs)
        elif isinstance(obj, Iterable):
            return [self.minimize(x, **kwargs) for x in obj]
        else:
            raise NotImplementedError(
                f"GulpOptimizer only implemented for Crystal, Molecule types not {obj.__class__.__name__}"
            )

    def single_point(self, obj, **kwargs):
        if isinstance(obj, Crystal):
            return self.single_point_crystal(obj, **kwargs)
        elif isinstance(obj, Molecule):
            return self.single_point_molecule(obj, **kwargs)
        elif isinstance(obj, Iterable):
            return [self.single_point(x, **kwargs) for x in obj]
        else:
            raise NotImplementedError(
                f"GulpOptimizer only implemented for Crystal, Molecule types not {obj.__class__.__name__}"
            )

    def __call__(self, obj, **kwargs):
        return self.minimize(obj)


class GulpEnergyEvaluator(GulpOptimizer):
    def __call__(self, obj, **kwargs):
        return self.single_point(obj, **kwargs)

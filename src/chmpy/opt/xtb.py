from chmpy import Crystal, Molecule
from collections.abc import Iterable
from chmpy.exe.xtb import Xtb
from chmpy.exe import ReturnCodeError, TimeoutExpired
from chmpy.fmt.xtb import turbomole_string, load_turbomole_string
from os.path import join, exists
from pathlib import Path
import os
import logging
import time
from tempfile import TemporaryDirectory
import re
import numpy as np

LOG = logging.getLogger(__name__)
NUMERIC_CONST_PATTERN = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
energy_regex = re.compile(r"total\s+energy\s+({})\s*Eh".format(NUMERIC_CONST_PATTERN))


def find_energies(stdout):
    matches = energy_regex.findall(stdout)
    if not matches:
        return np.nan, np.nan
    return float(matches[0]), float(matches[-1])


class XtbOptimizer:

    xtb_param_fmt = "param_gfn{}-xtb.txt"

    def __init__(self, gfn=0, name="molecule", **kwargs):
        LOG.debug("Initializing XtbOptimizer: GFN%s-XTB", gfn)
        xtb_home = os.environ.get("XTBHOME", os.environ.get("HOME"))
        xtb_path = os.environ.get("XTBPATH", xtb_home)
        self.maxcycle = kwargs.get("maxcycle", 1000)
        LOG.debug("XTBHOME: %s", xtb_home)
        LOG.debug("XTBPATH: %s", xtb_path)
        self.gfn = gfn
        gfnstr = "" if str(gfn) == "1" else str(gfn)
        xtb_param_file = self.xtb_param_fmt.format(gfnstr)
        self.param_file_loc = join(xtb_path, xtb_param_file)
        home_param = exists(self.param_file_loc)
        LOG.debug("Found %s: %s", self.param_file_loc, home_param)
        if not home_param:
            LOG.debug("No parameter data for GFN%s-XTB, will likely fail", self.gfn)
            # raise RuntimeError(f"Missing parameter file for Xtb: {xtb_param_file}")
        self.name = name
        self.kwargs = kwargs
        self.last_output_contents = None
        self.last_log_contents = None

    def single_point_crystal(self, crystal, **kwargs):
        input_contents = turbomole_string(crystal)
        LOG.debug("Input contents:\n%s", input_contents)
        result = None
        if self.gfn > 1:
            LOG.error(
                "Currently GFN%s-XTB is unsupported for periodic systems", self.gfn
            )
            raise ValueError("Must use GFN0-XTB or GFN-XTB (i.e. 0 or 1)")
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                opt=False,
                **self.kwargs,
            )
            self.last_input_contents = input_contents
            Path(exe.input_file).write_text(input_contents)
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                Path("xtb.coord").write_text(input_contents)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_log_contents = exe.opt_log_contents
            self.last_log_contents = exe.opt_coord_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
                crystal.properties["lattice_energy"] = final
                crystal.properties["lattice_energy_method"] = f"GFN{self.gfn}-XTB"
                crystal.properties["lattice_energy_units"] = "au"
        return result

    def minimize_crystal(self, crystal, engine="inertial"):
        input_contents = turbomole_string(
            crystal, opt=dict(engine=engine, maxcycle=self.maxcycle,),
        )
        LOG.debug("Input contents:\n%s", input_contents)
        result = None
        if self.gfn > 1:
            LOG.error(
                "Currently GFN%s-XTB is unsupported for periodic systems", self.gfn
            )
            raise ValueError("Must use GFN0-XTB or GFN-XTB (i.e. 0 or 1)")
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                **self.kwargs,
            )
            self.last_input_contents = input_contents
            Path(exe.input_file).write_text(input_contents)
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_log_contents = exe.opt_log_contents
            self.last_log_contents = exe.opt_coord_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
            if exe.opt and success:
                result = load_turbomole_string(exe.opt_coord_contents)
                result = Crystal(**result)
                result.properties["lattice_energy"] = final
                result.properties["density"] = result.density
                result.properties["lattice_energy_method"] = f"GFN{self.gfn}-XTB"
                result.properties["lattice_energy_units"] = "au"
        return result

    def minimize_molecule(self, molecule, engine="rf"):
        input_contents = turbomole_string(
            molecule, opt=dict(engine=engine, maxcycle=self.maxcycle,),
        )
        LOG.debug("Input contents:\n%s", input_contents)
        result = None
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                **self.kwargs,
            )
            Path(exe.input_file).write_text(input_contents)
            if molecule.charge != 0:
                Path(exe.charge_file).write_text(str(molecule.charge))
            if molecule.multiplicity != 1:
                Path(exe.uhf_file).write_text(str(molecule.multiplicity))

            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_log_contents = exe.opt_log_contents
            self.last_log_contents = exe.opt_coord_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
                LOG.debug("Energy change %.5f -> %.5f Eh", init, final)
            if exe.opt and success:
                result = load_turbomole_string(exe.opt_coord_contents)
                result = Molecule(**result)
                result.properties["name"] = molecule.name + "_xtbopt"
                result.properties["scf_energy"] = final * 2625.499639
                result.properties["scf_energy_units"] = "kj/mol"
                result.properties["scf_energy_method"] = f"GFN{self.gfn}-XTB"
                result.charge = molecule.charge
                result.multiplicity = molecule.multiplicity
        return result

    def single_point_molecule(self, molecule, engine="rf"):
        input_contents = turbomole_string(molecule)
        LOG.debug("Input contents:\n%s", input_contents)
        result = None
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            exe = Xtb(
                input_contents,
                gfn=self.gfn,
                name=self.name,
                working_directory=tmpdirname,
                **self.kwargs,
            )
            Path(exe.input_file).write_text(input_contents)
            t1 = time.time()
            try:
                exe.run()
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in XTB minimization: %s", exc)
                return None
            t2 = time.time()
            success = exe.output_contents is not None
            self.last_output_contents = exe.output_contents
            self.last_log_contents = exe.opt_log_contents
            self.last_log_contents = exe.opt_coord_contents
            if success:
                init, final = find_energies(exe.output_contents)
                result = final
                LOG.debug("Energy change %.5f -> %.5f Eh", init, final)
            if exe.opt and success:
                molecule.properties["scf_energy"] = final * 2625.499639
                molecule.properties["scf_energy_units"] = "kj/mol"
                molecule.properties["scf_energy_method"] = f"GFN{self.gfn}-XTB"
        return result

    def minimize(self, obj, **kwargs):
        if isinstance(obj, Crystal):
            return self.minimize_crystal(obj, **kwargs)
        elif isinstance(obj, Molecule):
            return self.minimize_molecule(obj, **kwargs)
        elif isinstance(obj, Iterable):
            return [self.minimize(x, **kwargs) for x in obj]
        else:
            raise NotImplementedError(
                f"XtbOptimizer only implemented for Crystal, Molecule types not {obj.__class__.__name__}"
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
                f"XtbOptimizer only implemented for Crystal, Molecule types not {obj.__class__.__name__}"
            )

    def __call__(self, obj, **kwargs):
        return self.minimize(obj)


class XtbEnergyEvaluator(XtbOptimizer):
    def __call__(self, obj, **kwargs):
        return self.single_point(obj, **kwargs)

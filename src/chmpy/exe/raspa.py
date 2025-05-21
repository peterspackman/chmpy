import copy
import json
import logging
import shutil
from os import environ
from pathlib import Path
from tempfile import TemporaryFile

from chmpy.util.exe import which

from .exe import AbstractExecutable, ReturnCodeError

RASPA_EXEC = which("raspa3")
LOG = logging.getLogger("raspa")


class Raspa(AbstractExecutable):
    _simulation_file = "simulation.json"
    _force_field_file = "force_field.json"
    _output_file = "raspa_output.txt"
    _executable_location = RASPA_EXEC
    _timeout = 10086400.0

    def __init__(
        self,
        simulation_json,
        force_field_json=None,
        framework_file=None,
        component_files=None,
        *args,
        working_directory=".",
        **kwargs,
    ):
        """
        Initialize a RASPA3 simulation.

        Args:
            simulation_json: Simulation configuration as dict or JSON string
            force_field_json: Force field configuration as dict or JSON string
            framework_file: Path to framework file (CIF)
            component_files: Dict mapping component names to their definition files
                             e.g. {"CO2": "path/to/CO2.json"}
            working_directory: Directory to run the simulation in
            kwargs: Additional arguments
                - timeout: Maximum execution time in seconds
                - name: Job name
                - threads: Number of threads to use
        """
        self._timeout = kwargs.get("timeout", self._timeout)
        self.name = kwargs.get("name", "raspa_job")
        self.threads = kwargs.get("threads", 1)

        # Store inputs
        self.simulation_json = simulation_json
        self.force_field_json = force_field_json
        self.framework_file = framework_file
        self.component_files = component_files or {}

        # Store outputs
        self.output_contents = None
        self.error_contents = None

        # Setup
        self.kwargs = kwargs.copy()
        self.working_directory = working_directory
        LOG.debug("Initializing RASPA3 calculation, timeout = %s", self.timeout)

        # Framework and component information
        self._framework_name = None
        self._component_names = []

        # Extract framework and component names from simulation JSON
        if isinstance(simulation_json, dict):
            sim_data = simulation_json
        else:
            try:
                sim_data = json.loads(simulation_json)
            except (json.JSONDecodeError, TypeError):
                sim_data = {}

        # Extract framework name and component names if present
        if "Systems" in sim_data and len(sim_data["Systems"]) > 0:
            if "Name" in sim_data["Systems"][0]:
                self._framework_name = sim_data["Systems"][0]["Name"]

        if "Components" in sim_data:
            for component in sim_data["Components"]:
                if "Name" in component:
                    self._component_names.append(component["Name"])

    @property
    def simulation_file(self):
        return Path(self.working_directory, self._simulation_file)

    @property
    def force_field_file(self):
        return Path(self.working_directory, self._force_field_file)

    @property
    def output_file(self):
        return Path(self.working_directory, self._output_file)

    def _get_framework_path(self):
        """Get the path to save the framework file"""
        if self._framework_name:
            return Path(self.working_directory, f"{self._framework_name}.cif")
        elif self.framework_file:
            # Use the original filename if no name specified
            return Path(self.working_directory, Path(self.framework_file).name)
        return None

    def _get_component_path(self, component_name):
        """Get the path to save a component file"""
        return Path(self.working_directory, f"{component_name}.json")

    def resolve_dependencies(self):
        """Write all necessary files before running the job"""
        LOG.debug("Writing RASPA3 simulation files to %s", self.working_directory)

        # Write simulation.json
        if isinstance(self.simulation_json, dict):
            simulation_content = json.dumps(self.simulation_json, indent=2)
        else:
            simulation_content = self.simulation_json

        Path(self.simulation_file).write_text(simulation_content)

        # Write force_field.json if provided
        if self.force_field_json:
            if isinstance(self.force_field_json, dict):
                force_field_content = json.dumps(self.force_field_json, indent=2)
            else:
                force_field_content = self.force_field_json

            Path(self.force_field_file).write_text(force_field_content)

        # Copy framework file if provided
        if self.framework_file:
            framework_dest = self._get_framework_path()
            if framework_dest:
                # Handle both file paths and content strings
                if (
                    isinstance(self.framework_file, str | Path)
                    and Path(self.framework_file).exists()
                ):
                    shutil.copy2(self.framework_file, framework_dest)
                else:
                    # Assume it's the content as string
                    Path(framework_dest).write_text(self.framework_file)

        # Copy component files if provided
        for component_name, component_file in self.component_files.items():
            component_dest = self._get_component_path(component_name)

            # Handle both file paths and content (dict or string)
            if isinstance(component_file, dict):
                component_content = json.dumps(component_file, indent=2)
                Path(component_dest).write_text(component_content)
            elif (
                isinstance(component_file, str | Path) and Path(component_file).exists()
            ):
                shutil.copy2(component_file, component_dest)
            else:
                # Assume it's the content as string
                Path(component_dest).write_text(component_file)

    def result(self):
        """Return the output contents"""
        return self.output_contents

    def post_process(self):
        """Process output after the job completes"""
        # Find and read output files
        output_dir = Path(self.working_directory, "output")
        if output_dir.exists() and output_dir.is_dir():
            # Find the most recent .txt output file
            output_files = list(output_dir.glob("*.txt"))
            if output_files:
                latest_output = max(output_files, key=lambda p: p.stat().st_mtime)
                self.output_contents = latest_output.read_text()
                return

        # If no output files found, check for standard output redirect
        if Path(self.output_file).exists():
            self.output_contents = Path(self.output_file).read_text()
        else:
            LOG.warning("No output files found after RASPA3 execution")
            self.output_contents = ""

    def run(self, *args, **kwargs):
        """Run RASPA3 with the simulation file"""
        LOG.debug("Running %s in %s", self._executable_location, self.working_directory)
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
                # RASPA3 is run without arguments - it automatically looks for simulation.json
                self._run_raw(stderr=tmp, env=env)
                tmp.seek(0)
                self.error_contents = tmp.read().decode("utf-8")
        except ReturnCodeError as e:
            from shutil import copytree

            from chmpy.util.path import list_directory

            LOG.error("RASPA3 execution failed: %s", e)
            self.post_process()
            LOG.error("output: %s", self.output_contents)
            LOG.error("Directory contents\n%s", list_directory(self.working_directory))
            copytree(self.working_directory, "failed_job")
            raise e

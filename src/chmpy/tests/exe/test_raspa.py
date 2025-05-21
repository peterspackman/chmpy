import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from chmpy.exe import Raspa
from chmpy.util.exe import which

# Path to example data
HQ_CO2_EXAMPLE = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    ),
    "extra_data",
    "HQ-CO2-simulation",
)

RASPA_AVAILABLE = which("raspa3") is not None


@unittest.skipIf(not RASPA_AVAILABLE, "RASPA3 executable not found")
class RaspaTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Check if example data is available
        if not HQ_CO2_EXAMPLE.exists():
            self.skipTest("HQ-CO2 example data not found")

        # Load test data
        self.simulation_json = (HQ_CO2_EXAMPLE / "simulation.json").read_text()
        self.force_field_json = (HQ_CO2_EXAMPLE / "force_field.json").read_text()
        self.framework_file_path = HQ_CO2_EXAMPLE / "HQ.cif"
        self.component_file_path = HQ_CO2_EXAMPLE / "CO2.json"

        # Parse simulation JSON to extract names
        sim_data = json.loads(self.simulation_json)
        self.framework_name = sim_data["Systems"][0]["Name"]
        self.component_names = [comp["Name"] for comp in sim_data["Components"]]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test the initialization of the RASPA wrapper"""
        raspa = Raspa(
            simulation_json=self.simulation_json,
            force_field_json=self.force_field_json,
            framework_file=self.framework_file_path,
            component_files={"CO2": self.component_file_path},
            working_directory=self.test_dir,
        )

        self.assertEqual(raspa._framework_name, self.framework_name)
        self.assertEqual(raspa._component_names, self.component_names)

    def test_resolve_dependencies(self):
        """Test that all required files are written correctly"""
        raspa = Raspa(
            simulation_json=self.simulation_json,
            force_field_json=self.force_field_json,
            framework_file=self.framework_file_path,
            component_files={"CO2": self.component_file_path},
            working_directory=self.test_dir,
        )

        raspa.resolve_dependencies()

        # Check simulation.json
        simulation_file = Path(self.test_dir, "simulation.json")
        self.assertTrue(simulation_file.exists(), "simulation.json file not created")

        # Check force_field.json
        force_field_file = Path(self.test_dir, "force_field.json")
        self.assertTrue(force_field_file.exists(), "force_field.json file not created")

        # Check framework file
        framework_file = Path(self.test_dir, f"{self.framework_name}.cif")
        self.assertTrue(
            framework_file.exists(), f"{self.framework_name}.cif file not created"
        )

        # Check component files
        for component_name in self.component_names:
            component_file = Path(self.test_dir, f"{component_name}.json")
            self.assertTrue(
                component_file.exists(), f"{component_name}.json file not created"
            )

    @patch("chmpy.exe.exe.run_subprocess")
    def test_run_mock(self, mock_run_subprocess):
        """Test running RASPA with mocked subprocess"""
        # Mock the subprocess call
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run_subprocess.return_value = mock_process

        # Create output directory and file to test post-processing
        output_dir = Path(self.test_dir, "output")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "output_323_1e+05.s0.txt"
        output_file.write_text("Test RASPA output")

        # Run RASPA
        raspa = Raspa(
            simulation_json=self.simulation_json,
            force_field_json=self.force_field_json,
            framework_file=self.framework_file_path,
            component_files={"CO2": self.component_file_path},
            working_directory=self.test_dir,
        )

        raspa.run()

        # Check that subprocess was called
        mock_run_subprocess.assert_called_once()

        # Test post-processing
        self.assertEqual(raspa.output_contents, "Test RASPA output")

    @unittest.skipIf(not RASPA_AVAILABLE, "RASPA3 executable not found")
    def test_run_real(self):
        """Test running actual RASPA executable (skip by default)"""
        # Create a simple test with minimal settings for a quick run
        simple_sim = {
            "SimulationType": "MonteCarlo",
            "NumberOfCycles": 10,  # Very few cycles for quick test
            "NumberOfInitializationCycles": 0,
            "PrintEvery": 1,
            "Systems": [
                {
                    "Type": "Framework",
                    "Name": "HQ",
                    "NumberOfUnitCells": [1, 1, 1],
                    "ExternalTemperature": 300.0,
                    "ExternalPressure": 0.0,
                }
            ],
            "Components": [
                {
                    "Name": "CO2",
                    "MoleculeDefinition": "ExampleDefinitions",
                    "CreateNumberOfMolecules": 0,
                }
            ],
        }

        # Run RASPA with minimal settings
        raspa = Raspa(
            simulation_json=simple_sim,
            force_field_json=self.force_field_json,
            framework_file=self.framework_file_path,
            component_files={"CO2": self.component_file_path},
            working_directory=self.test_dir,
        )

        try:
            raspa.run()
            # If we get here without exception, consider it a success
            self.assertIsNotNone(raspa.output_contents)
        except Exception as e:
            self.fail(f"RASPA execution failed: {e}")

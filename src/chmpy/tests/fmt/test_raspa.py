import os
import shutil
import tempfile
import unittest
from pathlib import Path

from chmpy.fmt import raspa

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


class RaspaParserTestCase(unittest.TestCase):
    def setUp(self):
        # Check if example data is available
        if not HQ_CO2_EXAMPLE.exists():
            self.skipTest("HQ-CO2 example data not found")

        # Find output files
        self.json_output = HQ_CO2_EXAMPLE / "output" / "output_323_1e+05.s0.json"
        self.txt_output = HQ_CO2_EXAMPLE / "output" / "output_323_1e+05.s0.txt"

        if not self.json_output.exists() or not self.txt_output.exists():
            self.skipTest("RASPA example output files not found")

    def test_parse_raspa_json(self):
        """Test parsing a RASPA JSON output file"""
        parsed_data = raspa.parse_raspa_json(self.json_output)

        # Check that key sections are present
        self.assertIn("initialization", parsed_data)
        self.assertIn("output", parsed_data)
        self.assertIn("properties", parsed_data)

        # Check version
        self.assertEqual(parsed_data.get("version"), "3.0.1")

        # Check some specific values from components
        self.assertIn("components", parsed_data["initialization"])
        self.assertIn("CO2", parsed_data["initialization"]["components"])

        co2_data = parsed_data["initialization"]["components"]["CO2"]
        self.assertEqual(co2_data["criticalTemperature"], 304.1282)
        self.assertEqual(co2_data["criticalPressure"], 7377300.0)
        self.assertEqual(co2_data["acentricFactor"], 0.22394)

    def test_parse_raspa_txt(self):
        """Test parsing a RASPA text output file"""
        parsed_data = raspa.parse_raspa_txt(self.txt_output)

        # Check that key sections are present
        self.assertIn("general", parsed_data)
        self.assertIn("adsorption", parsed_data)
        self.assertIn("energy", parsed_data)
        self.assertIn("pressure", parsed_data)
        self.assertIn("enthalpy", parsed_data)

        # Check general values
        self.assertEqual(parsed_data["general"].get("version"), "3.0.1")
        self.assertEqual(parsed_data["general"].get("temperature"), 323.0)
        self.assertEqual(parsed_data["general"].get("pressure"), 100000.0)

        # Check component information is not directly extracted now (focused on adsorption data)
        self.assertIn("CO2", parsed_data["adsorption"])

        # Check adsorption data
        self.assertIn("CO2", parsed_data["adsorption"])
        adsorption = parsed_data["adsorption"]["CO2"]
        self.assertIn("absolute", adsorption)
        self.assertIn("excess", adsorption)

        # Check absolute adsorption values
        abs_ads = adsorption["absolute"]
        self.assertAlmostEqual(abs_ads["molecules"], 3.0, places=1)
        self.assertGreater(abs_ads["mol_per_kg"], 3.0)
        self.assertGreater(abs_ads["mg_per_g"], 130.0)

        # Check energy data
        self.assertIn("total_potential", parsed_data["energy"])

        # Check pressure data
        self.assertIn("total", parsed_data["pressure"])
        self.assertIn("ideal_gas", parsed_data["pressure"])
        self.assertIn("excess", parsed_data["pressure"])

    def test_extract_adsorption_data(self):
        """Test extracting adsorption data from parsed RASPA output"""
        parsed_data = raspa.parse_raspa_txt(self.txt_output)
        adsorption_data = raspa.extract_adsorption_data(parsed_data)

        self.assertIn("CO2", adsorption_data)
        self.assertIn("absolute", adsorption_data["CO2"])
        self.assertIn("excess", adsorption_data["CO2"])

        # Check specific values
        abs_ads = adsorption_data["CO2"]["absolute"]
        self.assertAlmostEqual(abs_ads["molecules"], 3.0, places=1)

    def test_extract_energy_data(self):
        """Test extracting energy data from parsed RASPA output"""
        parsed_data = raspa.parse_raspa_txt(self.txt_output)
        energy_data = raspa.extract_energy_data(parsed_data)

        self.assertIn("total_potential", energy_data)
        # There should be several energy components
        self.assertGreater(len(energy_data), 5)

    def test_parse_output_directory(self):
        """Test parsing all output files in a directory"""
        # Create a temporary directory with copies of the output files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy(self.json_output, tmp_path / self.json_output.name)
            shutil.copy(self.txt_output, tmp_path / self.txt_output.name)

            # Parse the directory
            parsed_data = raspa.parse_output_directory(tmp_path)

            # Check results
            self.assertIn("text_outputs", parsed_data)
            self.assertIn("json_outputs", parsed_data)
            self.assertEqual(len(parsed_data["text_outputs"]), 1)
            self.assertEqual(len(parsed_data["json_outputs"]), 1)


if __name__ == "__main__":
    unittest.main()

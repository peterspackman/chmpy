import unittest
from pathlib import Path

import numpy as np

from chmpy.core import Molecule
from chmpy.crystal import Crystal
from chmpy.fmt.aims import (
    crystal_to_geometry_string,
    molecule_to_geometry_string,
    parse_geometry_string,
    to_geometry_string,
)

from .. import TEST_FILES


class AimsFormatTestCase(unittest.TestCase):
    def setUp(self):
        self.acetic_acid = Crystal.load(TEST_FILES["acetic_acid.cif"])
        self.water = Molecule.load(TEST_FILES["water.xyz"])

    def test_crystal_to_geometry_string_fractional(self):
        """Test converting a crystal to geometry.in format with fractional coordinates."""
        geom_str = crystal_to_geometry_string(self.acetic_acid, use_cartesian=False)

        # Check that it contains lattice vectors
        self.assertIn("lattice_vector", geom_str)

        # Check that it uses fractional coordinates
        self.assertIn("atom_frac", geom_str)
        self.assertNotIn("atom ", geom_str.replace("atom_frac", ""))

        # Count lines
        lines = [l for l in geom_str.splitlines() if l.strip()]
        lattice_lines = len([l for l in lines if l.startswith("lattice_vector")])
        self.assertEqual(lattice_lines, 3)

    def test_crystal_to_geometry_string_cartesian(self):
        """Test converting a crystal to geometry.in format with Cartesian coordinates."""
        geom_str = crystal_to_geometry_string(self.acetic_acid, use_cartesian=True)

        # Check that it contains lattice vectors
        self.assertIn("lattice_vector", geom_str)

        # Check that it uses Cartesian coordinates
        lines = [l for l in geom_str.splitlines() if l.strip() and not l.startswith("lattice_vector")]
        atom_lines = [l for l in lines if l.startswith("atom ")]
        self.assertGreater(len(atom_lines), 0)

    def test_molecule_to_geometry_string(self):
        """Test converting a molecule to geometry.in format."""
        geom_str = molecule_to_geometry_string(self.water)

        # Should not have lattice vectors for molecules
        self.assertNotIn("lattice_vector", geom_str)

        # Should have atom lines
        self.assertIn("atom ", geom_str)

        # Count atoms
        atom_lines = [l for l in geom_str.splitlines() if l.strip().startswith("atom ")]
        self.assertEqual(len(atom_lines), len(self.water.elements))

    def test_to_geometry_string_wrapper(self):
        """Test the generic wrapper function."""
        # Test with crystal
        geom_str = to_geometry_string(self.acetic_acid)
        self.assertIn("lattice_vector", geom_str)
        self.assertIn("atom_frac", geom_str)

        # Test with molecule
        geom_str = to_geometry_string(self.water)
        self.assertNotIn("lattice_vector", geom_str)
        self.assertIn("atom ", geom_str)

    def test_parse_geometry_string_fractional(self):
        """Test parsing a geometry.in string with fractional coordinates."""
        # Create a simple test structure
        test_input = """lattice_vector 10.0 0.0 0.0
lattice_vector 0.0 10.0 0.0
lattice_vector 0.0 0.0 10.0

atom_frac 0.0 0.0 0.0 C
atom_frac 0.5 0.5 0.5 H
"""

        result = parse_geometry_string(test_input)

        # Check lattice
        self.assertIn("lattice", result)
        np.testing.assert_array_almost_equal(
            result["lattice"], [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )

        # Check atoms
        self.assertEqual(len(result["elements"]), 2)
        self.assertEqual(result["elements"][0].symbol, "C")
        self.assertEqual(result["elements"][1].symbol, "H")
        self.assertTrue(result["fractional"])

    def test_parse_geometry_string_cartesian(self):
        """Test parsing a geometry.in string with Cartesian coordinates."""
        test_input = """lattice_vector 5.0 0.0 0.0
lattice_vector 0.0 5.0 0.0
lattice_vector 0.0 0.0 5.0

atom 1.0 2.0 3.0 O
atom 4.0 5.0 6.0 H
"""

        result = parse_geometry_string(test_input)

        # Check atoms
        self.assertEqual(len(result["elements"]), 2)
        self.assertEqual(result["elements"][0].symbol, "O")
        self.assertEqual(result["elements"][1].symbol, "H")
        self.assertFalse(result["fractional"])
        np.testing.assert_array_almost_equal(result["positions"][0], [1.0, 2.0, 3.0])

    def test_parse_geometry_string_comments(self):
        """Test that comments are ignored."""
        test_input = """# This is a comment
lattice_vector 10.0 0.0 0.0  # inline comment should work in real parser
lattice_vector 0.0 10.0 0.0
lattice_vector 0.0 0.0 10.0

# Another comment
atom_frac 0.0 0.0 0.0 C
"""

        result = parse_geometry_string(test_input)

        # Should parse successfully
        self.assertEqual(len(result["elements"]), 1)
        self.assertIn("lattice", result)

    def test_roundtrip_fractional(self):
        """Test that we can write and read back a structure with fractional coordinates."""
        # Write
        geom_str = crystal_to_geometry_string(self.acetic_acid, use_cartesian=False)

        # Parse
        result = parse_geometry_string(geom_str)

        # Verify we got the same number of atoms
        uc_atoms = self.acetic_acid.unit_cell_atoms()
        self.assertEqual(len(result["elements"]), len(uc_atoms["element"]))

    def test_roundtrip_cartesian(self):
        """Test that we can write and read back a structure with Cartesian coordinates."""
        # Write
        geom_str = crystal_to_geometry_string(self.acetic_acid, use_cartesian=True)

        # Parse
        result = parse_geometry_string(geom_str)

        # Verify we got the same number of atoms
        uc_atoms = self.acetic_acid.unit_cell_atoms()
        self.assertEqual(len(result["elements"]), len(uc_atoms["element"]))
        self.assertFalse(result["fractional"])

    def test_parse_real_geometry_file(self):
        """Test parsing an actual FHI-aims geometry.in file from the test data."""
        # Try to load one of the example files if available
        test_geom_path = Path("scaling/scaling_test/aspirin/128/geometry.in")
        if test_geom_path.exists():
            geom_str = test_geom_path.read_text()
            result = parse_geometry_string(geom_str)

            # Should have parsed correctly
            self.assertGreater(len(result["elements"]), 0)
            self.assertIn("lattice", result)
            self.assertTrue(result["fractional"])

    def test_crystal_load_geometry_in(self):
        """Test loading a geometry.in file using Crystal.load()."""
        test_geom_path = Path("scaling/scaling_test/aspirin/128/geometry.in")
        if test_geom_path.exists():
            crystal = Crystal.load(str(test_geom_path))

            # Verify it loaded correctly
            self.assertIsInstance(crystal, Crystal)
            uc_atoms = crystal.unit_cell_atoms()
            self.assertEqual(len(uc_atoms["element"]), 84)

    def test_crystal_load_roundtrip(self):
        """Test that we can save and reload a crystal via geometry.in format."""
        # Save to geometry.in format
        geom_str = crystal_to_geometry_string(self.acetic_acid)

        # Load it back
        crystal = Crystal.from_aims_string(geom_str)

        # Verify we got the same structure
        self.assertIsInstance(crystal, Crystal)
        orig_atoms = self.acetic_acid.unit_cell_atoms()
        new_atoms = crystal.unit_cell_atoms()
        self.assertEqual(len(orig_atoms["element"]), len(new_atoms["element"]))

        # Check unit cell is similar
        np.testing.assert_array_almost_equal(
            self.acetic_acid.unit_cell.lattice, crystal.unit_cell.lattice, decimal=5
        )

    def test_crystal_from_aims_file(self):
        """Test Crystal.from_aims_file() method directly."""
        test_geom_path = Path("scaling/scaling_test/aspirin/128/geometry.in")
        if test_geom_path.exists():
            crystal = Crystal.from_aims_file(str(test_geom_path))

            self.assertIsInstance(crystal, Crystal)
            self.assertGreater(len(crystal.asymmetric_unit.elements), 0)


if __name__ == "__main__":
    unittest.main()

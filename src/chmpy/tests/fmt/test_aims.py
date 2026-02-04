import unittest
from pathlib import Path

import numpy as np

from chmpy.core import Molecule
from chmpy.crystal import Crystal
from chmpy.fmt.aims import (
    AimsOutput,
    crystal_to_geometry_string,
    generate_control_in,
    geometry_string_to_crystal,
    geometry_string_to_molecule,
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

    def test_generate_control_in_basic(self):
        """Test basic control.in generation."""
        species_dir = Path("~/src/FHIaims-250822/species_defaults").expanduser()
        if not species_dir.exists():
            self.skipTest("Species defaults directory not found")

        control_str = generate_control_in(
            self.acetic_acid,
            species_defaults_dir=species_dir,
            basis="light",
            xc="pbe",
        )

        # Check header
        self.assertIn("FHI-aims control.in file", control_str)
        self.assertIn("xc                                 pbe", control_str)

        # Check species blocks
        self.assertIn("species        C", control_str)
        self.assertIn("species        H", control_str)
        self.assertIn("species        O", control_str)

    def test_generate_control_in_with_relaxation(self):
        """Test control.in with geometry and cell relaxation."""
        species_dir = Path("~/src/FHIaims-250822/species_defaults").expanduser()
        if not species_dir.exists():
            self.skipTest("Species defaults directory not found")

        control_str = generate_control_in(
            self.acetic_acid,
            species_defaults_dir=species_dir,
            basis="light",
            xc="pbe",
            k_grid=(4, 4, 4),
            relax_geometry=True,
            relax_unit_cell="full",
            output_options=["hirshfeld"],
        )

        # Check keywords
        self.assertIn("relax_geometry trm", control_str)
        self.assertIn("relax_unit_cell                    full", control_str)
        self.assertIn("k_grid                             4 4 4", control_str)
        self.assertIn("output                             hirshfeld", control_str)

    def test_generate_control_in_different_basis(self):
        """Test control.in with different basis sets."""
        species_dir = Path("~/src/FHIaims-250822/species_defaults").expanduser()
        if not species_dir.exists():
            self.skipTest("Species defaults directory not found")

        # Test lightdense
        if (species_dir / "defaults_2020" / "lightdense").exists():
            control_str = generate_control_in(
                self.acetic_acid,
                species_defaults_dir=species_dir,
                basis="lightdense",
                xc="b86bpbe-25",
            )
            self.assertIn("Basis set: lightdense", control_str)
            self.assertIn("xc                                 b86bpbe-25", control_str)

    def test_generate_control_in_extra_keywords(self):
        """Test control.in with extra keywords."""
        species_dir = Path("~/src/FHIaims-250822/species_defaults").expanduser()
        if not species_dir.exists():
            self.skipTest("Species defaults directory not found")

        control_str = generate_control_in(
            self.acetic_acid,
            species_defaults_dir=species_dir,
            basis="light",
            xc="b86bpbe-25",
            extra_keywords={"xdm": "0.69125110 1.57470830", "output_level": "MD_light"},
        )

        self.assertIn("xdm", control_str)
        self.assertIn("0.69125110 1.57470830", control_str)
        self.assertIn("output_level", control_str)
        self.assertIn("MD_light", control_str)

    def test_generate_control_in_molecule(self):
        """Test control.in generation for a molecule (non-periodic)."""
        species_dir = Path("~/src/FHIaims-250822/species_defaults").expanduser()
        if not species_dir.exists():
            self.skipTest("Species defaults directory not found")

        control_str = generate_control_in(
            self.water, species_defaults_dir=species_dir, basis="light", xc="pbe"
        )

        # Should not have k_grid for molecules
        self.assertNotIn("k_grid", control_str)

        # Should have species
        self.assertIn("species        H", control_str)
        self.assertIn("species        O", control_str)

    def test_geometry_string_to_molecule(self):
        """Test converting geometry string to Molecule."""
        test_input = """# Molecule geometry
atom 0.0 0.0 0.0 O
atom 0.96 0.0 0.0 H
atom -0.24 0.93 0.0 H
"""
        mol = geometry_string_to_molecule(test_input)

        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.elements), 3)
        self.assertEqual(mol.elements[0].symbol, "O")
        self.assertEqual(mol.elements[1].symbol, "H")
        self.assertEqual(mol.elements[2].symbol, "H")

    def test_geometry_string_to_molecule_rejects_periodic(self):
        """Test that geometry_string_to_molecule raises error for periodic systems."""
        test_input = """lattice_vector 10.0 0.0 0.0
lattice_vector 0.0 10.0 0.0
lattice_vector 0.0 0.0 10.0

atom 0.0 0.0 0.0 C
"""
        with self.assertRaises(ValueError) as context:
            geometry_string_to_molecule(test_input)

        self.assertIn("lattice vectors", str(context.exception))

    def test_geometry_string_to_crystal(self):
        """Test converting geometry string to Crystal."""
        test_input = """lattice_vector 5.0 0.0 0.0
lattice_vector 0.0 5.0 0.0
lattice_vector 0.0 0.0 5.0

atom_frac 0.0 0.0 0.0 Na
atom_frac 0.5 0.5 0.5 Cl
"""
        crystal = geometry_string_to_crystal(test_input)

        self.assertIsInstance(crystal, Crystal)
        uc_atoms = crystal.unit_cell_atoms()
        self.assertEqual(len(uc_atoms["element"]), 2)

    def test_molecule_from_aims_string(self):
        """Test Molecule.from_aims_string() method."""
        test_input = """atom 1.0 2.0 3.0 C
atom 4.0 5.0 6.0 H
"""
        mol = Molecule.from_aims_string(test_input)

        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.elements), 2)
        np.testing.assert_array_almost_equal(mol.positions[0], [1.0, 2.0, 3.0])


class AimsOutputTestCase(unittest.TestCase):
    """Tests for the AimsOutput parser."""

    def test_aims_output_parse_periodic_optimization(self):
        """Test parsing a periodic optimization output."""
        # Check if we have the test file
        aims_out_path = Path.home() / "work/embed/aims-jobs/opt/taurine/aims.out"
        if not aims_out_path.exists():
            self.skipTest("Test aims.out file not found")

        output = AimsOutput.from_file(aims_out_path)

        self.assertTrue(output.is_periodic)
        self.assertTrue(output.is_optimization)
        self.assertTrue(output.converged)
        self.assertGreater(output.n_steps, 0)
        self.assertIsNotNone(output.final_energy)
        self.assertIsNotNone(output.final_structure)

        # Check final structure has lattice
        self.assertIsNotNone(output.final_structure.lattice)
        self.assertEqual(output.final_structure.lattice.shape, (3, 3))

    def test_aims_output_parse_molecule_optimization(self):
        """Test parsing a molecule optimization output."""
        aims_out_path = Path.home() / "work/embed/aims-jobs/opt/monomers/sulfamic/aims.out"
        if not aims_out_path.exists():
            self.skipTest("Test aims.out file not found")

        output = AimsOutput.from_file(aims_out_path)

        self.assertFalse(output.is_periodic)
        self.assertTrue(output.is_optimization)
        self.assertTrue(output.converged)
        self.assertIsNotNone(output.final_energy)
        self.assertIsNotNone(output.final_structure)

        # Check that molecule has no lattice
        self.assertIsNone(output.final_structure.lattice)

    def test_aims_output_get_energies(self):
        """Test extracting energies from aims.out."""
        aims_out_path = Path.home() / "work/embed/aims-jobs/opt/taurine/aims.out"
        if not aims_out_path.exists():
            self.skipTest("Test aims.out file not found")

        output = AimsOutput.from_file(aims_out_path)
        energies = output.get_energies()

        self.assertGreater(len(energies), 0)
        # Energies should decrease or stay roughly constant during optimization
        # Just check they are all negative (typical for DFT total energies)
        for e in energies:
            self.assertLess(e, 0)

    def test_aims_output_get_trajectory(self):
        """Test getting trajectory from aims.out."""
        aims_out_path = Path.home() / "work/embed/aims-jobs/opt/taurine/aims.out"
        if not aims_out_path.exists():
            self.skipTest("Test aims.out file not found")

        output = AimsOutput.from_file(aims_out_path)
        trajectory = output.get_trajectory()

        self.assertGreater(len(trajectory), 0)
        # Should be Crystal objects for periodic system
        for struct in trajectory:
            self.assertIsInstance(struct, Crystal)

    def test_aims_output_to_crystal(self):
        """Test converting final structure to Crystal."""
        aims_out_path = Path.home() / "work/embed/aims-jobs/opt/taurine/aims.out"
        if not aims_out_path.exists():
            self.skipTest("Test aims.out file not found")

        output = AimsOutput.from_file(aims_out_path)
        crystal = output.final_structure.to_crystal()

        self.assertIsInstance(crystal, Crystal)
        uc_atoms = crystal.unit_cell_atoms()
        self.assertGreater(len(uc_atoms["element"]), 0)

    def test_aims_output_to_molecule(self):
        """Test converting final structure to Molecule for non-periodic."""
        aims_out_path = Path.home() / "work/embed/aims-jobs/opt/monomers/sulfamic/aims.out"
        if not aims_out_path.exists():
            self.skipTest("Test aims.out file not found")

        output = AimsOutput.from_file(aims_out_path)
        mol = output.final_structure.to_molecule()

        self.assertIsInstance(mol, Molecule)
        self.assertEqual(len(mol.elements), 8)  # Sulfamic acid has 8 atoms

    def test_aims_output_synthetic(self):
        """Test parsing synthetic aims.out content."""
        synthetic_output = """
------------------------------------------------------------
          Invoking FHI-aims ...
------------------------------------------------------------

  | Number of atoms                   :        3

  | Total energy uncorrected      :         -0.760000000000000E+02 eV

  Self-consistency cycle converged.

  Geometry optimization: Attempting to predict improved coordinates.

  || Forces on atoms   || =   0.100000E-01 eV/A.
  Maximum force component is  0.100000E-01 eV/A.
  Present geometry is not yet converged.

  Updated atomic structure:
                         x [A]             y [A]             z [A]
            atom         0.00000000        0.00000000        0.00000000  O
            atom         0.96000000        0.00000000        0.00000000  H
            atom        -0.24000000        0.93000000        0.00000000  H
------------------------------------------------------------

  | Total energy uncorrected      :         -0.760500000000000E+02 eV

  Self-consistency cycle converged.

  Geometry optimization: Attempting to predict improved coordinates.

  || Forces on atoms   || =   0.400000E-02 eV/A.
  Maximum force component is  0.400000E-02 eV/A.
  Present geometry is converged.

  Final atomic structure:
                         x [A]             y [A]             z [A]
            atom         0.00000000        0.00000000        0.00000000  O
            atom         0.95700000        0.00000000        0.00000000  H
            atom        -0.24000000        0.92800000        0.00000000  H
------------------------------------------------------------

  | Total energy uncorrected      :         -0.760510000000000E+02 eV
"""

        output = AimsOutput.from_string(synthetic_output)

        self.assertFalse(output.is_periodic)
        self.assertTrue(output.is_optimization)
        self.assertTrue(output.converged)
        self.assertEqual(output.n_atoms, 3)
        self.assertEqual(output.n_steps, 1)
        self.assertIsNotNone(output.final_structure)
        self.assertEqual(len(output.final_structure.elements), 3)
        self.assertAlmostEqual(output.final_structure.positions[1][0], 0.957, places=3)


if __name__ == "__main__":
    unittest.main()

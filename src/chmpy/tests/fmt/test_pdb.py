from chmpy.fmt.pdb import Pdb


class TestPdb:
    def test_pdb_write_simple(self):
        """Test basic PDB writing functionality."""
        test_pdb_content = """HEADER    Test structure
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
ATOM      1  C   MOL A   1       1.000   2.000   3.000  1.00 20.00           C
ATOM      2  N   MOL A   1       4.000   5.000   6.000  1.00 25.00           N
END"""

        pdb = Pdb.from_string(test_pdb_content)
        output = pdb.to_string()

        assert "HEADER    Test structure" in output
        assert "CRYST1" in output
        assert "ATOM      1 C    MOL A   1" in output
        assert "ATOM      2 N    MOL A   1" in output
        assert output.endswith("END")

    def test_pdb_roundtrip(self, tmp_path):
        """Test PDB read-write roundtrip."""
        test_pdb_content = """HEADER    Test structure
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
ATOM      1  C   MOL A   1       1.000   2.000   3.000  1.00 20.00           C
END"""

        pdb = Pdb.from_string(test_pdb_content)

        # Write to file
        test_file = tmp_path / "test.pdb"
        pdb.to_file(str(test_file))

        # Read back
        pdb2 = Pdb.from_file(str(test_file))

        assert pdb2.unit_cell["a"] == 10.0
        assert pdb2.unit_cell["b"] == 10.0
        assert pdb2.unit_cell["c"] == 10.0
        assert len(pdb2.atoms["serial"]) == 1
        assert pdb2.atoms["name"][0] == "C"
        assert pdb2.atoms["x"][0] == 1.0

    def test_crystal_line_formatting(self):
        """Test CRYST1 line formatting."""
        pdb = Pdb({})
        pdb.unit_cell = {
            "a": 10.123,
            "b": 20.456,
            "c": 30.789,
            "alpha": 90.0,
            "beta": 95.5,
            "gamma": 105.2,
        }
        pdb.space_group = "P 1"
        pdb.z = 2

        cryst_line = pdb.format_crystal_line()
        assert cryst_line.startswith("CRYST1")
        assert "10.123" in cryst_line
        assert "20.456" in cryst_line
        assert "30.789" in cryst_line
        assert "P 1" in cryst_line

    def test_atom_line_formatting(self):
        """Test ATOM line formatting."""
        pdb = Pdb({})
        pdb.atoms = {
            "serial": [1],
            "name": ["C"],
            "alt_loc": [" "],
            "res_name": ["MOL"],
            "chain_id": ["A"],
            "res_seq": [1],
            "icode": [" "],
            "x": [1.234],
            "y": [2.567],
            "z": [3.890],
            "occupancy": [1.00],
            "temp_factor": [20.00],
            "element": ["C"],
            "charge": [0.0],
        }

        atom_line = pdb.format_atom_line(0)
        assert atom_line.startswith("ATOM  ")
        assert "1.234" in atom_line
        assert "2.567" in atom_line
        assert "3.890" in atom_line
        assert "MOL" in atom_line

    def test_from_crystal(self):
        """Test creating PDB from Crystal object."""
        import numpy as np

        from chmpy.core.element import Element
        from chmpy.crystal.asymmetric_unit import AsymmetricUnit
        from chmpy.crystal.crystal import Crystal
        from chmpy.crystal.space_group import SpaceGroup
        from chmpy.crystal.unit_cell import UnitCell

        # Create a simple cubic crystal with one carbon atom
        unit_cell = UnitCell.from_lengths_and_angles(
            [10.0, 10.0, 10.0], [90.0, 90.0, 90.0], unit="degrees"
        )
        space_group = SpaceGroup.from_symbol("P 1")
        positions = np.array([[0.5, 0.5, 0.5]])  # Center of unit cell
        elements = [Element.from_atomic_number(6)]  # Carbon
        asym_unit = AsymmetricUnit(elements, positions)

        crystal = Crystal(unit_cell, space_group, asym_unit)

        # Convert to PDB
        pdb = Pdb.from_crystal(crystal, header="Test crystal")

        assert pdb.header == "Test crystal"
        assert pdb.unit_cell["a"] == 10.0
        assert pdb.unit_cell["b"] == 10.0
        assert pdb.unit_cell["c"] == 10.0
        assert len(pdb.atoms["serial"]) == 1
        assert pdb.atoms["element"][0] == "C"
        assert pdb.atoms["res_name"][0] == "UNL"

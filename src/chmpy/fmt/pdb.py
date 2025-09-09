from pathlib import Path


class Pdb:
    """
    Very basic PDB parser for crystallographic and atomic data.
    """

    def __init__(self, pdb_data):
        self.data = pdb_data
        self.content_lines = []
        self.header = None
        self.unit_cell = None
        self.space_group = None
        self.z = None
        self.atoms = None
        self.line_index = 0

    def parse_crystal_line(self, line):
        """Parse CRYST1 record containing unit cell parameters."""
        line_length = len(line)
        self.unit_cell = {
            "a": float(line[6:14]),
            "b": float(line[15:23]),
            "c": float(line[24:32]),
            "alpha": float(line[33:39]),
            "beta": float(line[40:46]),
            "gamma": float(line[47:53]),
        }
        self.space_group = "P 1"  # default
        if line_length > 55:
            self.space_group = line[55 : min(line_length, 65)].strip()
        self.z = 1  # default
        if line_length > 66:
            z_str = line[66 : min(line_length, 69)].strip()
            if z_str:
                self.z = int(z_str)

    def parse_atom_lines(self):
        """Parse ATOM and HETATM records."""
        # Initialize atom data structure
        self.atoms = {
            "serial": [],
            "name": [],
            "alt_loc": [],
            "res_name": [],
            "chain_id": [],
            "res_seq": [],
            "icode": [],
            "x": [],
            "y": [],
            "z": [],
            "occupancy": [],
            "temp_factor": [],
            "element": [],
            "charge": [],
        }

        # Process all atom lines until we hit a non-atom record or end of file
        total_lines = len(self.content_lines)

        while self.line_index < total_lines:
            line = self.content_lines[self.line_index]

            # Check for MODEL/ENDMDL records (handle multiple models)
            if line.startswith("MODEL "):
                self.line_index += 1
                continue
            elif line.startswith("ENDMDL") or line.startswith("END"):
                self.line_index += 1
                break

            # Process atom records
            if line[:6] in ("ATOM  ", "HETATM"):
                try:
                    self.atoms["serial"].append(int(line[6:11]))
                    self.atoms["name"].append(line[12:16].strip())
                    self.atoms["alt_loc"].append(line[16])
                    self.atoms["res_name"].append(line[17:20].strip())
                    self.atoms["chain_id"].append(line[21])
                    self.atoms["res_seq"].append(int(line[22:26]))
                    self.atoms["icode"].append(line[26])
                    self.atoms["x"].append(float(line[30:38]))
                    self.atoms["y"].append(float(line[38:46]))
                    self.atoms["z"].append(float(line[46:54]))
                    self.atoms["occupancy"].append(float(line[54:60]))
                    self.atoms["temp_factor"].append(float(line[60:66]))
                    self.atoms["element"].append(line[76:78].strip())

                    # Parse charge safely
                    chg = line[78:80].strip()
                    self.atoms["charge"].append(float(chg[::-1]) if chg else 0.0)
                except (ValueError, IndexError) as e:
                    # Handle malformed lines gracefully
                    print(
                        f"Warning: Error parsing atom at line {self.line_index + 1}: {e}"
                    )
            else:
                # If we hit a non-atom record, break the parsing loop
                break

            self.line_index += 1

    def parse_header(self):
        """Parse the PDB header to find crystallographic information."""
        total_lines = len(self.content_lines)

        while self.line_index < total_lines:
            line = self.content_lines[self.line_index]
            record_type = line[:6]

            if record_type == "CRYST1":
                self.parse_crystal_line(line)
                self.line_index += 1
            elif record_type == "HEADER":
                self.header = line[10:].strip()
                self.line_index += 1
            elif record_type in ("ATOM  ", "HETATM", "MODEL "):
                # We've reached the atom section
                break
            else:
                # Skip other header records
                self.line_index += 1

        # Debug info - print crystallographic data
        if self.unit_cell:
            print(self.unit_cell)
            print(self.space_group)
            print(self.z)

    def parse(self):
        """Parse the entire PDB contents."""
        self.line_index = 0

        # Parse header section
        self.parse_header()

        # Parse atom records
        self.parse_atom_lines()

        # Skip any remaining records
        while self.line_index < len(self.content_lines):
            self.line_index += 1

        return self.data

    def format_crystal_line(self):
        """Format CRYST1 record from unit cell parameters."""
        if not self.unit_cell:
            return None

        space_group = self.space_group or "P 1"
        z = self.z or 1

        return (
            f"CRYST1{self.unit_cell['a']:9.3f}{self.unit_cell['b']:9.3f}"
            f"{self.unit_cell['c']:9.3f}{self.unit_cell['alpha']:7.2f}"
            f"{self.unit_cell['beta']:7.2f}{self.unit_cell['gamma']:7.2f} "
            f"{space_group:<11}{z:4d}"
        )

    def format_atom_line(self, idx, record_type="ATOM"):
        """Format an ATOM or HETATM record."""
        if not self.atoms or idx >= len(self.atoms["serial"]):
            return None

        # Get atom data with defaults
        serial = self.atoms["serial"][idx]
        name = self.atoms["name"][idx]
        alt_loc = self.atoms["alt_loc"][idx] if self.atoms["alt_loc"][idx] else " "
        res_name = self.atoms["res_name"][idx]
        chain_id = self.atoms["chain_id"][idx] if self.atoms["chain_id"][idx] else " "
        res_seq = self.atoms["res_seq"][idx]
        icode = self.atoms["icode"][idx] if self.atoms["icode"][idx] else " "
        x = self.atoms["x"][idx]
        y = self.atoms["y"][idx]
        z = self.atoms["z"][idx]
        occupancy = (
            self.atoms["occupancy"][idx] if self.atoms["occupancy"][idx] else 1.00
        )
        temp_factor = (
            self.atoms["temp_factor"][idx] if self.atoms["temp_factor"][idx] else 0.00
        )
        element = self.atoms["element"][idx] if self.atoms["element"][idx] else ""
        charge = self.atoms["charge"][idx] if self.atoms["charge"][idx] else 0.0

        # Format charge
        charge_str = ""
        if charge != 0.0:
            charge_str = f"{abs(charge):.0f}{'+' if charge > 0 else '-'}"

        return (
            f"{record_type:<6}{serial:5d} {name:<4}{alt_loc}{res_name:>3} "
            f"{chain_id}{res_seq:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}{charge_str:>2}"
        )

    def to_string(self):
        """Convert the parsed PDB data back to a PDB format string."""
        lines = []

        # Add header if present
        if self.header:
            lines.append(f"HEADER    {self.header}")

        # Add crystal information
        cryst_line = self.format_crystal_line()
        if cryst_line:
            lines.append(cryst_line)

        # Add atom records
        if self.atoms and self.atoms["serial"]:
            for i in range(len(self.atoms["serial"])):
                atom_line = self.format_atom_line(i)
                if atom_line:
                    lines.append(atom_line)

        lines.append("END")
        return "\n".join(lines)

    def to_file(self, filename):
        """Write the PDB data to a file."""
        Path(filename).write_text(self.to_string())

    @classmethod
    def from_crystal(cls, crystal, header=None):
        """Initialize a PDB from a Crystal object."""
        pdb = cls({})

        # Set header
        pdb.header = header or "Crystal structure"

        # Set unit cell parameters
        import numpy as np

        pdb.unit_cell = {
            "a": crystal.unit_cell.a,
            "b": crystal.unit_cell.b,
            "c": crystal.unit_cell.c,
            "alpha": np.degrees(crystal.unit_cell.alpha),
            "beta": np.degrees(crystal.unit_cell.beta),
            "gamma": np.degrees(crystal.unit_cell.gamma),
        }

        # Keep the original space group and Z
        pdb.space_group = crystal.space_group.crystal17_spacegroup_symbol()
        pdb.z = 1  # Default for now

        # Use asymmetric unit atoms (not full unit cell)
        positions = crystal.to_cartesian(crystal.site_positions)
        atomic_numbers = crystal.site_atoms

        # Initialize atom data structure
        pdb.atoms = {
            "serial": [],
            "name": [],
            "alt_loc": [],
            "res_name": [],
            "chain_id": [],
            "res_seq": [],
            "icode": [],
            "x": [],
            "y": [],
            "z": [],
            "occupancy": [],
            "temp_factor": [],
            "element": [],
            "charge": [],
        }

        # Populate atom data
        for i, (pos, atomic_num) in enumerate(
            zip(positions, atomic_numbers, strict=False)
        ):
            from chmpy.core.element import Element

            element = Element.from_atomic_number(atomic_num)

            pdb.atoms["serial"].append(i + 1)
            pdb.atoms["name"].append(element.symbol)
            pdb.atoms["alt_loc"].append(" ")
            pdb.atoms["res_name"].append("UNL")  # Unknown ligand
            pdb.atoms["chain_id"].append("A")
            pdb.atoms["res_seq"].append(1)
            pdb.atoms["icode"].append(" ")
            pdb.atoms["x"].append(float(pos[0]))
            pdb.atoms["y"].append(float(pos[1]))
            pdb.atoms["z"].append(float(pos[2]))
            pdb.atoms["occupancy"].append(1.00)
            pdb.atoms["temp_factor"].append(0.00)
            pdb.atoms["element"].append(element.symbol)
            pdb.atoms["charge"].append(0.0)

        return pdb

    @classmethod
    def from_file(cls, filename):
        """Initialize a PDB parser from a file path."""
        return cls.from_string(Path(filename).read_text())

    @classmethod
    def from_string(cls, contents):
        """Initialize a PDB parser from string contents."""
        pdb = cls({})
        pdb.content_lines = [line.strip() for line in contents.split("\n")]
        pdb.parse()
        return pdb

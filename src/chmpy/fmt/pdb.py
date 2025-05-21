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
            self.z = int(line[66 : min(line_length, 69)])

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

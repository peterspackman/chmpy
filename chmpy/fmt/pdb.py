import numpy as np
from pathlib import Path

class Pdb:
    "Very basic PDB parser"
    def __init__(self, pdb_data):
        self.data = pdb_data
        self.content_lines = []
        self.header = None

    def parse_crystal_line(self, line):
        line_length = len(line)
        self.unit_cell = {
            "a": float(line[6:14]),
            "b": float(line[15:23]),
            "c": float(line[24:32]),
            "alpha": float(line[33:39]),
            "beta": float(line[40:46]),
            "gamma": float(line[47:53]),
        }

        self.space_group = "P 1"

        if line_length > 55:
            self.space_group = line[55:min(line_length, 65)].strip()

        self.z = 1
        if line_length > 66:
            self.z = int(line[66:min(line_length, 69)])

    def parse_atom_lines(self):
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
            "charge": []
        }

        # TODO handle MODEL/ENDMDL
        for self.line_index in range(self.line_index, len(self.content_lines)):
            line = self.content_lines[self.line_index]
            if line[:6] in ("ATOM  ", "HETATM"):
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
                self.atoms["element"].append(line[76:78])
                chg = line[78:80].strip()
                chg = float(chg[::-1]) if chg else 0.0
                self.atoms["charge"].append(chg)


    def parse_header(self):
        for self.line_index, line in enumerate(self.content_lines):
            record_type = line[:6]
            if record_type == "CRYST1":
                self.parse_crystal_line(line)
            if record_type in ("ATOM  ", "HETATM", "MODEL "):
                break
        print(self.unit_cell)
        print(self.space_group)
        print(self.z)
        self.parse_atom_lines()

    def parse(self):
        "parse the entire PDB contents"
        self.line_index = 0
        line_count = len(self.content_lines)
        self.parse_header()

        while self.line_index < line_count:
            line = self.content_lines[self.line_index].strip()
            self.line_index += 1
        return self.data

    @classmethod
    def from_file(cls, filename):
        "initialize a :obj:`Cif` from a file path"
        return cls.from_string(Path(filename).read_text())

    @classmethod
    def from_string(cls, contents):
        "initialize a :obj:`Cif` from string contents"
        c = cls({})
        c.content_lines = contents.split("\n")
        c.parse()
        return c


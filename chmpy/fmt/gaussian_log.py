import numpy as np
import logging

LOG = logging.getLogger(__name__)


class GaussianLogFile:
    optimized_tag = "Optimized Parameters"
    std_xyz_tag = "Standard orientation"
    inp_xyz_tag = "Input orientation"

    def __init__(self, filename=None):
        self.filename = filename
        if self.filename is not None:
            self.read_contents()

    def read_contents(self):
        with open(self.filename) as f:
            self.content_lines = f.readlines()

    @property
    def excitations(self):
        import re

        excitation_regex = re.compile(
            r"Excited State\s+(\d+):\s+([\w\d\-\.]+)\s+([+1]?[\d\.]+)"
            r"\s+eV\s+([+-]?[\d\.]+)\s+nm\s+f=([\d\.]+)\s+<S\*\*2>=([\d\.]+)"
        )
        matches = excitation_regex.findall("\n".join(self.content_lines))
        excitations = {
            "n": [],
            "A": [],
            "eV": [],
            "nm": [],
            "f": [],
            "S2": [],
        }
        for i, A, eV, nm, f, s2 in matches:
            excitations["n"].append(int(i))
            excitations["A"].append(A)
            excitations["eV"].append(float(eV))
            excitations["nm"].append(float(nm))
            excitations["f"].append(float(f))
            excitations["S2"].append(float(s2))
        return {k: np.array(v) for k, v in excitations.items()}

    @property
    def geometries(self):
        if hasattr(self, "_geometries"):
            return self._geometries
        lines_iter = iter(self.content_lines)
        parsing_xyz = None
        geometries = {"inp": [], "std": []}
        for line in lines_iter:
            # if optimized.. not found and we are not looking for original,
            # loop here.
            if self.inp_xyz_tag in line:
                # skip 4 lines
                for _ in range(4):
                    next(lines_iter)
                    parsing_xyz = "inp"
            elif self.std_xyz_tag in line:
                # skip 4 lines
                for _ in range(4):
                    next(lines_iter)
                    parsing_xyz = "std"
            if parsing_xyz:
                atomic_numbers = []
                positions = []
                while True:
                    next_line = next(lines_iter)
                    if "----" in next_line:
                        break
                    tokens = next_line.strip().split()
                    atomic_numbers.append(int(tokens[1]))
                    positions.append(tuple(map(float, tokens[3:6])))
                geometries[parsing_xyz].append(
                    {
                        "elements": np.asarray(atomic_numbers),
                        "positions": np.asarray(positions),
                    }
                )
                parsing_xyz = None
        self._geometries = geometries
        return geometries

    @property
    def final_geometry(self, kind="inp"):
        return self.geometries[kind][-1]

    def parse_archive_block(self):
        block_lines = []
        adding_lines = False
        for line in self.content_lines:
            # can't end raw string with backslash
            if adding_lines:
                block_lines.append(line.strip())
            if r"1\1" in line:
                adding_lines = True
                block_lines.append(line.strip())
            if r"\\@" in line:
                break
        self._archive_block_sections = "".join(block_lines).split("\\")
        for section in self._archive_block_sections:
            if "=" in section:
                tokens = section.split("=")
                if len(tokens) == 2:
                    l, r = tokens
                    setattr(self, l, r)

    @property
    def final_energy(self):
        if not hasattr(self, "_archive_block_sections"):
            self.parse_archive_block()
        return float(getattr(self, "HF", "nan"))

    @property
    def dipole(self):
        if not hasattr(self, "_archive_block_sections"):
            self.parse_archive_block()
        return np.array(
            [float(x) for x in getattr(self, "Dipole", "nan,nan,nan").split(",")]
        )

    @property
    def quadrupole(self):
        if not hasattr(self, "_archive_block_sections"):
            self.parse_archive_block()
        return np.array(
            [
                float(x)
                for x in getattr(self, "Quadrupole", "nan,nan,nan,nan,nan,nan").split(
                    ","
                )
            ]
        )

    @property
    def normal_termination(self):
        return "Normal termination" in self.content_lines[-1]

    @classmethod
    def from_string(cls, contents):
        res = cls()
        res.content_lines = contents.splitlines()
        return res

import numpy as np
from io import StringIO


class FchkFile:

    UNIT_CONVERSIONS = {"kj/mol": 2625.499638, "hartree": 1.0}
    CONVERSIONS = {"R": float, "I": int, "C": str}

    def __init__(self, file_content, parse=False):
        self.filename = "string"
        self.file_content = file_content
        if parse:
            self._parse()
        self._buf = StringIO(file_content)

    def _parse(self):
        energy = 0.0
        contents = {}
        with StringIO(self.file_content) as f:
            contents["header"] = (f.readline(), f.readline())
            line = f.readline()

            while line:
                name = line[:43].strip()
                kind = line[43]
                num = 1
                if line[47:49] == "N=":
                    num = int(line[49:].strip())
                    value = []
                    line = f.readline()
                    valid = True
                    while valid:
                        tokens = line.strip().split()
                        convert = self.CONVERSIONS[kind]
                        value += [convert(x) for x in tokens]
                        line = f.readline()
                        valid = line and line[0] == " "
                else:
                    value = self.CONVERSIONS[kind](line[48:].strip())
                    line = f.readline()
                contents[name] = value
        self.contents = contents

    @classmethod
    def from_file(cls, filename, parse=False):
        from pathlib import Path

        contents = Path(filename).read_text()
        fchk = cls(contents)
        fchk.filename = filename
        if parse:
            fchk._parse()
        return fchk

    def __getitem__(self, key):
        return self.contents[key]

    def _parse_energy_only(self):
        with open(self.filename) as f:
            for line in f:
                if line.startswith("Total Energy"):
                    return float(line[49:].strip())

    @classmethod
    def scf_energy(cls, fname, units="hartree"):
        fchk = cls(fname, parse=False)
        return cls.UNIT_CONVERSIONS[units] * fchk._parse_energy_only()

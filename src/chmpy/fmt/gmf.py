from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GMF:
    hkl: np.ndarray
    cuts: np.ndarray
    energies: np.ndarray

    @classmethod
    def from_file(cls, filename):
        lines = Path(filename).read_text().splitlines()
        start_line = 0
        for i, line in enumerate(lines):
            start_line = i
            if not line:
                continue
            if "miller: " in line:
                break

        hkls = []
        energies = []
        cuts = []
        for j in range(start_line, len(lines), 2):
            merged = lines[j] + lines[j + 1]
            tokens = merged.split()
            hkls.append(tuple(int(x) for x in tokens[1:4]))
            energies.append(float(tokens[7]))
            cuts.append(float(tokens[4]))

        return cls(np.array(hkls), np.array(energies), np.array(cuts))

from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class GMF:
    hkl: np.ndarray
    cuts: np.ndarray
    energies: np.ndarray

    @classmethod
    def from_file(cls, filename):
        lines = Path(filename).read_text().splitlines()
        for i, line in enumerate(lines):
            if not line: 
                continue
            if "miller: " in line:
                break

        hkls = []
        energies = []
        cuts = []
        for i in range(i, len(lines), 2):
            merged = lines[i] + lines[i+1]
            tokens = merged.split()
            hkls.append(tuple(int(x) for x in tokens[1:4]))
            energies.append(float(tokens[7]))
            cuts.append(float(tokens[4]))

        return cls(np.array(hkls), np.array(energies), np.array(cuts))
        

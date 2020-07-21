import logging
from pathlib import Path
from io import StringIO
import numpy as np

LOG = logging.getLogger(__name__)


class PhonopyAscii:
    def __init__(self, filename=None):
        if filename is not None:
            self._parse_file(filename)

    def _parse_buf(self, buf):
        from chmpy import Element

        lines = buf.read().splitlines()
        latt = np.zeros((3, 3))
        a, b1, b2 = (float(x) for x in lines[1].split())
        c1, c2, c3 = (float(x) for x in lines[2].split())
        latt[0, 0] = a
        latt[1, 0:2] = b1, b2
        latt[2, :] = c1, c2, c3
        LOG.debug("Read lattice: %s", latt)
        i = 3
        positions = []
        elements = []
        line = lines[i]
        while not line.startswith("#"):
            x, y, z, el = line.split()
            positions.append((float(x), float(y), float(z)))
            elements.append(el)
            i += 1
            line = lines[i]
        positions = np.array(positions)
        elements = [Element[x] for x in elements]
        metadata = "".join(lines[i:]).replace(" \\#", "").replace(";", ",")
        for mode in metadata.split("#metaData: qpt=[")[1:]:
            arr = np.fromstring(mode.rstrip("]"), sep=",")
            qpoint = arr[:3]
            eigenvalue = arr[3] ** 2
            arr = arr[4:].reshape((-1, 6))
        self.displacements = arr
        self.elements = elements
        self.positions = positions
        self.qpoints = np.repeat(qpoint[np.newaxis, :], len(elements), axis=0)

    @classmethod
    def from_string(cls, string):
        cube = cls()
        cube._parse_buf(StringIO(string))
        return cube

    def _parse_file(self, filename):
        with Path(filename).open() as f:
            self._parse_buf(f)

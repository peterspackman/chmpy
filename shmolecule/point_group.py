from typing import Tuple
from dataclasses import dataclass

@dataclass
class PointGroup:
    table: Tuple[int]
    symbol: str
    schoenflies: str
    crystal_system: str
    laue: str

    def __repr__(self):
        return f"<PointGroup: {self.symbol}>"


POINT_GROUP_DATA = (
    None,
    PointGroup((0, 0, 0, 0, 0, 1, 0, 0, 0, 0), "1", "C1", "triclinic", "-1",),
    PointGroup((0, 0, 0, 0, 1, 1, 0, 0, 0, 0), "-1", "Ci", "triclinic", "-1",),
    PointGroup((0, 0, 0, 0, 0, 1, 1, 0, 0, 0), "2", "C2", "monoclinic", "2/m",),
    PointGroup((0, 0, 0, 1, 0, 1, 0, 0, 0, 0), "m", "Cs", "monoclinic", "2/m",),
    PointGroup((0, 0, 0, 1, 1, 1, 1, 0, 0, 0), "2/m", "C2h", "monoclinic", "2/m",),
    PointGroup((0, 0, 0, 0, 0, 1, 3, 0, 0, 0), "222", "D2", "orthorhombic", "mmm",),
    PointGroup((0, 0, 0, 2, 0, 1, 1, 0, 0, 0), "mm2", "C2v", "orthorhombic", "mmm",),
    PointGroup((0, 0, 0, 3, 1, 1, 3, 0, 0, 0), "mmm", "D2h", "orthorhombic", "mmm",),
    PointGroup((0, 0, 0, 0, 0, 1, 1, 0, 2, 0), "4", "C4", "tetragonal", "4/m",),
    PointGroup((0, 2, 0, 0, 0, 1, 1, 0, 0, 0), "-4", "S4", "tetragonal", "4/m",),
    PointGroup((0, 2, 0, 1, 1, 1, 1, 0, 2, 0), "4/m", "C4h", "tetragonal", "4/m",),
    PointGroup((0, 0, 0, 0, 0, 1, 5, 0, 2, 0), "422", "D4", "tetragonal", "4/mmm",),
    PointGroup((0, 0, 0, 4, 0, 1, 1, 0, 2, 0), "4mm", "C4v", "tetragonal", "4/mmm",),
    PointGroup((0, 2, 0, 2, 0, 1, 3, 0, 0, 0), "-42m", "D2d", "tetragonal", "4/mmm",),
    PointGroup((0, 2, 0, 5, 1, 1, 5, 0, 2, 0), "4/mmm", "D4h", "tetragonal", "4/mmm",),
    PointGroup((0, 0, 0, 0, 0, 1, 0, 2, 0, 0), "3", "C3", "trigonal", "-3",),
    PointGroup((0, 0, 2, 0, 1, 1, 0, 2, 0, 0), "-3", "C3i", "trigonal", "-3",),
    PointGroup((0, 0, 0, 0, 0, 1, 3, 2, 0, 0), "32", "D3", "trigonal", "-3m",),
    PointGroup((0, 0, 0, 3, 0, 1, 0, 2, 0, 0), "3m", "C3v", "trigonal", "-3m",),
    PointGroup((0, 0, 2, 3, 1, 1, 3, 2, 0, 0), "-3m", "D3d", "trigonal", "-3m",),
    PointGroup((0, 0, 0, 0, 0, 1, 1, 2, 0, 2), "6", "C6", "hexagonal", "6/m",),
    PointGroup((2, 0, 0, 1, 0, 1, 0, 2, 0, 0), "-6", "C3h", "hexagonal", "6/m",),
    PointGroup((2, 0, 2, 1, 1, 1, 1, 2, 0, 2), "6/m", "C6h", "hexagonal", "6/m",),
    PointGroup((0, 0, 0, 0, 0, 1, 7, 2, 0, 2), "622", "D6", "hexagonal", "6/mmm",),
    PointGroup((0, 0, 0, 6, 0, 1, 1, 2, 0, 2), "6mm", "C6v", "hexagonal", "6/mmm",),
    PointGroup((2, 0, 0, 4, 0, 1, 3, 2, 0, 0), "-6m2", "D3h", "hexagonal", "6/mmm",),
    PointGroup((2, 0, 2, 7, 1, 1, 7, 2, 0, 2), "6/mmm", "D6h", "hexagonal", "6/mmm",),
    PointGroup((0, 0, 0, 0, 0, 1, 3, 8, 0, 0), "23", "T", "cubic", "m3",),
    PointGroup((0, 0, 8, 3, 1, 1, 3, 8, 0, 0), "m-3", "Th", "cubic", "m3",),
    PointGroup((0, 0, 0, 0, 0, 1, 9, 8, 6, 0), "432", "O", "cubic", "m3m",),
    PointGroup((0, 6, 0, 6, 0, 1, 3, 8, 0, 0), "-43m", "Td", "cubic", "m3m",),
    PointGroup((0, 6, 8, 9, 1, 1, 9, 8, 6, 0), "m-3m", "Oh", "cubic", "m3m",),
)

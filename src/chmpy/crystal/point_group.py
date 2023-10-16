from typing import Tuple
from dataclasses import dataclass
from chmpy.crystal.symmetry_operation import SymmetryOperation


@dataclass
class PointGroup:
    number: int
    symops_string: str
    symbol: str
    schoenflies: str
    crystal_system: str
    laue_group: str
    choice: str

    def __repr__(self):
        return f"<PointGroup: {self.symbol}>"

    @property
    def symmetry_operations(self):
        return [
            SymmetryOperation.from_string_code(x) for x in self.symops_string.split(";")
        ]

    @classmethod
    def from_number(cls, number, choice=None):
        if number < 1 or number > 32:
            raise ValueError("Point group number must be between [1, 32]")
        options = POINT_GROUP_FROM_NUMBER[number]
        if not choice:
            return options[0]
        else:
            for option in options:
                if choice == option.choice:
                    return option
                    break
            else:
                raise ValueError(
                    f"Could not find choice '{choice}' for point group {number}"
                )


POINT_GROUP_DATA = (
    PointGroup(1, "x,y,z", "1", "C1", "triclinic", "-1", ""),
    PointGroup(2, "-x,-y,-z", "-1", "Ci", "triclinic", "-1", ""),
    PointGroup(3, "-x,y,-z", "2", "C2", "monoclinic", "2/m", "b"),
    PointGroup(3, "-x,-y,z", "112", "C2", "monoclinic", "2/m", "c"),
    PointGroup(4, "x,-y,z", "m", "Cs", "monoclinic", "2/m", "b"),
    PointGroup(4, "x,y,-z", "11m", "Cs", "monoclinic", "2/m", "c"),
    PointGroup(5, "-x,y,-z; -x,-y,-z", "2/m", "C2h", "monoclinic", "2/m", "b"),
    PointGroup(5, "-x,-y,z; x,y,-z", "112/m", "C2h", "monoclinic", "2/m", "c"),
    PointGroup(6, "-x,-y,z; x,-y,-z", "222", "D2", "orthorhombic", "mmm", ""),
    PointGroup(7, "-x,-y,z; -x,y,z", "mm2", "C2v", "orthorhombic", "mmm", "mm2"),
    PointGroup(7, "x,-y,-z; x,-y,z", "2mm", "C2v", "orthorhombic", "mmm", "2mm"),
    PointGroup(7, "-x,y,-z; x,y,-z", "m2m", "C2v", "orthorhombic", "mmm", "m2m"),
    PointGroup(8, "-x,-y,-z;-x,-y,z;x,-y,-z", "mmm", "D2h", "orthorhombic", "mmm", ""),
    PointGroup(9, "-y,x,z", "4", "C4", "tetragonal", "4/m", ""),
    PointGroup(10, "y,-x,-z", "-4", "S4", "tetragonal", "4/m", ""),
    PointGroup(11, "-y,x,z;-x,-y,-z", "4/m", "C4h", "tetragonal", "4/m", ""),
    PointGroup(12, "-y,x,z; x,-y,-z", "422", "D4", "tetragonal", "4/mmm", ""),
    PointGroup(13, "-y,x,z; -x,y,z", "4mm", "C4v", "tetragonal", "4/mmm", ""),
    PointGroup(14, "y,-x,-z; x,-y,-z", "-42m", "D2d", "tetragonal", "4/mmm", "-42m"),
    PointGroup(14, "y,-x,-z; y,x,-z", "-4m2", "D2d", "tetragonal", "4/mmm", "-4m2"),
    PointGroup(
        15, "-y,x,z; x,y,-z; x,-y,-z", "4/mmm", "D4h", "tetragonal", "4/mmm", ""
    ),
    PointGroup(16, "-y,x-y,z", "3", "C3", "trigonal", "-3", "H"),
    PointGroup(16, "z,x,y", "3r", "C3", "trigonal", "-3", "R"),
    PointGroup(17, "y,y-x,-z", "-3", "C3i", "trigonal", "-3", "H"),
    PointGroup(17, "-z,-x,-y", "-3r", "C3i", "trigonal", "-3", "R"),
    PointGroup(18, "-y,x-y,z; x-y,-y,-z", "32", "D3", "trigonal", "-3m", "32"),
    PointGroup(18, "-y,x-y,z; x-y,-y,-z", "321", "D3", "trigonal", "-3m", "321"),
    PointGroup(18, "-y,x-y,z; x,x-y,-z", "3m1", "D3", "trigonal", "-3m", "3m1"),
    PointGroup(18, "z,x,y; -y,-x,-z", "32r", "D3", "trigonal", "-3m", "R"),
    PointGroup(19, "-y,x-y,z; y-x,y,z", "3m", "C3v", "trigonal", "-3m", "3m"),
    PointGroup(19, "-y,x-y,z; y-x,y,z", "3m1", "C3v", "trigonal", "-3m", "3m1"),
    PointGroup(19, " -y,x-y,z; -x,y-x,z", "31m", "C3v", "trigonal", "-3m", "31m"),
    PointGroup(19, "z,x,y; y,x,z", "3mr", "C3v", "trigonal", "-3m", "R"),
    PointGroup(20, "y,y-x,-z; x-y,-y,-z", "-3m", "D3d", "trigonal", "-3m", "-3m"),
    PointGroup(20, "y,y-x,-z; x-y,-y,-z", "-3m1", "D3d", "trigonal", "-3m", "-3m1"),
    PointGroup(20, "y,y-x,-z; x,x-y,-z", "-31m", "D3d", "trigonal", "-3m", "-31m"),
    PointGroup(20, "-z,-x,-y; y,x,z", "-3m", "D3d", "trigonal", "-3m", "R"),
    PointGroup(21, "x-y,x,z", "6", "C6", "hexagonal", "6/m", ""),
    PointGroup(22, "y-x,-x,-z", "-6", "C3h", "hexagonal", "6/m", ""),
    PointGroup(23, "x-y,x,z; -x,-y,-z", "6/m", "C6h", "hexagonal", "6/m", ""),
    PointGroup(24, "x-y,x,z; x-y,-y,-z", "622", "D6", "hexagonal", "6/mmm", ""),
    PointGroup(25, "x-y,x,z; y-x,y,z", "6mm", "C6v", "hexagonal", "6/mmm", ""),
    PointGroup(26, "y-x,-x,-z; y-x,y,z", "-6m2", "D3h", "hexagonal", "6/mmm", "-6m2"),
    PointGroup(26, "y-x,-x,-z; x-y,-y,-z", "-62m", "D3h", "hexagonal", "6/mmm", "-62m"),
    PointGroup(
        27, "x-y,x,z; x-y,-y,-z; -x,-y,-z", "6/mmm", "D6h", "hexagonal", "6/mmm", ""
    ),
    PointGroup(28, "z,x,y; -x,-y,z; x,-y,-z", "23", "T", "cubic", "m3", ""),
    PointGroup(29, "-z,-x,-y; -x,-y,z; x,-y,-z", "m-3", "Th", "cubic", "m3", ""),
    PointGroup(30, "z,x,y; -y,x,z; x,-y,-z", "432", "O", "cubic", "m3m", ""),
    PointGroup(31, "z,x,y; y,-x,-z; -y,-x,z", "-43m", "Td", "cubic", "m3m", ""),
    PointGroup(32, "-z,-x,-y; -y,x,z; y,x,-z", "m-3m", "Oh", "cubic", "m3m", ""),
)

POINT_GROUP_FROM_NUMBER = {
    i: [x for x in POINT_GROUP_DATA if x.number == i] for i in range(1, 33)
}

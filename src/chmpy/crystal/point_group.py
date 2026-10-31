from dataclasses import dataclass

from chmpy.crystal.hall_symbol import parse_hall_symbol
from chmpy.crystal.symmetry_operation import SymmetryOperation


@dataclass
class PointGroup:
    number: int
    symbol: str
    schoenflies: str
    crystal_system: str
    laue_group: str
    choice: str
    hall: str

    def __repr__(self):
        return f"<PointGroup: {self.symbol}>"

    @property
    def symmetry_operations(self) -> list[SymmetryOperation]:
        "The operations of this point group, in its crystallographic setting"
        return parse_hall_symbol(self.hall)

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
    PointGroup(1, "1", "C1", "triclinic", "-1", "", "P 1"),
    PointGroup(2, "-1", "Ci", "triclinic", "-1", "", "-P 1"),
    PointGroup(3, "2", "C2", "monoclinic", "2/m", "b", "P 2y"),
    PointGroup(3, "112", "C2", "monoclinic", "2/m", "c", "P 2"),
    PointGroup(4, "m", "Cs", "monoclinic", "2/m", "b", "P -2y"),
    PointGroup(4, "11m", "Cs", "monoclinic", "2/m", "c", "P -2"),
    PointGroup(5, "2/m", "C2h", "monoclinic", "2/m", "b", "-P 2y"),
    PointGroup(5, "112/m", "C2h", "monoclinic", "2/m", "c", "-P 2"),
    PointGroup(6, "222", "D2", "orthorhombic", "mmm", "", "P 2 2"),
    PointGroup(7, "mm2", "C2v", "orthorhombic", "mmm", "mm2", "P 2 -2"),
    PointGroup(7, "2mm", "C2v", "orthorhombic", "mmm", "2mm", "P -2 2"),
    PointGroup(7, "m2m", "C2v", "orthorhombic", "mmm", "m2m", "P -2 -2"),
    PointGroup(8, "mmm", "D2h", "orthorhombic", "mmm", "", "-P 2 2"),
    PointGroup(9, "4", "C4", "tetragonal", "4/m", "", "P 4"),
    PointGroup(10, "-4", "S4", "tetragonal", "4/m", "", "P -4"),
    PointGroup(11, "4/m", "C4h", "tetragonal", "4/m", "", "-P 4"),
    PointGroup(12, "422", "D4", "tetragonal", "4/mmm", "", "P 4 2"),
    PointGroup(13, "4mm", "C4v", "tetragonal", "4/mmm", "", "P 4 -2"),
    PointGroup(14, "-42m", "D2d", "tetragonal", "4/mmm", "-42m", "P -4 2"),
    PointGroup(14, "-4m2", "D2d", "tetragonal", "4/mmm", "-4m2", "P -4 -2"),
    PointGroup(15, "4/mmm", "D4h", "tetragonal", "4/mmm", "", "-P 4 2"),
    PointGroup(16, "3", "C3", "trigonal", "-3", "H", "P 3"),
    PointGroup(16, "3r", "C3", "trigonal", "-3", "R", "P 3*"),
    PointGroup(17, "-3", "C3i", "trigonal", "-3", "H", "-P 3"),
    PointGroup(17, "-3r", "C3i", "trigonal", "-3", "R", "-P 3*"),
    PointGroup(18, "32", "D3", "trigonal", "-3m", "32", 'P 3 2"'),
    PointGroup(18, "321", "D3", "trigonal", "-3m", "321", 'P 3 2"'),
    PointGroup(18, "312", "D3", "trigonal", "-3m", "312", "P 3 2"),
    PointGroup(18, "32r", "D3", "trigonal", "-3m", "R", "P 3* 2"),
    PointGroup(19, "3m", "C3v", "trigonal", "-3m", "3m", 'P 3 -2"'),
    PointGroup(19, "3m1", "C3v", "trigonal", "-3m", "3m1", 'P 3 -2"'),
    PointGroup(19, "31m", "C3v", "trigonal", "-3m", "31m", "P 3 -2"),
    PointGroup(19, "3mr", "C3v", "trigonal", "-3m", "R", "P 3* -2"),
    PointGroup(20, "-3m", "D3d", "trigonal", "-3m", "-3m", '-P 3 2"'),
    PointGroup(20, "-3m1", "D3d", "trigonal", "-3m", "-3m1", '-P 3 2"'),
    PointGroup(20, "-31m", "D3d", "trigonal", "-3m", "-31m", "-P 3 2"),
    PointGroup(20, "-3m", "D3d", "trigonal", "-3m", "R", "-P 3* 2"),
    PointGroup(21, "6", "C6", "hexagonal", "6/m", "", "P 6"),
    PointGroup(22, "-6", "C3h", "hexagonal", "6/m", "", "P -6"),
    PointGroup(23, "6/m", "C6h", "hexagonal", "6/m", "", "-P 6"),
    PointGroup(24, "622", "D6", "hexagonal", "6/mmm", "", "P 6 2"),
    PointGroup(25, "6mm", "C6v", "hexagonal", "6/mmm", "", "P 6 -2"),
    PointGroup(26, "-6m2", "D3h", "hexagonal", "6/mmm", "-6m2", "P -6 2"),
    PointGroup(26, "-62m", "D3h", "hexagonal", "6/mmm", "-62m", "P -6 -2"),
    PointGroup(27, "6/mmm", "D6h", "hexagonal", "6/mmm", "", "-P 6 2"),
    PointGroup(28, "23", "T", "cubic", "m3", "", "P 2 2 3"),
    PointGroup(29, "m-3", "Th", "cubic", "m3", "", "-P 2 2 3"),
    PointGroup(30, "432", "O", "cubic", "m3m", "", "P 4 2 3"),
    PointGroup(31, "-43m", "Td", "cubic", "m3m", "", "P -4 2 3"),
    PointGroup(32, "m-3m", "Oh", "cubic", "m3m", "", "-P 4 2 3"),
)

POINT_GROUP_FROM_NUMBER = {
    i: [x for x in POINT_GROUP_DATA if x.number == i] for i in range(1, 33)
}

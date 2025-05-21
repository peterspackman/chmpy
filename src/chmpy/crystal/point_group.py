from dataclasses import dataclass

import numpy as np

from chmpy.crystal.symmetry_operation import SymmetryOperation

_PG_ROTATION_AXES = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, 0],
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
        [0, -1, 2],
        [2, 0, -1],
        [-1, 2, 0],
        [0, -2, 1],
        [1, 0, -2],
        [-2, 1, 0],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2],
        [2, 1, -1],
        [-1, 2, 1],
        [1, -1, 2],
        [2, -1, 1],
        [1, 2, -1],
        [-1, 1, 2],
        [3, 1, 2],
        [2, 3, 1],
        [1, 2, 3],
        [3, 2, 1],
        [1, 3, 2],
        [2, 1, 3],
        [3, -1, 2],
        [2, 3, -1],
        [-1, 2, 3],
        [3, -2, 1],
        [1, 3, -2],
        [-2, 1, 3],
        [3, -1, -2],
        [-2, 3, -1],
        [-1, -2, 3],
        [3, -2, -1],
        [-1, 3, -2],
        [-2, -1, 3],
        [3, 1, -2],
        [-2, 3, 1],
        [1, -2, 3],
        [3, 2, -1],
        [-1, 3, 2],
        [2, -1, 3],
        [1, 1, 3],
        [-1, 1, 3],
        [1, -1, 3],
        [-1, -1, 3],
        [1, 3, 1],
        [-1, 3, 1],
        [1, 3, -1],
        [-1, 3, -1],
        [3, 1, 1],
        [3, 1, -1],
        [3, -1, 1],
        [3, -1, -1],
    ]
)


def _get_pg_rotation_matrix(axis, angle):
    """Generate a rotation matrix for a given axis and angle."""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def _generate_pg_operations(number, symbol):
    """Generate symmetry ops for a given point group."""
    ops = [np.eye(3)]  # Identity operation

    if number == 1:  # C1
        return ops

    if number == 2:  # Ci
        ops.append(-np.eye(3))  # Inversion
        return ops

    if symbol in ["2", "m", "2/m"]:  # C2, Cs, C2h
        axis = _PG_ROTATION_AXES[2]  # z-axis
        ops.append(_get_pg_rotation_matrix(axis, np.pi))
        if "m" in symbol:
            ops.append(np.diag([1, 1, -1]))  # Reflection
        return ops

    if symbol in ["222", "mm2", "mmm"]:  # D2, C2v, D2h
        for axis in _PG_ROTATION_AXES[:3]:
            ops.append(_get_pg_rotation_matrix(axis, np.pi))
        if "m" in symbol:
            for i in range(3):
                reflection = np.eye(3)
                reflection[i, i] = -1
                ops.append(reflection)
        return ops

    if symbol in [
        "4",
        "-4",
        "4/m",
        "422",
        "4mm",
        "-42m",
        "4/mmm",
    ]:  # C4, S4, C4h, D4, C4v, D2d, D4h
        axis = _PG_ROTATION_AXES[2]  # z-axis
        ops.append(_get_pg_rotation_matrix(axis, np.pi / 2))
        ops.append(_get_pg_rotation_matrix(axis, np.pi))
        ops.append(_get_pg_rotation_matrix(axis, 3 * np.pi / 2))
        if "-4" in symbol or "4/m" in symbol:
            ops.append(-_get_pg_rotation_matrix(axis, np.pi / 2))
        if "422" in symbol or "4mm" in symbol or "-42m" in symbol or "4/mmm" in symbol:
            for axis in _PG_ROTATION_AXES[:2]:
                ops.append(_get_pg_rotation_matrix(axis, np.pi))
        if "m" in symbol:
            ops.append(np.diag([1, 1, -1]))  # Reflection in xy-plane
            if "mm" in symbol or "4/mmm" in symbol:
                ops.append(np.diag([1, -1, 1]))  # Reflection in xz-plane
                ops.append(np.diag([-1, 1, 1]))  # Reflection in yz-plane
        return ops

    if symbol in ["3", "-3", "32", "3m", "-3m"]:  # C3, C3i, D3, C3v, D3d
        axis = _PG_ROTATION_AXES[9]  # [1, 1, 1] axis
        ops.append(_get_pg_rotation_matrix(axis, 2 * np.pi / 3))
        ops.append(_get_pg_rotation_matrix(axis, 4 * np.pi / 3))
        if "-3" in symbol:
            ops.append(-np.eye(3))  # Inversion
        if "32" in symbol or "-3m" in symbol:
            ops.append(_get_pg_rotation_matrix(_PG_ROTATION_AXES[0], np.pi))
            ops.append(_get_pg_rotation_matrix(_PG_ROTATION_AXES[5], np.pi))
            ops.append(_get_pg_rotation_matrix(_PG_ROTATION_AXES[1], np.pi))
        if "m" in symbol:
            ops.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))  # Reflection
        return ops

    if symbol in [
        "6",
        "-6",
        "6/m",
        "622",
        "6mm",
        "-6m2",
        "6/mmm",
    ]:  # C6, C3h, C6h, D6, C6v, D3h, D6h
        axis = _PG_ROTATION_AXES[2]  # z-axis
        ops.append(_get_pg_rotation_matrix(axis, np.pi / 3))
        ops.append(_get_pg_rotation_matrix(axis, 2 * np.pi / 3))
        ops.append(_get_pg_rotation_matrix(axis, np.pi))
        ops.append(_get_pg_rotation_matrix(axis, 4 * np.pi / 3))
        ops.append(_get_pg_rotation_matrix(axis, 5 * np.pi / 3))
        if "-6" in symbol or "6/m" in symbol:
            ops.append(-_get_pg_rotation_matrix(axis, np.pi / 3))
        if "622" in symbol or "6mm" in symbol or "-6m2" in symbol or "6/mmm" in symbol:
            for axis in [
                _PG_ROTATION_AXES[0],
                _PG_ROTATION_AXES[5],
                _PG_ROTATION_AXES[1],
            ]:
                ops.append(_get_pg_rotation_matrix(axis, np.pi))
        if "m" in symbol:
            ops.append(np.diag([1, 1, -1]))  # Reflection in xy-plane
            if "mm" in symbol or "6/mmm" in symbol:
                ops.append(np.diag([1, -1, 1]))  # Reflection in xz-plane
                ops.append(np.diag([-1, 1, 1]))  # Reflection in yz-plane
        return ops

    if symbol in ["23", "m-3", "432", "-43m", "m-3m"]:  # T, Th, O, Td, Oh
        # Rotations around [1, 0, 0], [0, 1, 0], [0, 0, 1]
        for axis in _PG_ROTATION_AXES[:3]:
            ops.append(_get_pg_rotation_matrix(axis, np.pi / 2))
            ops.append(_get_pg_rotation_matrix(axis, np.pi))
            ops.append(_get_pg_rotation_matrix(axis, 3 * np.pi / 2))
        # Rotations around [1, 1, 1] and equivalent directions
        for axis in [
            _PG_ROTATION_AXES[9],
            _PG_ROTATION_AXES[10],
            _PG_ROTATION_AXES[11],
            _PG_ROTATION_AXES[12],
        ]:
            ops.append(_get_pg_rotation_matrix(axis, 2 * np.pi / 3))
            ops.append(_get_pg_rotation_matrix(axis, 4 * np.pi / 3))
        if "m" in symbol:
            ops.append(-np.eye(3))  # Inversion
            for i in range(3):
                reflection = np.eye(3)
                reflection[i, i] = -1
                ops.append(reflection)
        if "-43m" in symbol or "m-3m" in symbol:
            # Add improper rotations
            ops.append(-_get_pg_rotation_matrix(_PG_ROTATION_AXES[0], np.pi / 2))
            ops.append(-_get_pg_rotation_matrix(_PG_ROTATION_AXES[1], np.pi / 2))
            ops.append(-_get_pg_rotation_matrix(_PG_ROTATION_AXES[2], np.pi / 2))
        return ops

    raise ValueError(f"Unsupported point group: {symbol}")


@dataclass
class PointGroup:
    number: int
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
            SymmetryOperation(x, np.zeros(3))
            for x in _generate_pg_operations(self.number, self.symbol)
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
    PointGroup(1, "1", "C1", "triclinic", "-1", ""),
    PointGroup(2, "-1", "Ci", "triclinic", "-1", ""),
    PointGroup(3, "2", "C2", "monoclinic", "2/m", "b"),
    PointGroup(3, "112", "C2", "monoclinic", "2/m", "c"),
    PointGroup(4, "m", "Cs", "monoclinic", "2/m", "b"),
    PointGroup(4, "11m", "Cs", "monoclinic", "2/m", "c"),
    PointGroup(5, "2/m", "C2h", "monoclinic", "2/m", "b"),
    PointGroup(5, "112/m", "C2h", "monoclinic", "2/m", "c"),
    PointGroup(6, "222", "D2", "orthorhombic", "mmm", ""),
    PointGroup(7, "mm2", "C2v", "orthorhombic", "mmm", "mm2"),
    PointGroup(7, "2mm", "C2v", "orthorhombic", "mmm", "2mm"),
    PointGroup(7, "m2m", "C2v", "orthorhombic", "mmm", "m2m"),
    PointGroup(8, "mmm", "D2h", "orthorhombic", "mmm", ""),
    PointGroup(9, "4", "C4", "tetragonal", "4/m", ""),
    PointGroup(10, "-4", "S4", "tetragonal", "4/m", ""),
    PointGroup(11, "4/m", "C4h", "tetragonal", "4/m", ""),
    PointGroup(12, "422", "D4", "tetragonal", "4/mmm", ""),
    PointGroup(13, "4mm", "C4v", "tetragonal", "4/mmm", ""),
    PointGroup(14, "-42m", "D2d", "tetragonal", "4/mmm", "-42m"),
    PointGroup(14, "-4m2", "D2d", "tetragonal", "4/mmm", "-4m2"),
    PointGroup(15, "4/mmm", "D4h", "tetragonal", "4/mmm", ""),
    PointGroup(16, "3", "C3", "trigonal", "-3", "H"),
    PointGroup(16, "3r", "C3", "trigonal", "-3", "R"),
    PointGroup(17, "-3", "C3i", "trigonal", "-3", "H"),
    PointGroup(17, "-3r", "C3i", "trigonal", "-3", "R"),
    PointGroup(18, "32", "D3", "trigonal", "-3m", "32"),
    PointGroup(18, "321", "D3", "trigonal", "-3m", "321"),
    PointGroup(18, "3m1", "D3", "trigonal", "-3m", "3m1"),
    PointGroup(18, "32r", "D3", "trigonal", "-3m", "R"),
    PointGroup(19, "3m", "C3v", "trigonal", "-3m", "3m"),
    PointGroup(19, "3m1", "C3v", "trigonal", "-3m", "3m1"),
    PointGroup(19, "31m", "C3v", "trigonal", "-3m", "31m"),
    PointGroup(19, "3mr", "C3v", "trigonal", "-3m", "R"),
    PointGroup(20, "-3m", "D3d", "trigonal", "-3m", "-3m"),
    PointGroup(20, "-3m1", "D3d", "trigonal", "-3m", "-3m1"),
    PointGroup(20, "-31m", "D3d", "trigonal", "-3m", "-31m"),
    PointGroup(20, "-3m", "D3d", "trigonal", "-3m", "R"),
    PointGroup(21, "6", "C6", "hexagonal", "6/m", ""),
    PointGroup(22, "-6", "C3h", "hexagonal", "6/m", ""),
    PointGroup(23, "6/m", "C6h", "hexagonal", "6/m", ""),
    PointGroup(24, "622", "D6", "hexagonal", "6/mmm", ""),
    PointGroup(25, "6mm", "C6v", "hexagonal", "6/mmm", ""),
    PointGroup(26, "-6m2", "D3h", "hexagonal", "6/mmm", "-6m2"),
    PointGroup(26, "-62m", "D3h", "hexagonal", "6/mmm", "-62m"),
    PointGroup(27, "6/mmm", "D6h", "hexagonal", "6/mmm", ""),
    PointGroup(28, "23", "T", "cubic", "m3", ""),
    PointGroup(29, "m-3", "Th", "cubic", "m3", ""),
    PointGroup(30, "432", "O", "cubic", "m3m", ""),
    PointGroup(31, "-43m", "Td", "cubic", "m3m", ""),
    PointGroup(32, "m-3m", "Oh", "cubic", "m3m", ""),
)

POINT_GROUP_FROM_NUMBER = {
    i: [x for x in POINT_GROUP_DATA if x.number == i] for i in range(1, 33)
}

import logging
import numpy as np
from numpy import zeros, allclose as close

LOG = logging.getLogger(__name__)


class UnitCell:
    """
    Storage class for the lattice vectors of a crystal i.e. its unit cell.

    Attributes:
        direct (np.ndarray): the direct matrix of this unit cell
            i.e. the lattice vectors
        reciprocal_lattice (np.ndarray): the reciprocal matrix of
            this unit cell i.e. the reciprocal lattice vectors
        inverse (np.ndarray): the inverse matrix of this unit
            cell i.e. the transpose of `reciprocal_lattice`
        lattice (np.ndarray): an alias for `direct`
    """

    def __init__(self, vectors):
        """
        Create a UnitCell object from a list of lattice vectors or
        a row major direct matrix. Unless otherwise specified, length
        units are Angstroms, and angular units are radians.

        Args:
            vectors (array_like): (3, 3) array of lattice vectors, row major i.e. vectors[0, :] is
                lattice vector A etc.
        """
        self.set_vectors(vectors)

    @property
    def lattice(self) -> np.ndarray:
        "The direct matrix of this unit cell i.e. vectors of the lattice"
        return self.direct

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        "The reciprocal matrix of this unit cell i.e. vectors of the reciprocal lattice"
        return self.inverse.T

    @property
    def direct_homogeneous(self) -> np.ndarray:
        "The direct matrix in homogeneous coordinates"
        T = np.eye(4)
        T[:3, :3] = self.direct
        return T

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from fractional space (a, b, c)
        to Cartesian space (x, y, z). The x-direction will be aligned
        along lattice vector A.

        Args:
            coords (array_like): (N, 3) array of fractional coordinates

        Returns:
            np.ndarray: (N, 3) array of Cartesian coordinates
        """
        return np.dot(coords, self.direct)

    def to_fractional(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from Cartesian space (x, y, z)
        to fractional space (a, b, c). The x-direction will is assumed
        be aligned along lattice vector A.

        Args:
            coords (array_like): an (N, 3) array of Cartesian coordinates

        Returns:
            np.ndarray: (N, 3) array of fractional coordinates
        """
        return np.dot(coords, self.inverse)

    def set_lengths_and_angles(self, lengths, angles):
        """
        Modify this unit cell by setting the lattice vectors
        according to lengths a, b, c and angles alpha, beta, gamma of
        a parallelipiped.

        Args:
            lengths (array_like): array of (a, b, c), the unit cell side lengths in Angstroms.
            angles (array_like): array of (alpha, beta, gamma), the unit cell angles lengths
                in radians.
        """
        self.lengths = lengths
        self.angles = angles
        a, b, c = self.lengths
        ca, cb, cg = np.cos(self.angles)
        sg = np.sin(self.angles[2])
        v = self.volume()
        self.direct = np.array((
            (a, 0, 0),
            (b * cg, b * sg, 0),
            (c * cb, c * (ca - cb * cg) / sg, v / (a * b * sg))
        ))
        self.inverse = np.array((
            (1.0 / a, 0.0, 0.0),
            (-cg / (a * sg), 1 / (b * sg), 0),
            (
                b * c * (ca * cg - cb) / v / sg,
                a * c * (cb * cg - ca) / v / sg,
                a * b * sg / v,
            )
        ))
        self._set_cell_type()

    def set_vectors(self, vectors):
        """
        Modify this unit cell by setting the lattice vectors
        according to those provided. This is performed by setting the
        lattice parameters (lengths and angles) based on the provided vectors,
        such that it results in a consistent basis without directly
        matrix inverse (and typically losing precision), and
        as the SHELX file/CIF output will be relying on these
        lengths/angles anyway, it is important to have these consistent.


        Args:
            vectors (array_like): (3, 3) array of lattice vectors, row major i.e. vectors[0, :] is
                lattice vector A etc.
        """
        self.direct = vectors
        params = zeros(6)
        a, b, c = np.linalg.norm(self.direct, axis=1)
        u_a = vectors[0, :] / a
        u_b = vectors[1, :] / b
        u_c = vectors[2, :] / c
        alpha = np.arccos(np.clip(np.vdot(u_b, u_c), -1, 1))
        beta = np.arccos(np.clip(np.vdot(u_c, u_a), -1, 1))
        gamma = np.arccos(np.clip(np.vdot(u_a, u_b), -1, 1))
        params[3:] = np.degrees([alpha, beta, gamma])
        self.lengths = [a, b, c]
        self.angles = [alpha, beta, gamma]
        self.inverse = np.linalg.inv(self.direct)
        self._set_cell_type()

    def _set_cell_type(self):
        if self.is_cubic:
            self.cell_type_index = 6
            self.cell_type = "cubic"
            self.unique_parameters = (self.a,)
            self.unique_parameters_deg = self.unique_parameters
        elif self.is_rhombohedral:
            self.cell_type_index = 4
            self.cell_type = "rhombohedral"
            self.unique_parameters = self.a, self.alpha
            self.unique_parameters_deg = (self.a, np.degrees(self.alpha))
        elif self.is_hexagonal:
            self.cell_type_index = 5
            self.cell_type = "hexagonal"
            self.unique_parameters = self.a, self.c
            self.unique_parameters_deg = self.unique_parameters
        elif self.is_tetragonal:
            self.cell_type_index = 3
            self.cell_type = "tetragonal"
            self.unique_parameters = self.a, self.c
            self.unique_parameters_deg = self.unique_parameters
        elif self.is_orthorhombic:
            self.cell_type_index = 2
            self.cell_type = "orthorhombic"
            self.unique_parameters = self.a, self.b, self.c
            self.unique_parameters_deg = self.unique_parameters
        elif self.is_monoclinic:
            self.cell_type_index = 1
            self.cell_type = "monoclinic"
            self.unique_parameters = self.a, self.b, self.c, self.beta
            self.unique_parameters_deg = (self.a, self.b, np.degrees(self.beta))
        else:
            self.cell_type_index = 0
            self.cell_type = "triclinic"
            self.unique_parameters = (
                self.a,
                self.b,
                self.c,
                self.alpha,
                self.beta,
                self.gamma,
            )
            self.unique_parameters_deg = (
                self.a,
                self.b,
                self.c,
                np.degrees(self.alpha),
                np.degrees(self.beta),
                np.degrees(self.gamma),
            )

    def volume(self) -> float:
        """The volume of the unit cell, in cubic Angstroms"""
        a, b, c = self.lengths
        ca, cb, cg = np.cos(self.angles)
        return a * b * c * np.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)

    @property
    def abc_equal(self) -> bool:
        "are the lengths a, b, c all equal?"
        return close(np.array(self.lengths) - self.lengths[0], zeros(3))

    @property
    def abc_different(self) -> bool:
        "are all of the lengths a, b, c different?"
        return not (
            close(self.a, self.b) or close(self.a, self.c) or close(self.b, self.c)
        )

    @property
    def orthogonal(self) -> bool:
        "returns true if the lattice vectors are orthogonal"
        return close(np.abs(self.angles) - np.pi / 2, zeros(3))

    @property
    def angles_different(self) -> bool:
        "are all of the angles alpha, beta, gamma different?"
        return not (
            close(self.alpha, self.beta)
            or close(self.alpha, self.gamma)
            or close(self.beta, self.gamma)
        )

    @property
    def is_triclinic(self) -> bool:
        """Returns true if and lengths are different"""
        return self.abc_different and self.angles_different

    @property
    def is_monoclinic(self) -> bool:
        """Returns true if angles alpha and gamma are equal"""
        return close(self.alpha, self.gamma) and self.abc_different

    @property
    def is_cubic(self) -> bool:
        """Returns true if all lengths are equal and all angles are 90 degrees"""
        return self.abc_equal and self.orthogonal

    @property
    def is_orthorhombic(self) -> bool:
        """Returns true if all angles are 90 degrees"""
        return self.orthogonal and self.abc_different

    @property
    def is_tetragonal(self) -> bool:
        """Returns true if a, b are equal and all angles are 90 degrees"""
        return close(self.a, self.b) and (not close(self.a, self.c)) and self.orthogonal

    @property
    def is_rhombohedral(self) -> bool:
        """Returns true if all lengths are equal and all angles are equal"""
        return (
            self.abc_equal
            and close(np.array(self.angles) - self.angles[0], zeros(3))
            and (not close(self.alpha, np.pi / 2))
        )

    @property
    def is_hexagonal(self) -> bool:
        """Returns true if lengths a == b, a != c, alpha and beta == 90 and gamma == 120"""
        return (
            close(self.a, self.b)
            and (not close(self.a, self.c))
            and close(self.angles[:2], np.pi / 2)
            and close(self.gamma, 2 * np.pi / 3)
        )

    @property
    def a(self) -> float:
        "Length of lattice vector a"
        return self.lengths[0]

    @property
    def v_a(self) -> np.ndarray:
        "lattice vector a"
        return self.direct[0]

    @property
    def v_a_star(self) -> np.ndarray:
        "reciprocal lattice vector a*"
        return self.inverse[:, 0]

    @property
    def a_star(self) -> float:
        "length of reciprocal lattice vector a*"
        return self.b * self.c * np.sin(self.alpha) / self.volume()

    @property
    def alpha(self) -> float:
        "Angle between lattice vectors b and c"
        return self.angles[0]

    @property
    def alpha_star(self) -> float:
        "Angle between reciprocal lattice vectors b* and c*"
        return np.arccos(
            (np.cos(self.beta) * np.cos(self.gamma) - np.cos(self.alpha))
            / (np.sin(self.beta) * np.sin(self.gamma))
        )

    @property
    def b(self) -> float:
        "Length of lattice vector b"
        return self.lengths[1]

    @property
    def v_b(self) -> np.ndarray:
        "lattice vector a"
        return self.direct[1]

    @property
    def v_b_star(self) -> np.ndarray:
        "reciprocal lattice vector b*"
        return self.inverse[:, 1]

    @property
    def b_star(self) -> float:
        "length of reciprocal lattice vector b*"
        return self.a * self.c * np.sin(self.beta) / self.volume()

    @property
    def beta(self) -> float:
        "Angle between lattice vectors a and c"
        return self.angles[1]

    @property
    def beta_star(self) -> float:
        "Angle between reciprocal lattice vectors a* and c*"
        return np.arccos(
            (np.cos(self.alpha) * np.cos(self.gamma) - np.cos(self.beta))
            / (np.sin(self.alpha) * np.sin(self.gamma))
        )

    @property
    def c(self) -> float:
        "Length of lattice vector c"
        return self.lengths[2]

    @property
    def v_c(self) -> np.ndarray:
        "lattice vector a"
        return self.direct[2]

    @property
    def v_c_star(self) -> np.ndarray:
        "reciprocal lattice vector c*"
        return self.inverse[:, 2]

    @property
    def c_star(self) -> float:
        "length of reciprocal lattice vector c*"
        return self.a * self.b * np.sin(self.gamma) / self.volume()

    @property
    def gamma(self) -> float:
        "Angle between lattice vectors a and b"
        return self.angles[2]

    @property
    def gamma_star(self) -> float:
        "Angle between reciprocal lattice vectors a* and c*"
        return np.arccos(
            (np.cos(self.alpha) * np.cos(self.beta) - np.cos(self.gamma))
            / (np.sin(self.alpha) * np.sin(self.beta))
        )

    @property
    def alpha_deg(self) -> float:
        "Angle between lattice vectors b and c in degrees"
        return np.degrees(self.angles[0])

    @property
    def beta_deg(self) -> float:
        "Angle between lattice vectors a and c in degrees"
        return np.degrees(self.angles[1])

    @property
    def gamma_deg(self) -> float:
        "Angle between lattice vectors a and b in degrees"
        return np.degrees(self.angles[2])

    @property
    def parameters(self) -> np.ndarray:
        "single vector of lattice side lengths and angles in degrees"
        atol = 1e-6
        l = np.array(self.lengths)
        deg = np.degrees(self.angles)
        len_diffs = np.abs(l[:, np.newaxis] - l[np.newaxis, :]) < atol
        ang_diffs = np.abs(deg[:, np.newaxis] - deg[np.newaxis, :]) < atol
        for i in range(3):
            l[len_diffs[i]] = l[i]
            deg[ang_diffs[i]] = deg[i]
        return np.hstack((l, deg))

    @classmethod
    def from_lengths_and_angles(cls, lengths, angles, unit="radians"):
        """
        Construct a new UnitCell from the provided lengths and angles.

        Args:
            lengths (array_like): Lattice side lengths (a, b, c) in Angstroms.
            angles (array_like): Lattice angles (alpha, beta, gamma) in provided units (default radians)
            unit (str, optional): Unit for angles i.e. 'radians' or 'degrees' (default radians).

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """
        uc = cls(np.eye(3))
        if unit == "radians":
            if np.any(np.abs(angles) > np.pi):
                LOG.warn(
                    "Large angle in UnitCell.from_lengths_and_angles, "
                    "are you sure your angles are not in degrees?"
                )
            uc.set_lengths_and_angles(lengths, angles)
        else:
            uc.set_lengths_and_angles(lengths, np.radians(angles))
        return uc

    @classmethod
    def cubic(cls, length):
        """
        Construct a new cubic UnitCell from the provided side length.

        Args:
            length (float): Lattice side length a in Angstroms.

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """
        return cls(np.eye(3) * length)

    @classmethod
    def from_unique_parameters(cls, params, cell_type="triclinic", **kwargs):
        """
        Construct a new unit cell from the unique parameters and
        the specified cell type.

        Args:
            params (Tuple): tuple of floats of unique parameters
            cell_type (str, optional): the desired cell type
        """
        return getattr(cls, cell_type)(*params)

    @classmethod
    def triclinic(cls, *params, **kwargs):
        """
        Construct a new UnitCell from the provided side lengths and angles.

        Args:
            params (array_like): Lattice side lengths and angles (a, b, c, alpha, beta, gamma)

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """

        assert len(params) == 6, "Requre three lengths and angles for Triclinic cell"
        return cls.from_lengths_and_angles(params[:3], params[3:], **kwargs)

    @classmethod
    def monoclinic(cls, *params, **kwargs):
        """
        Construct a new UnitCell from the provided side lengths and angle.

        Args:
            params (array_like): Lattice side lengths and angles (a, b, c, beta)

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """

        assert (
            len(params) == 4
        ), "Requre three lengths and one angle for Monoclinic cell"
        unit = kwargs.get("unit", "radians")
        if unit != "radians":
            alpha, gamma = 90, 90
        else:
            alpha, gamma = np.pi / 2, np.pi / 2
        return cls.from_lengths_and_angles(
            params[:3], (alpha, params[3], gamma), **kwargs
        )

    @classmethod
    def tetragonal(cls, *params, **kwargs):
        """
        Construct a new UnitCell from the provided side lengths and angles.

        Args:
            params (array_like): Lattice side lengths (a, c)

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """
        assert len(params) == 2, "Requre 2 lengths for Tetragonal cell"
        unit = kwargs.get("unit", "radians")
        if unit != "radians":
            angles = [90] * 3
        else:
            angles = [np.pi / 2] * 3
        return cls.from_lengths_and_angles(
            (params[0], params[0], params[1]), angles, **kwargs
        )

    @classmethod
    def hexagonal(cls, *params, **kwargs):
        """
        Construct a new UnitCell from the provided side lengths and angles.

        Args:
            params (array_like): Lattice side lengths (a, c)

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """
        assert len(params) == 2, "Requre 2 lengths for Hexagonal cell"
        unit = kwargs.pop("unit", "radians")
        unit = "radians"
        angles = [np.pi / 2, np.pi / 2, 2 * np.pi / 3]
        return cls.from_lengths_and_angles(
            (params[0], params[0], params[1]), angles, unit=unit, **kwargs
        )

    @classmethod
    def rhombohedral(cls, *params, **kwargs):
        """
        Construct a new UnitCell from the provided side lengths and angles.

        Args:
            params (array_like): Lattice side length a and angle alpha c

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """
        assert len(params) == 2, "Requre 1 length and 1 angle for Rhombohedral cell"
        return cls.from_lengths_and_angles([params[0]] * 3, [params[1]] * 3, **kwargs)

    @classmethod
    def orthorhombic(cls, *lengths, **kwargs):
        """
        Construct a new orthorhombic UnitCell from the provided side lengths.

        Args:
            lengths (array_like): Lattice side lengths (a, b, c) in Angstroms.

        Returns:
            UnitCell: A new unit cell object representing the provided lattice.
        """

        assert len(lengths) == 3, "Requre three lengths for Orthorhombic cell"
        return cls(np.diag(lengths))

    def as_rhombohedral(
        self, T=((-1 / 3, 1 / 3, 1 / 3), (2 / 3, 1 / 3, 1 / 3), (-1 / 3, -2 / 3, 1 / 3))
    ):
        if not (self.is_hexagonal):
            raise ValueError("Only hexagonal cells can be converted to rhombohedral")
        T = np.array(T)
        return UnitCell(np.dot(T, self.direct))

    def as_hexagonal(self, T=((-1, 1, 0), (1, 0, -1), (1, 1, 1))):
        if not self.is_rhombohedral:
            raise ValueError("Only rhombohedral cells can be converted to hexagonal")
        # Crystal17 convention =  ((1, -1,  0), (0, 1, -1), (1, 1, 1))
        T = np.array(T)
        return UnitCell(np.dot(T, self.direct))

    def to_mesh(self):
        from trimesh import Trimesh
        verts = np.array([
            np.zeros(3),
            self.v_c,
            self.v_b,
            self.v_b + self.v_c,
            self.v_a,
            self.v_a + self.v_c,
            self.v_a + self.v_b,
            self.v_a + self.v_b + self.v_c,
        ])

        faces = np.array([
            [1, 3, 0],
            [4, 1, 0],
            [0, 3, 2],
            [2, 4, 0],
            [1, 7, 3],
            [5, 1, 4],
            [5, 7, 1],
            [3, 7, 2],
            [6, 4, 2],
            [2, 7, 6],
            [6, 5, 4],
            [7, 5, 6]
        ])
        return Trimesh(vertices=verts, faces=faces)

    def __repr__(self):
        cell = self.cell_type
        unique = self.unique_parameters
        s = "<{{}}: {{}} ({})>".format(",".join("{:.3f}" for p in unique))
        return s.format(self.__class__.__name__, cell, *unique)

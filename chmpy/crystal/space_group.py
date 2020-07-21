import numpy as np
from collections import namedtuple
import os
import json
import re
from copy import deepcopy
from fractions import Fraction
from chmpy.util.num import cartesian_product
from chmpy.util.text import subscript, overline
from .point_group import POINT_GROUP_DATA
import logging

LOG = logging.getLogger(__name__)

_sgdata = namedtuple(
    "_sgdata",
    "number short schoenflies "
    "full international pointgroup "
    "choice centering symops centrosymmetric",
)

with open(os.path.join(os.path.dirname(__file__), "sgdata.json")) as f:
    _sgdata_dict = json.load(f)

SG_FROM_NUMBER = {k: [_sgdata._make(x) for x in v] for k, v in _sgdata_dict.items()}

SG_FROM_SYMOPS = {tuple(x.symops): x for k, sgs in SG_FROM_NUMBER.items() for x in sgs}
SG_CHOICES = {int(k): [x.choice for x in v] for k, v in SG_FROM_NUMBER.items()}

SYMM_STR_SYMBOL_REGEX = re.compile(r".*?([+-]*[xyz0-9\/\.]+)")

LATTICE_TYPE_TRANSLATIONS = {
    1: (),
    2: ((1 / 2, 1 / 2, 1 / 2),),  # P
    3: ((2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)),  # R
    4: ((0, 1 / 2, 1 / 2), (1 / 2, 0, 1 / 2), (1 / 2, 1 / 2, 0)),  # F
    5: ((0, 1 / 2, 1 / 2),),  # A
    6: ((1 / 2, 0, 1 / 2),),  # B
    7: ((1 / 2, 1 / 2, 0),),  # C
}


def encode_symm_str(rotation, translation):
    """
    Encode a rotation matrix (of -1, 0, 1s) and (rational) translation vector
    into string form e.g. 1/2-x,z-1/3,-y-1/6

    >>> encode_symm_str(((-1, 0, 0), (0, 0, 1), (0, 1, 0)), (0, 0.5, 1/3))
    '-x,1/2+z,1/3+y'
    >>> encode_symm_str(((1, 1, 1), (1, 0, 1), (0, 1, 0)), (0, 0.5, 1/3))
    '+x+y+z,1/2+x+z,1/3+y'

    Parameters:
        rotation (array_like): (3,3) matrix of -1, 0, or 1s encoding the rotation component
            of the symmetry operation
        translation (array_like): (3) vector of rational numbers encoding the translation component
            of the symmetry operation

    Returns:
        str: the encoded symmetry operation
    """
    symbols = "xyz"
    res = []
    for i in (0, 1, 2):
        t = Fraction(translation[i]).limit_denominator(12)
        v = ""
        if t != 0:
            v += str(t)
        for j in range(0, 3):
            c = rotation[i][j]
            if c != 0:
                s = "-" if c < 0 else "+"
                v += s + symbols[j]
        res.append(v)
    res = ",".join(res)
    return res


def decode_symm_str(s):
    """
    Decode a symmetry operation represented in the string
    form e.g. '1/2 + x, y, -z -0.25' into a rotation matrix
    and translation vector.
    
    >>> encode_symm_str(*decode_symm_str("x,y,z"))
    '+x,+y,+z'
    >>> encode_symm_str(*decode_symm_str("1/2 - x,y-0.3333333,z"))
    '1/2-x,2/3+y,+z'

    Parameters:
        s (str): the encoded symmetry operation string

    Returns:
        Tuple[np.ndarray, np.ndarray]: a (3,3) rotation matrix and a (3) translation vector
    """
    rotation = np.zeros((3, 3), dtype=np.float64)
    translation = np.zeros((3,), dtype=np.float64)
    tokens = s.lower().replace(" ", "").split(",")
    for i, row in enumerate(tokens):
        fac = 1
        row = row.strip()
        symbols = re.findall(SYMM_STR_SYMBOL_REGEX, row)
        for symbol in symbols:
            if "x" in symbol:
                idx = 0
                fac = -1 if "-x" in symbol else 1
                rotation[i, idx] = fac
            elif "y" in symbol:
                idx = 1
                fac = -1 if "-y" in symbol else 1
                rotation[i, idx] = fac
            elif "z" in symbol:
                idx = 2
                fac = -1 if "-z" in symbol else 1
                rotation[i, idx] = fac
            else:
                if "/" in symbol:
                    numerator, denominator = symbol.split("/")
                    translation[i] = Fraction(
                        Fraction(numerator), Fraction(denominator)
                    )
                else:
                    translation[i] += float(Fraction(symbol))
    translation = translation % 1
    return rotation, translation


def decode_symm_int(coded_integer):
    """
    Decode an integer encoded symmetry operation. 
    
    A space group operation is compressed using ternary numerical system for
    rotation and duodecimal system for translation. This is achieved because
    each element of rotation matrix can have only one of {-1,0,1}, and the
    translation can have one of {0,2,3,4,6,8,9,10} divided by 12.  Therefore
    3^9 * 12^3 = 34012224 different values can map space group operations. In
    principle, octal numerical system can be used for translation, but
    duodecimal system is more convenient.

    >>> encode_symm_str(*decode_symm_int(16484))
    '+x,+y,+z'

    Parameters:
        coded_integer (int): integer encoding a symmetry operation

    Returns:
        Tuple[np.ndarray, np.ndarray]: (3,3) rotation matrix, (3) translation vector
    """
    r = coded_integer % 19683  # 19683 = 3**9
    shift = 6561  # 6561 = 3**8
    rotation = np.empty((3, 3), dtype=np.float64)
    translation = np.empty(3, dtype=np.float64)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            # we need integer division here
            rotation[i, j] = (r % (shift * 3)) // shift - 1
            shift //= 3

    t = coded_integer // 19683
    shift = 144
    for i in (0, 1, 2):
        # we need integer division here by shift
        translation[i] = ((t % (shift * 12)) // shift) / 12
        shift //= 12
    return rotation, translation


def encode_symm_int(rotation, translation):
    """
    Encode an integer encoded symmetry from a rotation matrix and translation
    vector.
    
    A space group operation is compressed using ternary numerical system for
    rotation and duodecimal system for translation. This is achieved because
    each element of rotation matrix can have only one of {-1,0,1}, and the
    translation can have one of {0,2,3,4,6,8,9,10} divided by 12.  Therefore
    3^9 * 12^3 = 34012224 different values can map space group operations. In
    principle, octal numerical system can be used for translation, but
    duodecimal system is more convenient.

    >>> encode_symm_int(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
    16484
    >>> encode_symm_int(((1, 0, 0), (0, 1, 0), (0, 1, 1)), (0, 0.5, 0))
    1433663

     Parameters:
        rotation (array_like): (3,3) matrix of -1, 0, or 1s encoding the rotation component
            of the symmetry operation
        translation (array_like): (3) vector of rational numbers encoding the translation component
            of the symmetry operation

    Returns:
        int: the encoded symmetry operation
    """

    r = 0
    shift = 1
    # encode rotation component
    rotation = np.round(np.array(rotation)).astype(int) + 1
    for i in (2, 1, 0):
        for j in (2, 1, 0):
            r += rotation[i, j] * shift
            shift *= 3
    t = 0
    shift = 1
    translation = np.round(np.array(translation) * 12).astype(int)
    for i in (2, 1, 0):
        t += translation[i] * shift
        shift *= 12
    return r + t * 19683


class SymmetryOperation:
    """
    Class to represent a crystallographic symmetry operation,
    composed of a rotation and a translation.

    Attributes:
        rotation (np.ndarray): (3, 3) rotation matrix in fractional coordinates
        translation (np.ndarray): (3) translation vector in fractional coordinates
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __init__(self, rotation, translation):
        """
        Construct a new symmetry operation from a rotation matrix and
        a translation vector

        Arguments:
            rotation (np.ndarray): (3, 3) rotation matrix
            translation (np.ndarray): (3) translation vector

        Returns:
            SymmetryOperation: a new SymmetryOperation
        """
        self.rotation = rotation
        self.translation = translation % 1

    @property
    def seitz_matrix(self) -> np.ndarray:
        "The Seitz matrix form of this SymmetryOperation"
        s = np.eye(4, dtype=np.float64)
        s[:3, :3] = self.rotation
        s[:3, 3] = self.translation
        return s

    @property
    def integer_code(self) -> int:
        "Represent this SymmetryOperation as a packed integer"
        if not hasattr(self, "_integer_code"):
            setattr(
                self, "_integer_code", encode_symm_int(self.rotation, self.translation)
            )
        return getattr(self, "_integer_code")

    @property
    def cif_form(self) -> str:
        "Represent this SymmetryOperation in string form e.g. '+x,+y,+z'"
        return str(self)

    def inverted(self):
        """"
        A copy of this symmetry operation under inversion

        Returns:
            SymmetryOperation: an inverted copy of this symmetry operation
        """
        return SymmetryOperation(-self.rotation, -self.translation)

    def __add__(self, value: np.ndarray):
        """
        Add a vector to this symmetry operation's translation vector.
        
        Returns:
            SymmetryOperation: a copy of this symmetry operation under additional translation"
        """
        return SymmetryOperation(self.rotation, self.translation + value)

    def __sub__(self, value: np.ndarray):
        """
        Subtract a vector from this symmetry operation's translation.
        
        Returns:
            SymmetryOperation: a copy of this symmetry operation under additional translation"
        """
        return SymmetryOperation(self.rotation, self.translation - value)

    def apply(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Apply this symmetry operation to a set of fractional coordinates.
        
        Parameters:
            coordinates (np.ndarray): (N,3) or (N,4) array of fractional coordinates or homogeneous
                fractional coordinates.

        Returns:
            np.ndarray: (N, 3) array of transformed coordinates
        """
        if coordinates.shape[1] == 4:
            return np.dot(coordinates, self.seitz_matrix.T)
        else:
            return np.dot(coordinates, self.rotation.T) + self.translation

    def __str__(self):
        if not hasattr(self, "_string_code"):
            setattr(
                self, "_string_code", encode_symm_str(self.rotation, self.translation)
            )
        return getattr(self, "_string_code")

    def __lt__(self, other):
        return self.integer_code < other.integer_code

    def __eq__(self, other):
        return self.integer_code == other.integer_code

    def __hash__(self):
        return int(self.integer_code)

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self)

    def __call__(self, coordinates):
        return self.apply(coordinates)

    @classmethod
    def from_integer_code(cls, code: int):
        """
        Alternative constructor from an integer-encoded
        symmetry operation e.g. 16484

        See also  the `encode_symm_int`, `decode_symm_int` methods.

        Parameters:
            code (int): integer-encoded symmetry operation

        Returns:
            SymmetryOperation: a new symmetry operation from the provided integer code
        """

        rot, trans = decode_symm_int(code)
        s = SymmetryOperation(rot, trans)
        setattr(s, "_integer_code", code)
        return s

    @classmethod
    def from_string_code(cls, code: str):
        """
        Alternative constructor from a string encoded
        symmetry operation e.g. '+x,+y,+z'.

        See also the `encode_symm_str`, `decode_symm_str` methods.

        Parameters:
            code (str): string-encoded symmetry operation

        Returns:
            SymmetryOperation: a new symmetry operation from the provided string code
        """
        rot, trans = decode_symm_str(code)
        s = SymmetryOperation(rot, trans)
        setattr(s, "_string_code", code)
        return s

    def is_identity(self) -> bool:
        "Returns true if this is the identity symmetry operation '+x,+y,+z'"
        return self.integer_code == 16484

    @classmethod
    def identity(cls):
        "Alternative constructor for the the identity symop i.e. x,y,z"
        return cls.from_integer_code(16484)


def expanded_symmetry_list(reduced_symops, lattice_type):
    """
    Create an expanded list of symmetry operations from the minimum
    specification given a certain lattice type.

    Parameters:
        reduced_symops (List[SymmetryOperation]): reduced list of symmetry operations
        lattice_type (int): integer encoded lattice type with SHELX conventions, i.e.
            ```
            1: P,
            2: I,
            3: rhombohedral obverse on hexagonal axes,
            4: F,
            5: A,
            6: B,
            7: C
            ```
    Returns:
        List[SymmetryOperation]: an expanded list of symmetry operations given lattice type
    """
    lattice_type_value = abs(lattice_type)
    translations = LATTICE_TYPE_TRANSLATIONS[lattice_type_value]

    identity = SymmetryOperation.identity()
    if identity not in reduced_symops:
        LOG.debug("Appending identity symop %s to reduced_symops")
        reduced_symops.append(identity)
    LOG.debug("Reduced symmetry list contains %d symops", len(reduced_symops))

    full_symops = []

    for symop in reduced_symops:
        full_symops.append(symop)
        for t in translations:
            full_symops.append(symop + t)

    if lattice_type > 0:
        full_symops += [x.inverted() for x in full_symops]

    LOG.debug("Expanded symmetry list contains %d symops", len(full_symops))
    return full_symops


def reduced_symmetry_list(full_symops, lattice_type):
    """
    Reduce an expanded list of symmetry operations to the minimum
    specification given a certain lattice type.

    Parameters:
        full_symops (List[SymmetryOperation]): list of symmetry operations
        lattice_type (int): integer encoded lattice type with SHELX conventions, i.e.
            ```
            1: P,
            2: I,
            3: rhombohedral obverse on hexagonal axes,
            4: F,
            5: A,
            6: B,
            7: C
            ```
    Returns:
        List[SymmetryOperation]: minimal list of symmetry operations given lattice type
    """
    lattice_type_value = abs(lattice_type)
    translations = LATTICE_TYPE_TRANSLATIONS[lattice_type_value]

    reduced_symops = [SymmetryOperation.identity()]
    symops_to_process = list(full_symops)

    inversion = lattice_type > 0

    while symops_to_process:
        next_symop = symops_to_process.pop(0)
        if next_symop in reduced_symops:
            continue
        if inversion and next_symop.inverted() in reduced_symops:
            continue
        for t in translations:
            x = next_symop + t
            if inversion and x.inverted() in reduced_symops:
                break
            if x in reduced_symops:
                break
        else:
            reduced_symops.append(next_symop)

    LOG.debug("Reduced symmetry list contains %d symops", len(reduced_symops))
    return reduced_symops


class SpaceGroup:
    """
    Represent a crystallographic space group, including
    all necessary symmetry operations in fractional coordinates,
    the international tables number from 1-230, and the international
    tables symbol.

    Attributes:
        symbol (str): The international tables short space group symbol 
        full_symbol (str): The full international tables space group symbol
        choice (str): The space group choice (if applicable)
        centering (str): The space group centering (if applicable)
        schoenflies (str): The Schoenflies space group symbol
        centrosymmetric (bool): Whether or not the space group is centrosymmetric
        symmetry_operations (List[SymmetryOperation]): List of symmetry operations making up this space group
    """

    def __init__(self, international_tables_number, choice=""):
        if international_tables_number < 1 or international_tables_number > 230:
            raise ValueError("Space group number must be between [1, 230]")
        self.international_tables_number = international_tables_number
        if not choice:
            sgdata = SG_FROM_NUMBER[str(international_tables_number)][0]
        else:
            candidates = SG_FROM_NUMBER[str(international_tables_number)]
            for candidate in candidates:
                if choice == candidate.choice:
                    sgdata = candidate
                    break
            else:
                raise ValueError("Could not find choice {}".format(choice))
        self.symbol = sgdata.short
        self.full_symbol = sgdata.full
        self.choice = sgdata.choice
        self.centering = sgdata.centering
        self._point_group = POINT_GROUP_DATA[sgdata.pointgroup]
        self.schoenflies = sgdata.schoenflies
        self.centrosymmetric = sgdata.centrosymmetric
        symops = sgdata.symops
        self.symmetry_operations = [
            SymmetryOperation.from_integer_code(s) for s in symops
        ]

    @property
    def cif_section(self) -> str:
        "Representation of the SpaceGroup in CIF files"
        return "\n".join(
            "{} {}".format(i, sym.cif_form)
            for i, sym in enumerate(self.symmetry_operations, start=1)
        )

    @property
    def crystal_system(self) -> str:
        "The crystal system of the space group e.g. triclinic, monoclinic etc."
        sg = self.international_tables_number
        if sg <= 0 or sg >= 231:
            raise ValueError("International spacegroup number must be between 1-230")
        if sg <= 2:
            return "triclinic"
        if sg <= 16:
            return "monoclinic"
        if sg <= 74:
            return "orthorhombic"
        if sg <= 142:
            return "tetragonal"
        if sg <= 167:
            return "trigonal"
        if sg <= 194:
            return "hexagonal"
        return "cubic"

    @property
    def point_group(self):
        "the point group of this space group"
        return self._point_group

    @property
    def pg(self):
        "alias for `self.point_group`"
        return self._point_group

    @property
    def sym(self) -> str:
        "alias for `self.symbol`"
        return self.symbol

    @property
    def symbol_unicode(self) -> str:
        "the space group symbol with unicode subscripts"
        symbol = deepcopy(self.symbol)
        if "_" in symbol:
            tokens = symbol.split("_")
            symbol = tokens[0] + "".join(subscript(x[0]) + x[1:] for x in tokens[1:])
        if "-" in symbol:
            tokens = symbol.split("-")
            symbol = tokens[0] + "".join(overline(x[0]) + x[1:] for x in tokens[1:])
        return symbol

    @property
    def symops(self):
        "alias for `self.symmetry_operations`"
        return self.symmetry_operations

    @property
    def laue_class(self) -> str:
        "the Laue class of the point group associated with this space group"
        return self._point_group.laue

    @property
    def lattice_type(self) -> str:
        "the lattice type of this space group e.g. rhombohedral, hexagonal etc."
        inum = self.international_tables_number
        if inum < 143 or inum > 194:
            return self.crystal_system
        if inum in (146, 148, 155, 160, 161, 166, 167):
            if self.choice == "H":
                return "hexagonal"
            elif self.choice == "R":
                return "rhombohedral"
        else:
            return "hexagonal"

    @property
    def latt(self) -> int:
        """
        The SHELX LATT number associated with this space group. Returns
        a negative if there is no inversion.

        Options are
        ```
        1: P,
        2: I,
        3: rhombohedral obverse on hexagonal axes,
        4: F,
        5: A,
        6: B,
        7: C
        ```
        
        Examples:
            >>> P1 = SpaceGroup(1)
            >>> P21c = SpaceGroup(14)
            >>> I41 = SpaceGroup(14)
            >>> R3bar = SpaceGroup(148)
            >>> P1.latt
            -1
            >>> P21c.latt
            1
            >>> R3bar.latt
            3

        Returns:
            int: the SHELX LATT number of this space group
        """
        centering_to_latt = {
            "primitive": 1,  # P
            "body": 2,  # I
            "rcenter": 3,  # R
            "face": 4,  # F
            "aface": 5,  # A
            "bface": 6,  # B
            "cface": 7,  # C
        }
        if not self.centrosymmetric:
            return -centering_to_latt[self.centering]
        return centering_to_latt[self.centering]

    def __len__(self):
        return len(self.symmetry_operations)

    def ordered_symmetry_operations(self):
        "The symmetry operations of this space group in order (with identiy first)"
        # make sure we do the unit symop first
        unity = 0
        for i, s in enumerate(self.symmetry_operations):
            if s.is_identity():
                unity = i
                break
        else:
            raise ValueError(
                "Could not find identity symmetry_operation -- invalide space group"
            )
        other_symops = (
            self.symmetry_operations[:unity] + self.symmetry_operations[unity + 1 :]
        )
        return [self.symmetry_operations[unity]] + other_symops

    def apply_all_symops(self, coordinates: np.ndarray):
        """
        For a given set of coordinates, apply all symmetry
        operations in this space group, yielding a set subject
        to only translational symmetry (i.e. a unit cell).
        Assumes the input coordinates are fractional.

        Parameters:
            coordinates (np.ndarray): (N, 3) set of fractional coordinates

        Returns:
            Tuple[np.ndarray, np.ndarray]: a (MxN) array of generator symop integers
                and an (MxN, 3) array of coordinates where M is the number of symmetry
                operations in this space group.
        """
        nsites = len(coordinates)
        transformed = np.empty((nsites * len(self), 3))
        generator_symop = np.empty(nsites * len(self), dtype=np.int32)

        # make sure we do the unit symop first
        unity = 0
        for i, s in enumerate(self.symmetry_operations):
            if s.integer_code == 16484:
                unity = i
                break
        transformed[0:nsites] = coordinates
        generator_symop[0:nsites] = 16484
        other_symops = (
            self.symmetry_operations[:unity] + self.symmetry_operations[unity + 1 :]
        )
        for i, s in enumerate(other_symops, start=1):
            transformed[i * nsites : (i + 1) * nsites] = s(coordinates)
            generator_symop[i * nsites : (i + 1) * nsites] = s.integer_code
        return generator_symop, transformed

    def __repr__(self):
        return "<{} {}: {}>".format(
            self.__class__.__name__, self.international_tables_number, self.symbol
        )

    def __eq__(self, other):
        return (
            self.international_tables_number == other.international_tables_number
        ) and (self.choice == other.choice)

    def __hash__(self):
        return hash((self.international_tables_number, self.choice))

    def reduced_symmetry_operations(self):
        "returns a reduced list of symmetry operations"
        return reduced_symmetry_list(self.symmetry_operations, self.latt)

    def has_hexagonal_rhombohedral_choices(self) -> bool:
        "returns true if this space group could be represented as hexagonal or rhombohedral"
        return self.international_tables_number in (146, 148, 155, 160, 161, 166, 167)

    @classmethod
    def from_symmetry_operations(cls, symops, expand_latt=None):
        """
        Find a matching spacegroup for a given set of symmetry
        operations, optionally treating them as a reduced set of
        symmetry operations and expanding them based on the lattice
        type.

        Parameters:
            symops (List[SymmetryOperation]): a reduced or full list of symmetry operations
            expand_latt (int, optional): the SHELX LATT number to expand this list of symmetry operations

        Returns:
            SpaceGroup: the matching `SpaceGroup` for the provided symmetry operations and LATT
        
        """
        if expand_latt is not None:
            if not -8 < expand_latt < 8:
                raise ValueError("expand_latt must be between [-7, 7]")
            symops = expanded_symmetry_list(symops, expand_latt)
        encoded = tuple(sorted(s.integer_code for s in symops))
        if encoded not in SG_FROM_SYMOPS:
            raise ValueError(
                "Could not find matching spacegroup for "
                "the following symops:\n{}".format(
                    "\n".join(str(s) for s in sorted(symops))
                )
            )
        else:
            sgdata = SG_FROM_SYMOPS[encoded]
            return SpaceGroup(sgdata.number, choice=sgdata.choice)

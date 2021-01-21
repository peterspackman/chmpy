from fractions import Fraction
import logging
import numpy as np
import re
from collections import namedtuple
from copy import deepcopy

LOG = logging.getLogger(__name__)


SYMM_STR_SYMBOL_REGEX = re.compile(r".*?([+-]*[xyz0-9\/\.]+)")


LATTICE_TYPE_TRANSLATIONS = {
    1: (),  # P
    2: ((1 / 2, 1 / 2, 1 / 2),),  # I
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

    Args:
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

    Args:
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

    Args:
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

     Args:
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
        """ "
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

        Args:
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

        Args:
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

        Args:
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

    Args:
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

    Args:
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

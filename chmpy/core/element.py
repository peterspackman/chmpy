"""Module for static information about chemical elements."""

import re
import functools
from collections import Counter
import numbers
import numpy as np

_SYMBOL_REGEX = re.compile("([A-Z]+).*", re.IGNORECASE)


_ELEMENT_DATA = (
    # name symbol cov vdw mass
    ("hydrogen", "H", 0.23, 1.09, 1.00794),
    ("helium", "He", 1.50, 1.40, 4.002602),
    ("lithium", "Li", 1.28, 1.82, 6.941),
    ("beryllium", "Be", 0.96, 2.00, 9.012182),
    ("boron", "B", 0.83, 2.00, 10.811),
    ("carbon", "C", 0.68, 1.70, 12.0107),
    ("nitrogen", "N", 0.68, 1.55, 14.0067),
    ("oxygen", "O", 0.68, 1.52, 15.9994),
    ("fluorine", "F", 0.64, 1.47, 18.998403),
    ("neon", "Ne", 1.50, 1.54, 20.1797),
    ("sodium", "Na", 1.66, 2.27, 22.98977),
    ("magnesium", "Mg", 1.41, 1.73, 24.305),
    ("aluminium", "Al", 1.21, 2.00, 26.981538),
    ("silicon", "Si", 1.20, 2.10, 28.0855),
    ("phosphorus", "P", 1.05, 1.80, 30.973761),
    ("sulfur", "S", 1.02, 1.80, 32.065),
    ("chlorine", "Cl", 0.99, 1.75, 35.453),
    ("argon", "Ar", 1.51, 1.88, 39.948),
    ("potassium", "K", 2.03, 2.75, 39.0983),
    ("calcium", "Ca", 1.76, 2.00, 40.078),
    ("scandium", "Sc", 1.70, 2.00, 44.95591),
    ("titanium", "Ti", 1.60, 2.00, 47.867),
    ("vanadium", "V", 1.53, 2.00, 50.9415),
    ("chromium", "Cr", 1.39, 2.00, 51.9961),
    ("manganese", "Mn", 1.61, 2.00, 54.938049),
    ("iron", "Fe", 1.52, 2.00, 55.845),
    ("cobalt", "Co", 1.26, 2.00, 58.9332),
    ("nickel", "Ni", 1.24, 1.63, 58.6934),
    ("copper", "Cu", 1.32, 1.40, 63.546),
    ("zinc", "Zn", 1.22, 1.39, 65.409),
    ("gallium", "Ga", 1.22, 1.87, 69.723),
    ("germanium", "Ge", 1.17, 2.00, 72.64),
    ("arsenic", "As", 1.21, 1.85, 74.9216),
    ("selenium", "Se", 1.22, 1.90, 78.96),
    ("bromine", "Br", 1.21, 1.85, 79.904),
    ("krypton", "Kr", 1.50, 2.02, 83.798),
    ("rubidium", "Rb", 2.20, 2.00, 85.4678),
    ("strontium", "Sr", 1.95, 2.00, 87.62),
    ("yttrium", "Y", 1.90, 2.00, 88.90585),
    ("zirconium", "Zr", 1.75, 2.00, 91.224),
    ("niobium", "Nb", 1.64, 2.00, 92.90638),
    ("molybdenum", "Mo", 1.54, 2.00, 95.94),
    ("technetium", "Tc", 1.47, 2.00, 98.0),
    ("ruthenium", "Ru", 1.46, 2.00, 101.07),
    ("rhodium", "Rh", 1.45, 2.00, 102.9055),
    ("palladium", "Pd", 1.39, 1.63, 106.42),
    ("silver", "Ag", 1.45, 1.72, 107.8682),
    ("cadmium", "Cd", 1.44, 1.58, 112.411),
    ("indium", "In", 1.42, 1.93, 114.818),
    ("tin", "Sn", 1.39, 2.17, 118.71),
    ("antimony", "Sb", 1.39, 2.00, 121.76),
    ("tellurium", "Te", 1.47, 2.06, 127.6),
    ("iodine", "I", 1.40, 1.98, 126.90447),
    ("xenon", "Xe", 1.50, 2.16, 131.293),
    ("caesium", "Cs", 2.44, 2.00, 132.90545),
    ("barium", "Ba", 2.15, 2.00, 137.327),
    ("lanthanum", "La", 2.07, 2.00, 138.9055),
    ("cerium", "Ce", 2.04, 2.00, 140.116),
    ("praseodymium", "Pr", 2.03, 2.00, 140.90765),
    ("neodymium", "Nd", 2.01, 2.00, 144.24),
    ("promethium", "Pm", 1.99, 2.00, 145.0),
    ("samarium", "Sm", 1.98, 2.00, 150.36),
    ("europium", "Eu", 1.98, 2.00, 151.964),
    ("gadolinium", "Gd", 1.96, 2.00, 157.25),
    ("terbium", "Tb", 1.94, 2.00, 158.92534),
    ("dysprosium", "Dy", 1.92, 2.00, 162.5),
    ("holmium", "Ho", 1.92, 2.00, 164.93032),
    ("erbium", "Er", 1.89, 2.00, 167.259),
    ("thulium", "Tm", 1.90, 2.00, 168.93421),
    ("Ytterbium", "Yb", 1.87, 2.00, 173.04),
    ("lutetium", "Lu", 1.87, 2.00, 174.967),
    ("hafnium", "Hf", 1.75, 2.00, 178.49),
    ("tantalum", "Ta", 1.70, 2.00, 180.9479),
    ("tungsten", "W", 1.62, 2.00, 183.84),
    ("rhenium", "Re", 1.51, 2.00, 186.207),
    ("osmium", "Os", 1.44, 2.00, 190.23),
    ("iridium", "Ir", 1.41, 2.00, 192.217),
    ("platinum", "Pt", 1.36, 1.72, 195.078),
    ("gold", "Au", 1.50, 1.66, 196.96655),
    ("mercury", "Hg", 1.32, 1.55, 200.59),
    ("thallium", "Tl", 1.45, 1.96, 204.3833),
    ("lead", "Pb", 1.46, 2.02, 207.2),
    ("bismuth", "Bi", 1.48, 2.00, 208.98038),
    ("polonium", "Po", 1.40, 2.00, 290.0),
    ("astatine", "At", 1.21, 2.00, 210.0),
    ("radon", "Rn", 1.50, 2.00, 222.0),
    ("francium", "Fr", 2.60, 2.00, 223.0),
    ("radium", "Ra", 2.21, 2.00, 226.0),
    ("actinium", "Ac", 2.15, 2.00, 227.0),
    ("thorium", "Th", 2.06, 2.00, 232.0381),
    ("protactinium", "Pa", 2.00, 2.00, 231.03588),
    ("uranium", "U", 1.96, 1.86, 238.02891),
    ("neptunium", "Np", 1.90, 2.00, 237.0),
    ("plutonium", "Pu", 1.87, 2.00, 244.0),
    ("americium", "Am", 1.80, 2.00, 243.0),
    ("curium", "Cm", 1.69, 2.00, 247.0),
    ("berkelium", "Bk", 1.54, 2.00, 247.0),
    ("californium", "Cf", 1.83, 2.00, 251.0),
    ("einsteinium", "Es", 1.50, 2.00, 252.0),
    ("fermium", "Fm", 1.50, 2.00, 257.0),
    ("mendelevium", "Md", 1.50, 2.00, 258.0),
    ("nobelium", "No", 1.50, 2.00, 259.0),
    ("lawrencium", "Lr", 1.50, 2.00, 262.0),
)

_EL_FROM_SYM = {
    s: (i, n, s, rcov, rvdw, m)
    for i, (n, s, rcov, rvdw, m) in enumerate(_ELEMENT_DATA, start=1)
}

_EL_FROM_NAME = {
    n: (i, n, s, rcov, rvdw, m)
    for i, (n, s, rcov, rvdw, m) in enumerate(_ELEMENT_DATA, start=1)
}

_EL_COLORS = (
    (255, 255, 255, 255),
    (217, 255, 255, 255),
    (204, 128, 255, 255),
    (194, 255, 0, 255),
    (255, 181, 181, 255),
    (144, 144, 144, 255),
    (48, 80, 248, 255),
    (255, 13, 13, 255),
    (144, 224, 80, 255),
    (179, 227, 245, 255),
    (171, 92, 242, 255),
    (138, 255, 0, 255),
    (191, 166, 166, 255),
    (240, 200, 160, 255),
    (255, 128, 0, 255),
    (255, 255, 48, 255),
    (31, 240, 31, 255),
    (128, 209, 227, 255),
    (143, 64, 212, 255),
    (61, 255, 0, 255),
    (230, 230, 230, 255),
    (191, 194, 199, 255),
    (166, 166, 171, 255),
    (138, 153, 199, 255),
    (156, 122, 199, 255),
    (224, 102, 51, 255),
    (240, 144, 160, 255),
    (80, 208, 80, 255),
    (200, 128, 51, 255),
    (125, 128, 176, 255),
    (194, 143, 143, 255),
    (102, 143, 143, 255),
    (189, 128, 227, 255),
    (255, 161, 0, 255),
    (166, 41, 41, 255),
    (92, 184, 209, 255),
    (112, 46, 176, 255),
    (0, 255, 0, 255),
    (148, 255, 255, 255),
    (148, 224, 224, 255),
    (115, 194, 201, 255),
    (84, 181, 181, 255),
    (59, 158, 158, 255),
    (36, 143, 143, 255),
    (10, 125, 140, 255),
    (0, 105, 133, 255),
    (192, 192, 192, 255),
    (255, 217, 143, 255),
    (166, 117, 115, 255),
    (102, 128, 128, 255),
    (158, 99, 181, 255),
    (212, 122, 0, 255),
    (148, 0, 148, 255),
    (66, 158, 176, 255),
    (87, 23, 143, 255),
    (0, 201, 0, 255),
    (112, 212, 255, 255),
    (255, 255, 199, 255),
    (217, 255, 199, 255),
    (199, 255, 199, 255),
    (163, 255, 199, 255),
    (143, 255, 199, 255),
    (97, 255, 199, 255),
    (69, 255, 199, 255),
    (48, 255, 199, 255),
    (31, 255, 199, 255),
    (0, 255, 156, 255),
    (0, 230, 117, 255),
    (0, 212, 82, 255),
    (0, 191, 56, 255),
    (0, 171, 36, 255),
    (77, 194, 255, 255),
    (77, 166, 255, 255),
    (33, 148, 214, 255),
    (38, 125, 171, 255),
    (38, 102, 150, 255),
    (23, 84, 135, 255),
    (208, 208, 224, 255),
    (255, 209, 35, 255),
    (184, 184, 208, 255),
    (166, 84, 77, 255),
    (87, 89, 97, 255),
    (158, 79, 181, 255),
    (171, 92, 0, 255),
    (117, 79, 69, 255),
    (66, 130, 150, 255),
    (66, 0, 102, 255),
    (0, 125, 0, 255),
    (112, 171, 250, 255),
    (0, 186, 255, 255),
    (0, 161, 255, 255),
    (0, 143, 255, 255),
    (0, 128, 255, 255),
    (0, 107, 255, 255),
    (84, 92, 242, 255),
    (120, 92, 227, 255),
    (138, 79, 227, 255),
    (161, 54, 212, 255),
    (179, 31, 212, 255),
    (179, 31, 186, 255),
    (179, 13, 166, 255),
    (189, 13, 135, 255),
    (199, 0, 102, 255),
    (204, 0, 89, 255),
    (209, 0, 79, 255),
    (217, 0, 69, 255),
    (224, 0, 56, 255),
    (230, 0, 46, 255),
    (235, 0, 38, 255),
)


class _ElementMeta(type):
    def __getitem__(cls, val):
        if isinstance(val, numbers.Integral):
            return cls.from_atomic_number(val)
        elif isinstance(val, str):
            return cls.from_string(val)
        else:
            raise ValueError("cannot construct element from provided type")


@functools.total_ordering
class Element(metaclass=_ElementMeta):
    """Storage class for information about a chemical element.

    Examples:
        >>> h = Element.from_string("H")
        >>> c = Element.from_string("C")
        >>> n = Element.from_atomic_number(7)
        >>> f = Element.from_string("F")

        Element implements an ordering for sorting in e.g.
        molecular formulae where carbon and hydrogen come first,
        otherwise elements are sorted in order of atomic number.

        >>> sorted([h, f, f, c, n])
        [C, H, N, F, F]
    """

    def __init__(self, atomic_number, name, symbol, cov, vdw, mass):
        """Initialize an Element from its chemical data."""
        self.atomic_number = atomic_number
        self.name = name
        self.symbol = symbol
        self.cov = cov
        self.vdw = vdw
        self.mass = mass

    @staticmethod
    def from_string(s: str) -> "Element":
        """Create an element from a given element symbol.

        Args:
            s (str): a string representation of an element in the periodic table

        Returns:
            Element: an Element object if the conversion was successful, otherwise an exception is raised

        Examples:
            >>> Element.from_string("h")
            H
            >>> Element["rn"].name
            'radon'
            >>> Element["AC"].cov
            2.15
        """
        symbol = s.strip().capitalize()
        if symbol == "D":
            symbol = "H"
        if symbol.isdigit():
            return Element.from_atomic_number(int(symbol))
        if symbol not in _EL_FROM_SYM:
            name = symbol.lower()
            if name not in _EL_FROM_NAME:
                return Element.from_label(s)
            else:
                return Element(*_EL_FROM_NAME[name])
        return Element(*_EL_FROM_SYM[symbol])

    @staticmethod
    def from_label(label: str) -> "Element":
        """Create an element from a label e.g. 'C1', 'H2_F2___i' etc.

        Args:
            l (str): a string representation of an element in the periodic table

        Returns:
            Element: an Element object if the conversion was successful, otherwise an exception is raised

        Examples:
            >>> Element.from_label("C1")
            C
            >>> Element.from_label("H")
            H
            >>> Element["LI2_F2____1____i"]
            Li

            An ambiguous case, will make this Calcium not Carbon
            >>> Element.from_label("Ca2_F2____1____i")
            Ca
        """
        m = re.match(_SYMBOL_REGEX, label)
        if m is None:
            raise ValueError("Could not determine symbol from {}".format(label))
        sym = m.group(1).strip().capitalize()
        if sym not in _EL_FROM_SYM:
            raise ValueError("Could not determine symbol from {}".format(label))
        return Element(*_EL_FROM_SYM[sym])

    @staticmethod
    def from_atomic_number(n: int) -> "Element":
        """Create an element from a given atomic number.

        Args:
            n (int): the atomic number of the element

        Returns:
            Element: an Element object if atomic number was valid, otherwise an exception is raised

        Examples:
            >>> Element.from_atomic_number(2)
            He
            >>> Element[79].name
            'gold'
        """
        return Element(n, *_ELEMENT_DATA[n - 1])

    @property
    def vdw_radius(self) -> float:
        """The van der Waals radius in angstroms."""
        return self.vdw

    @property
    def color(self):
        """The color RGBA color of this element."""
        return _EL_COLORS[self.atomic_number - 1]

    @property
    def ball_stick_radius(self) -> float:
        """The radius of this element in a ball and stick representation."""
        if self.symbol == "H":
            return self.covalent_radius
        return self.cov * 0.5

    @property
    def covalent_radius(self) -> float:
        """The covalent radius in angstroms."""
        return self.cov

    def __repr__(self):
        """Represent this element as a string for REPL."""
        return self.symbol

    def __hash__(self):
        """Hash of this element (its atomic number)."""
        return int(self.atomic_number)

    def _is_valid_operand(self, other):
        return hasattr(other, "atomic_number")

    def __eq__(self, other):
        """Check if two Elements have the same atomic number."""
        if not self._is_valid_operand(other):
            raise NotImplementedError
        return self.atomic_number == other.atomic_number

    def __lt__(self, other):
        """Check which element comes before the other in chemical formulae (C first, then order of atomic number)."""
        if not self._is_valid_operand(other):
            raise NotImplementedError
        n1, n2 = self.atomic_number, other.atomic_number
        if n1 == n2:
            return False
        if n1 == 6:
            return True
        elif n2 == 6:
            return False
        else:
            return n1 < n2


def chemical_formula(elements, subscript=False):
    """Calculate the chemical formula for the given list of elements.

    Examples:
        >>> chemical_formula(['O', 'C', 'O'])
        'CO2'
        >>> chemical_formula(['C', 'H', 'O', 'B'])
        'BCHO'

    Args:
        elements (List[Element or str]): a list of elements or element symbols.
            Note that if a list of strings are provided the order of chemical
            symbols may not match convention.
        subscript (bool, optoinal): toggle to use unicode subscripts for the chemical formula string

    Returns:
        str: the chemical formula
    """
    count = Counter(sorted(elements))
    if subscript:
        blocks = []
        for el, c in count.items():
            c = "".join(chr(0x2080 + int(i)) for i in str(c)) if c > 1 else ""
            blocks.append(f"{el}{c}")
    else:
        blocks = []
        for el, c in count.items():
            c = c if c > 1 else ""
            blocks.append(f"{el}{c}")
    return "".join(blocks)


def cov_radii(atomic_numbers):
    """Return the covalent radii for the given atomic numbers.

    Args:
        atomic_numbers (array_like): the (N,) length integer array of atomic numbers

    Returns:
        np.ndarray: (N,) array of floats representing covalent radii
    """
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return np.array([_ELEMENT_DATA[i - 1][2] for i in atomic_numbers], dtype=np.float32)


def vdw_radii(atomic_numbers):
    """Return the van der Waals radii for the given atomic numbers.

    Args:
        atomic_numbers (array_like): the (N,) length integer array of atomic numbers

    Returns:
        np.ndarray: (N,) array of floats representing van der Waals radii
    """
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return np.array([_ELEMENT_DATA[i - 1][3] for i in atomic_numbers], dtype=np.float32)


def element_names(atomic_numbers):
    """Return the element names for the given atomic numbers.

    Args:
        atomic_numbers (array_like): the (N,) length integer array of atomic numbers

    Returns:
        List[str]: (N,) list of strings representing element names
    """
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return [_ELEMENT_DATA[i - 1][0] for i in atomic_numbers]


def element_symbols(atomic_numbers):
    """Return the element symbols for the given atomic numbers.

    Args:
        atomic_numbers (array_like): the (N,) length integer array of atomic numbers

    Returns:
        List[str]: (N,) list of strings representing element symbols
    """
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 103):
        raise ValueError("All elements must be atomic numbers between [1,103]")
    return [_ELEMENT_DATA[i - 1][1] for i in atomic_numbers]

import numpy as np
from collections import namedtuple
import os
import json
import re
from copy import deepcopy
from chmpy.util.text import subscript, overline
from .point_group import PointGroup
from .symmetry_operation import (
    SymmetryOperation,
    expanded_symmetry_list,
    reduced_symmetry_list,
)
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
SG_FROM_SYMBOL = {x.international.split("=")[-1]: x for k, sgs in SG_FROM_NUMBER.items() for x in sgs}
SG_CHOICES = {int(k): [x.choice for x in v] for k, v in SG_FROM_NUMBER.items()}
SG_DEFAULT_SETTING_CHOICE = {
    48: "2",
    50: "2",
    59: "2",
    68: "2",
    70: "2",
    85: "2",
    86: "2",
    88: "2",
    125: "2",
    126: "2",
    129: "2",
    130: "2",
    133: "2",
    134: "2",
    137: "2",
    138: "2",
    141: "2",
    142: "2",
    201: "2",
    203: "2",
    222: "2",
    224: "2",
    227: "2",
    228: "2",
}


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
        if international_tables_number in SG_DEFAULT_SETTING_CHOICE and not choice:
            choice = SG_DEFAULT_SETTING_CHOICE[international_tables_number]
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
        if choice not in ("b", "c", "R"):
            choice = None
        self._point_group = PointGroup.from_number(sgdata.pointgroup, choice=choice)
        self.schoenflies = sgdata.schoenflies
        self.centrosymmetric = sgdata.centrosymmetric
        symops = sgdata.symops
        self.symmetry_operations = [
            SymmetryOperation.from_integer_code(s) for s in symops
        ]
        self._sgdata = sgdata

    @property
    def cif_section(self) -> str:
        "Representation of the SpaceGroup in CIF files"
        return "\n".join(
            "{} {}".format(i, sym.cif_form)
            for i, sym in enumerate(self.symmetry_operations, start=1)
        )

    def crystal17_spacegroup_symbol(self):
        tokens = []
        s = self._sgdata.international.upper()
        if "=" in s:
            s = s.split("=")[-1]
        s = iter(s)
        for ch in s:
            if ch == "_":
                tokens[-1] += next(s)
            elif ch == "/":
                tokens[-1] += ch + next(s)
            elif ch == "-":
                tokens.append(ch + next(s))
            else:
                tokens.append(ch)
        return " ".join(tokens)

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
        symbol = deepcopy(self.full_symbol)
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
        return self._point_group.laue_group

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

        Args:
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
            self.__class__.__name__, self.international_tables_number, self.full_symbol
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

        Args:
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

    @classmethod
    def from_symbol(cls, symbol):
        symbol = symbol.replace(" ", "")
        sgdata = SG_FROM_SYMBOL.get(symbol, None)
        special_cases = {
            "P21/a": "P12_1/a1",
            "P21/n": "P12_1/n1",
        }
        symbol = special_cases.get(symbol, symbol)
        if sgdata is None:
            for number, groups in SG_FROM_NUMBER.items():
                if int(number) > 14:
                    continue
                for g in groups:
                    intl = g.international.split("=")[-1]
                    if symbol == intl:
                        sgdata = g
                        break
                    elif symbol == intl.replace("_", ""):
                        sgdata = g
                        break
                if sgdata is not None:
                    break
            else:
                raise ValueError("Could not find matching space group for '{}'".format(symbol))

            
        return SpaceGroup(sgdata.number, choice=sgdata.choice)


"""
Site symmetry analysis for crystal structures.

This module provides data structures for tracking the site symmetry
(stabilizer subgroup) of each atom in the asymmetric unit, which
corresponds to the Wyckoff position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .symmetry_operation import SymmetryOperation

if TYPE_CHECKING:
    from .crystal import Crystal

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteSymmetry:
    """
    Site symmetry information for a single asymmetric unit atom.

    The site symmetry is the stabilizer subgroup - the set of symmetry
    operations that leave the site invariant (up to a lattice translation).

    Attributes:
        multiplicity: The Wyckoff multiplicity = |G| / |stabilizer|
                     This is how many equivalent copies exist in the unit cell.
        stabilizer_symop_codes: Tuple of integer codes for symops that fix this site
        n_stabilizer: Size of stabilizer subgroup = len(stabilizer_symop_codes)
        is_general_position: True if site is on a general position (stabilizer = identity only)
    """

    multiplicity: int
    stabilizer_symop_codes: tuple[int, ...]
    n_stabilizer: int
    is_general_position: bool

    @classmethod
    def from_position(
        cls,
        position: np.ndarray,
        symops: list[SymmetryOperation],
        tolerance: float = 1e-4,
    ) -> SiteSymmetry:
        """
        Compute the site symmetry for a given position.

        A symmetry operation g belongs to the stabilizer if:
            g(position) - position = lattice_vector (integer components)

        Args:
            position: (3,) fractional coordinates of the site
            symops: List of space group symmetry operations
            tolerance: Tolerance for detecting lattice translations

        Returns:
            SiteSymmetry with stabilizer information
        """
        stabilizer_codes = []
        position = np.asarray(position)

        for symop in symops:
            # Apply symmetry operation
            transformed = symop.apply(position.reshape(1, 3)).flatten()

            # Check if difference is a lattice translation (integer values)
            diff = transformed - position
            rounded = np.round(diff)

            if np.allclose(diff, rounded, atol=tolerance):
                stabilizer_codes.append(symop.integer_code)

        n_stabilizer = len(stabilizer_codes)
        n_symops = len(symops)
        multiplicity = n_symops // n_stabilizer

        # General position has only identity in stabilizer
        is_general = n_stabilizer == 1

        return cls(
            multiplicity=multiplicity,
            stabilizer_symop_codes=tuple(stabilizer_codes),
            n_stabilizer=n_stabilizer,
            is_general_position=is_general,
        )


@dataclass
class SiteSymmetryTable:
    """
    Site symmetry information for all atoms in an asymmetric unit.

    This table stores the stabilizer subgroup (site symmetry) for each
    asymmetric unit atom. Sites on special positions have larger stabilizers
    and lower multiplicities.

    Attributes:
        site_symmetries: Tuple of SiteSymmetry, one per asymmetric unit atom
        general_position_mask: (N_asym,) bool array, True if site is general
        n_asym: Number of atoms in asymmetric unit
        n_symops: Number of symmetry operations in space group
    """

    site_symmetries: tuple[SiteSymmetry, ...]
    general_position_mask: np.ndarray
    n_asym: int
    n_symops: int

    @classmethod
    def from_crystal(
        cls, crystal: Crystal, tolerance: float = 1e-4
    ) -> SiteSymmetryTable:
        """
        Build a SiteSymmetryTable from a Crystal object.

        Args:
            crystal: The crystal structure to analyze
            tolerance: Tolerance for detecting lattice translations

        Returns:
            SiteSymmetryTable with site symmetry for each asymmetric unit atom
        """
        asym = crystal.asymmetric_unit
        symops = crystal.space_group.symmetry_operations

        n_asym = len(asym)
        n_symops = len(symops)

        site_symmetries = []
        general_mask = np.zeros(n_asym, dtype=bool)

        for i, position in enumerate(asym.positions):
            site_sym = SiteSymmetry.from_position(position, symops, tolerance)
            site_symmetries.append(site_sym)
            general_mask[i] = site_sym.is_general_position

        return cls(
            site_symmetries=tuple(site_symmetries),
            general_position_mask=general_mask,
            n_asym=n_asym,
            n_symops=n_symops,
        )

    def get_multiplicity(self, asym_idx: int) -> int:
        """Get the Wyckoff multiplicity for an asymmetric unit atom."""
        return self.site_symmetries[asym_idx].multiplicity

    def get_stabilizer_codes(self, asym_idx: int) -> tuple[int, ...]:
        """Get the stabilizer symop codes for an asymmetric unit atom."""
        return self.site_symmetries[asym_idx].stabilizer_symop_codes

    def is_general_position(self, asym_idx: int) -> bool:
        """Check if an asymmetric unit atom is on a general position."""
        return self.general_position_mask[asym_idx]

    def total_multiplicity(self) -> int:
        """
        Sum of all multiplicities.

        For a valid structure, this should equal N_uc (number of atoms in unit cell).
        """
        return sum(s.multiplicity for s in self.site_symmetries)

    def n_special_positions(self) -> int:
        """Count how many atoms are on special positions."""
        return int(np.sum(~self.general_position_mask))

    def n_general_positions(self) -> int:
        """Count how many atoms are on general positions."""
        return int(np.sum(self.general_position_mask))

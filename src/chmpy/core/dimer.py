"""Module for pairs of molecules, handling symmetry relations and more."""
import numpy as np
import logging
from chmpy.core import Molecule

LOG = logging.getLogger(__name__)


class Dimer:
    """Storage class for symmetry information about a dimers.

    Dimers are two molecules that may or may not be symmetry related.

    Args:
            mol_a (Molecule):
                one of the molecules in the pair (symmetry unique)
            mol_b (Molecule): the neighbouring molecule (may be symmetry related to mol_a)
            separation (float, optional): set the separation of the molecules (otherwise it
                will be calculated)
            transform_ab (np.ndarray, optional): specify the transform from mol_a to mol_b
                (otherwise it will be calculated)
            frac_shift (np.ndarray, optional): specify the offset in fractions of a unit cell,
                which combined with transform_ab will yield mol_b
    """

    seitz_b = None
    symm_str = None
    crystal_transform = False

    def __init__(
        self, mol_a, mol_b, separation=None, transform_ab=None, frac_shift=None
    ):
        """Initialize a Dimer."""
        self.a = mol_a
        self.b = mol_b
        self.a_idx = self.a.properties.get("asym_mol_idx", 0)
        self.b_idx = self.b.properties.get("asym_mol_idx", 0)
        self.frac_shift = frac_shift
        if "generator_symop" in self.a.properties:
            self.symop_a = self.a.properties["generator_symop"]

        if "generator_symop" in self.b.properties:
            self.symop_b = self.b.properties["generator_symop"]

        if separation is not None:
            self.separation = separation
        else:
            self.separation = mol_a.distance_to(mol_b)
        if transform_ab == "calculate":
            self.calculate_transform()
        else:
            self.transform_ab = transform_ab
        self.closest_separation = self.a.distance_to(self.b, method="nearest_atom")
        self.centroid_separation = self.a.distance_to(self.b, method="centroid")
        self.com_separation = self.a.distance_to(self.b, method="center_of_mass")

    def calculate_transform(self):
        """Calculate the transform (if any) from mol_a to mol_b."""
        from chmpy.util.num import kabsch_rotation_matrix

        if len(self.a) != len(self.b):
            self.transform_ab = None
            return

        if not np.all(self.a.atomic_numbers == self.b.atomic_numbers):
            self.transform_ab = None
            return

        v_a = self.a.centroid
        v_b = self.b.centroid
        v_ab = v_b - v_a
        pos_a = self.a.positions - v_a
        pos_b = self.b.positions - v_b
        R = kabsch_rotation_matrix(pos_b, pos_a)
        self.transform_ab = (R, v_ab)

        if (
            self.frac_shift is not None
            and self.symop_a is not None
            and self.symop_b is not None
        ):
            self.crystal_transform = True
            from chmpy.crystal.symmetry_operation import (
                SymmetryOperation,
                encode_symm_str,
            )

            s_b = SymmetryOperation.from_integer_code(self.symop_b[0])
            t_ab = np.zeros((4, 4))
            t_ab[:3, 3] = self.frac_shift
            self.seitz_b = s_b.seitz_matrix.copy()
            self.seitz_b[:3, 3] += self.frac_shift
            self.symm_str = encode_symm_str(self.seitz_b[:3, :3], self.seitz_b[:3, 3])
        return self.transform_ab

    def supermolecule(self):
        return Molecule.from_arrays(
            np.hstack((self.a.atomic_numbers, self.b.atomic_numbers)),
            np.vstack((self.a.positions, self.b.positions)),
        )

    def scale_separation(self, scale_factor):
        v_a = self.a.centroid
        v_b = self.b.centroid
        v_ab = v_b - v_a
        self.b.positions -= v_ab
        v_ab *= scale_factor
        self.b.positions += v_ab

    @property
    def separations(self):
        """The closest atom, centroid-centroid, and center of mass - center of mass separations of mol_a and mol_b."""
        return np.array(
            (self.closest_separation, self.centroid_separation, self.com_separation)
        )

    def __eq__(self, other):
        """Return true if all separations are identical."""
        return np.allclose(self.separations, other.separations)

    def transform_string(self):
        """The transform from mol_a to mol_b as a string (e.g. x,-y,z)."""
        if self.transform_ab is None:
            return "none"
        if self.crystal_transform:
            return self.symm_str
        return str(self.transform_ab)

    def __repr__(self):
        """Represent the Dimer for a REPL or similar."""
        return f"<Dimer: d={self.separation:.2f} symm={self.transform_string()}>"

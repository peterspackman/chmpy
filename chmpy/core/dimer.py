import numpy as np
import logging

LOG = logging.getLogger(__name__)

class Dimer:
    seitz_b = None
    symm_str = None
    crystal_transform = False

    def __init__(self, mol_a, mol_b, separation=None, transform_ab=None, frac_shift=None):
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

        if self.frac_shift is not None and self.symop_a is not None and self.symop_b is not None:
            self.crystal_transform = True
            from chmpy.crystal.symmetry_operation import SymmetryOperation, encode_symm_str
            s_a = SymmetryOperation.from_integer_code(self.symop_a[0])
            s_b = SymmetryOperation.from_integer_code(self.symop_b[0])
            seitz_a = s_a.seitz_matrix
            t_ab = np.zeros((4, 4))
            t_ab[:3, 3] = self.frac_shift
            self.seitz_b = s_b.seitz_matrix + t_ab
            self.symm_str = encode_symm_str(self.seitz_b[:3, :3], self.seitz_b[:3, 3])
        return self.transform_ab

    @property
    def separations(self):
       return np.array((self.closest_separation, self.centroid_separation, self.com_separation))

    def __eq__(self, other):
        return np.allclose(self.separations, other.separations)

    def transform_string(self):
        if self.transform_ab is None:
            return "none"
        if self.crystal_transform:
            return self.symm_str
        return str(self.transform_ab)

    def __repr__(self):
        return f"<Dimer: d={self.separation:.2f} symm={self.transform_string()}>"

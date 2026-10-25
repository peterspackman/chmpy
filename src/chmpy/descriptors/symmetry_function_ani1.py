from itertools import combinations_with_replacement

import numpy as np
from scipy.spatial import cKDTree as KDTree

from chmpy.core.element import Element

SYMF_DEFAULT_PARAMETERS = {
    "radial_eta": 50,
    "angular_eta": 5,
    "radial_cutoff": 9.3,
    "angular_cutoff": 6.27,
    "zeta": 70,
    "radial_rs": np.linspace(1, 9.3, 32),
    "angular_rs": np.linspace(1, 6.27, 8),
    "theta": np.linspace(0, np.pi, 8),
    "only_intermolecular": False,
    "separate_radial": True,
    "separate_angular": True,
}


class SymmetryFunctionsANI1:
    """Atomic environment symmetry functions, based on the ANI-1
    descriptors [1]

    References
    ----------
    [1] J.S. Smith et al. Chem. Sci., 2017, 8, 3192-3203
        https://dx.doi.org/10.1039/C6SC05720A
    """

    def __init__(self, labels, n_radial, n_angular, n_theta):
        self.labels = labels
        self.unique_atoms = np.unique(labels).tolist()
        self.pairs = [
            tuple(sorted(x))
            for x in combinations_with_replacement(self.unique_atoms, 2)
        ]
        # atomic numbers matching the ordering of unique_atoms and pairs,
        # used to lay out the radial and angular blocks
        self.species = [Element[x].atomic_number for x in self.unique_atoms]
        self.species_pairs = [
            (Element[a].atomic_number, Element[b].atomic_number) for a, b in self.pairs
        ]
        self.radial = np.zeros(
            (self.n_atoms, self.n_atom_types * n_radial), dtype=np.float64
        )
        self.angular = np.zeros(
            (self.n_atoms, self.n_pairs * n_angular * n_theta), dtype=np.float64
        )
        self.nangular = n_angular * n_theta
        self.nradial = n_radial

    @property
    def n_atoms(self):
        return len(self.labels)

    @property
    def n_atom_types(self):
        return len(self.unique_atoms)

    @property
    def n_pairs(self):
        return len(self.pairs)

    def set_radial(self, i, arr):
        self.radial[i, :] = arr[:]

    def set_angular(self, i, arr):
        self.angular[i, :] = arr[:]

    def get_angular(self, i, pair):
        idx = self.pairs.index(tuple(sorted(pair)))
        l, u = idx * self.nangular, (idx + 1) * self.nangular
        return self.angular[i, l:u]

    def get_radial(self, i, atom):
        idx = self.unique_atoms.index(atom)
        l, u = idx * self.nradial, (idx + 1) * self.nradial
        return self.radial[i, l:u]

    def as_flat_matrix(self):
        return np.hstack((self.radial, self.angular))

    def __repr__(self):
        return f"<SymmetryFunctionsANI1: natom={self.n_atoms}>"

    @classmethod
    def from_crystal(cls, crys, **kwargs):
        molecules = crys.symmetry_unique_molecules()
        if len(molecules) > 1:
            raise NotImplementedError("Only implemented for Z'=1 crystals")
        args = SYMF_DEFAULT_PARAMETERS.copy()
        args.update(kwargs)

        assert len(args["angular_rs"]) == len(args["theta"]), (
            "Angular R_s and Theta lists must be the same length."
        )

        # Get xyz data
        mol = molecules[0]

        # Generate a finite cluster containing every atom within the largest
        # cutoff of any atom in the molecule
        nearest_atoms = atoms_in_radius(
            crys, max((args["radial_cutoff"], args["angular_cutoff"])), molecule=mol
        )
        cm_xyz = mol.positions.copy()
        labels = [e.symbol for e in mol.elements]

        if not args["separate_radial"]:
            raise NotImplementedError
        if not args["separate_angular"]:
            raise NotImplementedError

        symf = cls(
            labels, len(args["radial_rs"]), len(args["angular_rs"]), len(args["theta"])
        )

        for idx, cm_atom_xyz in enumerate(cm_xyz):
            symf.set_radial(
                idx,
                calc_radial_function(
                    cm_xyz,
                    nearest_atoms,
                    cm_atom_xyz,
                    args["radial_cutoff"],
                    args["radial_rs"],
                    args["radial_eta"],
                    args["only_intermolecular"],
                    args["separate_radial"],
                    species=symf.species,
                ),
            )
            symf.set_angular(
                idx,
                calc_angular_function(
                    cm_xyz,
                    nearest_atoms,
                    cm_atom_xyz,
                    args["angular_cutoff"],
                    args["angular_rs"],
                    args["theta"],
                    args["angular_eta"],
                    args["zeta"],
                    args["only_intermolecular"],
                    args["separate_angular"],
                    species_pairs=symf.species_pairs,
                ),
            )
        return symf


class NearestAtoms:
    def __init__(self, atomic_numbers, atomic_positions):
        self.nums = atomic_numbers
        self.positions = atomic_positions
        self.tree = KDTree(self.positions)

    def neighbours(self, pt, r):
        idx = self.tree.query_ball_point(pt, r)
        return self.nums[idx], self.positions[idx]


def atoms_in_radius(c, radius, molecule=None):
    """
    Periodic atoms within `radius` of the Cartesian origin or, if `molecule`
    is given, within `radius` of any atom in `molecule`.
    """
    if molecule is None:
        atoms_dict = c.atoms_in_radius(radius)
    else:
        centroid = molecule.positions.mean(axis=0)
        extent = np.max(np.linalg.norm(molecule.positions - centroid, axis=1))
        atoms_dict = c.atoms_in_radius(radius + extent, origin=centroid)
    return NearestAtoms(atoms_dict["element"], atoms_dict["cart_pos"])


def calc_radial_function(
    cm_xyz,
    nearest_atoms,
    cm_atom_xyz,
    cutoff,
    r_s_lst,
    eta,
    only_intermolecular=False,
    separate_radial=True,
    species=None,
):
    """
    Radial function (eq (3) of Smith et al. (2017))
    For a given atom returns a list of radial functions of length r_s

    mol_xyz : list of xyz of all atoms in a finite cluster larger than cutoff
        not in the central molecule
    cm_atom_xyz : xyz of a given atom in the central molecule
    cutoff : float, max L1 distance where atom contributions are included
    r_s : list of hyperparameters controlling position of each gaussian
    eta : float, hyperparameter controlling width of each gaussian
    species : atomic numbers giving the order of the per-element blocks
        (default: the unique elements among the neighbours)
    """
    els, pos = nearest_atoms.neighbours(cm_atom_xyz, cutoff)
    centered = pos - cm_atom_xyz

    distances = np.linalg.norm(centered, axis=1)
    center_atom = distances < 1e-3
    fc = 0.5 * np.cos(distances * np.pi / cutoff) + 0.5
    fc[center_atom] = 0

    if only_intermolecular:
        for cm_atom_xyz in cm_xyz:
            fc[(np.linalg.norm(pos - cm_atom_xyz, axis=1) < 1e-3)] = 0

    g = np.square(distances[:, np.newaxis] - r_s_lst)

    unique_atoms = np.unique(els) if species is None else species

    if separate_radial:
        G1_list = []
        for atom in unique_atoms:
            fc_temp = np.copy(fc)
            fc_temp[els != atom] = 0
            G1_list.append(fc_temp.dot(np.exp(g * -eta)))
        G1_list = np.array(G1_list)
        return np.ravel(G1_list)

    G1 = fc.dot(np.exp(g * -eta))

    return np.ravel(G1)


def calc_angular_function(
    cm_xyz,
    nearest_atoms,
    cm_atom_xyz,
    cutoff,
    r_s_lst,
    theta_lst,
    eta,
    zeta,
    only_intermolecular=False,
    separate_angular=True,
    species_pairs=None,
):
    """
    Angular function (eq (4) of Smith et al. (2017))
    For a given atom returns a list of angular functions of length r_s
    (and theta - they must be the same length)

    mol_xyz : list of xyz of all atoms in a finite cluster larger than cutoff
        not in the central molecule
    cm_atom_xyz : xyz of a given atom in the central molecule
    cutoff : float, max L1 distance where atom contributions are included
    r_s : list of hyperparameters controlling position of each gaussian
    theta: list of hyperparameters contrlling the angular position of each gaussian
    eta : float, hyperparameter controlling width of each gaussian
    zeta : float, hyperparameter controlling the width of each gaussian
    species_pairs : (Z1, Z2) pairs giving the order of the per-pair blocks
        (default: unique pairs among the neighbours, ordered by atomic number)
    """

    front_factor = 2 ** (1 - zeta)

    els, pos = nearest_atoms.neighbours(cm_atom_xyz, cutoff)
    centered = pos - cm_atom_xyz
    distances = np.linalg.norm(centered, axis=1)

    if np.min(distances) > 0.001:
        raise Exception(
            """
        Central Atom is probabaly not at (0, 0, 0). This means that atoms in
        the original molecule cannot be found in the cluster (ball).
        Check line 37 if "direct" should be replaced by "direct.T"
        """
        )

    center_atom = distances < 1e-3

    distances.shape[0]

    fc = 0.5 * np.cos(distances * np.pi / cutoff) + 0.5
    fc[center_atom] = 0
    distances[center_atom] = 1

    if only_intermolecular:
        for cm_atom_xyz in cm_xyz:
            fc[(np.linalg.norm(pos - cm_atom_xyz, axis=1) < 1e-3)] = 0

    normed = centered[:] / distances[:, np.newaxis]
    angles = np.arccos(np.clip(np.matmul(normed, normed.T), -1, 1))
    angle_diff = (1 + np.cos(angles[:, :, np.newaxis] - np.array(theta_lst))) ** zeta
    r1r2 = 0.5 * (distances[:, np.newaxis] + distances)
    r1r2 = np.exp(-eta * (r1r2[:, :, np.newaxis] - r_s_lst) ** 2)

    fjfk = np.outer(fc, fc)

    # Diagonal terms correspond to the angle of an atom with itself.
    # Pushed them to zero by the following line

    fjfk = np.triu(fjfk, k=1)

    if species_pairs is None:
        unique_atoms = np.unique(els)
        species_pairs = [
            (a, b) for i, a in enumerate(unique_atoms) for b in unique_atoms[i:]
        ]

    if separate_angular is True:
        G2_list = []
        for a, b in species_pairs:
            is_a, is_b = els == a, els == b
            mask = np.outer(is_a, is_b) | np.outer(is_b, is_a)
            fjfk_temp = np.where(mask, fjfk, 0)
            G2 = front_factor * np.einsum("ij,ijk,ijl->kl", fjfk_temp, r1r2, angle_diff)
            G2_list.append(G2)

        return np.ravel(G2_list)

    G2 = front_factor * np.einsum("ij,ijk,ijl->kl", fjfk, r1r2, angle_diff)

    return np.ravel(G2)

from collections import namedtuple
import numpy as np

EEM_KAPPA = 0.529176
EEM_PARAMETERS = {
    "H": (0.20606, 1.31942),
    "Li": (0.36237, 0.65932),
    "B": (0.36237, 0.65932),
    "C": (0.36237, 0.65932),
    "N": (0.49279, 0.69038),
    "O": (0.73013, 1.08856),
    "F": (0.72052, 1.45328),
    "Na": (0.36237, 0.65932),
    "Mg": (0.36237, 0.65932),
    "Si": (0.36237, 0.65932),
    "P": (0.36237, 0.65932),
    "S": (0.62020, 0.41280),
    "Cl": (0.36237, 0.65932),
    "K": (0.36237, 0.65932),
    "Ca": (0.36237, 0.65932),
    "Fe": (0.36237, 0.65932),
    "Cu": (0.36237, 0.65932),
    "Zn": (0.36237, 0.65932),
    "Br": (0.70052, 1.09108),
    "I": (0.68052, 0.61328),
    "*": (0.20606, 1.31942),
}


class EEM:
    "Class to handle calculation of electronegativity equilibration method charges"

    @staticmethod
    def calculate_charges(mol):
        """
        Calculate the partial atomic charges based on the EEM method.

        Args:
            mol (Molecule): The molecule with atoms where partial charges are desired

        Returns:
            np.ndarray: the partial charges associated the atoms in `mol`
        """
        A = []
        B = []
        for el in mol.elements:
            a, b = EEM_PARAMETERS.get(el.symbol, EEM_PARAMETERS["*"])
            A.append(a)
            B.append(b)
        N = len(mol)
        M = np.zeros((N + 1, N + 1))
        M[-1, :-1] = 1
        M[:-1, -1] = -1
        dists = mol.distance_matrix
        idx = np.triu_indices(N, k=1)
        M[idx] = EEM_KAPPA / dists[idx]
        idx = np.tril_indices(N, k=-1)
        M[idx] = EEM_KAPPA / dists[idx]
        np.fill_diagonal(M, B)
        M[N, N] = 0.0
        y = np.zeros(N + 1)
        y[:N] -= A
        y[N] = mol.charge
        return np.linalg.solve(M, y)[:N]

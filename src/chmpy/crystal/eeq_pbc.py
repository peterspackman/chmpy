import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.special import erf, erfc

from chmpy.core.eeq import CHI, COVALENT_D3, ETA, KCN_PARAM, WIDTH
from chmpy.util.unit import ANGSTROM_TO_BOHR


def find_neighbors_pbc(crystal, cutoff=6.0):
    """
    Find all neighbors within a cutoff distance considering periodic boundary conditions.
    This is based on the unit_cell_connectivity method but optimized for EEQ calculations.

    Args:
        crystal: The Crystal instance
        cutoff (float): Maximum distance to consider for neighbors in Angstroms

    Returns:
        list: List of tuples (i, j, distance, cell_translation) for all neighbor pairs
    """

    frac_cutoff = cutoff / np.array(crystal.unit_cell.lengths)
    bounds = (
        np.floor(-frac_cutoff).astype(int) - 1,
        np.ceil(frac_cutoff).astype(int) + 1,
    )

    slab = crystal.slab(bounds=bounds)

    n_uc = slab["n_uc"]
    uc_nums = slab["element"][:n_uc]
    uc_pos = crystal.to_cartesian(slab["frac_pos"][:n_uc])
    nums = slab["element"]
    neighbour_pos = crystal.to_cartesian(slab["frac_pos"][n_uc:])

    tree_uc = KDTree(uc_pos)

    dist_uc = tree_uc.sparse_distance_matrix(tree_uc, max_distance=cutoff)

    neighbors = [[] for _ in range(n_uc)]
    for (i, j), d in dist_uc.items():
        if d > 1e-3:
            neighbors[i].append((j, nums[j], d))

    tree_neighbors = KDTree(neighbour_pos)
    dist_neighbors = tree_uc.sparse_distance_matrix(tree_neighbors, max_distance=cutoff)

    for (uc_atom, neighbor_atom), d in dist_neighbors.items():
        uc_idx = neighbor_atom % n_uc
        if d > 1e-3:
            neighbors[uc_atom].append((uc_idx, uc_nums[uc_idx], d))

    return [np.array(n) for n in neighbors]


def calculate_coordination_numbers_crystal(crystal, cutoff=None):
    """
    Calculate coordination numbers for all atoms in a crystal using the EEQ method.

    Args:
        crystal: The Crystal instance
        cutoff (float): Cutoff distance for neighbor search in Angstroms

    Returns:
        np.ndarray: Array of coordination numbers for each atom
    """
    kcn_value = 7.5  # Constant from the C++ implementation

    # Get atomic numbers
    uc_atoms = crystal.unit_cell_atoms()
    atomic_numbers = uc_atoms["element"]
    N = len(atomic_numbers)
    uc_cov = COVALENT_D3[atomic_numbers]

    # Initialize coordination numbers
    cn = np.zeros(N)

    if cutoff is None:
        cutoff = np.max(uc_cov) * 4  # 1.4 * sum of cov should give around erf(-3)

    # Find all neighbors within cutoff distance
    neighbors = find_neighbors_pbc(crystal, cutoff)

    for i, atom_neighbors in enumerate(neighbors):
        if atom_neighbors.shape[0] < 1:
            continue
        rc = (
            uc_cov[i] + COVALENT_D3[atom_neighbors[:, 1].astype(int)]
        ) * ANGSTROM_TO_BOHR

        dists = atom_neighbors[:, 2] * ANGSTROM_TO_BOHR

        count = 0.5 * (1.0 + erf(-kcn_value * (dists - rc) / rc))

        cn[i] += 0.5 * count.sum()
        cn[atom_neighbors[:, 0].astype(int)] += 0.5 * count[:]

    return cn


def build_a_matrix_crystal(crystal, cutoff=12.0, wolf_eta=0.2):
    uc_atoms = crystal.unit_cell_atoms()
    atomic_numbers = uc_atoms["element"]
    N = len(atomic_numbers)
    widths2 = WIDTH[atomic_numbers] ** 2

    neighbors = find_neighbors_pbc(crystal, cutoff=cutoff)

    wolf_eta /= ANGSTROM_TO_BOHR
    cutoff *= ANGSTROM_TO_BOHR
    wolf_self = erfc(wolf_eta * cutoff) / cutoff

    A = np.zeros((N + 1, N + 1))

    for i, atom_neighbors in enumerate(neighbors):
        if atom_neighbors.shape[0] < 1:
            continue

        j = atom_neighbors[:, 0].astype(int)
        ri_squared = widths2[i]
        rj_squared = widths2[j]
        r = atom_neighbors[:, 2]
        r2 = (r * ANGSTROM_TO_BOHR) ** 2

        gamma = 1.0 / (ri_squared + rj_squared)
        w_term = erfc(wolf_eta * r) / r
        values = erf(np.sqrt(r2 * gamma)) / np.sqrt(r2) * (w_term - wolf_self)
        A[i, j] += 0.5 * values
        A[j, i] += 0.5 * values

    sqrt_pi_fac = np.sqrt(2.0 / np.pi)
    diagonal_values = ETA[atomic_numbers] + sqrt_pi_fac / WIDTH[atomic_numbers]
    np.fill_diagonal(A[:N, :N], diagonal_values)

    A[N, :N] = 1.0
    A[:N, N] = 1.0
    A[N, N] = 0.0

    return A


def build_x_vector_crystal(crystal, cn, charge=0.0):
    """
    Build the X vector for EEQ charge calculation in periodic systems.

    Args:
        crystal: The Crystal instance
        cn (np.ndarray): Array of coordination numbers for each atom
        charge (float): Total charge of the system

    Returns:
        np.ndarray: X vector for the EEQ calculation
    """
    # Get atomic numbers from unit cell
    uc_atoms = crystal.unit_cell_atoms()
    atomic_numbers = uc_atoms["element"]
    N = len(atomic_numbers)

    eps = 1e-14  # Avoid singularity with 0
    X = np.empty(N + 1)

    # Chemical potential terms with coordination number correction
    X[:N] = -CHI[atomic_numbers] + cn * KCN_PARAM[atomic_numbers] / np.sqrt(cn + eps)

    # Total charge constraint
    X[N] = charge

    return X


def calculate_eeq_charges_crystal(crystal, charge=0.0, cutoff=12.0):
    """
    Calculate EEQ partial charges for a crystal structure.

    Args:
        crystal: The Crystal instance
        charge (float): Total charge of the system
        cutoff (float): Cutoff distance for neighbor search in Angstroms

    Returns:
        np.ndarray: Array of partial charges for each atom in the unit cell
    """
    # Calculate coordination numbers
    cn = calculate_coordination_numbers_crystal(crystal, cutoff=None)

    # Build A matrix
    A = build_a_matrix_crystal(crystal, cutoff=cutoff)

    # Build X vector
    X = build_x_vector_crystal(crystal, cn, charge)

    # Solve linear system A * Q = X
    Q = np.linalg.solve(A, X)

    # Return charges (excluding Lagrange multiplier)
    return Q[:-1]

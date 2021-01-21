import numpy as np
from scipy.spatial.distance import pdist
from chmpy.util.unit import units

def surface_charge(charges, epsilon):
    return charges * (epsilon - 1) / (epsilon + 0.5)

def c_matrix(points, areas):
    #k = 1.07
    # from Tomasi J, Mennucci B, Cammi R, Chem. Rev. 2005, 105, 2999-3093
    # on pp. 3013
    k = 1.0694
    N = areas.shape[0]
    c = np.empty((N, N))
    c[np.triu_indices(N, k=1)] = 1 / pdist(points)
    c += c.T
    np.fill_diagonal(c, 0.0)
    diag = k * np.sqrt(4 * np.pi / areas)
    return diag, c

def minimize_cosmo_energy(points, areas, charges, unit="angstrom", max_iter=50, epsilon=78.39):
    from chmpy.util.unit import BOHR_TO_ANGSTROM, AU_TO_KJ_PER_MOL
    if unit.lower() == "angstrom":
        points = points / BOHR_TO_ANGSTROM
        areas = areas / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM)

    diis_start = 1
    diis_threshold = 1.0e-6
    inverse = False
    surface_area_minimum = 1.0e-6
    threshold = 1.0e-9
    initial_charge_scale_factor = 0.05

    q = surface_charge(charges, epsilon)
    diag, C = c_matrix(points, areas)
    d0 = 1.0 / diag

    qprev = initial_charge_scale_factor * q * d0

    prev_q = []
    prev_dq = []

    N = q.shape[0]

    #print("{:>3s} {:>14s} {:>9s} {:>16s}".format("N", "Energy", "Q", "Error"))
    for k in range(1, max_iter):
        vpot = np.sum(qprev * C, axis=1)
        qcur = (q - vpot) * d0
        dq = qcur - qprev

        if k >= diis_start:
            prev_q.append(qcur)
            prev_dq.append(dq)

        ndiis = len(prev_dq)
        if ndiis > 1:
            rhs = np.zeros(ndiis + 1)
            rhs[ndiis] = -1

            # setup system of linear equations
            B = np.zeros((ndiis + 1, ndiis + 1))
            B[:][ndiis] = -1
            B = B.transpose()
            B[:][ndiis] = -1
            B[ndiis][ndiis] = 0

            for i, qi in enumerate(prev_dq):
                for j, qj in enumerate(prev_dq):
                    B[i, j] = qi.dot(qj)

            # solve the system of linear equations
            c = np.linalg.solve(B, rhs)[:ndiis]

            # better guess
            qcur = np.zeros(np.shape(dq))
            for i in range(ndiis):
                qcur += c[i] * prev_q[i]

            # remove, if needed, items from the DIIS space
            sel = np.where(np.abs(c) < diis_threshold)[0]
            for i in reversed(sel):
                prev_q.pop(i)
                prev_dq.pop(i)

        abs_err = np.sqrt(dq.dot(dq)/N)
        e_q = -0.5 * q.dot(qcur)
        #print("{:3d} {:14.8f} {:9.5f} {:16.9f}".format(k, e_q, qcur.sum(), abs_err))
        if abs_err < threshold:
            break
        qprev = qcur[:]


    G = -0.5 * q.dot(qcur)
    print("Energy: {:16.9f} kJ/mol".format(G * AU_TO_KJ_PER_MOL))
    return qcur

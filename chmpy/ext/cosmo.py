import logging
import numpy as np
from collections import namedtuple
from scipy.spatial.distance import pdist
from chmpy.util.unit import units
from chmpy.ext.solvation_parameters import DIELECTRIC_CONSTANTS

LOG = logging.getLogger(__name__)
COSMOResult = namedtuple("COSMOResult", "qinit qmin total_energy")


def surface_charge(charges, epsilon, x=0.5):
    return charges * (epsilon - 1) / (epsilon + x)


def coulomb_matrix(points):
    N = points.shape[0]
    C = np.zeros((N, N))
    np.fill_diagonal(C, 0.0)
    C[np.triu_indices(N, k=1)] = 1 / pdist(points)
    C += C.T
    return C


def self_interaction_term(areas, k=1.0694):
    Sii = 3.8 / np.sqrt(areas)
    return Sii


def minimize_cosmo_energy(points, areas, charges, **kwargs):

    from chmpy.util.unit import BOHR_TO_ANGSTROM, AU_TO_KJ_PER_MOL

    if kwargs.get("unit", "angstrom").lower() == "angstrom":
        points = points / BOHR_TO_ANGSTROM
        areas = areas / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM)

    diis_tolerance = kwargs.get("diis_tolerance", 1e-6)
    diis_start = 1
    surface_area_minimum = 1.0e-6
    convergence = kwargs.get("convergence", 1.0e-6)
    initial_charge_scale_factor = 0.0694

    solvent = kwargs.get("solvent", "water")
    epsilon = DIELECTRIC_CONSTANTS.get(solvent, 0.0)
    LOG.debug("Using dielectric constant of %.2f for solvent '%s'", epsilon, solvent)
    qinit = surface_charge(charges, epsilon)
    C = coulomb_matrix(points)
    Sii = self_interaction_term(areas)
    d0 = 1.0 / Sii

    qprev = initial_charge_scale_factor * qinit * d0
    prev_q = []
    prev_dq = []

    N = qinit.shape[0]

    qcur = np.empty_like(qprev)
    LOG.debug("{:>3s} {:>14s} {:>9s} {:>16s}".format("N", "Energy", "Q", "Error"))

    for k in range(1, kwargs.get("max_iter", 50)):
        vpot = np.sum(qprev * C, axis=1)
        qcur = (qinit - vpot) * d0
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
            B[:, ndiis] = -1
            B = B.T
            B[:, ndiis] = -1
            B[ndiis, ndiis] = 0

            for i, qi in enumerate(prev_dq):
                for j, qj in enumerate(prev_dq):
                    B[i, j] = qi.dot(qj)

            c = np.linalg.solve(B, rhs)[:ndiis]

            qcur = 0.0
            for i in range(ndiis):
                qcur += c[i] * prev_q[i]

            sel = np.where(np.abs(c) < diis_tolerance)[0]
            for i in reversed(sel):
                prev_q.pop(i)
                prev_dq.pop(i)

        rms_err = np.sqrt(dq.dot(dq) / N)
        e_q = -0.5 * qinit.dot(qcur)
        LOG.debug("{:3d} {:14.8f} {:9.5f} {:16.9f}".format(k, e_q, qcur.sum(), rms_err))
        if rms_err < convergence:
            break
        qprev[:] = qcur[:]

    G = -0.5 * qinit.dot(qcur)
    LOG.debug("Energy: %16.9f kJ/mol", G * AU_TO_KJ_PER_MOL)
    return COSMOResult(qinit, qcur, G)

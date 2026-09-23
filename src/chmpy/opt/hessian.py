"""A starting curvature model, built from the geometry alone.

Bonds are stiff and lattice modes are soft, by two orders of magnitude in a
molecular crystal. An optimiser starting from the identity has to discover that
from curvature pairs, taking short steps along the soft directions until it
does; starting from a model of it costs no energy evaluations at all.

The model here is the usual stretch one -- a spring along every interatomic
vector, stiffer for close pairs -- in the exponential form of Packwood et al.'s
preconditioner, `k(r) = mu exp(-A (r / r_nn - 1))`.

It is built for any `Coordinates` without per-class code. The model energy
depends only on interatomic
distances, so its Hessian is `G^T diag(k) G`, where `G[p, k]` is how much pair
`p` stretches per unit of degree of freedom `k`. That row of `G` is obtained by
moving each degree of freedom and watching the pair vectors -- which costs
nothing, since no energy is evaluated. Symmetry-adapted coordinates, cell
strains and rigid bodies all come out right, including the periodic images,
because the pair vectors carry their lattice translations with them.
"""

from __future__ import annotations

import numpy as np

from chmpy.calc.neighbors import NeighborList

#: how fast the stiffness falls off with distance, in units of the
#: nearest-neighbour distance
DEFAULT_DECAY = 3.0
#: added to the diagonal, as a fraction of its mean, so the model is definite
#: along directions no pair constrains
DEFAULT_RIDGE = 0.05
#: overall stiffness after normalisation; 1 matches the identity the optimiser
#: would otherwise start from
DEFAULT_GAMMA = 1.0


def stretch_hessian(
    coordinates,
    gamma: float = DEFAULT_GAMMA,
    decay: float = DEFAULT_DECAY,
    cutoff: float | None = None,
    ridge: float = DEFAULT_RIDGE,
    step: float = 1e-5,
) -> np.ndarray:
    """A model Hessian in the optimiser's scaled coordinates.

    The model is normalised to a mean diagonal of `gamma`, so what it supplies
    is the *shape* of the curvature -- which directions are stiff relative to
    which -- and not its magnitude. That distinction is the difference between
    this helping and hurting. Absolute spring constants in eV/A^2 are wrong by
    a large factor for any given material (measured: about ninety times too
    stiff for Lennard-Jones argon), and a model that is uniformly too stiff
    makes every Newton step too short by that factor, so the optimiser crawls.
    The anisotropy, meanwhile, is roughly right, and it is what the identity
    gets wrong; BFGS learns an overall scale within a couple of steps but takes
    far longer to learn a hundredfold spread across directions.

    Args:
        coordinates: the parameterisation to build it for
        gamma: mean diagonal of the returned model
        decay: exponential fall-off of the stiffness with distance
        cutoff: pair cutoff in Angstroms; defaults to three times the
            nearest-neighbour distance
        ridge: added to the diagonal as a fraction of its mean, so that
            directions no pair constrains still have positive curvature
        step: displacement used to measure how pairs stretch per degree of
            freedom. No energies are evaluated, so this is cheap and its only
            requirement is to stay in the linear regime.

    Returns:
        (n_dof, n_dof) positive-definite curvature model, ready to hand to
        `TrustRegion(..., hessian=...)`
    """
    system = coordinates.system
    x = coordinates.get()
    n_dof = coordinates.n_dof
    if n_dof == 0:
        return np.zeros((0, 0))

    nearest = _nearest_neighbour_distance(system)
    neighbors = NeighborList(cutoff or 3.0 * nearest, full=False)
    pairs = neighbors.compute(system)
    if len(pairs) == 0:
        return np.eye(n_dof)

    stiffness = np.exp(-decay * (pairs.distances / nearest - 1.0))
    directions = pairs.vectors / pairs.distances[:, None]

    # how much each pair stretches per unit of each degree of freedom
    stretch = np.zeros((len(pairs), n_dof))
    for k in range(n_dof):
        shifted = x.copy()
        shifted[k] = x[k] + step
        coordinates.set(shifted)
        plus = _pair_vectors(coordinates.system, pairs)
        shifted[k] = x[k] - step
        coordinates.set(shifted)
        minus = _pair_vectors(coordinates.system, pairs)
        stretch[:, k] = np.einsum("pi,pi->p", directions, plus - minus) / (2 * step)
    coordinates.set(x)

    hessian = stretch.T @ (stiffness[:, None] * stretch)

    scale = coordinates.scale()
    hessian /= np.outer(scale, scale)

    diagonal = float(np.mean(np.diag(hessian)))
    if not diagonal > 0:
        return np.eye(n_dof)
    hessian *= gamma / diagonal
    hessian += ridge * gamma * np.eye(n_dof)
    return 0.5 * (hessian + hessian.T)


def _pair_vectors(system, pairs) -> np.ndarray:
    """The same pairs, re-measured at the current geometry.

    The pair list is deliberately *not* rebuilt: the model wants the derivative
    of a fixed set of distances, and a rebuilt list at a displaced geometry
    might not contain the same pairs.
    """
    positions = system.positions
    vectors = positions[pairs.j] - positions[pairs.i]
    if np.any(system.pbc):
        vectors = vectors + pairs.shifts @ system.cell
    return vectors


def _nearest_neighbour_distance(system, fallback: float = 1.5) -> float:
    """The shortest interatomic distance, which sets the stiffness scale."""
    for cutoff in (3.0, 6.0, 12.0):
        pairs = NeighborList(cutoff, full=False).compute(system)
        if len(pairs):
            return max(float(pairs.distances.min()), 0.5)
    return fallback

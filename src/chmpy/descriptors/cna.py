"""Common neighbour analysis (CNA) of local crystal structure.

Each bond from an atom to one of its neighbours is labelled by three numbers:
how many neighbours the pair has in common (`n_cn`), how many bonds those
common neighbours share (`n_b`), and how many bonds the largest connected
cluster of those holds (`n_lcb`). An atom's multiset of labels identifies its
local structure:

    FCC  12 x (4, 2, 1)
    HCP   6 x (4, 2, 1) + 6 x (4, 2, 2)
    ICO  12 x (5, 5, 5)
    BCC   8 x (6, 6, 6) + 6 x (4, 4, 4)

Everything about one atom happens inside its own neighbour shell, so the
analysis is built from layers, each usable on its own:

* `chmpy.calc.nearest.nearest_neighbors` finds the shell.
* `neighbor_bonds` records which neighbours are bonded to which as bitmasks,
  one integer per neighbour, bit `b` of entry `a` set when neighbours `a` and
  `b` are bonded. The common neighbours of the atom and neighbour `a` are then
  exactly the set bits of entry `a`.
* `cna_signatures` turns the bitmasks into `(n_cn, n_b, n_lcb)`, and
  `classify` matches them against `PATTERNS`.

Three ways of drawing the bonds are provided:

* `fixed_cna` uses one cutoff throughout (Honeycutt and Andersen, J. Phys.
  Chem. 91, 4950 (1987)).
* `adaptive_cna` sets each atom's cutoff from its own neighbour distances
  (Stukowski, Modelling Simul. Mater. Sci. Eng. 20, 045021 (2012)).
* `interval_cna` needs no cutoff, and tolerates far more thermal disorder
  (Larsen, arXiv:2003.08879). Bonds of equal length do not bound an
  interval (`INTERVAL_TOLERANCE`), so that the answer for an atom does not
  depend on the rounding error between them, or on the orientation of the
  structure.

Atoms are processed in chunks, on `workers` threads; numpy releases the GIL
for the work inside each chunk.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from functools import cache

import numpy as np

from chmpy.calc.nearest import nearest_neighbors

#: atoms processed together; small enough that the temporaries stay in cache
CHUNK = 2048

#: the most neighbours a shell may have, set by the width of the bitmasks
MAX_NEIGHBORS = 32

#: adaptive cutoff relative to the mean nearest-neighbour distance in FCC/HCP,
#: or to the lattice parameter estimated from the first two shells in BCC
FCC_CUTOFF_FACTOR = (1.0 + np.sqrt(2.0)) / 2.0
BCC_CUTOFF_FACTOR = 1.207

#: interval CNA considers bonds up to this multiple of the first-shell
#: distance, two thirds of the way to the second shell
INTERVAL_FRACTION = 1.0 / 3.0 + 2.0 / 3.0 * np.sqrt(2.0)

#: narrowest interval interval CNA accepts, relative to the bond length; bonds
#: of equal length differ by rounding error, and an interval between them is
#: not one
INTERVAL_TOLERANCE = 1e-9


class StructureType(IntEnum):
    """Local structures CNA identifies, with their conventional numbering."""

    OTHER = 0
    FCC = 1
    HCP = 2
    BCC = 3
    ICO = 4


#: the signatures that make up each structure, keyed by shell size
PATTERNS = {
    12: {
        StructureType.FCC: {(4, 2, 1): 12},
        StructureType.HCP: {(4, 2, 1): 6, (4, 2, 2): 6},
        StructureType.ICO: {(5, 5, 5): 12},
    },
    14: {
        StructureType.BCC: {(6, 6, 6): 8, (4, 4, 4): 6},
    },
}


# -- the layers ---------------------------------------------------------------


def neighbor_bonds(vectors, cutoff, workers=-1) -> np.ndarray:
    """Which neighbours of each atom are bonded to each other.

    Args:
        vectors: (N, k) neighbour vectors from each atom, as in
            `NearestNeighbors.vectors`; NaN entries bond to nothing
        cutoff: bond length cutoff, a scalar or one per atom (N,)
        workers: threads to use, -1 for one per CPU

    Returns:
        (N, k) uint32 bitmasks: bit `b` of entry `[i, a]` is set when
        neighbours `a` and `b` of atom `i` are no further apart than the cutoff
    """
    flat, n, k = _flatten(vectors)
    cutoff2 = np.broadcast_to(np.asarray(cutoff, dtype=np.float64) ** 2, (n,))
    out = np.empty((n, k), dtype=np.uint32)

    def work(start, stop):
        distance2 = _pair_distances2(flat[start:stop], k)
        out[start:stop] = _pack(distance2 <= cutoff2[start:stop, None], k)

    _for_chunks(n, work, workers)
    return out


def cna_signatures(bonds, workers=-1) -> np.ndarray:
    """The CNA signature of every bond from every atom.

    Args:
        bonds: (N, k) bitmasks from `neighbor_bonds`
        workers: threads to use, -1 for one per CPU

    Returns:
        (N, k, 3) int `(n_cn, n_b, n_lcb)` for the bond from each atom to each
        of its neighbours
    """
    bonds = np.asarray(bonds, dtype=np.uint32)
    n, k = bonds.shape
    out = np.empty((n, k, 3), dtype=np.int64)

    def work(start, stop):
        chunk = bonds[start:stop]
        common, shared, chain = _signature(chunk)
        _resolve_chains(chunk, chain)
        out[start:stop] = np.stack([common, shared, chain], axis=-1)

    _for_chunks(n, work, workers)
    return out


def classify(bonds, patterns, workers=-1) -> np.ndarray:
    """Match each atom's signatures against a set of structures.

    Only atoms whose common-neighbour counts fit some structure have their
    signatures computed, and only the bonds whose first two numbers fit have
    their chains followed, so a disordered atom costs little.

    Args:
        bonds: (N, k) bitmasks from `neighbor_bonds`
        patterns: `{structure: {(n_cn, n_b, n_lcb): count}}`, the counts of
            each structure summing to k, as in `PATTERNS`
        workers: threads to use, -1 for one per CPU

    Returns:
        (N,) int8 structure of each atom, `StructureType.OTHER` where none
        matches
    """
    bonds = np.asarray(bonds, dtype=np.uint32)
    out = np.zeros(len(bonds), dtype=np.int8)

    def work(start, stop):
        out[start:stop] = _classify_chunk(bonds[start:stop], patterns)

    _for_chunks(len(bonds), work, workers)
    return out


# -- the analyses -------------------------------------------------------------


def adaptive_cna(positions, cell=None, pbc=None, neighbors=None, workers=-1):
    """Adaptive common neighbour analysis.

    FCC, HCP and icosahedral atoms are tested on their 12 nearest neighbours
    with a cutoff of `(1 + sqrt 2) / 2` times their mean distance; atoms that
    are none of those are tested for BCC on their 14 nearest, with the cutoff
    scaled from the lattice parameter those imply.

    Args:
        positions: (N, 3) Cartesian positions, Angstroms
        cell: (3, 3) lattice vectors as rows, or None for an isolated system
        pbc: (3,) periodic directions, as for `nearest_neighbors`
        neighbors: `NearestNeighbors` with at least 14 per atom, to reuse a
            search already done; found from the other arguments if omitted
        workers: threads to use, -1 for one per CPU

    Returns:
        (N,) int8 `StructureType` of each atom
    """
    neighbors = _neighbors(neighbors, 14, positions, cell, pbc, workers, "adaptive")
    vectors, distances = neighbors.vectors, neighbors.distances
    types = np.zeros(len(neighbors), dtype=np.int8)

    atoms = np.flatnonzero(np.isfinite(distances[:, 11]))
    cutoff = distances[atoms, :12].mean(axis=1) * FCC_CUTOFF_FACTOR
    bonds = neighbor_bonds(vectors[atoms, :12], cutoff, workers)
    types[atoms] = classify(bonds, PATTERNS[12], workers)

    atoms = np.flatnonzero(
        (types == StructureType.OTHER) & np.isfinite(distances[:, 13])
    )
    first, second = distances[atoms, :8], distances[atoms, 8:14]
    lattice = (first.sum(axis=1) / np.sqrt(0.75) + second.sum(axis=1)) / 14
    bonds = neighbor_bonds(vectors[atoms, :14], lattice * BCC_CUTOFF_FACTOR, workers)
    types[atoms] = classify(bonds, PATTERNS[14], workers)
    return types


def interval_cna(positions, cell=None, pbc=None, neighbors=None, workers=-1):
    """Interval common neighbour analysis, which needs no cutoff.

    Rather than fix a cutoff, the bonds among an atom's neighbours are added
    shortest first. Each structure is recognised at the point where the bonds
    drawn so far give its signatures, and the gap to the next bond is the
    interval of cutoffs over which that holds. The structure with the widest
    interval wins.

    The 12 nearest neighbours are tested for FCC, HCP and icosahedral order
    and the 14 nearest for BCC. Only bonds shorter than two thirds of the way
    from the first neighbour shell to the second are considered, and the
    interval after the last of those ends at a longer bond chosen by
    `_interval_ends`.

    Args:
        positions: (N, 3) Cartesian positions, Angstroms
        cell: (3, 3) lattice vectors as rows, or None for an isolated system
        pbc: (3,) periodic directions, as for `nearest_neighbors`
        neighbors: `NearestNeighbors` with at least 14 per atom, to reuse a
            search already done; found from the other arguments if omitted
        workers: threads to use, -1 for one per CPU

    Returns:
        (N,) int8 `StructureType` of each atom
    """
    neighbors = _neighbors(neighbors, 14, positions, cell, pbc, workers, "interval")
    distances = neighbors.distances
    types = np.zeros(len(neighbors), dtype=np.int8)
    full = np.isfinite(distances[:, 13])
    for shell, atoms in (
        (14, np.flatnonzero(full)),
        (12, np.flatnonzero(~full & np.isfinite(distances[:, 11]))),
    ):
        flat, n, _ = _flatten(neighbors.vectors[atoms, :shell])
        lengths = distances[atoms, :shell]
        out = np.zeros(n, dtype=np.int8)

        def work(start, stop, flat=flat, lengths=lengths, out=out):
            out[start:stop] = _interval_chunk(flat[start:stop], lengths[start:stop])

        _for_chunks(n, work, workers)
        types[atoms] = out
    return types


def fixed_cna(positions, cutoff, cell=None, pbc=None, neighbors=None, workers=-1):
    """Conventional common neighbour analysis with one cutoff.

    Atoms with exactly 12 neighbours inside the cutoff are tested for FCC,
    HCP and icosahedral order, those with exactly 14 for BCC, and the rest are
    `StructureType.OTHER`.

    Args:
        positions: (N, 3) Cartesian positions, Angstroms
        cutoff: bond length cutoff, Angstroms
        cell: (3, 3) lattice vectors as rows, or None for an isolated system
        pbc: (3,) periodic directions, as for `nearest_neighbors`
        neighbors: `NearestNeighbors` with at least 15 per atom, to reuse a
            search already done; found from the other arguments if omitted
        workers: threads to use, -1 for one per CPU

    Returns:
        (N,) int8 `StructureType` of each atom
    """
    neighbors = _neighbors(neighbors, 15, positions, cell, pbc, workers, "fixed")
    coordination = (neighbors.distances <= cutoff).sum(axis=1)
    types = np.zeros(len(neighbors), dtype=np.int8)
    for shell, patterns in PATTERNS.items():
        atoms = np.flatnonzero(coordination == shell)
        bonds = neighbor_bonds(neighbors.vectors[atoms, :shell], cutoff, workers)
        types[atoms] = classify(bonds, patterns, workers)
    return types


# -- running in chunks --------------------------------------------------------


def _for_chunks(n, work, workers):
    """Call `work(start, stop)` on consecutive chunks of `n` atoms.

    Each call must write only to its own slice of any shared output.
    """
    chunks = [(start, min(start + CHUNK, n)) for start in range(0, n, CHUNK)]
    workers = (os.cpu_count() or 1) if workers == -1 else int(workers)
    if workers < 1:
        raise ValueError(f"workers must be positive or -1, got {workers}")
    if workers == 1 or len(chunks) < 2:
        for chunk in chunks:
            work(*chunk)
        return
    with ThreadPoolExecutor(min(workers, len(chunks))) as pool:
        for _ in pool.map(lambda chunk: work(*chunk), chunks):
            pass


def _neighbors(neighbors, k, positions, cell, pbc, workers, name):
    if neighbors is None:
        return nearest_neighbors(positions, k, cell=cell, pbc=pbc, workers=workers)
    if neighbors.k < k:
        raise ValueError(f"{name} CNA needs {k} neighbours, got {neighbors.k}")
    return neighbors


# -- bonds --------------------------------------------------------------------


@cache
def _pairs(k):
    return np.triu_indices(k, 1)


@cache
def _bond_operators(k):
    """Linear maps from neighbour vectors to pair differences, and pair bonds
    to bitmasks.

    `difference` takes the (3k,) flattened vectors of one atom to the (3P,)
    components of `v_a - v_b` for each of its P neighbour pairs, component
    major. `pack` takes the (P,) 0/1 bond flags to the (k,) bitmasks: a bond
    between `a` and `b` adds bit `b` to entry `a` and bit `a` to entry `b`.
    Each bit is added at most once, so the sums are exact.
    """
    a, b = _pairs(k)
    n_pairs = len(a)
    pair = np.arange(n_pairs)
    difference = np.zeros((k, 3, 3, n_pairs))
    for c in range(3):
        difference[a, c, c, pair] = 1.0
        difference[b, c, c, pair] = -1.0
    pack = np.zeros((n_pairs, k))
    pack[pair, a] = 2.0**b
    pack[pair, b] = 2.0**a
    return difference.reshape(3 * k, 3 * n_pairs), pack


@cache
def _incidence(k):
    """(P, k) matrix summing a quantity over the pairs each neighbour is in."""
    a, b = _pairs(k)
    incidence = np.zeros((len(a), k))
    incidence[np.arange(len(a)), a] = 1.0
    incidence[np.arange(len(a)), b] = 1.0
    return incidence


def _flatten(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    n, k = vectors.shape[:2]
    if k > MAX_NEIGHBORS:
        raise ValueError(f"at most {MAX_NEIGHBORS} neighbours, got {k}")
    return vectors.reshape(n, 3 * k), n, k


def _pair_distances2(flat, k):
    """(n, P) squared distances between each atom's neighbours, in the pair
    order of `_pairs(k)`."""
    difference, _ = _bond_operators(k)
    components = flat @ difference
    components *= components
    return components.reshape(len(flat), 3, -1).sum(axis=1)


def _pack(bonded, k):
    """(n, P) pair bond flags to (n, k) bitmasks."""
    _, pack = _bond_operators(k)
    return (bonded.astype(np.float64) @ pack).astype(np.uint32)


# -- signatures ---------------------------------------------------------------


def _signature(bonds):
    """`(n_cn, n_b, n_lcb)` for a chunk, with `n_lcb` -1 where undecided.

    The common neighbours of the atom and neighbour `a` are the set bits of
    `bonds[a]`; a common neighbour `b` has `|bonds[a] & bonds[b]|` bonds to
    the others. Summing over the pairs `(a, b)` that are themselves bonded
    gives, for each `a`, twice the number of bonds among its common
    neighbours, the number of those neighbours that hold any bond, and the
    number of pairs of bonds that meet at a neighbour.

    The largest cluster follows without a search when there are at most two
    bonds; when the bonded neighbours are too few for the bonds to fall into
    more than one cluster (`v` vertices split into two parts hold at most
    `C(v - 1, 2)` edges); and when every common neighbour holds exactly two
    bonds, making the bonds disjoint cycles, of which there is one below six
    vertices and at six either a hexagon or two triangles.
    """
    k = bonds.shape[1]
    a, b = _pairs(k)
    incidence = _incidence(k)
    first, second = bonds[:, a], bonds[:, b]
    adjacent = (first >> b.astype(np.uint32)) & 1
    # float, so the sums over pairs go through BLAS
    degree = (np.bitwise_count(first & second) * adjacent).astype(np.float64)

    common = np.bitwise_count(bonds).astype(np.int64)
    shared = (degree @ incidence).astype(np.int64) // 2
    vertices = ((degree > 0).astype(np.float64) @ incidence).astype(np.int64)
    meeting = ((degree * (degree - 1)) @ incidence).astype(np.int64) // 2

    chain = np.full(bonds.shape, -1, dtype=np.int64)
    small = shared <= 1
    chain[small] = shared[small]
    two = shared == 2
    chain[two] = np.where(meeting[two] > 0, 2, 1)
    connected = ~small & ~two & (2 * shared > (vertices - 1) * (vertices - 2))
    chain[connected] = shared[connected]

    cycles = (chain < 0) & (vertices == common) & (shared == common)
    cycles &= meeting == common
    one = cycles & (common < 6)
    chain[one] = shared[one]
    atom, neighbor = np.nonzero(cycles & (common == 6))
    if len(atom):
        chain[atom, neighbor] = np.where(
            _in_triangle(bonds[atom], bonds[atom, neighbor]), 3, 6
        )
    return common, shared, chain


def _in_triangle(rows, common):
    """Whether the lowest common neighbour's two bonded neighbours are bonded.

    Args:
        rows: (P, k) bitmasks of the atom each bond belongs to
        common: (P,) common neighbours of each bond, each holding two bonds
    """
    index = np.arange(len(common))
    seed = common & -common
    ends = rows[index, np.bitwise_count(seed - 1)] & common
    first = ends & -ends
    return (rows[index, np.bitwise_count(first - 1)] & (ends ^ first)) != 0


def _resolve_chains(bonds, chain):
    """Fill the entries of `chain` that `_signature` left undecided."""
    atom, neighbor = np.nonzero(chain < 0)
    if len(atom):
        rows = bonds[atom]
        chain[atom, neighbor] = _largest_cluster(
            rows[np.arange(len(atom)), neighbor], rows
        )


def _largest_cluster(common, rows):
    """Bonds in the largest connected cluster among common neighbours.

    Args:
        common: (P,) bitmask of the common neighbours of each bond
        rows: (P, k) bitmasks of the atom each bond belongs to

    Returns:
        (P,) bond count of the largest cluster
    """
    k = rows.shape[1]
    bits = np.arange(k, dtype=np.uint32)
    member = ((common[:, None] >> bits) & 1).astype(bool)
    edges = np.where(member, rows & common[:, None], 0).astype(np.uint32)
    degree = np.bitwise_count(edges).astype(np.int64)
    remaining = degree.sum(axis=1) // 2
    unvisited = np.bitwise_or.reduce(edges, axis=1)
    largest = np.zeros(len(common), dtype=np.int64)

    # grow one cluster at a time from the lowest unvisited vertex, until the
    # bonds left over could not form a larger one
    active = np.flatnonzero(remaining > 0)
    while len(active):
        cluster = _flood(edges[active], unvisited[active] & -unvisited[active], bits)
        inside = ((cluster[:, None] >> bits) & 1).astype(bool)
        size = np.where(inside, degree[active], 0).sum(axis=1) // 2
        largest[active] = np.maximum(largest[active], size)
        remaining[active] -= size
        unvisited[active] &= ~cluster
        active = active[remaining[active] > largest[active]]
    return largest


def _flood(edges, seed, bits):
    """Every vertex connected to `seed`, as a bitmask per row."""
    reach = seed
    rows = np.arange(len(seed))
    while len(rows):
        inside = ((reach[rows, None] >> bits) & 1).astype(bool)
        grown = reach[rows] | np.bitwise_or.reduce(
            np.where(inside, edges[rows], 0), axis=1
        )
        changed = grown != reach[rows]
        reach[rows] = grown
        rows = rows[changed]
    return reach


def _classify_chunk(bonds, patterns):
    k = bonds.shape[1]
    common = np.bitwise_count(bonds)
    types = np.zeros(len(bonds), dtype=np.int8)

    for structure, pattern in patterns.items():
        if sum(pattern.values()) != k:
            raise ValueError(f"{structure.name} pattern does not have {k} bonds")
        # common-neighbour counts first: they cost nothing and rule out most
        expected = {}
        for (n_cn, _, _), count in pattern.items():
            expected[n_cn] = expected.get(n_cn, 0) + count
        fits = types == StructureType.OTHER
        for n_cn, count in expected.items():
            fits &= (common == n_cn).sum(axis=1) == count
        atoms = np.flatnonzero(fits)
        if len(atoms) == 0:
            continue

        n_cn, n_b, chain = _signature(bonds[atoms])
        fits = np.zeros(n_cn.shape, dtype=bool)
        for c, s in {(c, s) for c, s, _ in pattern}:
            fits |= (n_cn == c) & (n_b == s)
        whole = fits.all(axis=1)
        atoms, n_cn, n_b, chain = atoms[whole], n_cn[whole], n_b[whole], chain[whole]
        _resolve_chains(bonds[atoms], chain)

        matches = np.ones(len(atoms), dtype=bool)
        for (c, s, l), count in pattern.items():
            hits = (n_cn == c) & (n_b == s) & (chain == l)
            matches &= hits.sum(axis=1) == count
        types[atoms[matches]] = structure
    return types


# -- interval CNA -------------------------------------------------------------


def _interval_chunk(flat, lengths):
    """Interval CNA of atoms that all have the same number of neighbours.

    Args:
        flat: (n, 3k) flattened neighbour vectors, k being 12 or 14
        lengths: (n, k) their lengths

    Returns:
        (n,) int8 structure of each atom
    """
    n, k = lengths.shape
    _, b = _pairs(k)
    # the short path bonds only the 12 nearest, whose pairs come first in the
    # same order as `_pairs(12)`
    in_short_shell = b < 12
    length = np.sqrt(_pair_distances2(flat, k))

    short_scale = lengths[:, :12].mean(axis=1)
    short = in_short_shell & (length < INTERVAL_FRACTION * short_scale[:, None])
    candidates = [(12, short)]
    if k == 14:
        long_scale = (
            lengths[:, :8].sum(axis=1) / np.sqrt(0.75) + lengths[:, 8:].sum(axis=1)
        ) / 14
        long = length < INTERVAL_FRACTION * long_scale[:, None]
        candidates.append((14, long))
    else:
        long = np.zeros_like(short)
    short_end, long_end = _interval_ends(np.where(short | long, np.inf, length))
    ends = {12: short_end, 14: long_end}

    best = np.zeros(n)
    types = np.zeros(n, dtype=np.int8)
    for shell, on_path in candidates:
        path = np.where(on_path, length, np.inf)
        if shell < k:
            path = path[:, in_short_shell]
        ordered = np.sort(path, axis=1)
        count = on_path.sum(axis=1)
        for edges, group in _by_bond_count(PATTERNS[shell]).items():
            width, last, rows = _interval(ordered, count, ends[shell], edges)
            found = _classify_chunk(_pack(path[rows] <= last[:, None], shell), group)
            wider = (found != StructureType.OTHER) & (width > best[rows])
            best[rows[wider]] = width[wider]
            types[rows[wider]] = found[wider]
    return types


def _interval_ends(outside):
    """Where the short and long paths through the bonds end.

    Both end at bonds beyond their thresholds, picked walking the pairs in
    order: each such bond shorter than every one before it becomes the long
    end, and any other the short end if shorter than the short end so far.

    Args:
        outside: (n, P) lengths of bonds beyond both thresholds, inf elsewhere

    Returns:
        (short_end, long_end): (n,) lengths, inf where there is none
    """
    running = np.minimum.accumulate(outside, axis=1)
    before = np.empty_like(running)
    before[:, 0] = np.inf
    before[:, 1:] = running[:, :-1]
    demoted = np.where(outside >= before, outside, np.inf)
    return demoted.min(axis=1), running[:, -1]


def _interval(ordered, count, end, edges):
    """The gap after `edges` steps along a path of bonds.

    Args:
        ordered: (n, P) lengths of the bonds on the path, ascending, then inf
        count: (n,) bonds on the path
        end: (n,) length at which the path ends
        edges: steps taken

    Returns:
        (width, last, rows): the interval width and the length of the last
        bond taken, for the rows whose path is long enough and whose interval
        is wider than rounding error
    """
    rows = np.flatnonzero(count >= edges)
    last = ordered[rows, edges - 1]
    following = np.where(count[rows] > edges, ordered[rows, edges], end[rows])
    width = following - last
    keep = np.isfinite(width) & (width > INTERVAL_TOLERANCE * last)
    return width[keep], last[keep], rows[keep]


def _by_bond_count(patterns):
    """Structures grouped by how many bonds their neighbour shell holds."""
    groups = {}
    for structure, pattern in patterns.items():
        edges = sum(n_cn * count for (n_cn, _, _), count in pattern.items()) // 2
        groups.setdefault(edges, {})[structure] = pattern
    return groups

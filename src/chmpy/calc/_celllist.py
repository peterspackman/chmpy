"""Pair enumeration by binning, in numpy.

This is the fallback under `chmpy.calc.neighbors` when `vesin` is not
installed, and it is meant to be quick enough that installing vesin is an
optimisation rather than a requirement.

The approach is the usual linked-cell list, written without a Python loop over
atoms: atoms are binned on a grid at least as coarse as the cutoff, and the
enumeration runs one vectorised pass per bin offset. Three things keep the
element count down, and they are the difference between this and a numpy
neighbour list that is ten times slower.

* **No replicated coordinate array.** The obvious numpy approach tiles the
  positions over every periodic image and builds one tree over `N * n_images`
  points; for a small cell and a 6 A cutoff that is a 125-fold blow-up in
  memory before a single distance is computed. Here periodicity is a wrap of
  the *bin index*, and the lattice translation falls out of that wrap as an
  integer, so the coordinates are never copied.

* **Half the bin offsets.** A pair found at offset `o` is the same pair found
  at `-o` from the other end, so only the lexicographically positive half of
  the offsets is visited (13 of 27, plus the zero offset restricted to
  `j > i`). Every candidate distance is therefore computed once rather than
  twice. A full list is the half list plus its mirror, which costs a copy
  rather than a second enumeration.

* **Shifts built only for survivors.** The lattice translation of a pair is an
  integer 3-vector, and assembling it for every *candidate* costs more than
  the distances do. Instead the per-atom part of the translation is folded
  into a shifted query position, so a candidate costs one subtraction and one
  dot product, and the integer shifts are assembled afterwards for the ~15%
  of candidates that survive the cutoff.

The pairs come out in a different order from vesin's, and a half list may pick
the opposite representative of a pair -- `(j, i, -S)` where vesin gives
`(i, j, S)`. Both describe the same set of interactions, which is all any
consumer of a half list can rely on.
"""

from __future__ import annotations

import numpy as np

#: cap on the number of bins, relative to the atom count, so a large cell with
#: few atoms does not allocate a grid far bigger than its contents
MAX_BINS_PER_ATOM = 8


def cell_list_pairs(positions, cell, pbc, cutoff, full=False, self_pairs=True):
    """Enumerate pairs within `cutoff`.

    Args:
        positions: (N, 3) Cartesian coordinates, Angstroms
        cell: (3, 3) lattice vectors as rows
        pbc: (3,) which directions are periodic
        cutoff: maximum pair distance
        full: emit each pair from both ends
        self_pairs: allow an atom to pair with its own periodic image

    Returns:
        (i, j, shifts): (P,) int64, (P,) int64, (P, 3) float64, where the pair
        vector is ``positions[j] - positions[i] + shifts @ cell``
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)

    if bool(np.all(pbc)):
        i, j, shifts = _periodic_pairs(
            positions, np.asarray(cell, dtype=np.float64), cutoff, self_pairs
        )
    elif not np.any(pbc):
        i, j, shifts = _free_pairs(positions, cutoff)
    else:
        raise NotImplementedError("mixed periodicity is not handled by the cell list")

    if full:
        i, j = np.concatenate([i, j]), np.concatenate([j, i])
        shifts = np.concatenate([shifts, -shifts])
    return i, j, shifts


# -- the grid -----------------------------------------------------------------


def _grid_shape(spacing, cutoff, n_atoms):
    """Bins along each direction: as fine as the cutoff allows, within reason.

    The cap matters for a large cell holding few atoms, where the cutoff would
    otherwise ask for more bins than there are atoms to put in them. Scaling
    all three counts down at once rather than decrementing the largest one
    repeatedly keeps that cheap: a 1000 A cell with a 1 A cutoff wants 10^9
    bins, and walking down to the limit one bin at a time took 4 ms.
    """
    counts = np.maximum(1, (spacing / cutoff).astype(int))
    limit = max(8, MAX_BINS_PER_ATOM * n_atoms)
    if counts.prod() > limit:
        counts = np.maximum(
            1, (counts * (limit / counts.prod()) ** (1 / 3)).astype(int)
        )
    while counts.prod() > limit and counts.max() > 1:
        counts[counts.argmax()] -= 1
    return counts


def _reach(cutoff, width):
    """How many bins away a pair within the cutoff can be.

    Bins `D` apart along a direction hold atoms at least `(D - 1) * width`
    apart, so the last offset worth visiting is the largest `D` with
    `(D - 1) * width < cutoff`, i.e. `ceil(cutoff / width)`. That is 1 for a
    grid at least as coarse as the cutoff; more only when the cell is thinner
    than the cutoff, where the extra offsets reach genuinely different
    periodic images rather than repeating one.
    """
    # the epsilon keeps an exact ratio of 1 from rounding up to 2 offsets
    return np.maximum(1, np.ceil(cutoff / width - 1e-12).astype(int))


def _half_offsets(reach):
    """Bin offsets in the lexicographically non-negative half.

    Visiting `o` and `-o` would find the same pair from both ends, so only one
    of each is enumerated. The zero offset stays in and is the one the caller
    restricts to `j > i`.
    """
    ranges = [np.arange(-r, r + 1) for r in reach]
    offsets = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    a, b, c = offsets.T
    return offsets[(a > 0) | ((a == 0) & (b > 0)) | ((a == 0) & (b == 0) & (c >= 0))]


def _within(target, source, i, j, cutoff2):
    """Which candidate pairs are inside the cutoff.

    `target` and `source` are (3, N) arrays of coordinates -- the components
    laid out contiguously, so each gather reads a contiguous array rather than
    a strided row of an (N, 3) one.
    """
    dx = target[0][j]
    dx -= source[0][i]
    dx *= dx
    for axis in (1, 2):
        d = target[axis][j]
        d -= source[axis][i]
        d *= d
        dx += d
    return dx <= cutoff2


def _bin_atoms(bin_index, n_bins):
    """Sort atom indices by bin.

    Returns:
        (order, start, count): `order` lists atom indices bin by bin,
        `start[b]` is where bin `b` begins in it, `count[b]` how many it holds
    """
    order = np.argsort(bin_index, kind="stable")
    count = np.bincount(bin_index, minlength=n_bins)
    start = np.zeros(n_bins, dtype=np.int64)
    np.cumsum(count[:-1], out=start[1:])
    return order, start, count


def _candidates(order, start, count, target_bin):
    """Every (source atom, target atom) pair for one bin offset.

    `target_bin[a]` is the bin atom `a` should look in. Returns the source
    index repeated once per occupant, and the occupants themselves.

    The slot of the k-th occupant of atom a's target bin is
    `start[target] + k`, and `k` for a flat run is `arange(total)` minus where
    that run began. Folding the second term into the first lets one `repeat`
    do the work of two, which matters: these arrays are the largest ones the
    enumeration touches.
    """
    counts = count[target_bin]
    total = int(counts.sum())
    if total == 0:
        return None, None
    base = start[target_bin] - (np.cumsum(counts) - counts)
    slots = np.repeat(base, counts)
    slots += np.arange(total, dtype=np.int64)
    source = np.repeat(np.arange(len(target_bin), dtype=np.int64), counts)
    return source, order[slots]


# -- periodic -----------------------------------------------------------------


def _periodic_pairs(positions, cell, cutoff, self_pairs):
    n_atoms = len(positions)
    if n_atoms == 0:
        return _empty()

    reciprocal = np.linalg.inv(cell)
    fractional = positions @ reciprocal
    # wrap into [0, 1): `wrap` records how many cells each atom moved, so the
    # shift reported for a pair is in terms of the ORIGINAL positions
    wrap = -np.floor(fractional).astype(np.int64)
    fractional = fractional + wrap
    wrapped_positions = fractional @ cell
    # everything gathered per candidate is kept as contiguous (3, N) components:
    # gathering a contiguous column is about twice as quick as gathering the
    # strided rows of an (N, 3) array, and these gathers are the inner loop
    wrapped = np.ascontiguousarray(wrapped_positions.T)
    wrap_t = np.ascontiguousarray(wrap.T)
    displaced = bool(np.any(wrap))

    spacing = 1.0 / np.linalg.norm(reciprocal, axis=0)
    n = _grid_shape(spacing, cutoff, n_atoms)
    reach = _reach(cutoff, spacing / n)

    coords = np.minimum((fractional * n).astype(np.int64), n - 1)
    strides = np.array([n[1] * n[2], n[2], 1], dtype=np.int64)
    order, start, count = _bin_atoms(coords @ strides, int(n.prod()))

    cutoff2 = cutoff * cutoff
    i_out, j_out, s_out = [], [], []
    for offset in _half_offsets(reach):
        target = coords + offset
        cells = np.floor_divide(target, n)
        target_bin = (target - cells * n) @ strides
        cells_t = np.ascontiguousarray(cells.T)

        i, j = _candidates(order, start, count, target_bin)
        if i is None:
            continue
        if not offset.any():
            keep = j > i
            i, j = i[keep], j[keep]
            if len(i) == 0:
                continue

        # Fold the per-atom lattice translation into the query position, so a
        # candidate costs three subtractions rather than a gather of integer
        # shifts and a matrix product. The coordinates are kept as three flat
        # arrays so that rejecting a candidate never involves materialising a
        # (candidates, 3) temporary -- only about one in seven survives.
        query = wrapped_positions - cells @ cell
        inside = _within(wrapped, query.T, i, j, cutoff2)
        if not inside.all():
            i, j = i[inside], j[inside]
            if len(i) == 0:
                continue

        if displaced:
            shifts = np.c_[
                cells_t[0][i] + wrap_t[0][j] - wrap_t[0][i],
                cells_t[1][i] + wrap_t[1][j] - wrap_t[1][i],
                cells_t[2][i] + wrap_t[2][j] - wrap_t[2][i],
            ]
        else:
            shifts = np.c_[cells_t[0][i], cells_t[1][i], cells_t[2][i]]
        shifts = shifts.astype(np.float64)
        if not self_pairs:
            keep = i != j
            i, j, shifts = i[keep], j[keep], shifts[keep]
            if len(i) == 0:
                continue
        i_out.append(i)
        j_out.append(j)
        s_out.append(shifts)

    return _stack(i_out, j_out, s_out)


# -- isolated -----------------------------------------------------------------


def _free_pairs(positions, cutoff):
    n_atoms = len(positions)
    if n_atoms < 2:
        return _empty()

    low = positions.min(axis=0)
    extent = positions.max(axis=0) - low
    n = _grid_shape(np.maximum(extent, cutoff), cutoff, n_atoms)
    width = np.where(extent > 0, extent / n, cutoff)
    # beyond n - 1 bins away every target is outside the grid
    reach = np.minimum(_reach(cutoff, width), n - 1)

    coords = np.clip(((positions - low) / width).astype(np.int64), 0, n - 1)
    strides = np.array([n[1] * n[2], n[2], 1], dtype=np.int64)
    order, start, count = _bin_atoms(coords @ strides, int(n.prod()))

    flat = np.ascontiguousarray(positions.T)
    cutoff2 = cutoff * cutoff
    i_out, j_out = [], []
    for offset in _half_offsets(reach):
        target = coords + offset
        valid = np.all((target >= 0) & (target < n), axis=1)
        if not valid.any():
            continue
        source = np.flatnonzero(valid)

        slot, j = _candidates(order, start, count, target[valid] @ strides)
        if slot is None:
            continue
        i = source[slot]
        if not offset.any():
            keep = j > i
            i, j = i[keep], j[keep]
            if len(i) == 0:
                continue

        inside = _within(flat, flat, i, j, cutoff2)
        if inside.any():
            i_out.append(i[inside])
            j_out.append(j[inside])

    if not i_out:
        return _empty()
    i = np.concatenate(i_out)
    j = np.concatenate(j_out)
    return i, j, np.zeros((len(i), 3))


# -- odds and ends ------------------------------------------------------------


def _stack(i_out, j_out, s_out):
    if not i_out:
        return _empty()
    return (
        np.concatenate(i_out),
        np.concatenate(j_out),
        np.concatenate(s_out),
    )


def _empty():
    return (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 3)),
    )

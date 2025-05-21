"""Marching cubes implementation of volumetric data.

This file is a modification of the original _marching_cubes_lewiner.pyx
file from the scikit-image project, retrieved from git revision:

    129af33b9c118dd87efd4a39ce623e70f8188ce8

As such the following copyright notice *must* be included here, and
we should include the acknowledgement in the LICENSE.txt of this
project.

Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in
       the documentation and/or other materials provided with the
       distribution.
    3. Neither the name of skimage nor the names of its contributors may be
       used to endorse or promote products derived from this software without
       specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE), ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import base64

import numpy as np

from . import lookup_tables
from ._mc_lewiner import LutProvider, remove_degenerate_faces
from ._mc_lewiner import marching_cubes as _marching_cubes


def marching_cubes(
    volume,
    level=None,
    spacing=(1.0, 1.0, 1.0),
    gradient_direction="descent",
    step_size=1,
    allow_degenerate=True,
    use_classic=False,
):
    """Lewiner marching cubes algorithm to find surfaces in 3d volumetric data.

    In contrast to ``marching_cubes_classic()``, this algorithm is faster,
    resolves ambiguities, and guarantees topologically correct results.
    Therefore, this algorithm generally a better choice, unless there
    is a specific need for the classic algorithm.

    Args:
        volume: (M, N, P) array
            Input data volume to find isosurfaces. Will internally be
            converted to float32 if necessary.
        level: float, optional
            Contour value to search for isosurfaces in `volume`. If not
            given or None, the average of the min and max of vol is used.
        spacing: length-3 tuple of floats, optional
            Voxel spacing in spatial dimensions corresponding to numpy array
            indexing dimensions (M, N, P) as in `volume`.
        gradient_direction: string, optional
            Controls if the mesh was generated from an isosurface with gradient
            descent toward objects of interest (the default), or the opposite,
            considering the *left-hand* rule.
            The two options are:
            * descent : Object was greater than exterior
            * ascent : Exterior was greater than object
        step_size: int, optional
            Step size in voxels. Default 1. Larger steps yield faster but
            coarser results. The result will always be topologically correct
            though.
        allow_degenerate: bool, optional
            Whether to allow degenerate (i.e. zero-area) triangles in the
            end-result. Default True. If False, degenerate triangles are
            removed, at the cost of making the algorithm slower.
        use_classic: bool, optional
            If given and True, the classic marching cubes by Lorensen (1987)
            is used. This option is included for reference purposes. Note
            that this algorithm has ambiguities and is not guaranteed to
            produce a topologically correct result. The results with using
            this option are *not* generally the same as the
            ``marching_cubes_classic()`` function.

    Returns
    -------
    verts : (V, 3) array
        Spatial coordinates for V unique mesh vertices. Coordinate order
        matches input `volume` (M, N, P).
    faces : (F, 3) array
        Define triangular faces via referencing vertex indices from ``verts``.
        This algorithm specifically outputs triangles, so each face has
        exactly three indices.
    normals : (V, 3) array
        The normal direction at each vertex, as calculated from the
        data.
    values : (V, ) array
        Gives a measure for the maximum value of the data in the local region
        near each vertex. This can be used by visualization tools to apply
        a colormap to the mesh.

    Notes
    -----
    The algorithm [1] is an improved version of Chernyaev's Marching
    Cubes 33 algorithm. It is an efficient algorithm that relies on
    heavy use of lookup tables to handle the many different cases,
    keeping the algorithm relatively easy. This implementation is
    written in Cython, ported from Lewiner's C++ implementation.

    To quantify the area of an isosurface generated by this algorithm, pass
    verts and faces to `skimage.measure.mesh_surface_area`.

    References
    ----------
    [1] Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
        Tavares. Efficient implementation of Marching Cubes' cases with
        topological guarantees. Journal of Graphics Tools 8(2)
        pp. 1-15 (Dec 2003).
        https://dx.doi.org/10.1080/10867651.2003.10487582

    See Also
    --------
    skimage.measure.marching_cubes_classic
    skimage.measure.mesh_surface_area
    """
    # Check volume and ensure its in the format that the alg needs
    if not isinstance(volume, np.ndarray) or (volume.ndim != 3):
        raise ValueError("Input volume should be a 3D numpy array.")
    if volume.shape[0] < 2 or volume.shape[1] < 2 or volume.shape[2] < 2:
        raise ValueError("Input array must be at least 2x2x2.")
    volume = np.ascontiguousarray(volume, np.float32)  # no copy if not necessary

    # Check/convert other inputs:
    # level
    if level is None:
        level = 0.5 * (volume.min() + volume.max())
    else:
        level = float(level)
        if level < volume.min() or level > volume.max():
            raise ValueError("Surface level must be within volume data range.")
    # spacing
    if len(spacing) != 3:
        raise ValueError("`spacing` must consist of three floats.")
    # step_size
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError("step_size must be at least one.")
    # use_classic
    use_classic = bool(use_classic)

    # Get LutProvider class (reuse if possible)
    L = _get_lookup_tables()

    # Apply algorithm
    func = _marching_cubes
    vertices, faces, normals, values = func(volume, level, L, step_size, use_classic)

    if not len(vertices):
        raise RuntimeError("No surface found at the given iso value.")

    # Output in z-y-x order, as is common in skimage
    vertices = np.fliplr(vertices)
    normals = np.fliplr(normals)

    # Finishing touches to output
    faces.shape = -1, 3
    if gradient_direction == "descent":
        # MC implementation is right-handed, but gradient_direction is left-handed
        faces = np.fliplr(faces)
    elif not gradient_direction == "ascent":
        raise ValueError(
            f"Incorrect input {gradient_direction} in `gradient_direction`, see "
            "docstring."
        )
    if not np.array_equal(spacing, (1, 1, 1)):
        vertices = vertices * np.r_[spacing]

    if allow_degenerate:
        return vertices, faces, normals, values
    else:
        fun = remove_degenerate_faces
        return fun(vertices.astype(np.float32), faces, normals, values)


def _to_array(args):
    shape, text = args
    byts = base64.decodebytes(text.encode("utf-8"))
    ar = np.frombuffer(byts, dtype="int8")
    ar.shape = shape
    return ar


# Map an edge-index to two relative pixel positions. The ege index
# represents a point that lies somewhere in between these pixels.
# Linear interpolation should be used to determine where it is exactly.
#   0
# 3   1   ->  0x
#   2         xx
EDGETORELATIVEPOSX = np.array(
    [
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
    ],
    "int8",
)
EDGETORELATIVEPOSY = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
    ],
    "int8",
)
EDGETORELATIVEPOSZ = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ],
    "int8",
)


def _get_lookup_tables():
    """Kind of lazy obtaining of the lookup tables."""
    if not hasattr(lookup_tables, "THE_LUTS"):
        lookup_tables.THE_LUTS = LutProvider(
            EDGETORELATIVEPOSX,
            EDGETORELATIVEPOSY,
            EDGETORELATIVEPOSZ,
            _to_array(lookup_tables.CASESCLASSIC),
            _to_array(lookup_tables.CASES),
            _to_array(lookup_tables.TILING1),
            _to_array(lookup_tables.TILING2),
            _to_array(lookup_tables.TILING3_1),
            _to_array(lookup_tables.TILING3_2),
            _to_array(lookup_tables.TILING4_1),
            _to_array(lookup_tables.TILING4_2),
            _to_array(lookup_tables.TILING5),
            _to_array(lookup_tables.TILING6_1_1),
            _to_array(lookup_tables.TILING6_1_2),
            _to_array(lookup_tables.TILING6_2),
            _to_array(lookup_tables.TILING7_1),
            _to_array(lookup_tables.TILING7_2),
            _to_array(lookup_tables.TILING7_3),
            _to_array(lookup_tables.TILING7_4_1),
            _to_array(lookup_tables.TILING7_4_2),
            _to_array(lookup_tables.TILING8),
            _to_array(lookup_tables.TILING9),
            _to_array(lookup_tables.TILING10_1_1),
            _to_array(lookup_tables.TILING10_1_1_),
            _to_array(lookup_tables.TILING10_1_2),
            _to_array(lookup_tables.TILING10_2),
            _to_array(lookup_tables.TILING10_2_),
            _to_array(lookup_tables.TILING11),
            _to_array(lookup_tables.TILING12_1_1),
            _to_array(lookup_tables.TILING12_1_1_),
            _to_array(lookup_tables.TILING12_1_2),
            _to_array(lookup_tables.TILING12_2),
            _to_array(lookup_tables.TILING12_2_),
            _to_array(lookup_tables.TILING13_1),
            _to_array(lookup_tables.TILING13_1_),
            _to_array(lookup_tables.TILING13_2),
            _to_array(lookup_tables.TILING13_2_),
            _to_array(lookup_tables.TILING13_3),
            _to_array(lookup_tables.TILING13_3_),
            _to_array(lookup_tables.TILING13_4),
            _to_array(lookup_tables.TILING13_5_1),
            _to_array(lookup_tables.TILING13_5_2),
            _to_array(lookup_tables.TILING14),
            _to_array(lookup_tables.TEST3),
            _to_array(lookup_tables.TEST4),
            _to_array(lookup_tables.TEST6),
            _to_array(lookup_tables.TEST7),
            _to_array(lookup_tables.TEST10),
            _to_array(lookup_tables.TEST12),
            _to_array(lookup_tables.TEST13),
            _to_array(lookup_tables.SUBCONFIG13),
        )

    return lookup_tables.THE_LUTS

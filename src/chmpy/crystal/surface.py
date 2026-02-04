"""Surface calculation functions for Crystal structures.

Includes promolecule density isosurfaces, void surfaces,
stockholder weight (Hirshfeld) isosurfaces, and mesh scene generation.
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree
from trimesh import Trimesh

from chmpy.core.molecule import Molecule


def _nearest_molecule_idx(vertices, el, pos):
    from scipy.sparse.csgraph import connected_components

    m = Molecule.from_arrays(el, pos)
    m.guess_bonds()
    nfrag, labels = connected_components(m.bonds)
    tree = KDTree(pos)
    d, idxs = tree.query(vertices, k=1)
    l = labels[idxs]
    u, idxs = np.unique(l, return_inverse=True)
    return np.arange(len(u), dtype=np.uint8)[idxs]


def _nearest_atom_idx(vertices, el, pos):
    tree = KDTree(pos)
    d, idxs = tree.query(vertices, k=1)
    return idxs


def promolecule_density_isosurfaces(crystal, **kwargs) -> list[Trimesh]:
    """
    Calculate promolecule electron density isosurfaces
    for each symmetry unique molecule in this crystal.

    Args:
        crystal: Crystal instance
        kwargs: Keyword arguments used by `Molecule.promolecule_density_isosurface`.

            Options are:
            ```
            isovalue (float, optional): level set value for the isosurface
                (default=0.002) in au.
            separation (float, optional): separation between density grid
                used in the surface calculation (default 0.2) in Angstroms.
            color (str, optional): surface property to use for vertex coloring,
                one of ('d_norm_i', 'd_i', 'd_norm_e', 'd_e')
            colormap (str, optional): matplotlib colormap to use for surface
                coloring (default 'viridis_r')
            midpoint (float, optional): midpoint of the segmented
                colormap (if applicable)
            ```

    Returns:
        A list of meshes representing the promolecule density isosurfaces
    """
    if kwargs.get("color", None) == "fragment_patch":
        color = kwargs.pop("color")
        surfaces = [
            mol.promolecule_density_isosurface(**kwargs)
            for mol in crystal.symmetry_unique_molecules()
        ]
        radius = kwargs.get("fragment_patch_radius", 6.0)
        from chmpy.util.color import property_to_color
        from chmpy.util.mesh import face_centroids

        for i, (_mol, n_e, n_p) in enumerate(
            crystal.molecule_environments(radius=radius)
        ):
            surf = surfaces[i]
            prop = _nearest_molecule_idx(surf.vertices, n_e, n_p)
            color = property_to_color(prop, cmap=kwargs.get("colormap", color))
            face_points = face_centroids(surf)
            surf.visual.vertex_colors = color
            surf.vertex_attributes["fragment_patch"] = prop
            surf.face_attributes["fragment_patch"] = _nearest_molecule_idx(
                face_points, n_e, n_p
            )
    else:
        surfaces = [
            mol.promolecule_density_isosurface(**kwargs)
            for mol in crystal.symmetry_unique_molecules()
        ]
    return surfaces


def void_surface(crystal, *args, **kwargs) -> Trimesh:
    """
    Calculate void surface based on promolecule electron density
    for the unit cell of this crystal.

    Args:
        crystal: Crystal instance
        kwargs: Keyword arguments used in the evaluation of the surface.

            Options are:
            ```
            isovalue (float, optional): level set value for the
                isosurface (default=0.002) in au.
            separation (float, optional): separation between density grid
                used in the surface calculation (default 0.2) in Angstroms.
            ```

    Returns:
        the mesh representing the promolecule density void isosurface
    """
    import trimesh

    from chmpy import PromoleculeDensity
    from chmpy.mc import marching_cubes

    vertex_color = kwargs.get("color", None)

    atoms = crystal.slab(bounds=((-1, -1, -1), (1, 1, 1)))
    density = PromoleculeDensity((atoms["element"], atoms["cart_pos"]))
    sep = kwargs.get("separation", kwargs.get("resolution", 0.5))
    isovalue = kwargs.get("isovalue", 3e-4)
    grid_type = kwargs.get("grid_type", "uc")
    if grid_type == "uc":
        seps = sep / np.array(crystal.unit_cell.lengths)
        x_grid = np.arange(0, 1.0, seps[0], dtype=np.float32)
        y_grid = np.arange(0, 1.0, seps[1], dtype=np.float32)
        z_grid = np.arange(0, 1.0, seps[2], dtype=np.float32)
        x, y, z = np.meshgrid(x_grid, y_grid, z_grid)
        shape = x.shape
        pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
        pts = pts.astype(np.float32)
        pts = crystal.to_cartesian(pts)
    elif grid_type == "box":
        ((x0, y0, z0), (x1, y1, z1)) = kwargs.get(
            "box_corners", ((0.0, 0.0, 0.0), (5.0, 5.0, 5.0))
        )
        x, y, z = np.mgrid[x0:x1:sep, y0:y1:sep, z0:z1:sep]
        pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
        pts = pts.astype(np.float32)
        shape = x.shape
        seps = (sep, sep, sep)
    else:
        raise NotImplementedError("Only uc grid supported currently")
    tree = KDTree(atoms["cart_pos"])
    distances, _ = tree.query(pts)
    values = np.ones(pts.shape[0], dtype=np.float32)
    mask = distances > 1.0  # minimum bigger than 1 angstrom
    rho = density.rho(pts[mask])
    values[mask] = rho
    values = values.reshape(shape)
    verts, faces, normals, _ = marching_cubes(
        values, isovalue, spacing=seps, gradient_direction="ascent"
    )
    if grid_type == "uc":
        verts = crystal.to_cartesian(np.c_[verts[:, 1], verts[:, 0], verts[:, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)

    if kwargs.get("subdivide", False):
        for _ in range(int(kwargs.get("subdivide", False))):
            mesh = mesh.subdivide()

    if vertex_color == "esp":
        from chmpy.util.color import property_to_color

        asym_charges = crystal.asymmetric_unit_partial_charges()
        mol = Molecule.from_arrays(atoms["element"], atoms["cart_pos"])
        partial_charges = np.empty(len(mol), dtype=np.float32)
        partial_charges = asym_charges[atoms["asym_atom"]]
        mol._partial_charges = partial_charges
        prop = mol.electrostatic_potential(mesh.vertices)
        mesh.visual.vertex_colors = property_to_color(
            prop, cmap=kwargs.get("cmap", "esp")
        )
    return mesh


def mesh_scene(crystal, **kwargs):
    """
    Calculate a scene of this meshes of unit cell molecules in this crystal,
    along with optional void surface.

    Args:
        crystal: Crystal instance
        kwargs: optional arguments used in the generation of this scene.

    Returns:
        trimesh.scene.Scene: trimesh scene object.
    """
    from trimesh import Scene

    meshes = {}
    for i, m in enumerate(crystal.unit_cell_molecules()):
        mesh = m.to_mesh(representation=kwargs.get("representation", "ball_stick"))
        n = m.molecular_formula
        for k, v in mesh.items():
            meshes[f"mol_{i}_{n}.{k}"] = v

    if kwargs.get("void", False):
        void_kwargs = kwargs.get("void_kwargs", {})
        meshes["void_surface"] = void_surface(crystal, **void_kwargs)
    if kwargs.get("axes", False):
        from trimesh.creation import axis

        meshes["axes"] = axis(
            transform=crystal.unit_cell.direct_homogeneous.T, axis_length=1.0
        )
    return Scene(meshes)


def stockholder_weight_isosurfaces(crystal, kind="mol", **kwargs) -> list[Trimesh]:
    """
    Calculate stockholder weight isosurfaces (i.e. Hirshfeld surfaces)
    for each symmetry unique molecule or atom in this crystal.

    Args:
        crystal: Crystal instance
        kind (str, optional): dictates whether we calculate surfaces
            for each unique molecule or for each unique atom
        kwargs: keyword arguments passed to `stockholder_weight_isosurface`.

            Options include:
            ```
            isovalue: float, optional
                level set value for the isosurface (default=0.5). Must be between
                0 and 1, but values other than 0.5 probably won't make sense anyway.
            separation: float, optional
                separation between density grid used in the surface calculation
                (default 0.2) in Angstroms.
            radius: float, optional
                maximum distance for contributing neighbours for the stockholder
                weight calculation
            color: str, optional
                surface property to use for vertex coloring, one of ('d_norm_i',
                'd_i', 'd_norm_e', 'd_e', 'd_norm', 'fragment_patch')
            colormap: str, optional
                matplotlib colormap to use for surface coloring
                (default 'viridis_r')
            midpoint: float, optional, default 0.0 if using d_norm
                use the midpoint norm (as is used in CrystalExplorer)
            ```

    Returns:
        A list of meshes representing the stockholder weight isosurfaces
    """
    import trimesh

    from chmpy import StockholderWeight
    from chmpy.surface import stockholder_weight_isosurface
    from chmpy.util.color import property_to_color

    sep = kwargs.get("separation", kwargs.get("resolution", 0.2))
    radius = kwargs.get("radius", 12.0)
    vertex_color = kwargs.get("color", "d_norm")
    isovalue = kwargs.get("isovalue", 0.5)
    meshes = []
    extra_props = {}
    isos = []

    def nearest_atomic_number(pos, n_e, n_p):
        return np.array(n_e[_nearest_atom_idx(pos, n_e, n_p)], dtype=np.uint8)

    if kind == "atom":
        for surrounds in crystal.atomic_surroundings(radius=radius):
            n = surrounds["centre"]["element"]
            pos = surrounds["centre"]["cart_pos"]
            neighbour_els = surrounds["neighbours"]["element"]
            neighbour_pos = surrounds["neighbours"]["cart_pos"]
            s = StockholderWeight.from_arrays(
                [n], [pos], neighbour_els, neighbour_pos
            )
            iso = stockholder_weight_isosurface(s, isovalue=isovalue, sep=sep)
            isos.append(iso)
    elif kind == "mol":
        for _i, (mol, n_e, n_p) in enumerate(
            crystal.molecule_environments(radius=radius)
        ):
            extra_props = {}
            if vertex_color == "esp":
                extra_props["esp"] = mol.electrostatic_potential
            elif vertex_color == "fragment_patch":
                extra_props["fragment_patch"] = (
                    lambda x, _n_e=n_e, _n_p=n_p: _nearest_molecule_idx(
                        x, _n_e, _n_p
                    )
                )
            extra_props["nearest_atom_external"] = (
                lambda x, _n_e=n_e, _n_p=n_p: nearest_atomic_number(x, _n_e, _n_p)
            )
            extra_props["nearest_atom_internal"] = (
                lambda x,
                _atomic_nums=mol.atomic_numbers,
                _positions=mol.positions: nearest_atomic_number(
                    x, _atomic_nums, _positions
                )
            )
            s = StockholderWeight.from_arrays(
                mol.atomic_numbers, mol.positions, n_e, n_p
            )
            iso = stockholder_weight_isosurface(
                s, isovalue=isovalue, sep=sep, extra_props=extra_props
            )
            isos.append(iso)
    else:
        for arr in crystal.functional_group_surroundings(radius=radius, kind=kind):
            s = StockholderWeight.from_arrays(*arr)
            iso = stockholder_weight_isosurface(s, isovalue=isovalue, sep=sep)
            isos.append(iso)

    for iso in isos:
        prop = iso.vertex_prop[vertex_color]
        color = property_to_color(prop, cmap=kwargs.get("cmap", vertex_color))
        mesh = trimesh.Trimesh(
            vertices=iso.vertices,
            faces=iso.faces,
            normals=iso.normals,
            vertex_colors=color,
        )
        for k, v in iso.vertex_prop.items():
            mesh.vertex_attributes[k] = v
        meshes.append(mesh)
    return meshes

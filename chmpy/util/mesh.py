import logging

LOG = logging.getLogger(__name__)


def save_mesh(mesh, filename):
    """
    Save the given Trimesh to a file.

    Args:
        mesh (trimesh.Trimesh): The mesh to save.
        filename (str): The path to the destination file.
    """
    ext = filename.split(".")[-1]
    with open(filename, "wb") as f:
        mesh.export(f, ext)

    LOG.debug("Saved mesh %s to %s", mesh, filename)


def molecule_to_meshes(molecule, **kwargs):
    """
    Convert the provided molecule into a list
    of trimesh Meshes representing the molecule
    either as van der Waals spheres or as a CPK
    representation.

    Args:
        molecule (Molecule): The molecule to represent
        kwargs (dict): Optional Keyword arguments

    Returns:
        list: a list of meshes representing atoms and (optionally) bonds
    """

    from trimesh.creation import icosphere, cylinder
    from trimesh import Scene
    from trimesh import Trimesh
    import numpy as np
    from chmpy import Element
    from copy import deepcopy

    representation = kwargs.get("representation", "ball_stick")
    base_sphere = icosphere(subdivisions=3)
    mesh_primitives = []
    n_points = len(base_sphere.vertices)
    vertices = []
    faces = []
    colors = []
    offset = 0
    meshes = {}
    for i, (el, pos) in enumerate(molecule):
        m = base_sphere.copy()
        m.apply_scale(getattr(el, f"{representation}_radius"))
        m.apply_translation(pos)
        m.visual.vertex_colors = np.repeat([el.color], n_points, axis=0)
        meshes[f"atom_{molecule.labels[i]}"] = m
    if representation == "ball_stick":
        bond_thickness = 0.12
        for i, (a, b, d) in enumerate(molecule.unique_bonds):
            x1 = molecule.positions[a]
            x3 = molecule.positions[b]
            cyl = cylinder(bond_thickness, d, segment=(x1, x3))
            cyl.visual.vertex_colors = np.repeat(
                [(100, 100, 100, 255),], cyl.vertices.shape[0], axis=0,
            )
            bond_label = f"bond_{molecule.labels[a]}_{molecule.labels[b]}"
            meshes[bond_label] = cyl
    return meshes


def face_centroids(mesh):
    import numpy as np

    return np.sum(mesh.vertices[mesh.faces], axis=1) / mesh.faces.shape[1]


def color_mesh(f, mesh, faces=False, **kwargs):
    from chmpy.util.color import property_to_color

    is_function = hasattr(f, "__call__")

    if faces:
        if is_function:
            pts = face_centroids(mesh)
            prop = f(pts)
        else:
            prop = f
        mesh.visual.face_colors = property_to_color(prop, **kwargs) * 255
    else:
        if is_function:
            prop = f(mesh.vertices)
        else:
            prop = f
        mesh.visual.vertex_colors = property_to_color(prop, **kwargs) * 255

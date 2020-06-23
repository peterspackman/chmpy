import logging

LOG = logging.getLogger(__name__)


def save_mesh(mesh, filename):
    """Save the given Trimesh to a file.

    Parameters
    ----------
    mesh: :obj:`trimesh.Trimesh`
        The mesh to save.
    filename: str
        The path to the destination file.
    """
    ext = filename.split(".")[-1]
    with open(filename, "wb") as f:
        mesh.export(f, ext)

    LOG.debug("Saved mesh %s to %s", mesh, filename)


def molecule_to_meshes(molecule, **kwargs):
    from trimesh.creation import icosphere, cylinder
    from trimesh import Scene
    from trimesh import Trimesh
    import numpy as np
    from chmpy import Element
    from copy import deepcopy

    base_sphere = icosphere(subdivisions=3)
    mesh_primitives = []
    n_points = len(base_sphere.vertices)
    vertices = []
    faces = []
    colors = []
    offset = 0
    for i, (el, pos) in enumerate(molecule):
        vertices.append(base_sphere.vertices * el.ball_stick_radius + pos)
        faces.append(offset + base_sphere.faces)
        colors.append(np.repeat([el.color], n_points, axis=0))
        offset += n_points

    bond_thickness = 0.9 * Element["H"].ball_stick_radius
    for i, (a, b, d) in enumerate(molecule.unique_bonds):
        x1 = molecule.positions[a]
        x3 = molecule.positions[b]
        x2 = 0.5 * (x3 + x1)
        cyl = cylinder(bond_thickness, d, segment=(x1, x2))
        n = len(cyl.vertices)
        vertices.append(cyl.vertices)
        faces.append(cyl.faces + offset)
        colors.append(np.repeat([molecule.elements[a].color], n, axis=0))
        offset += n
        cyl2 = cylinder(bond_thickness, d, segment=(x2, x3))
        vertices.append(cyl2.vertices)
        faces.append(cyl2.faces + offset)
        colors.append(np.repeat([molecule.elements[b].color], n, axis=0))
        offset += n

    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    colors = np.vstack(colors)
    return Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)

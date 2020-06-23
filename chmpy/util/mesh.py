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
    from copy import deepcopy

    base_sphere = icosphere(subdivisions=3)
    mesh_primitives = []
    for el, pos in molecule:
        site = deepcopy(base_sphere)
        site.vertices *= el.covalent_radius
        site.vertices += pos
        mesh_primitives.append(site)

    scene = Scene()
    for l, m in zip(molecule.labels, mesh_primitives):
        scene.add_geometry(m, node_name=l)
    return scene

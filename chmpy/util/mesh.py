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

import logging
from collections import defaultdict
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import dok_matrix
import numpy as np
import os
from .element import Element


LOG = logging.getLogger(__name__)


class Molecule:
    """class to represent information about a
    molecule i.e. a set of atoms with 3D coordinates
    joined by covalent bonds

    e.g. int, float, str etc. Will handle uncertainty values
    contained in parentheses.

    Parameters
    ----------
    elements: List[:obj:`Element`]
        list of element information for each atom in this molecule
    positions: :obj:`np.ndarray`
        (N, 3) array of Cartesian coordinates for each atom in this molecule (Angstroms)
    bonds: :obj:`np.ndarray`
        (N, N) adjacency matrix of bond lengths for connected atoms, 0 otherwise.
        If not provided this will be calculated.
    labels: :obj:`np.ndarray`
        (N,) vector of string labels for each atom in this molecule
        If not provided this will assigned default labels i.e. numbered in order.

    Keyword arguments will be stored in the `properties` member.
    """

    positions: np.ndarray
    elements: np.ndarray
    labels: np.ndarray
    properties: dict

    def __init__(self, elements, positions, bonds=None, labels=None, **kwargs):
        self.positions = positions
        self.elements = elements
        self.properties = {}
        self.properties.update(kwargs)
        self.bonds = None

        if bonds is None:
            if kwargs.get("guess_bonds", False):
                self.guess_bonds()
        else:
            self.bonds = dok_matrix(bonds)

        if labels is None:
            self.assign_default_labels()
        else:
            self.labels = labels

    def __iter__(self):
        for atom in zip(self.elements, self.positions):
            yield atom

    def __len__(self):
        return len(self.elements)

    def guess_bonds(self, tolerance=0.40):
        """Use geometric distances and covalent radii
        to determine bonding information for this molecule.
        Will set the bonds member.

        Bonding is determined by the distance between
        sites being closer than the sum of covalent radii + `tolerance`

        Parameters
        ----------
        tolerance: float, optional
            Additional tolerance for attributing two sites as 'bonded'.
            The default is 0.4 angstroms, which is recommended by the CCDC
        """
        tree = KDTree(self.positions)
        covalent_radii = np.array([x.cov for x in self.elements])
        max_cov = np.max(covalent_radii)
        thresholds = (
            covalent_radii[:, np.newaxis] + covalent_radii[np.newaxis, :] + tolerance
        )
        max_distance = max_cov * 2 + tolerance
        dist = tree.sparse_distance_matrix(tree, max_distance=max_distance).toarray()
        mask = (dist > 0) & (dist < thresholds)
        self.bonds = np.zeros(dist.shape)
        self.bonds[mask] = dist[mask]
        self.bonds = dok_matrix(self.bonds)

    def assign_default_labels(self):
        "Assign the default labels to atom sites in this molecule (number them by element)"
        counts = defaultdict(int)
        labels = []
        for el, _ in self:
            counts[el] += 1
            labels.append("{}{}".format(el.symbol, counts[el]))
        self.labels = np.asarray(labels)

    def distance_to(self, other, method="centroid"):
        """Calculate the euclidean distance between this
        molecule and another. May use the distance between
        centres-of-mass, centroids, or nearest atoms.

        Parameters
        ----------
        other: :obj:`Molecule`
            the molecule to calculate distance to
        method: str, optional
            one of 'centroid', 'center_of_mass', 'nearest_atom'
        """
        method = method.lower()
        if method == "centroid":
            return np.linalg.norm(self.centroid - other.centroid)
        elif method == "center_of_mass":
            return np.linalg.norm(self.center_of_mass - other.center_of_mass)
        elif method == "nearest_atom":
            return np.min(cdist(self.positions, other.positions))
        else:
            raise ValueError(f"Unknown method={method}")

    @property
    def atomic_numbers(self):
        "Atomic numbers for each atom in this molecule"
        return np.array([e.atomic_number for e in self.elements])

    @property
    def centroid(self):
        "Mean cartesian position of atoms in this molecule"
        return np.mean(self.positions, axis=0)

    @property
    def center_of_mass(self):
        "Mean cartesian position of atoms in this molecule, weighted by atomic mass"
        masses = np.asarray([x.mass for x in self.elements])
        return np.sum(self.positions * masses[:, np.newaxis] / np.sum(masses), axis=0)

    @property
    def molecular_formula(self):
        "string of the molecular formula for this molecule"
        from .element import chemical_formula

        return chemical_formula(self.elements, subscript=False)

    def __repr__(self):
        return "<{}: {}({:.2f},{:.2f},{:.2f})>".format(
            self.__class__.__name__, self.molecular_formula, *self.center_of_mass
        )

    @classmethod
    def from_xyz_file(cls, filename, **kwargs):
        """construct a molecule from the provided xmol .xyz file. kwargs
        will be passed through to the Molecule constructor.

        Parameters
        ----------
        filename: str
            path to the .xyz file
        """
        from .xyz_file import parse_xyz_file

        xyz_dict = parse_xyz_file(filename)
        elements = []
        positions = []
        for label, position in xyz_dict["atoms"]:
            elements.append(Element[label])
            positions.append(position)
        return cls(
            elements, np.asarray(positions), comment=xyz_dict["comment"], **kwargs
        )

    @classmethod
    def load(cls, filename, **kwargs):
        """construct a molecule from the provided file. kwargs
        will be passed through to the Molecule constructor.

        Parameters
        ----------
        filename: str
            path to the file (in xyz format)
        """
        extension_map = {".xyz": cls.from_xyz_file}
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename, **kwargs)

    def save(self, filename, header=True):
        """save this molecule to the destination file in xyz format,
        optionally discarding the typical header.

        Parameters
        ----------
        filename: str
            path to the destination file
        header: bool, optional
            optionally disable writing of the header (no. of atoms and a comment line)
        """
        from pathlib import Path

        if header:
            lines = [
                f"{len(self)}",
                self.properties.get("comment", self.molecular_formula),
            ]
        else:
            lines = []
        for el, (x, y, z) in zip(self.elements, self.positions):
            lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")
        Path(filename).write_text("\n".join(lines))

    @property
    def bbox_corners(self):
        "the lower, upper corners of a axis-aligned bounding box for this molecule"
        b_min = np.min(self.positions, axis=0)
        b_max = np.max(self.positions, axis=0)
        return (b_min, b_max)

    @property
    def bbox_size(self):
        "the dimensions of the axis-aligned bounding box for this molecule"
        b_min, b_max = self.bbox_corners
        return np.abs(b_max - b_min)

    @property
    def asym_symops(self):
        "the symmetry operations which generate this molecule (default x,y,z if not set)"
        return self.properties.get("generator_symop", [16484] * len(self))

    @classmethod
    def from_arrays(cls, elements, positions, **kwargs):
        """construct a molecule from the provided arrays. kwargs
        will be passed through to the Molecule constructor.


        Parameters
        ----------
        elements: :obj:`np.ndarray`
            (N,) array of atomic numbers for each atom in this molecule
        positions: :obj:`np.ndarray`
            (N, 3) array of Cartesian coordinates for each atom in this molecule (Angstroms)
        """
        return cls([Element[x] for x in elements], np.array(positions), **kwargs)

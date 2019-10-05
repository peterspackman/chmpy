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
    positions: np.ndarray
    elements: np.ndarray
    labels: np.ndarray
    properties: dict

    def __init__(
        self,
        elements,
        positions,
        bonds=None,
        labels=None,
        **kwargs,
    ):
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

        tolerance is default 0.4 angstroms, which is recommended
        by the CCDC
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
        counts = defaultdict(int)
        labels = []
        for el, _ in self:
            counts[el] += 1
            labels.append("{}{}".format(el.symbol, counts[el]))
        self.labels = np.asarray(labels)


    def distance_to(self, other, method="centroid"):
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
        return np.array([e.atomic_number for e in self.elements])

    @property
    def centroid(self):
        return np.mean(self.positions, axis=0)

    @property
    def center_of_mass(self):
        masses = np.asarray([x.mass for x in self.elements])
        return np.sum(self.positions * masses[:, np.newaxis] / np.sum(masses), axis=0)

    @property
    def molecular_formula(self):
        from .element import chemical_formula

        return chemical_formula(self.elements, subscript=False)

    def __repr__(self):
        return "<{}: {}({:.2f},{:.2f},{:.2f})>".format(
            self.__class__.__name__, self.molecular_formula, *self.center_of_mass
        )

    @classmethod
    def from_xyz_file(cls, filename, **kwargs):
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
        extension_map = {
            ".xyz": cls.from_xyz_file,
        }
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename, **kwargs)

    @property
    def bbox_corners(self):
        b_min = np.min(self.positions, axis=0)
        b_max = np.max(self.positions, axis=0)
        return (b_min, b_max)

    @property
    def bbox_size(self):
        b_min, b_max = self.bbox_corners
        return np.abs(b_max - b_min)

    @property
    def asym_symops(self):
        return self.properties.get("generator_symop", [16484] * len(self))

    @classmethod
    def from_arrays(cls, elements, positions, **kwargs):
        return cls([Element[x] for x in elements], np.array(positions), **kwargs)

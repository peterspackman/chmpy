import logging
from collections import defaultdict
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import dok_matrix
import numpy as np
import os
from .element import Element


_FUNCTIONAL_GROUP_SUBGRAPHS = {}
LOG = logging.getLogger(__name__)


class Molecule:
    """class to represent information about a
    molecule i.e. a set of atoms with 3D coordinates
    joined by covalent bonds

    e.g. int, float, str etc. Will handle uncertainty values
    contained in parentheses.

    Parameters
    ----------
    elements: list of :obj:`Element`
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
        try:
            import graph_tool as gt

            self.bond_graph()
        except ImportError as e:
            pass

    def connected_fragments(self):
        from shmolecule.util import cartesian_product
        from scipy.sparse.csgraph import connected_components

        if self.bonds is None:
            self.guess_bonds()

        nfrag, labels = connected_components(self.bonds)
        molecules = []
        for frag in range(nfrag):
            atoms = np.where(labels == frag)[0]
            na = len(atoms)
            sqidx = cartesian_product(atoms, atoms)
            molecules.append(
                Molecule(
                    [self.elements[i] for i in atoms],
                    self.positions[atoms],
                    labels=self.labels[atoms],
                    bonds=self.bonds[sqidx[:, 0], sqidx[:, 1]].reshape(na, na),
                )
            )
        return molecules

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
    def partial_charges(self):
        from openbabel.pybel import readstring
        m = readstring("xyz", self.to_xyz_string())
        charges = np.array(m.calccharges(), dtype=np.float32)
        return charges

    def electrostatic_potential(self, positions):
        from shmolecule.util import BOHR_PER_ANGSTROM
        v_pot = np.zeros(positions.shape[0])
        for charge, position in zip(self.partial_charges, self.positions):
            if charge == 0.0:
                continue
            r = np.linalg.norm(positions - position[np.newaxis, :], axis=1)
            v_pot += charge / (r * BOHR_PER_ANGSTROM)
        return v_pot

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
    def from_xyz_string(cls, contents, **kwargs):
        """construct a molecule from the provided xmol .xyz file. kwargs
        will be passed through to the Molecule constructor.

        Parameters
        ----------
        contents: str
            contents of the .xyz file to read
        """
        from .xyz_file import parse_xyz_string

        elements, positions = parse_xyz_string(contents)
        return cls(elements, np.asarray(positions), **kwargs)

    @classmethod
    def from_xyz_file(cls, filename, **kwargs):
        """construct a molecule from the provided xmol .xyz file. kwargs
        will be passed through to the Molecule constructor.

        Parameters
        ----------
        filename: str
            path to the .xyz file
        """
        from pathlib import Path

        return cls.from_xyz_string(Path(filename).read_text(), **kwargs)

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

    def to_xyz_string(self, header=True):
        if header:
            lines = [
                f"{len(self)}",
                self.properties.get("comment", self.molecular_formula),
            ]
        else:
            lines = []
        for el, (x, y, z) in zip(self.elements, self.positions):
            lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")
        return "\n".join(lines)

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
        Path(filename).write_text(self.to_xyz_string(header=header))

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

    def bond_graph(self):
        """Calculate the graph_tool.Graph object corresponding
        to this molecule. Requires the graph_tool library to be
        installed

        Returns
        -------
        :obj:`graph_tool.Graph`
            the (undirected) graph of this molecule
        """

        if hasattr(self, "_bond_graph"):
            return self._bond_graph
        try:
            import graph_tool as gt
        except ImportError as e:
            raise RuntimeError(
                "Please install the graph_tool library for graph operations"
            )
        if self.bonds is None:
            self.guess_bonds()
        g = gt.Graph(directed=False)
        v_el = g.new_vertex_property("int")
        g.add_edge_list(self.bonds.keys())
        e_w = g.new_edge_property("float")
        v_el.a[:] = self.atomic_numbers
        g.vertex_properties["element"] = v_el
        e_w.a[:] = list(self.bonds.values())
        g.edge_properties["bond_distance"] = e_w
        self._bond_graph = g
        return g

    def functional_groups(self, kind=None):
        """Find all indices of atom groups which constitute
        subgraph isomorphisms with stored functional group data

        Parameters
        ----------
        kind: str, optional
            Find only matches of the given kind

        Returns
        -------
        Either
            a dict with keys as functional group type and values as list of
            lists of indices, or a list of lists of indices if kind is specified.
        """
        global _FUNCTIONAL_GROUP_SUBGRAPHS
        try:
            import graph_tool.topology as top
        except ImportError as e:
            raise RuntimeError(
                "Please install the graph_tool library for graph operations"
            )
        if not _FUNCTIONAL_GROUP_SUBGRAPHS:
            from shmolecule.subgraphs import load_data

            _FUNCTIONAL_GROUP_SUBGRAPHS = load_data()

        if kind is not None:
            sub = _FUNCTIONAL_GROUP_SUBGRAPHS[kind]
            matches = self.matching_subgraph(sub)
            if kind == "ring":
                matches = list(
                    set(tuple(sorted(x)) for x in matches)
                )
            return matches

        matches = {}
        for n, sub in _FUNCTIONAL_GROUP_SUBGRAPHS.items():
            m = self.matching_subgraph(sub)
            if n == "ring":
                m = list(
                    set(tuple(sorted(x)) for x in m)
                )
            matches[n] = m
        return matches

    def matching_subgraph(self, sub):
        """Find all indices of atoms which match the given graph

        Parameters
        ----------
        sub: :obj:`graph_tool.Graph`
            the subgraph

        Returns
        -------
        list
            list of lists of atomic indices matching the atoms in sub
            to those in this molecule
        """

        try:
            import graph_tool.topology as top
        except ImportError as e:
            raise RuntimeError(
                "Please install the graph_tool library for graph operations"
            )

        g = self.bond_graph()
        matches = top.subgraph_isomorphism(
            sub,
            g,
            vertex_label=(
                sub.vertex_properties["element"],
                g.vertex_properties["element"],
            ),
        )
        return [tuple(x.a) for x in matches]

    def matching_fragments(self, fragment, method="connectivity"):
        """Find the indices of a matching fragment to the given
        molecular fragment

        Parameters
        ----------
        fragment: :obj:`Molecule`
            Molecule object containing the desired fragment

        Returns
        -------
        list of dict
            List of maps between matching indices in this molecule and those
            in the fragment
        """
        try:
            import graph_tool.topology as top
        except ImportError as e:
            raise RuntimeError(
                "Please install the graph_tool library for graph operations"
            )

        sub = fragment.bond_graph()
        g = self.bond_graph()
        matches = top.subgraph_isomorphism(
            sub,
            g,
            vertex_label=(
                sub.vertex_properties["element"],
                g.vertex_properties["element"],
            ),
        )
        return [list(x.a) for x in matches]

    def atomic_shape_descriptors(self, l_max=5, radius=6.0, background=1e-5):
        """Calculate the shape descriptors[1,2] for all
        atoms in this isolated molecule. If you wish to use
        the crystal environment please see the corresponding method
        in :obj:`shmolecule.crystal.Crystal`.

        Parameters
        ----------
        l_max: int, optional
            maximum level of angular momenta to include in the spherical harmonic
            transform of the shape function. (default=5)
        radius: float, optional
            Maximum distance in Angstroms between any atom in the molecule
            and the resulting neighbouring atoms (default=6.0)
        background: float, optional
            'background' density to ensure closed surfaces for isolated atoms
            (default=1e-5)

        Returns
        -------
        :obj:`np.ndarray`
            shape description vector

        References
        ----------
        [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
            https://dx.doi.org/10.1038/srep22204
        [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
            https://dx.doi.org/10.1002/anie.201906602
        """
        descriptors = []
        from .sht import SHT
        from .shape_descriptors import stockholder_weight_descriptor

        sph = SHT(l_max=l_max)
        elements = self.atomic_numbers
        positions = self.positions
        dists = cdist(self.positions, self.positions)

        for n in range(elements.shape[0]):
            els = elements[n : n + 1]
            pos = positions[n : n + 1, :]
            idxs = np.where((dists[n, :] < radius) & (dists[n, :] > 1e-3))[0]
            neighbour_els = elements[idxs]
            neighbour_pos = positions[idxs]
            ubound = Element[n].vdw_radius * 3
            descriptors.append(
                stockholder_weight_descriptor(
                    sph,
                    els,
                    pos,
                    neighbour_els,
                    neighbour_pos,
                    bounds=(0.2, ubound),
                    background=background,
                )
            )
        return np.asarray(descriptors)

    def atomic_stockholder_weight_isosurfaces(self, **kwargs):
        """Calculate the stockholder weight isosurfaces for the atoms
        in this molecule, with the provided background density.

        Keyword Args
        ------------
        background: float, optional
            'background' density to ensure closed surfaces for isolated atoms
            (default=1e-5)
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
            'd_i', 'd_norm_e', 'd_e', 'd_norm')
        colormap: str, optional
            matplotlib colormap to use for surface coloring (default 'viridis_r')
        midpoint: float, optional, default 0.0 if using d_norm
            use the midpoint norm (as is used in CrystalExplorer)

        Returns
        -------
        list of :obj:`trimesh.Trimesh`
            A list of meshes representing the stockholder weight isosurfaces
        """

        from .density import StockholderWeight
        from .surface import stockholder_weight_isosurface
        from matplotlib.cm import get_cmap
        import trimesh
        from .crystal import DEFAULT_COLORMAPS

        sep = kwargs.get("separation", kwargs.get("resolution", 0.2))
        radius = kwargs.get("radius", 12.0)
        background = kwargs.get("background", 1e-5)
        vertex_color = kwargs.get("color", "d_norm_i")
        isovalue = kwargs.get("isovalue", 0.5)
        midpoint = kwargs.get("midpoint", 0.0 if vertex_color == "d_norm" else None)
        meshes = []
        colormap = get_cmap(
            kwargs.get("colormap", DEFAULT_COLORMAPS.get(vertex_color, "viridis_r"))
        )
        isos = []
        elements = self.atomic_numbers
        positions = self.positions
        dists = cdist(self.positions, self.positions)

        for n in range(elements.shape[0]):
            els = elements[n : n + 1]
            pos = positions[n : n + 1, :]
            idxs = np.where((dists[n, :] < radius) & (dists[n, :] > 1e-3))[0]
            neighbour_els = elements[idxs]
            neighbour_pos = positions[idxs]

            s = StockholderWeight.from_arrays(
                els, pos, neighbour_els, neighbour_pos, background=background
            )
            iso = stockholder_weight_isosurface(s, isovalue=isovalue, sep=sep)
            isos.append(iso)
        for iso in isos:
            prop = iso.vertex_prop[vertex_color]
            norm = None
            if midpoint is not None:
                from matplotlib.colors import DivergingNorm

                norm = DivergingNorm(vmin=prop.min(), vcenter=midpoint, vmax=prop.max())
                prop = norm(prop)
            color = colormap(prop)
            mesh = trimesh.Trimesh(
                vertices=iso.vertices,
                faces=iso.faces,
                normals=iso.normals,
                vertex_colors=color,
            )
            meshes.append(mesh)
        return meshes

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

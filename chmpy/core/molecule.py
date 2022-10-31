import logging
from collections import defaultdict
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import dok_matrix
from pathlib import Path
import numpy as np
from .element import Element
from typing import Tuple, List, Union


_FUNCTIONAL_GROUP_SUBGRAPHS = {}
LOG = logging.getLogger(__name__)


class Molecule:
    """
    Class to represent information about a
    molecule i.e. a set of atoms with 3D coordinates
    joined by covalent bonds

    e.g. int, float, str etc. Will handle uncertainty values
    contained in parentheses.

    Attributes:
        elements: list of element information for each atom in this molecule
        positions: (N, 3) array of Cartesian coordinates for each atom in this molecule (Angstroms)
        bonds: (N, N) adjacency matrix of bond lengths for connected atoms, 0 otherwise.
            If not provided this will be calculated.
        labels: (N,) vector of string labels for each atom in this molecule
            If not provided this will assigned default labels i.e. numbered in order.
        proerties: Additional keyword arguments will be stored in the `properties` member, and
            some may be utilized in methods, raising an exception if they are not set.
    """

    positions: np.ndarray
    elements: np.ndarray
    labels: np.ndarray
    properties: dict
    bonds: dok_matrix

    def __init__(self, elements, positions, bonds=None, labels=None, **kwargs):
        """
        Initialize a new molecule.

        Arguments:
            elements (List[Element]): N length list of elements associated with the sites
            positions (array_like): (N, 3) array of site positions in Cartesian coordinates
            bonds (dok_matrix, optional): if bonds are already calculated provide them here
            labels (array_like, optional): labels (array_like): N length array of string labels for each site
            **kwargs: Additional properties (will populate the properties member) to store in this molecule
        """
        self.positions = positions
        self.elements = elements
        self.properties = {}
        self.properties.update(kwargs)
        self.bonds = None

        self.charge = kwargs.get("charge", 0)
        self.multiplicity = kwargs.get("multiplicity", 1)

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

    @property
    def distance_matrix(self) -> np.ndarray:
        "The (dense) pairwise distance matrix for this molecule"
        return cdist(self.positions, self.positions)

    @property
    def unique_bonds(self) -> List:
        """The unique bonds for this molecule. If bonds are not assigned,
        this will `None`"""
        if self.bonds is None:
            return None
        return tuple(
            (a, b, self.bonds[a, b])
            for a, b in set(tuple(sorted(x)) for x in self.bonds.keys())
        )

    def guess_bonds(self, tolerance=0.40):
        """
        Use geometric distances and covalent radii
        to determine bonding information for this molecule.

        Bonding is determined by the distance between
        sites being closer than the sum of covalent radii + `tolerance`

        Will set the `bonds` member.

        If the `graph_tool` library is available, this will call the
        `bond_graph` method to populate the connectivity graph.


        Args:
            tolerance (float, optional): Additional tolerance for attributing two sites as 'bonded'.
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
            self.bond_graph()
        except Exception:
            pass

    def connected_fragments(self) -> List:
        """
        Separate this molecule into fragments/molecules based
        on covalent bonding criteria.

        Returns:
            a list of connected `Molecule` objects
        """
        from chmpy.util.num import cartesian_product
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
        """
        Calculate the euclidean distance between this
        molecule and another. May use the distance between
        centres-of-mass, centroids, or nearest atoms.

        Parameters
            other (Molecule): the molecule to calculate distance to
            method (str, optional): one of 'centroid', 'center_of_mass', 'nearest_atom'
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
    def atomic_numbers(self) -> np.ndarray:
        "Atomic numbers for each atom in this molecule"
        return np.array([e.atomic_number for e in self.elements])

    @property
    def centroid(self) -> np.ndarray:
        "Mean cartesian position of atoms in this molecule"
        return np.mean(self.positions, axis=0)

    @property
    def center_of_mass(self) -> np.ndarray:
        "Mean cartesian position of atoms in this molecule, weighted by atomic mass"
        if len(self) > 0:
            masses = np.asarray([x.mass for x in self.elements])
            return np.sum(
                self.positions * masses[:, np.newaxis] / np.sum(masses), axis=0
            )
        return np.zeros(3)

    @property
    def partial_charges(self) -> np.ndarray:
        """The partial charges associated with atoms in this molecule.
        If `self._partial_charges` is not set, the charges will be
        assigned based on EEM method."""
        assert len(self) > 0, "Must have at least one atom to calculate partial charges"
        if not hasattr(self, "_partial_charges"):
            from chmpy.ext.charges import EEM

            charges = EEM.calculate_charges(self)
            self._partial_charges = charges.astype(np.float32)
        return self._partial_charges

    @partial_charges.setter
    def partial_charges(self, charges):
        self._partial_charges = charges

    @partial_charges.deleter
    def partial_charges(self):
        del self._partial_charges

    def electrostatic_potential_from_cube(self, cube, positions):
        LOG.info("Interpolating ESP using assigned cube data")
        interpolator = cube.interpolator()
        return interpolator.predict(positions)

    def electrostatic_potential(self, positions) -> np.ndarray:
        """
        Calculate the electrostatic potential based on the partial
        charges associated with this molecule. The potential will be
        in atomic units.

        Args:
            positions (np.ndarray): (N, 3) array of locations where the molecular ESP should
                be calculated. Assumed to be in Angstroms.

        Returns:
            np.ndarray: (N,) array of electrostatic potential values (atomic units) at the given
            positions.
        """
        if "esp_cube" in self.properties:
            return self.electrostatic_potential_from_cube(
                self.properties["esp_cube"], positions
            )

        from chmpy.util.unit import BOHR_TO_ANGSTROM

        v_pot = np.zeros(positions.shape[0])
        for charge, position in zip(self.partial_charges, self.positions):
            if charge == 0.0:
                continue
            r = np.linalg.norm(positions - position[np.newaxis, :], axis=1)
            v_pot += BOHR_TO_ANGSTROM * charge / r
        return v_pot

    @property
    def molecular_formula(self) -> str:
        "string of the molecular formula for this molecule"
        from .element import chemical_formula

        if len(self) > 0:
            return chemical_formula(self.elements, subscript=False)
        return "empty"

    def __repr__(self):
        x, y, z = self.center_of_mass
        return "<{name} ({formula})[{x:.2f} {y:.2f} {z:.2f}]>".format(
            name=self.name, formula=self.molecular_formula, x=x, y=y, z=z
        )

    @classmethod
    def from_xyz_string(cls, contents, **kwargs):
        """
        Construct a molecule from the provided xmol .xyz file. kwargs
        will be passed through to the Molecule constructor.

        Args:
            contents (str): the contents of the .xyz file to read
            kwargs: keyword arguments passed ot the `Molecule` constructor

        Returns:
            Molecule: A new `Molecule` object
        """
        from chmpy.fmt.xyz_file import parse_xyz_string

        elements, positions = parse_xyz_string(contents)
        return cls(elements, np.asarray(positions), **kwargs)

    @classmethod
    def from_xyz_file(cls, filename, **kwargs):
        """
        Construct a molecule from the provided xmol .xyz file. kwargs
        will be passed through to the Molecule constructor.

        Args:
            filename (str): the path to the .xyz file
            kwargs: keyword arguments passed ot the `Molecule` constructor

        Returns:
            Molecule: A new `Molecule` object
        """

        return cls.from_xyz_string(Path(filename).read_text(), **kwargs)

    @classmethod
    def from_turbomole_string(cls, contents, **kwargs):
        """
        Construct a molecule from the provided turbomole file contents. kwargs
        will be passed through to the Molecule constructor.

        Args:
            contents (str): the contents of the .xyz file to read
            kwargs: keyword arguments passed ot the `Molecule` constructor

        Returns:
            Molecule: A new `Molecule` object
        """
        from chmpy.fmt.tmol import parse_tmol_string

        elements, positions = parse_tmol_string(contents)
        return cls(elements, np.asarray(positions), **kwargs)

    @classmethod
    def from_turbomole_file(cls, filename, **kwargs):
        """
        Construct a molecule from the provided turbomole file. kwargs
        will be passed through to the Molecule constructor.

        Args:
            filename (str): the path to the .xyz file
            kwargs: keyword arguments passed ot the `Molecule` constructor

        Returns:
            Molecule: A new `Molecule` object
        """
        return cls.from_turbomole_string(Path(filename).read_text(), **kwargs)

    @classmethod
    def from_fchk_string(cls, fchk_contents, **kwargs):
        from chmpy.fmt.fchk import FchkFile
        from chmpy.util.unit import units

        fchk = FchkFile(fchk_contents, parse=True)
        elements = np.array(fchk["Atomic numbers"])
        positions = np.array(fchk["Current cartesian coordinates"]).reshape(
            elements.shape[0], 3
        )
        positions = units.angstrom(positions, unit="bohr")
        return cls.from_arrays(elements=elements, positions=positions, **kwargs)

    @classmethod
    def from_fchk_file(cls, filename, **kwargs):
        return cls.from_fchk_string(Path(filename).read_text(), **kwargs)

    @classmethod
    def from_mol2_string(cls, contents, **kwargs):
        from chmpy.fmt.mol2 import parse_mol2_string
        atoms, bonds = parse_mol2_string(contents)
        elements = [Element[x] for x in atoms.pop("type")]

        N = len(elements)

        positions = np.array([
            tuple(p) for p in zip(atoms.pop("x"), atoms.pop("y"), atoms.pop("z"))
        ])

        labels = None
        if "name" in atoms:
            labels = atoms.pop("name")

        bondlist = None
        if bonds != {}:
            bondlist = dok_matrix((N, N))

            for a, b, t in zip(bonds["origin"], bonds["target"], bonds["type"]):
                bondlist[a - 1, b - 1] = int(t)

        return cls(elements, positions, bonds=bondlist, labels=labels, **atoms, **kwargs)

    @classmethod
    def from_mol2_file(cls, filename, **kwargs):
        return cls.from_mol2_string(Path(filename).read_text(), **kwargs)

    @classmethod
    def _ext_load_map(cls):
        return {
            ".xyz": cls.from_xyz_file,
            ".sdf": cls.from_sdf_file,
            ".fchk": cls.from_fchk_file,
            ".coord": cls.from_turbomole_file,
            ".mol2": cls.from_mol2_file,
        }

    @classmethod
    def _fname_load_map(cls):
        return {}

    def _ext_save_map(self):
        return {".xyz": self.to_xyz_file,
                ".sdf": self.to_sdf_file}

    def _fname_save_map(self):
        return {}

    @classmethod
    def load(cls, filename, **kwargs):
        """
        Construct a molecule from the provided file.

        Args:
            filename (str): the path to the (xyz or SDF) file
            kwargs: keyword arguments passed ot the `Molecule` constructor

        Returns:
            Molecule: A new `Molecule` object
        """
        fpath = Path(filename)
        n = fpath.name
        fname_map = cls._fname_load_map()
        if n in fname_map:
            return fname_map[n](filename)
        extension_map = cls._ext_load_map()
        extension = kwargs.pop("fmt", fpath.suffix.lower())
        if not extension.startswith("."):
            extension = "." + extension
        return extension_map[extension](filename, **kwargs)

    def to_sdf_string(self) -> str:
        """
        Represent this molecule as a string in the format
        of an MDL .sdf file.

        Returns:
            contents (str) the contents of the .sdf file
        """
        from chmpy.fmt.sdf import to_sdf_string

        bonds_left = []
        bonds_right = []
        if self.bonds is not None:
            self.guess_bonds()
            for x, y in self.bonds.keys():
                bonds_left.append(x + 1)
                bonds_right.append(y + 1)


        sdf_dict = {
            "header": [self.name, "created by chmpy", ""],
            "atoms": {
                "x": self.positions[:, 0],
                "y": self.positions[:, 0],
                "z": self.positions[:, 0],
                "symbol": np.array([x.symbol for x in self.elements]),
            },
            "bonds": {
                "left": np.array(bonds_left),
                "right": np.array(bonds_right),
            }
        }
        return to_sdf_string(sdf_dict)

    def to_sdf_file(self, filename, **kwargs):
        """
        Represent this molecule as an
        of an MDL .sdf file. Keyword arguments are
        passed to `self.to_sdf_string`.

        Args:
            filename (str): The path in which store this molecule
            kwargs: Keyword arguments to be passed to `self.to_sdf_string`
        """
        Path(filename).write_text(self.to_sdf_string(**kwargs))


    def to_xyz_string(self, header=True) -> str:
        """
        Represent this molecule as a string in the format
        of an xmol .xyz file.

        Args:
            header (bool, optional):toggle whether or not to return the 'header' of the
                xyz file i.e. the number of atoms line and the comment line

        Returns:
            contents (str) the contents of the .xyz file
        """
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

    def to_xyz_file(self, filename, **kwargs):
        """
        Represent this molecule as an
        of an xmol .xyz file. Keyword arguments are
        passed to `self.to_xyz_string`.

        Args:
            filename (str): The path in which store this molecule
            kwargs: Keyword arguments to be passed to `self.to_xyz_string`
        """

        Path(filename).write_text(self.to_xyz_string(**kwargs))

    def save(self, filename, **kwargs):
        """
        Save this molecule to the destination file in xyz format,
        optionally discarding the typical header.

        Args:
            filename (str): path to the destination file
            kwargs: keyword arguments passed to the relevant method
        """
        fpath = Path(filename)
        n = fpath.name
        fname_map = self._fname_save_map()
        if n in fname_map:
            return fname_map[n](filename, **kwargs)
        extension_map = self._ext_save_map()
        extension = kwargs.pop("fmt", fpath.suffix.lower())
        if not extension.startswith("."):
            extension = "." + extension
        return extension_map[extension](filename, **kwargs)

    @property
    def bbox_corners(self) -> Tuple:
        "the lower, upper corners of a axis-aligned bounding box for this molecule"
        b_min = np.min(self.positions, axis=0)
        b_max = np.max(self.positions, axis=0)
        return (b_min, b_max)

    @property
    def bbox_size(self) -> np.ndarray:
        "the dimensions of the axis-aligned bounding box for this molecule"
        b_min, b_max = self.bbox_corners
        return np.abs(b_max - b_min)

    def bond_graph(self):
        """
        Calculate the `graph_tool.Graph` object corresponding
        to this molecule. Requires the graph_tool library to be
        installed

        Returns:
            graph_tool.Graph: the (undirected) graph of this molecule
        """

        if hasattr(self, "_bond_graph"):
            return getattr(self, "_bond_graph")
        try:
            import graph_tool as gt
        except ImportError:
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

    def functional_groups(self, kind=None) -> Union[dict, List]:
        """
        Find all indices of atom groups which constitute
        subgraph isomorphisms with stored functional group data

        Args:
            kind (str, optional):Find only matches of the given kind

        Returns:
            Either a dict with keys as functional group type and values as list of
            lists of indices, or a list of lists of indices if kind is specified.
        """
        global _FUNCTIONAL_GROUP_SUBGRAPHS
        try:
            import graph_tool.topology as top  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Please install the graph_tool library for graph operations"
            )
        if not _FUNCTIONAL_GROUP_SUBGRAPHS:
            from chmpy.subgraphs import load_data

            _FUNCTIONAL_GROUP_SUBGRAPHS = load_data()

        if kind is not None:
            sub = _FUNCTIONAL_GROUP_SUBGRAPHS[kind]
            matches = self.matching_subgraph(sub)
            if kind == "ring":
                matches = list(set(tuple(sorted(x)) for x in matches))
            return matches

        matches = {}
        for n, sub in _FUNCTIONAL_GROUP_SUBGRAPHS.items():
            m = self.matching_subgraph(sub)
            if n == "ring":
                m = list(set(tuple(sorted(x)) for x in m))
            matches[n] = m
        return matches

    def matching_subgraph(self, sub):
        """Find all indices of atoms which match the given graph.

        Args:
            sub (graph_tool.Graph): the subgraph

        Returns:
            List: list of lists of atomic indices matching the atoms in sub
                to those in this molecule
        """

        try:
            import graph_tool.topology as top
        except ImportError:
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
        """
        Find the indices of a matching fragment to the given
        molecular fragment

        Args:
            fragment (Molecule): Molecule object containing the desired fragment
            method (str, optional): the method for matching

        Returns:
            List[dict]: List of maps between matching indices in this molecule and those
                in the fragment
        """
        try:
            import graph_tool.topology as top
        except ImportError:
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

    def calculate_wavefunction(self, method="HF", basis_set="3-21G", program="nwchem"):
        from chmpy.fmt.nwchem import to_nwchem_input

        return to_nwchem_input(self, method=method, basis_set=basis_set)

    def atomic_shape_descriptors(
        self, l_max=5, radius=6.0, background=1e-5
    ) -> np.ndarray:
        """
        Calculate the shape descriptors`[1,2]` for all
        atoms in this isolated molecule. If you wish to use
        the crystal environment please see the corresponding method
        in :obj:`chmpy.crystal.Crystal`.

        Args:
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the shape function. (default=5)
            radius (float, optional): Maximum distance in Angstroms between any atom in the molecule
                and the resulting neighbouring atoms (default=6.0)
            background (float, optional): 'background' density to ensure closed surfaces for isolated atoms
                (default=1e-5)

        Returns:
            shape description vector

        References:
            [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
                https://dx.doi.org/10.1038/srep22204
            [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
                https://dx.doi.org/10.1002/anie.201906602
        """
        descriptors = []
        from chmpy.shape import SHT, stockholder_weight_descriptor

        sph = SHT(l_max)
        elements = self.atomic_numbers
        positions = self.positions
        dists = self.distance_matrix

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
        """
        Calculate the stockholder weight isosurfaces for the atoms
        in this molecule, with the provided background density.

        Keyword Args:
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

        Returns:
            List[trimesh.Trimesh]: A list of meshes representing the stockholder weight isosurfaces
        """

        from chmpy import StockholderWeight
        from chmpy.surface import stockholder_weight_isosurface
        from matplotlib.cm import get_cmap
        import trimesh
        from chmpy.util.color import DEFAULT_COLORMAPS

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
        dists = self.distance_matrix

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

    def shape_descriptors(self, l_max=5, **kwargs) -> np.ndarray:
        """
        Calculate the molecular shape descriptors`[1,2]` for
        this molecule using the promolecule density.

        Args:
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the molecular shape function.

        Keyword Args:
            with_property (str, optional): describe the combination of the radial shape function and a surface
                property in the real, imaginary channels of a complex function
            isovalue (float, optional): the isovalue for the promolecule density surface (default 0.0002 au)

        Returns:
            shape description vector

        References:
            [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
                https://dx.doi.org/10.1038/srep22204
            [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
                https://dx.doi.org/10.1002/anie.201906602
        """
        from chmpy.shape import SHT, promolecule_density_descriptor

        sph = SHT(l_max)
        return promolecule_density_descriptor(
            sph, self.atomic_numbers, self.positions, **kwargs
        )

    def promolecule_density_isosurface(self, **kwargs):
        """
        Calculate promolecule electron density isosurface
        for this molecule.

        Keyword Args:
            isovalue: float, optional
                level set value for the isosurface (default=0.002) in au.
            separation: float, optional
                separation between density grid used in the surface calculation
                (default 0.2) in Angstroms.
            color: str, optional
                surface property to use for vertex coloring, one of ('d_norm_i',
                'd_i', 'd_norm_e', 'd_e')
            colormap: str, optional
                matplotlib colormap to use for surface coloring (default 'viridis_r')
            midpoint: float, optional, default 0.0 if using d_norm
                use the midpoint norm (as is used in CrystalExplorer)

        Returns:
            trimesh.Trimesh: A mesh representing the promolecule density isosurface
        """
        from chmpy import PromoleculeDensity
        from chmpy.surface import promolecule_density_isosurface
        from chmpy.util.color import property_to_color
        import trimesh

        isovalue = kwargs.get("isovalue", 0.002)
        sep = kwargs.get("separation", kwargs.get("resolution", 0.2))
        vertex_color = kwargs.get("color", "d_norm_i")
        extra_props = {}
        pro = PromoleculeDensity((self.atomic_numbers, self.positions))
        if vertex_color == "esp":
            extra_props["esp"] = self.electrostatic_potential
        iso = promolecule_density_isosurface(
            pro, sep=sep, isovalue=isovalue, extra_props=extra_props
        )
        prop = iso.vertex_prop[vertex_color]
        color = property_to_color(prop, cmap=kwargs.get("cmap", vertex_color))
        mesh = trimesh.Trimesh(
            vertices=iso.vertices,
            faces=iso.faces,
            normals=iso.normals,
            vertex_colors=color,
        )
        return mesh

    def to_mesh(self, **kwargs):
        """
        Convert this molecule to a mesh of spheres and
        cylinders, colored by element. The origins of the spheres
        will be at the corresponding atomic position, and all units
        will be Angstroms.

        Returns:
            dict: a dictionary of `trimesh.Trimesh` objects representing this molecule.
        """
        from chmpy.util.mesh import molecule_to_meshes

        return molecule_to_meshes(self, **kwargs)

    @property
    def name(self):
        "The name of this molecule, checks 'GENERIC_NAME' and 'name' keys in `self.properties`"
        return self.properties.get(
            "GENERIC_NAME", self.properties.get("name", self.__class__.__name__)
        )

    @property
    def molecular_dipole_moment(self):
        if "molecular_dipole" in self.properties:
            return self.properties["molecular_dipole"]

        if hasattr(self, "_partial_charges"):
            from chmpy.util.unit import ANGSTROM_TO_BOHR

            net_charge = np.sum(self.partial_charges)
            if np.abs(net_charge) > 1e-3:
                LOG.warn(
                    "Molecular dipole will be origin dependent: molecule has a net charge (%f)",
                    net_charge,
                )
            r = ANGSTROM_TO_BOHR * (self.positions - self.center_of_mass)
            return np.sum(r * self.partial_charges[:, np.newaxis], axis=0)
        return np.zeros(3)

    @property
    def asym_symops(self):
        "the symmetry operations which generate this molecule (default x,y,z if not set)"
        return self.properties.get("generator_symop", [16484] * len(self))

    @classmethod
    def from_arrays(cls, elements, positions, **kwargs):
        """
        Construct a molecule from the provided arrays. kwargs
        will be passed through to the Molecule constructor.

        Args:
            elements (np.ndarray): (N,) array of atomic numbers for each atom in this molecule
            positions (np.ndarray): (N, 3) array of Cartesian coordinates for each atom in this molecule (Angstroms)

        Returns:
            Molecule: a new molecule object
        """
        return cls([Element[x] for x in elements], np.array(positions), **kwargs)

    def mask(self, mask, **kwargs):
        """
        Convenience method to construct a new molecule from this molecule with the given mask
        array.

        Args:
            mask (np.ndarray): a numpy mask array used to filter which atoms to keep in the new molecule.

        Returns:
            Molecule: a new `Molecule`, with atoms filtered by the mask.
        """
        return Molecule.from_arrays(
            self.atomic_numbers[mask], self.positions[mask], **kwargs
        )

    def rotate(self, rotation, origin=(0, 0, 0)):
        """
        Convenience method to rotate this molecule by a given
        rotation matrix

        Args:
            rotation (np.ndarray): A (3, 3) rotation matrix
        """

        if np.allclose(origin, (0, 0, 0)):
            np.dot(self.positions, rotation, out=self.positions)
        else:
            self.positions -= origin
            np.dot(self.positions, rotation, out=self.positions)
            self.positions += origin

    def axes(self, homogeneous=False, method="pca"):
        if method == "pca":
            axes, s, vh = np.linalg.svd((self.positions - self.center_of_mass).T)
        else:
            raise ValueError(f"Unknown molecular axis method '{method}'")
        if homogeneous:
            transform = np.eye(4)
            transform[:3, :3] = axes
            translation = -np.dot(axes, self.center_of_mass)
            transform[:3, 3] = translation
            transform[np.abs(transform) < 1e-15] = 0
            return transform
        return axes

    def inertia_tensor(self):
        masses = np.asarray([x.mass for x in self.elements])
        d = self.positions - self.center_of_mass
        d2 = d ** 2
        r2 = (d2).sum(axis=1)
        tensor = np.empty((3, 3))
        tensor[0, 0] = np.sum(masses * (d2[:, 1] + d2[:, 2]))
        tensor[1, 1] = np.sum(masses * (d2[:, 0] + d2[:, 2]))
        tensor[2, 2] = np.sum(masses * (d2[:, 0] + d2[:, 1]))
        tensor[0, 1] = - np.sum(masses * d[:, 0] * d[:, 1])
        tensor[1, 0] = tensor[0, 1]
        tensor[0, 2] = - np.sum(masses * d[:, 0] * d[:, 2])
        tensor[2, 0] = tensor[0, 2]
        tensor[1, 2] = - np.sum(masses * d[:, 1] * d[:, 2])
        tensor[2, 1] = tensor[1, 2]
        return tensor

    def principle_moments_of_inertia(self, units="amu angstrom^2"):
        t = self.inertia_tensor()
        return np.sort(np.linalg.eig(t)[0])

    def rotational_constants(self, unit="MHz"):
        from scipy.constants import Planck, speed_of_light, Avogadro
        from chmpy.util.unit import BOHR_TO_ANGSTROM
        # convert amu angstrom^2 to g cm^2
        moments = self.principle_moments_of_inertia() / Avogadro / 1e16

        # convert g cm^2 to kg m^2
        return 1e5 * Planck / (8 * np.pi * np.pi * speed_of_light * moments)


    def positions_in_molecular_axis_frame(self, method="pca"):
        if method not in ("pca",):
            raise NotImplementedError("Only pca implemented")
        if len(self) == 1:
            return np.array([[0.0, 0.0, 0.0]])
        axis = self.axes(method=method)
        return np.dot(self.positions - self.center_of_mass, axis.T)

    def oriented(self, method="pca"):
        from copy import deepcopy

        result = deepcopy(self)
        result.positions = self.positions_in_molecular_axis_frame(method=method)
        return result

    def rotated(self, rotation, origin=(0, 0, 0)):
        """
        Convenience method to construct a new copy of thismolecule
        rotated by a given rotation matrix

        Args:
            rotation (np.ndarray): A (3, 3) rotation matrix

        Returns:
            Molecule: a new copy of this `Molecule` rotated by the given rotation matrix.
        """
        from copy import deepcopy

        result = deepcopy(self)
        result.rotate(rotation, origin=origin)
        return result

    def translate(self, translation):
        """
        Convenience method to translate this molecule by a given
        translation vector

        Args:
            translation (np.ndarray): A (3,) vector of x, y, z coordinates of the translation
        """
        self.positions += translation

    def translated(self, translation):
        """
        Convenience method to construct a new copy of this molecule
        translated by a given translation vector

        Args:
            translation (np.ndarray): A (3,) vector of x, y, z coordinates of the translation

        Returns:
            Molecule: a new copy of this `Molecule` translated by the given vector.
        """
        import copy

        result = copy.deepcopy(self)
        result.positions += translation
        return result

    def transform(self, rotation=None, translation=None):
        """
        Convenience method to transform this molecule
        by rotation and translation.

        Args:
            rotation (np.ndarray): A (3,3) rotation matrix
            translation (np.ndarray): A (3,) vector of x, y, z coordinates of the translation
        """

        if rotation is not None:
            self.rotate(rotation, origin=(0, 0, 0))
        if translation is not None:
            self.translate(translation)

    def transformed(self, rotation=None, translation=None):
        """
        Convenience method to transform this molecule
        by rotation and translation.

        Args:
            rotation (np.ndarray): A (3,3) rotation matrix
            translation (np.ndarray): A (3,) vector of x, y, z coordinates of the translation

        Returns:
            Molecule: a new copy of this `Molecule` transformed by the provided matrix and vector.
        """

        from copy import deepcopy

        result = deepcopy(self)
        result.transform(rotation=rotation, translation=translation)
        return result

    @classmethod
    def from_sdf_dict(cls, sdf_dict, **kwargs) -> "Molecule":
        """
        Construct a molecule from the provided dictionary of
        sdf terms. Not intended for typical use cases, but as a
        helper method for `Molecule.from_sdf_file`

        Args:
            sdf_dict (dict): a dictionary containing the 'atoms', 'x', 'y', 'z',
                'symbol', 'bonds' members.

        Returns:
            Molecule: a new `Molecule` from the provided data
        """
        atoms = sdf_dict["atoms"]
        positions = np.c_[atoms["x"], atoms["y"], atoms["z"]]
        elements = [Element[x] for x in atoms["symbol"]]
        # TODO use bonds from SDF
        # bonds = sdf_dict["bonds"]
        m = cls(elements, positions, **sdf_dict["data"])
        if "sdf" in sdf_dict:
            m.properties["sdf"] = sdf_dict["sdf"]
        return m

    @classmethod
    def from_sdf_file(cls, filename, **kwargs):
        """
        Construct a molecule from the provided SDF file.
        Because an SDF file can have multiple molecules,
        an optional keyword argument 'progress' may be provided
        to track the loading of many molecules.

        Args:
            filename (str): the path of the SDF file to read.

        Returns:
            Molecule: a new `Molecule` or list of :obj:`Molecule` objects
            from the provided SDF file.
        """

        from chmpy.fmt.sdf import parse_sdf_file

        sdf_data = parse_sdf_file(filename, **kwargs)
        progress = kwargs.get("progress", False)
        update = lambda x: None

        if progress:
            from tqdm import tqdm

            pbar = tqdm(
                desc="Creating molecule objects", total=len(sdf_data), leave=False
            )
            update = pbar.update

        molecules = []
        for d in sdf_data:
            molecules.append(cls.from_sdf_dict(d, **kwargs))
            update(1)

        if progress:
            pbar.close()

        if len(molecules) == 1:
            return molecules[0]
        return molecules

    @classmethod
    def from_pdb_file(cls, filename, **kwargs):
        from chmpy.fmt.pdb import Pdb

        p = Pdb.from_file(filename)
        xyz = np.c_[p.data["x"], p.data["y"], p.data["z"]]
        elements = [Element[x] for x in p.data["element"]]
        return cls(elements, xyz)

import logging
from collections import defaultdict
import os
from scipy.spatial import cKDTree as KDTree
import numpy as np
from scipy.sparse import dok_matrix
import scipy.sparse.csgraph as csgraph
from pathlib import Path
from .cif import Cif
from .space_group import SpaceGroup, SymmetryOperation
from .element import Element, chemical_formula
from .molecule import Molecule
from .util import cartesian_product


LOG = logging.getLogger(__name__)


class UnitCell:
    def __init__(self, vectors):
        self.set_vectors(vectors)

    @property
    def lattice(self):
        return self.direct

    @property
    def reciprocal_lattice(self):
        return self.inverse.T

    def to_cartesian(self, coords):
        return np.dot(coords, self.direct)

    def to_fractional(self, coords):
        return np.dot(coords, self.inverse)

    def set_lengths_and_angles(self, lengths, angles):
        self.lengths = lengths
        self.angles = angles
        a, b, c = self.lengths
        ca, cb, cg = np.cos(self.angles)
        sg = np.sin(self.angles[2])
        v = self.volume()
        self.direct = np.transpose(
            [
                [a, b * cg, c * cb],
                [0, b * sg, c * (ca - cb * cg) / sg],
                [0, 0, v / (a * b * sg)],
            ]
        )
        r = [
            [1 / a, 0.0, 0.0],
            [-cg / (a * sg), 1 / (b * sg), 0],
            [
                b * c * (ca * cg - cb) / v / sg,
                a * c * (cb * cg - ca) / v / sg,
                a * b * sg / v,
            ],
        ]
        self.inverse = np.array(r)

    def set_vectors(self, vectors):
        """
        This is performed by setting the lattice parameters
        based on the provided vectors, as that results in
        a consistent basis without inverting a matrix, and
        as the res file/cif output will be relying on these
        lengths/angles anyway.

        """
        self.direct = vectors
        params = np.zeros(6)
        a, b, c = np.linalg.norm(self.direct, axis=1)
        u_a = vectors[0, :] / a
        u_b = vectors[1, :] / b
        u_c = vectors[2, :] / c
        alpha = np.arccos(np.clip(np.vdot(u_b, u_c), -1, 1))
        beta = np.arccos(np.clip(np.vdot(u_c, u_a), -1, 1))
        gamma = np.arccos(np.clip(np.vdot(u_a, u_b), -1, 1))
        params[3:] = np.degrees([alpha, beta, gamma])
        self.lengths = [a, b, c]
        self.angles = [alpha, beta, gamma]
        self.inverse = np.linalg.inv(self.direct)

    def volume(self):
        a, b, c = self.lengths
        ca, cb, cg = np.cos(self.angles)
        return a * b * c * np.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)

    @classmethod
    def from_lengths_and_angles(cls, lengths, angles, unit="radians"):
        uc = cls(np.eye(3))
        if unit == "radians":
            if np.any(np.abs(angles) > np.pi):
                LOG.warn(
                    "Large angle in UnitCell.from_lengths_and_angles, "
                    "are you sure your angles are not in degrees?"
                )
            uc.set_lengths_and_angles(lengths, angles)
        else:
            uc.set_lengths_and_angles(lengths, np.radians(angles))
        return uc

    def __repr__(self):
        return "<{}: ({:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f})>".format(
            self.__class__.__name__, *self.lengths, *self.angles
        )


class AsymmetricUnit:
    """Storage class for the coordinates and labels in a crystal
    asymmetric unit"""

    def __init__(self, elements, positions, labels=None, **kwargs):
        self.elements = elements
        self.atomic_numbers = np.asarray([x.atomic_number for x in elements])
        self.positions = np.asarray(positions)
        self.properties = {}
        self.properties.update(kwargs)
        if labels is None:
            self.labels = []
            label_index = defaultdict(int)
            for el in self.elements:
                label_index[el] += 1
                self.labels.append("{}{}".format(el, label_index[el]))
        else:
            self.labels = labels
        self.labels = np.array(self.labels)

    @property
    def formula(self):
        """Molecular formula for this asymmetric unit"""
        return chemical_formula(self.elements, subscript=False)

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return "<{}>".format(self.formula)


class Crystal:
    space_group: SpaceGroup
    unit_cell: UnitCell
    asymmetric_unit: AsymmetricUnit
    properties: dict

    def __init__(self, unit_cell, space_group, asymmetric_unit, **kwargs):
        self.space_group = space_group
        self.unit_cell = unit_cell
        self.asymmetric_unit = asymmetric_unit
        self.properties = {}
        self.properties.update(kwargs)

    @property
    def site_positions(self):
        "row major array of asymmetric unit atomic positions"
        return self.asymmetric_unit.positions

    @property
    def site_atoms(self):
        "array of asymmetric unit atomic numbers"
        return self.asymmetric_unit.atomic_numbers

    @property
    def nsites(self):
        return len(self.site_atoms)

    @property
    def symmetry_operations(self):
        return self.space_group.symmetry_operations

    def to_cartesian(self, coords):
        "Conver coordinates (row major) from fractional to cartesian"
        return self.unit_cell.to_cartesian(coords)

    def to_fractional(self, coords):
        "Conver coordinates (row major) from cartesian to fractional"
        return self.unit_cell.to_fractional(coords)

    def unit_cell_atoms(self, tolerance=1e-3):
        """Generate all atoms in the unit cell (i.e. with 
        fractional coordinates in [0, 1]) along with associated
        information about symmetry operations, occupation, elements
        related asymmetric_unit atom etc.
        
        Will merge atom sites within tolerance of each other, and
        sum their occupation numbers. A warning will be logged if
        any atom site in the unit cell has > 1.0 occupancy after
        this.
        """
        if hasattr(self, "_unit_cell_atom_dict"):
            return getattr(self, "_unit_cell_atom_dict")
        pos = self.site_positions
        atoms = self.site_atoms
        natom = self.nsites
        nsymops = len(self.space_group.symmetry_operations)
        occupation = np.tile(
            self.asymmetric_unit.properties.get("occupation", np.ones(natom)), nsymops
        )
        labels = np.tile(self.asymmetric_unit.labels, nsymops)
        uc_nums = np.tile(atoms, nsymops)
        asym = np.arange(len(uc_nums)) % natom
        sym, uc_pos = self.space_group.apply_all_symops(pos)
        translated = np.fmod(uc_pos + 7.0, 1)
        tree = KDTree(translated)
        dist = tree.sparse_distance_matrix(tree, max_distance=tolerance)
        mask = np.ones(len(uc_pos), dtype=bool)
        # because crystals may have partially occupied sites
        # on special positions, we need to merge some sites
        expected_natoms = np.sum(occupation)
        for (i, j), d in dist.items():
            if not (i < j):
                continue
            occupation[i] += occupation[j]
            mask[j] = False
        occupation = occupation[mask]
        if not np.isclose(np.sum(occupation), expected_natoms):
            LOG.warn("invalid total occupation after merging sites")
        if np.any(occupation > 1.0):
            LOG.warn("Some unit cell site occupations are > 1.0")
        setattr(
            self,
            "_unit_cell_atom_dict",
            {
                "asym_atom": asym[mask],
                "frac_pos": translated[mask],
                "element": uc_nums[mask],
                "symop": sym[mask],
                "label": labels[mask],
                "occupation": occupation,
                "cart_pos": self.to_cartesian(translated[mask]),
            },
        )
        return self._unit_cell_atom_dict

    def unit_cell_connectivity(self, tolerance=0.4, neighbouring_cells=1):
        """periodic connectiviy for the unit cell, populates _uc_graph
        with a networkx.Graph object, where nodes are indices into the
        _unit_cell_atom_dict arrays and the edges contain the translation
        (cell) for the image of the corresponding unit cell atom with the
        higher index to be bonded to the lower"""
        if hasattr(self, "_uc_graph"):
            return getattr(self, "_uc_graph")
        slab = self.slab(bounds=((-1, -1, -1), (1, 1, 1)))
        n_uc = slab["n_uc"]
        uc_pos = slab["frac_pos"][:n_uc]
        uc_nums = slab["element"][:n_uc]
        neighbour_pos = slab["frac_pos"][n_uc:]
        cart_uc_pos = self.to_cartesian(uc_pos)
        unique_elements = {x: Element.from_atomic_number(x) for x in np.unique(uc_nums)}
        # first establish all connections in the unit cell
        covalent_radii = np.array([unique_elements[x].cov for x in uc_nums])
        max_cov = np.max(covalent_radii)
        # TODO this needs to be sped up for large cells, tends to slow for > 1000 atoms
        # and the space storage will become a problem
        thresholds = (
            covalent_radii[:, np.newaxis] + covalent_radii[np.newaxis, :] + tolerance
        )
        tree = KDTree(cart_uc_pos)
        dist = tree.sparse_distance_matrix(tree, max_distance=2 * max_cov + tolerance)
        uc_edges = []

        for (i, j), d in dist.items():
            if not (i < j):
                continue
            if d > 1e-3 and d < (covalent_radii[i] + covalent_radii[j] + tolerance):
                uc_edges.append((i, j, d, (0, 0, 0)))

        idxs = np.arange(n_uc)
        asymmetric_unit_idx = slab["asym_atom"]
        cart_neighbour_pos = self.unit_cell.to_cartesian(neighbour_pos)
        tree2 = KDTree(cart_neighbour_pos)
        dist = tree.sparse_distance_matrix(tree2, max_distance=2 * max_cov + tolerance)
        # could be sped up if done outside python
        cells = slab["cell"][n_uc:]
        for (uc_atom, neighbour_atom), d in dist.items():
            uc_idx = neighbour_atom % n_uc
            if not (uc_atom < uc_idx):
                continue
            if d > 1e-3 and d < (
                covalent_radii[uc_atom] + covalent_radii[uc_idx] + tolerance
            ):
                cell = cells[neighbour_atom]
                uc_edges.append((uc_atom, uc_idx, d, tuple(cell)))

        properties = {}
        uc_graph = dok_matrix((n_uc, n_uc))
        for i, j, d, cell in uc_edges:
            uc_graph[i, j] = d
            properties[(i, j)] = cell

        setattr(self, "_uc_graph", (uc_graph, properties))
        return self._uc_graph

    def unit_cell_molecules(self):
        """Calculate the molecules for all sites in the unit cell,
        where the number of molecules will be equal to number of
        symmetry unique molecules times number of symmetry operations."""
        if hasattr(self, "_unit_cell_molecules"):
            return getattr(self, "_unit_cell_molecules")
        uc_graph, edge_cells = self.unit_cell_connectivity()
        n_uc_mols, uc_mols = csgraph.connected_components(
            csgraph=uc_graph, directed=False, return_labels=True
        )
        uc_frac = self._unit_cell_atom_dict["frac_pos"]
        uc_cartesian = self._unit_cell_atom_dict["cart_pos"]
        uc_elements = self._unit_cell_atom_dict["element"]
        uc_asym = self._unit_cell_atom_dict["asym_atom"]
        uc_symop = self._unit_cell_atom_dict["symop"]

        molecules = []

        n_uc = len(uc_frac)
        LOG.debug("%d molecules in unit cell", n_uc_mols)
        for i in range(n_uc_mols):
            nodes = np.where(uc_mols == i)[0]
            root = nodes[0]
            elements = uc_elements[nodes]
            shifts = np.zeros((n_uc, 3))
            ordered, pred = csgraph.breadth_first_order(
                csgraph=uc_graph, i_start=root, directed=False
            )
            for j in ordered[1:]:
                i = pred[j]
                if j < i:
                    shifts[j, :] = shifts[i, :] - edge_cells[(j, i)]
                else:
                    shifts[j, :] = shifts[i, :] + edge_cells[(i, j)]
            positions = self.to_cartesian((uc_frac + shifts)[nodes])
            asym_atoms = uc_asym[nodes]
            reorder = np.argsort(asym_atoms)
            asym_atoms = asym_atoms[reorder]

            mol = Molecule.from_arrays(
                elements=elements[reorder],
                positions=positions[reorder],
                guess_bonds=True,
                unit_cell_atoms=np.array(nodes)[reorder],
                asymmetric_unit_atoms=asym_atoms,
                asymmetric_unit_labels=self.asymmetric_unit.labels[asym_atoms],
                generator_symop=uc_symop[np.asarray(nodes)[reorder]],
            )
            molecules.append(mol)
        setattr(self, "_unit_cell_molecules", molecules)
        return molecules

    def symmetry_unique_molecules(self, bond_tolerance=0.4):
        """Calculate a list of connected molecules which contain
        every site in the asymmetric_unit"""
        if hasattr(self, "_symmetry_unique_molecules"):
            return getattr(self, "_symmetry_unique_molecules")
        uc_molecules = self.unit_cell_molecules()
        asym_atoms = np.zeros(len(self.asymmetric_unit), dtype=bool)
        molecules = []
        # sort by % of identity symop
        order = lambda x: len(np.where(x.asym_symops == 16484)[0]) / len(x)
        for i, mol in enumerate(sorted(uc_molecules, key=order, reverse=True)):
            asym_atoms_in_g = np.unique(mol.properties["asymmetric_unit_atoms"])
            if np.all(asym_atoms[asym_atoms_in_g]):
                continue
            asym_atoms[asym_atoms_in_g] = True
            molecules.append(mol)
            if np.all(asym_atoms):
                break
        LOG.debug("%d symmetry unique molecules", len(molecules))
        setattr(self, "_symmetry_unique_molecules", molecules)
        return molecules

    def slab(self, bounds=((-1, -1, -1), (1, 1, 1))):
        """Calculate the atoms and associated information
        for a slab consisting of multiple unit cells.
        
        If unit cell atoms have not been calculated, this populates
        their information."""
        uc_atoms = self.unit_cell_atoms()
        (hmin, kmin, lmin), (hmax, kmax, lmax) = bounds
        h = np.arange(hmin, hmax + 1)
        k = np.arange(kmin, kmax + 1)
        l = np.arange(lmin, lmax + 1)
        cells = cartesian_product(
            h[np.argsort(np.abs(h))], k[np.argsort(np.abs(k))], l[np.argsort(np.abs(l))]
        )
        ncells = len(cells)
        uc_pos = uc_atoms["frac_pos"]
        n_uc = len(uc_pos)
        pos = np.empty((ncells * n_uc, 3), dtype=np.float64)
        slab_cells = np.empty((ncells * n_uc, 3), dtype=np.float64)
        for i, cell in enumerate(cells):
            pos[i * n_uc : (i + 1) * n_uc, :] = uc_pos + cell
            slab_cells[i * n_uc : (i + 1) * n_uc] = cell
        slab_dict = {
            k: np.tile(v, ncells) for k, v in uc_atoms.items() if not k.endswith("pos")
        }
        slab_dict["frac_pos"] = pos
        slab_dict["cell"] = slab_cells
        slab_dict["n_uc"] = n_uc
        slab_dict["n_cells"] = ncells
        slab_dict["cart_pos"] = self.to_cartesian(pos)
        return slab_dict

    def atoms_in_radius(self, radius, origin=(0, 0, 0)):
        frac_origin = self.to_fractional(origin)
        frac_radius = radius / np.array(self.unit_cell.lengths)
        hmax, kmax, lmax = np.ceil(frac_radius + frac_origin).astype(int)
        hmin, kmin, lmin = np.floor(frac_origin - frac_radius).astype(int)
        slab = self.slab(bounds=((hmin, kmin, lmin), (hmax, kmax, lmax)))
        tree = KDTree(slab["cart_pos"])
        idxs = sorted(tree.query_ball_point(origin, radius))
        result = {k: v[idxs] for k, v in slab.items() if isinstance(v, np.ndarray)}
        result["uc_atom"] = np.tile(np.arange(slab["n_uc"]), slab["n_cells"])[idxs]
        return result

    def atomic_surroundings(self, radius=6.0):
        cart_asym = self.to_cartesian(self.asymmetric_unit.positions)
        hklmax = np.array([-np.inf, -np.inf, -np.inf])
        hklmin = np.array([np.inf, np.inf, np.inf])
        frac_radius = radius / np.array(self.unit_cell.lengths)
        for pos in self.asymmetric_unit.positions:
            hklmax = np.maximum(hklmax, np.ceil(frac_radius + pos))
            hklmin = np.minimum(hklmin, np.floor(pos - frac_radius))
        hmax, kmax, lmax = hklmax.astype(int)
        hmin, kmin, lmin = hklmin.astype(int)
        slab = self.slab(bounds=((hmin, kmin, lmin), (hmax, kmax, lmax)))
        tree = KDTree(slab["cart_pos"])
        results = []
        for i, (n, pos) in enumerate(zip(self.asymmetric_unit.elements, cart_asym)):
            idxs = tree.query_ball_point(pos, radius)
            positions = slab["cart_pos"][idxs]
            elements = slab["element"][idxs]
            d = np.linalg.norm(positions - pos, axis=1)
            keep = np.where(d > 1e-3)[0]
            results.append((
                n.atomic_number, pos, elements[keep], positions[keep]
            ))
        return results 

    def molecule_surroundings(self, radius=12.0):
        results = []
        for mol in self.symmetry_unique_molecules():
            hklmax = np.array([-np.inf, -np.inf, -np.inf])
            hklmin = np.array([np.inf, np.inf, np.inf])
            frac_radius = radius / np.array(self.unit_cell.lengths)
            for pos in self.to_fractional(mol.positions):
                hklmax = np.maximum(hklmax, np.ceil(frac_radius + pos))
                hklmin = np.minimum(hklmin, np.floor(pos - frac_radius))
            hmax, kmax, lmax = hklmax.astype(int)
            hmin, kmin, lmin = hklmin.astype(int)
            slab = self.slab(bounds=((hmin, kmin, lmin), (hmax, kmax, lmax)))
            elements = slab["element"]
            positions = slab["cart_pos"]
            tree = KDTree(positions)
            keep = np.zeros(positions.shape[0], dtype=bool)
            this_mol = []
            for i, (n, pos) in enumerate(zip(mol.elements, mol.positions)):
                idxs = tree.query_ball_point(pos, radius)
                d, nn = tree.query(pos)
                keep[idxs] = True
                if d < 1e-3:
                    this_mol.append(nn)
                    keep[this_mol] = False
            results.append((
                mol, elements[keep], positions[keep]
            ))
        return results 

    def stockholder_weight_surfaces(self):
        from .density import StockholderWeight
        from .surface import stockholder_weight_isosurface
        import trimesh

        i = 0
        for n, pos, neighbour_els, neighbour_pos in self.atomic_surroundings():
            s = StockholderWeight.from_arrays([n], [pos], neighbour_els, neighbour_pos)
            iso = stockholder_weight_isosurface(s, sep=0.2)
            mesh = trimesh.Trimesh(vertices=iso.vertices, faces=iso.faces, normals=iso.normals)
            i += 1

    def molecular_shape_descriptors(self, l_max=5):
        descriptors = []
        from .sht import SHT
        from .shape_descriptors import stockholder_weight_descriptor
        sph = SHT(l_max=l_max)
        for mol, neighbour_els, neighbour_pos in self.molecule_surroundings():
            c = np.array(mol.centroid, dtype=np.float32)
            dists = np.linalg.norm(mol.positions - c, axis=1)
            bounds = np.min(dists)/2, np.max(dists) + 10.0
            descriptors.append(
                stockholder_weight_descriptor(
                    sph, mol.atomic_numbers, mol.positions, neighbour_els, neighbour_pos,
                    origin=c,
                    bounds=bounds
                )
            )
        return descriptors

    def atomic_shape_descriptors(self, l_max=5):
        descriptors = []
        from .sht import SHT
        from .shape_descriptors import stockholder_weight_descriptor
        sph = SHT(l_max=l_max)
        for n, pos, neighbour_els, neighbour_pos in self.atomic_surroundings():
            ubound = Element[n].vdw_radius * 3
            descriptors.append(
                stockholder_weight_descriptor(
                    sph, [n], [pos], neighbour_els, neighbour_pos,
                    bounds=(0.2, ubound)
                )
            )
        return descriptors

    @property
    def site_labels(self):
        "array of labels for sites in the asymmetric_unit"
        return self.asymmetric_unit.labels

    def __repr__(self):
        if "lattice_energy" in self.properties and "density" in self.properties:
            return "<Crystal {} {} ({:.3f}, {:.3f})>".format(
                self.asymmetric_unit.formula,
                self.space_group.symbol,
                self.properties["density"],
                self.properties["lattice_energy"],
            )
        return "<Crystal {} {}>".format(
            self.asymmetric_unit.formula, self.space_group.symbol
        )

    @classmethod
    def load(cls, filename):
        """Load a crystal structure from file (.res, .cif)"""
        extension_map = {
            ".cif": cls.from_cif_file,
        }
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename)

    @classmethod
    def from_cif_data(cls, cif_data):
        """Initialize a crystal structure from a dictionary
        of CIF data"""
        labels = cif_data.get("atom_site_label", None)
        symbols = cif_data.get("atom_site_type_symbol", None)
        elements = [Element[x] for x in symbols]
        x = np.asarray(cif_data.get("atom_site_fract_x", []))
        y = np.asarray(cif_data.get("atom_site_fract_y", []))
        z = np.asarray(cif_data.get("atom_site_fract_z", []))
        occupation = np.asarray(cif_data.get("atom_site_occupancy", [1] * len(x)))
        frac_pos = np.array([x, y, z]).T
        asym = AsymmetricUnit(
            elements=elements, positions=frac_pos, labels=labels, occupation=occupation
        )
        lengths = [cif_data[f"cell_length_{x}"] for x in ("a", "b", "c")]
        angles = [cif_data[f"cell_angle_{x}"] for x in ("alpha", "beta", "gamma")]
        unit_cell = UnitCell.from_lengths_and_angles(lengths, angles, unit="degrees")
        if "symmetry_equiv_pos_as_xyz" in cif_data:
            space_group = SpaceGroup.from_symmetry_operations(
                [
                    SymmetryOperation.from_string_code(x)
                    for x in cif_data["symmetry_equiv_pos_as_xyz"]
                ]
            )
        elif "symmetry_Int_Tables_number" in cif_data:
            space_group = SpaceGroup(cif_data["symmetry_Int_Tables_number"])

        return Crystal(unit_cell, space_group, asym, cif_data=cif_data)

    @classmethod
    def from_cif_file(cls, filename, data_block_name=None):
        """Initialize a crystal structure from a CIF file"""
        cif = Cif.from_file(filename)
        if data_block_name is not None:
            return cls.from_cif_data(cif.data[data_block_name])

        crystals = {
            name: cls.from_cif_data(data) for name, data in cif.data.items()
        }
        keys = list(crystals.keys())
        if len(keys) == 1:
            return crystals[keys[0]]
        return crystals

    def from_cif_string(cls, file_content, **kwargs):
        cif_data = Cif.from_string(file_content).data
        return cls.from_cif_data(cif_data, **kwargs)

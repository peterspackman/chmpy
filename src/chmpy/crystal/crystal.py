import logging
from scipy.spatial import cKDTree as KDTree
import numpy as np
from scipy.sparse import dok_matrix
import scipy.sparse.csgraph as csgraph
from pathlib import Path
from chmpy.fmt.cif import Cif
from .unit_cell import UnitCell
from .space_group import SpaceGroup, SymmetryOperation
from .asymmetric_unit import AsymmetricUnit
from chmpy.core.element import Element
from chmpy.core.molecule import Molecule
from chmpy.util.num import cartesian_product
from typing import List, Tuple, Union, Dict
from trimesh import Trimesh

LOG = logging.getLogger(__name__)


def _nearest_molecule_idx(vertices, el, pos):
    from scipy.sparse.csgraph import connected_components
    import pandas as pd
    from time import time

    t1 = time()
    m = Molecule.from_arrays(el, pos)
    m.guess_bonds()
    nfrag, labels = connected_components(m.bonds)
    tree = KDTree(pos)
    d, idxs = tree.query(vertices, k=1)
    t2 = time()
    l = labels[idxs]
    u, idxs = np.unique(l, return_inverse=True)
    return np.arange(len(u), dtype=np.uint8)[idxs]


class Crystal:
    """
    Storage class for a molecular crystal structure.

    Attributes:
        unit_cell: the translational symmetry
        space_group: the symmetry within the unit cell
        asymmetric_unit: the symmetry unique set of sites in
            the crystal. Contains information on atomic positions,
            elements, labels etc.
        properties: variable collection of named properties for
            this crystal
    """

    space_group: SpaceGroup
    unit_cell: UnitCell
    asymmetric_unit: AsymmetricUnit
    properties: dict

    def __init__(
        self,
        unit_cell: UnitCell,
        space_group: SpaceGroup,
        asymmetric_unit: AsymmetricUnit,
        **kwargs,
    ):
        """
        Construct a new crystal.


        Arguments:
            unit_cell: The unit cell for this crystal i.e. the
                translational symmetry of the crystal structure.
            space_group: The space group symmetry of this crystal
                i.e. the generators for populating the unit cell given the
                asymmetric unit.
            asymmetric_unit: The asymmetric unit of this crystal.
                 The sites of this combined with the space group will generate all
                 translationally equivalent positions.
            **kwargs: Optional properties to (will populate the properties member) store
                about the the crystal structure.
        """

        self.space_group = space_group
        self.unit_cell = unit_cell
        self.asymmetric_unit = asymmetric_unit
        self.properties = {}
        self.properties.update(kwargs)

    @property
    def sg(self) -> SpaceGroup:
        "short accessor for `space_group`"
        return self.space_group

    @property
    def uc(self) -> UnitCell:
        "short accessor for `unit_cell`"
        return self.unit_cell

    @property
    def asym(self) -> AsymmetricUnit:
        "short accessor for `asymmetric_unit`"
        return self.asymmetric_unit

    @property
    def site_positions(self) -> np.ndarray:
        "Row major array of asymmetric unit atomic positions"
        return self.asymmetric_unit.positions

    @property
    def site_atoms(self) -> np.ndarray:
        "Array of asymmetric unit atomic numbers"
        return self.asymmetric_unit.atomic_numbers

    @property
    def nsites(self) -> int:
        """The number of sites in the asymmetric unit."""
        return len(self.site_atoms)

    @property
    def symmetry_operations(self) -> List[SymmetryOperation]:
        "Symmetry operations belonging to the space group symmetry of this crystal."
        return self.space_group.symmetry_operations

    def to_cartesian(self, coords) -> np.ndarray:
        """
        Convert coordinates (row major) from fractional to cartesian coordinates.

        Arguments:
            coords (np.ndarray): (N, 3) array of positions assumed to be in fractional coordinates

        Returns:
            (N, 3) array of positions transformed to cartesian (orthogonal) coordinates
            by the unit cell of this crystal.
        """
        return self.unit_cell.to_cartesian(coords)

    def to_fractional(self, coords) -> np.ndarray:
        """
        Convert coordinates (row major) from cartesian to fractional coordinates.

        Args:
            coords (np.ndarray): (N, 3) array of positions assumed to be in cartesian (orthogonal) coordinates

        Returns:
            (N, 3) array of positions transformed to fractional coordinates
            by the unit cell of this crystal.
        """

        return self.unit_cell.to_fractional(coords)

    def unit_cell_atoms(self, tolerance=1e-2) -> dict:
        """
        Generate all atoms in the unit cell (i.e. with
        fractional coordinates in [0, 1]) along with associated
        information about symmetry operations, occupation, elements
        related asymmetric_unit atom etc.

        Will merge atom sites within tolerance of each other, and
        sum their occupation numbers. A warning will be logged if
        any atom site in the unit cell has > 1.0 occupancy after
        this.

        Sets the `_unit_cell_atom_dict` member as this is an expensive
        operation and is worth caching the result. Subsequent calls
        to this function will be a no-op.

        Arguments:
            tolerance (float, optional): Minimum separation of sites in the unit
                cell, below which atoms/sites will be merged and their (partial)
                occupations added.

        Returns:
            A dictionary of arrays associated with all sites contained
            in the unit cell of this crystal, members are:

                asym_atom: corresponding asymmetric unit atom indices for all sites.
                frac_pos: (N, 3) array of fractional positions for all sites.
                cart_pos: (N, 3) array of cartesian positions for all sites.
                element: (N) array of atomic numbers for all sites.
                symop: (N) array of indices corresponding to the generator symmetry
                operation for each site.
                label: (N) array of string labels corresponding to each site
                occupation: (N) array of occupation numbers for each site. Will
                    warn if any of these are greater than 1.0
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
        # expected_natoms = np.sum(occupation)
        for (i, j), _ in dist.items():
            if not (i < j):
                continue
            occupation[i] += occupation[j]
            mask[j] = False
        occupation = occupation[mask]
        if np.any(occupation > 1.0):
            LOG.debug("Some unit cell site occupations are > 1.0")
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
        return getattr(self, "_unit_cell_atom_dict")

    def unit_cell_connectivity(self, tolerance=0.4, neighbouring_cells=1, **kwargs) -> Tuple:
        """
        Periodic connectiviy for the unit cell, populates _uc_graph
        with a networkx.Graph object, where nodes are indices into the
        _unit_cell_atom_dict arrays and the edges contain the translation
        (cell) for the image of the corresponding unit cell atom with the
        higher index to be bonded to the lower

        Bonding is determined by interatomic distances being less than the
        sum of covalent radii for the sites plus the tolerance (provided
        as a parameter)

        Arguments:
            tolerance (float, optional):
                Bonding tolerance (bonded if d < cov_a + cov_b + tolerance)
            neighbouring_cells (int, optional):
                Number of neighbouring cells in which to look for bonded atoms.
                We start at the (0, 0, 0) cell, so a value of 1 will look in the
                (0, 0, 1), (0, 1, 1), (1, 1, 1) i.e. all 26 neighbouring cells.
                1 is typically sufficient for organic systems.

        Returns:
            A tuple of (sparse_matrix in dict of keys format, dict)
            the (i, j) value in this matrix is the bond length from i,j
            the (i, j) value in the dict is the cell translation on j which
            bonds these two sites
        """

        if hasattr(self, "_uc_graph"):
            return getattr(self, "_uc_graph")
        slab = self.slab(bounds=((-1, -1, -1), (1, 1, 1)))
        n_uc = slab["n_uc"]
        uc_pos = slab["frac_pos"][:n_uc]
        uc_nums = slab["element"][:n_uc]
        neighbour_pos = slab["frac_pos"][n_uc:]
        cart_uc_pos = self.to_cartesian(uc_pos)
        covalent_radii_dict = {x: Element.from_atomic_number(x).cov for x in np.unique(uc_nums)}
        covalent_radii_dict.update(kwargs.get("covalent_radii", {}))
        # first establish all connections in the unit cell
        covalent_radii = np.array([covalent_radii_dict[x] for x in uc_nums])
        max_cov = np.max(covalent_radii)
        # TODO this needs to be sped up for large cells, tends to slow for > 1000 atoms
        # and the space storage will become a problem
        tree = KDTree(cart_uc_pos)
        dist = tree.sparse_distance_matrix(tree, max_distance=2 * max_cov + tolerance)
        uc_edges = []

        for (i, j), d in dist.items():
            if not (i < j):
                continue
            if d > 1e-3 and d < (covalent_radii[i] + covalent_radii[j] + tolerance):
                uc_edges.append((i, j, d, (0, 0, 0)))

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
        return getattr(self, "_uc_graph")

    def unit_cell_molecules(self, bond_tolerance=0.4, **kwargs) -> List[Molecule]:
        """
        Calculate the molecules for all sites in the unit cell,
        where the number of molecules will be equal to number of
        symmetry unique molecules times number of symmetry operations.

        Args:
            bond_tolerance (float, optional): Bonding tolerance (bonded if d < cov_a + cov_b + bond_tolerance)

        Returns:
            A list of all connected molecules in this crystal, which
            when translated by the unit cell would produce the full crystal.
            If the asymmetric is molecular, the list will be of length
            num_molecules_in_asymmetric_unit * num_symm_operations
        """

        if hasattr(self, "_unit_cell_molecules"):
            return getattr(self, "_unit_cell_molecules")
        uc_graph, edge_cells = self.unit_cell_connectivity(tolerance=bond_tolerance, **kwargs)
        n_uc_mols, uc_mols = csgraph.connected_components(
            csgraph=uc_graph, directed=False, return_labels=True
        )
        uc_dict = getattr(self, "_unit_cell_atom_dict")
        uc_frac = uc_dict["frac_pos"]
        uc_elements = uc_dict["element"]
        uc_asym = uc_dict["asym_atom"]
        uc_symop = uc_dict["symop"]

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
            centroid = mol.center_of_mass
            frac_centroid = self.to_fractional(centroid)
            new_centroid = np.fmod(frac_centroid + 7.0, 1.0)
            translation = self.to_cartesian(new_centroid - frac_centroid)
            mol.translate(translation)
            molecules.append(mol)
        setattr(self, "_unit_cell_molecules", molecules)
        return molecules

    def molecular_shell(
        self, mol_idx=0, radius=3.8, method="nearest_atom"
    ) -> List[Molecule]:
        """
        Calculate the neighbouring molecules around the molecule with index
        `mol_idx`, within the given `radius` using the specified `method`.

        Arguments:
            mol_idx (int, optional): The index (into `symmetry_unique_molecules`) of the central
                molecule for the shell
            radius (float, optional): The maximum distance (Angstroms) between the central
                molecule and the neighbours.
            method (str, optional): the method to use when determining inclusion of neighbours.

        Returns:
            A list of neighbouring molecules using the given method.
        """
        mol = self.symmetry_unique_molecules()[mol_idx]
        frac_origin = self.to_fractional(mol.center_of_mass)
        frac_radius = radius / np.array(self.unit_cell.lengths)
        hmax, kmax, lmax = np.ceil(frac_radius + frac_origin).astype(int) + 1
        hmin, kmin, lmin = np.floor(frac_origin - frac_radius).astype(int) - 1
        uc_mols = self.unit_cell_molecules()
        shifts = self.to_cartesian(
            cartesian_product(
                np.arange(hmin, hmax), np.arange(kmin, kmax), np.arange(lmin, lmax)
            )
        )
        neighbours = []
        for uc_mol in uc_mols:
            for shift in shifts:
                uc_mol_t = uc_mol.translated(shift)
                dist = mol.distance_to(uc_mol_t, method=method)
                if (dist < radius) and (dist > 1e-2):
                    neighbours.append(uc_mol_t)
        return neighbours

    def molecule_dict(self, **kwargs) -> dict:
        """
        A dictionary of `symmetry_unique_molecules`, grouped by
        their chemical formulae.

        Returns:
            the dictionary of molecules with chemical formula keys
            and list of molecule values.
        """
        result = {}
        mols = self.symmetry_unique_molecules()
        for m in mols:
            f = m.molecular_formula
            if f not in result:
                result[f] = []
            result[f].append(m)
        return result

    def symmetry_unique_molecules(self, bond_tolerance=0.4, **kwargs) -> List[Molecule]:
        """
        Calculate a list of connected molecules which contain
        every site in the asymmetric_unit

        Populates the _symmetry_unique_molecules member, subsequent
        calls to this function will be a no-op.

        Args:
            bond_tolerance (float, optional): Bonding tolerance (bonded if d < cov_a + cov_b + bond_tolerance)

        Returns:
            List of all connected molecules in the asymmetric_unit of this
            crystal, i.e. the minimum list of connected molecules which contain
            all sites in the asymmetric unit.

            If the asymmetric is molecular, the list will be of length
            num_molecules_in_asymmetric_unit and the total number of atoms
            will be equal to the number of atoms in the asymmetric_unit
        """

        if hasattr(self, "_symmetry_unique_molecules"):
            return getattr(self, "_symmetry_unique_molecules")
        uc_molecules = self.unit_cell_molecules(bond_tolerance=bond_tolerance, **kwargs)
        asym_atoms = np.zeros(len(self.asymmetric_unit), dtype=bool)
        molecules = []

        # sort by % of identity symop
        def order(x):
            return len(np.where(x.asym_symops == 16484)[0]) / len(x)

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
        for i, mol in enumerate(molecules):
            mol.properties["asym_mol_idx"] = i

        ak = "asymmetric_unit_atoms"
        for mol in self.unit_cell_molecules():
            if "asym_mol_idx" in mol.properties:
                continue
            else:
                for asym_mol in molecules:
                    if np.all(mol.properties[ak] == asym_mol.properties[ak]):
                        mol.properties["asym_mol_idx"] = asym_mol.properties[
                            "asym_mol_idx"
                        ]
                        break
                else:
                    LOG.warn(
                        "No equivalent asymmetric unit molecule found!? -- this should not happen!"
                    )
        return molecules

    def slab(self, bounds=((-1, -1, -1), (1, 1, 1))) -> dict:
        """
        Calculate the atoms and associated information
        for a slab consisting of multiple unit cells.

        If unit cell atoms have not been calculated, this calculates
        their information and caches it.

        Args:
            bounds (Tuple, optional): Tuple of upper and lower corners (hkl) describing the bounds
                of the slab.

        Returns:
            A dictionary of arrays associated with all sites contained
            in the unit cell of this crystal, members are:
                asym_atom: corresponding asymmetric unit atom indices for all sites.
                frac_pos: (N, 3) array of fractional positions for all sites.
                cart_pos: (N, 3) array of cartesian positions for all sites.
                element: (N) array of atomic numbers for all sites.
                symop: (N) array of indices corresponding to the generator symmetry
                    operation for each site.
                label: (N) array of string labels corresponding to each site
                occupation: (N) array of occupation numbers for each site. Will
                    warn if any of these are greater than 1.0
                cell: (N,3) array of cell indices for each site

            n_uc: number of atoms in the unit cell

            n_cells: number of cells in this slab

            occupation: (N) array of occupation numbers for each site. Will
            warn if any of these are greater than 1.0

        """
        uc_atoms = self.unit_cell_atoms()
        (hmin, kmin, lmin), (hmax, kmax, lmax) = bounds
        h = np.arange(hmin, hmax + 1)
        k = np.arange(kmin, kmax + 1)
        l = np.arange(lmin, lmax + 1)  # noqa: E741
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

    def atoms_in_radius(self, radius, origin=(0, 0, 0)) -> dict:
        """
        Calculate all (periodic) atoms within the given `radius` of the specified
        `origin`.

        Arguments:
            radius (float): the maximum distance (Angstroms) from the origin for inclusion
            origin (Tuple, optional): the origin in fractional coordinates

        Returns:
            A dictionary mapping (see the the `slab` method),
            of those atoms within `radius` of the `origin`.
        """
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

    def atomic_surroundings(self, radius=6.0) -> List[Dict]:
        """
        Calculate all atoms within the given `radius` of
        each atomic site in the asymmetric unit.

        Arguments:
            radius (float): the maximum distance (Angstroms) from the origin for inclusion

        Returns:
            A list of atomic number, Cartesian position for both the
            atomic site in question and the surroundings (as an array)
        """
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
            asym = slab["asym_atom"][idxs]
            d = np.linalg.norm(positions - pos, axis=1)
            keep = np.where(d > 1e-3)[0]
            results.append({
                    "centre": {
                        "element": n.atomic_number,
                        "cart_pos": pos,
                        "asym_atom": i,
                    },
                    "neighbours": {
                        "element": elements[keep],
                        "cart_pos": positions[keep],
                        "distance": d[keep],
                        "asym_atom": asym[keep],
                    }
                })
        return results

    def atom_group_surroundings(self, atoms, radius=6.0) -> Tuple:
        """
        Calculate all atoms within the given `radius` of the specified
        group of atoms in the asymetric unit.

        Arguments:
            radius (float): the maximum distance (Angstroms) from the origin for inclusion

        Returns:
            A list of atomic number, Cartesian position for both the
            atomic sites in question and their surroundings (as an array)
        """
        hklmax = np.array([-np.inf, -np.inf, -np.inf])
        hklmin = np.array([np.inf, np.inf, np.inf])
        frac_radius = radius / np.array(self.unit_cell.lengths)
        mol = self.symmetry_unique_molecules()[0]
        central_positions = self.to_fractional(mol.positions[atoms])
        central_elements = mol.atomic_numbers[atoms]
        central_cart_positions = mol.positions[atoms]

        for pos in central_positions:
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
        for pos in central_cart_positions:
            idxs = tree.query_ball_point(pos, radius)
            d, nn = tree.query(pos)
            keep[idxs] = True
            if d < 1e-3:
                this_mol.append(nn)
                keep[this_mol] = False
        return (
            (central_elements, central_cart_positions),
            (elements[keep], positions[keep]),
        )

    def molecule_environment(self, mol, radius=6.0, threshold=1e-3) -> Tuple:
        """
        Calculate the atomic information for all
        atoms surrounding the given molecule in this crystal
        within the given radius. Atoms closer than `threshold`
        to any atom in the provided molecule will be excluded and
        considered part of the molecule.

        Args:
            mol (Molecule): the molecule whose environment to calculate
            radius (float, optional): Maximum distance in Angstroms between any atom in the molecule
                and the resulting neighbouring atoms
            threshold (float, optional): tolerance for detecting the neighbouring sites as part of the
                given molecule.

        Returns:
            A list of tuples of (Molecule, elements, positions)
                where `elements` is an `np.ndarray` of atomic numbers,
                and `positions` is an `np.ndarray` of Cartesian atomic positions
        """

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
        for pos in mol.positions:
            idxs = tree.query_ball_point(pos, radius)
            d, nn = tree.query(pos)
            keep[idxs] = True
            if d < threshold:
                this_mol.append(nn)
                keep[this_mol] = False
        return (mol, elements[keep], positions[keep])

    def molecule_environments(self, radius=6.0, threshold=1e-3) -> List[Tuple]:
        """
        Calculate the atomic information for all
        atoms surrounding each symmetry unique molecule
        in this crystal within the given radius.

        Args:
            radius (float, optional): Maximum distance in Angstroms between any atom in the molecule
                and the resulting neighbouring atoms
            threshold (float, optional): tolerance for detecting the neighbouring sites as part of the
                given molecule.

        Returns:
            A list of tuples of (Molecule, elements, positions)
            where `elements` is an `np.ndarray` of atomic numbers,
            and `positions` is an `np.ndarray` of Cartesian atomic positions
        """
        return [
            self.molecule_environment(x, radius=radius, threshold=threshold)
            for x in self.symmetry_unique_molecules()
        ]

    def functional_group_surroundings(self, radius=6.0, kind="carboxylic_acid") -> List:
        """
        Calculate the atomic information for all
        atoms surrounding each functional group in each symmetry unique molecule
        in this crystal within the given radius.

        Args:
            radius (float, optional): Maximum distance in Angstroms between any atom in the molecule
                and the resulting neighbouring atoms
            kind (str, optional): the functional group type

        Returns:
            A list of tuples of (func_el, func_pos, neigh_el, neigh_pos)
            where `func_el` and `neigh_el` are `np.ndarray` of atomic numbers,
            and `func_pos` and `neigh_pos` are `np.ndarray` of Cartesian atomic positions
        """
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
            groups = mol.functional_groups(kind=kind)
            for fg in groups:
                fg = list(fg)
                keep = np.zeros(positions.shape[0], dtype=bool)
                inside = []
                for pos in mol.positions[fg]:
                    idxs = tree.query_ball_point(pos, radius)
                    d, nn = tree.query(pos)
                    keep[idxs] = True
                    if d < 1e-3:
                        inside.append(nn)
                        keep[inside] = False
                results.append(
                    (
                        mol.atomic_numbers[fg],
                        mol.positions[fg],
                        elements[keep],
                        positions[keep],
                    )
                )
        return results

    def promolecule_density_isosurfaces(self, **kwargs) -> List[Trimesh]:
        """
        Calculate promolecule electron density isosurfaces
        for each symmetry unique molecule in this crystal.

        Args:
            kwargs: Keyword arguments used by `Molecule.promolecule_density_isosurface`.

                Options are:
                ```
                isovalue (float, optional): level set value for the isosurface (default=0.002) in au.
                separation (float, optional): separation between density grid used in the surface calculation
                    (default 0.2) in Angstroms.
                color (str, optional): surface property to use for vertex coloring, one of ('d_norm_i',
                    'd_i', 'd_norm_e', 'd_e')
                colormap (str, optional): matplotlib colormap to use for surface coloring (default 'viridis_r')
                midpoint (float, optional): midpoint of the segmented colormap (if applicable)
                ```

        Returns:
            A list of meshes representing the promolecule density isosurfaces
        """
        if kwargs.get("color", None) == "fragment_patch":
            color = kwargs.pop("color")
            surfaces = [
                mol.promolecule_density_isosurface(**kwargs)
                for mol in self.symmetry_unique_molecules()
            ]
            radius = kwargs.get("fragment_patch_radius", 6.0)
            from chmpy.util.color import property_to_color
            from chmpy.util.mesh import face_centroids

            for i, (mol, n_e, n_p) in enumerate(
                self.molecule_environments(radius=radius)
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
                for mol in self.symmetry_unique_molecules()
            ]
        return surfaces

    def asymmetric_unit_partial_charges(self) -> np.ndarray:
        """
        Calculate the partial charges for the asymmetric unit of this
        crystal using the EEM method.

        Returns:
            an `ndarray` of atomic partial charges.
        """
        mols = self.symmetry_unique_molecules()
        charges = np.empty(len(self.asymmetric_unit), dtype=np.float32)
        for mol in mols:
            for idx, charge in zip(
                mol.properties["asymmetric_unit_atoms"], mol.partial_charges
            ):
                charges[idx] = charge
        return charges

    def void_surface(self, *args, **kwargs) -> Trimesh:
        """
        Calculate void surface based on promolecule electron density
        for the unit cell of this crystal

        Args:
            kwargs: Keyword arguments used in the evaluation of the surface.

                Options are:
                ```
                isovalue (float, optional): level set value for the isosurface (default=0.002) in au.
                separation (float, optional): separation between density grid used in the surface calculation
                    (default 0.2) in Angstroms.
                ```

        Returns:
            the mesh representing the promolecule density void isosurface
        """

        from chmpy import PromoleculeDensity
        import trimesh
        from chmpy.mc import marching_cubes

        vertex_color = kwargs.get("color", None)

        atoms = self.slab(bounds=((-1, -1, -1), (1, 1, 1)))
        density = PromoleculeDensity((atoms["element"], atoms["cart_pos"]))
        sep = kwargs.get("separation", kwargs.get("resolution", 0.5))
        isovalue = kwargs.get("isovalue", 3e-4)
        grid_type = kwargs.get("grid_type", "uc")
        if grid_type == "uc":
            seps = sep / np.array(self.unit_cell.lengths)
            x_grid = np.arange(0, 1.0, seps[0], dtype=np.float32)
            y_grid = np.arange(0, 1.0, seps[1], dtype=np.float32)
            z_grid = np.arange(0, 1.0, seps[2], dtype=np.float32)
            x, y, z = np.meshgrid(x_grid, y_grid, z_grid)
            shape = x.shape
            pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
            pts = pts.astype(np.float32)
            pts = self.to_cartesian(pts)
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
            verts = self.to_cartesian(np.c_[verts[:, 1], verts[:, 0], verts[:, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)

        if kwargs.get("subdivide", False):
            for _ in range(int(kwargs.get("subdivide", False))):
                mesh = mesh.subdivide()

        if vertex_color == "esp":
            from chmpy.util.color import property_to_color

            asym_charges = self.asymmetric_unit_partial_charges()
            mol = Molecule.from_arrays(atoms["element"], atoms["cart_pos"])
            partial_charges = np.empty(len(mol), dtype=np.float32)
            partial_charges = asym_charges[atoms["asym_atom"]]
            mol._partial_charges = partial_charges
            prop = mol.electrostatic_potential(mesh.vertices)
            mesh.visual.vertex_colors = property_to_color(
                prop, cmap=kwargs.get("cmap", "esp")
            )
        return mesh

    def mesh_scene(self, **kwargs):
        """
        Calculate a scene of this meshes of unit cell molecules in this crystal,
        along with optional void surface.

        Args:
            kwargs: optional arguments used in the generation of this scene.

        Returns:
            trimesh.scene.Scene: trimesh scene object.
        """
        from trimesh import Scene

        meshes = {}
        for i, m in enumerate(self.unit_cell_molecules()):
            mesh = m.to_mesh(representation=kwargs.get("representation", "ball_stick"))
            n = m.molecular_formula
            for k, v in mesh.items():
                meshes[f"mol_{i}_{n}.{k}"] = v

        if kwargs.get("void", False):
            void_kwargs = kwargs.get("void_kwargs", {})
            meshes["void_surface"] = self.void_surface(**void_kwargs)
        if kwargs.get("axes", False):
            from trimesh.creation import axis

            meshes["axes"] = axis(
                transform=self.unit_cell.direct_homogeneous.T, axis_length=1.0
            )
        return Scene(meshes)

    def hirshfeld_surfaces(self, **kwargs):
        "Alias for `self.stockholder_weight_isosurfaces`"
        return self.stockholder_weight_isosurfaces(**kwargs)

    def stockholder_weight_isosurfaces(self, kind="mol", **kwargs) -> List[Trimesh]:
        """
        Calculate stockholder weight isosurfaces (i.e. Hirshfeld surfaces)
        for each symmetry unique molecule or atom in this crystal.

        Args:
            kind (str, optional): dictates whether we calculate surfaces for each unique molecule
                or for each unique atom
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
                    matplotlib colormap to use for surface coloring (default 'viridis_r')
                midpoint: float, optional, default 0.0 if using d_norm
                    use the midpoint norm (as is used in CrystalExplorer)
                ```

        Returns:
            A list of meshes representing the stockholder weight isosurfaces
        """
        from chmpy import StockholderWeight
        from chmpy.surface import stockholder_weight_isosurface
        from chmpy.util.color import property_to_color
        import trimesh

        sep = kwargs.get("separation", kwargs.get("resolution", 0.2))
        radius = kwargs.get("radius", 12.0)
        vertex_color = kwargs.get("color", "d_norm")
        isovalue = kwargs.get("isovalue", 0.5)
        meshes = []
        extra_props = {}
        isos = []
        if kind == "atom":
            for surrounds in self.atomic_surroundings(radius=radius):
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
            for i, (mol, n_e, n_p) in enumerate(
                self.molecule_environments(radius=radius)
            ):
                if vertex_color == "esp":
                    extra_props["esp"] = mol.electrostatic_potential
                elif vertex_color == "fragment_patch":
                    extra_props["fragment_patch"] = lambda x: _nearest_molecule_idx(
                        x, n_e, n_p
                    )
                s = StockholderWeight.from_arrays(
                    mol.atomic_numbers, mol.positions, n_e, n_p
                )
                iso = stockholder_weight_isosurface(
                    s, isovalue=isovalue, sep=sep, extra_props=extra_props
                )
                isos.append(iso)
        else:
            for arr in self.functional_group_surroundings(radius=radius, kind=kind):
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

    def functional_group_shape_descriptors(
        self, l_max=5, radius=6.0, kind="carboxylic_acid"
    ) -> np.ndarray:
        """
        Calculate the shape descriptors `[1,2]` for the all atoms in the functional group
        given for all symmetry unique molecules in this crystal.

        Args:
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the molecular shape function. (default: 5)
            radius (float, optional): maximum distance (Angstroms) of neighbouring atoms to include in
                stockholder weight calculation (default: 5)
            kind (str, optional): Identifier for the functional group type (default: 'carboxylic_acid')

        Returns:
            shape description vector

        References:
        ```
        [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
            https://dx.doi.org/10.1038/srep22204
        [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
            https://dx.doi.org/10.1002/anie.201906602
        ```
        """
        descriptors = []
        from chmpy.shape import SHT, stockholder_weight_descriptor

        sph = SHT(l_max)
        for (
            in_els,
            in_pos,
            neighbour_els,
            neighbour_pos,
        ) in self.functional_group_surroundings(kind=kind, radius=radius):
            masses = np.asarray([Element[x].mass for x in in_els])
            c = np.sum(in_pos * masses[:, np.newaxis] / np.sum(masses), axis=0).astype(
                np.float32
            )
            dists = np.linalg.norm(in_pos - c, axis=1)
            bounds = np.min(dists) / 2, np.max(dists) + 10.0
            descriptors.append(
                stockholder_weight_descriptor(
                    sph,
                    in_els,
                    in_pos,
                    neighbour_els,
                    neighbour_pos,
                    origin=c,
                    bounds=bounds,
                )
            )
        return np.asarray(descriptors)

    def molecule_shape_descriptors(
        self, mol, l_max=5, radius=6.0, with_property=None
    ) -> np.ndarray:
        """
        Calculate the molecular shape descriptors `[1,2]` for
        the provided molecule in the crystal.

        Args:
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the molecular shape function.
            radius (float, optional): maximum distance (Angstroms) to include surroundings
                in the shape description
            with_property (str, optional): name of the surface property to include in the shape description

        Returns:
            shape description vector

        References:
        ```
        [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
            https://dx.doi.org/10.1038/srep22204
        [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
            https://dx.doi.org/10.1002/anie.201906602
        ```
        """
        from chmpy.shape import SHT, stockholder_weight_descriptor

        sph = SHT(l_max)
        mol, neighbour_els, neighbour_pos = self.molecule_environment(
            mol, radius=radius
        )
        c = np.array(mol.centroid, dtype=np.float32)
        dists = np.linalg.norm(mol.positions - c, axis=1)
        bounds = np.min(dists) / 2, np.max(dists) + 10.0
        return stockholder_weight_descriptor(
            sph,
            mol.atomic_numbers,
            mol.positions,
            neighbour_els,
            neighbour_pos,
            origin=c,
            bounds=bounds,
            with_property=with_property,
        )

    def molecular_shape_descriptors(
        self, l_max=5, radius=6.0, with_property=None, return_coefficients=False
    ) -> np.ndarray:
        """
        Calculate the molecular shape descriptors[1,2] for all symmetry unique
        molecules in this crystal.

        Args:
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the molecular shape function.
            radius (float, optional): maximum distance (Angstroms) to include surroundings
                in the shape description
            with_property (str, optional): name of the surface property to include in the shape description
            return_coefficients (bool, optional): also return the spherical harmonic coefficients

        Returns:
            shape description vector

        References:
        ```
        [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
            https://dx.doi.org/10.1038/srep22204
        [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
            https://dx.doi.org/10.1002/anie.201906602
        ```
        """
        descriptors = []
        coeffs = []
        from chmpy.shape import SHT, stockholder_weight_descriptor

        sph = SHT(l_max)
        for mol, neighbour_els, neighbour_pos in self.molecule_environments(
            radius=radius
        ):
            c = np.array(mol.centroid, dtype=np.float32)
            dists = np.linalg.norm(mol.positions - c, axis=1)
            bounds = np.min(dists) / 2, np.max(dists) + 10.0
            descriptor = stockholder_weight_descriptor(
                sph,
                mol.atomic_numbers,
                mol.positions,
                neighbour_els,
                neighbour_pos,
                origin=c,
                bounds=bounds,
                with_property=with_property,
                coefficients=return_coefficients,
            )

            if return_coefficients:
                coeffs.append(descriptor[0])
                descriptors.append(descriptor[1])
            else:
                descriptors.append(descriptor)
        if return_coefficients:
            return np.asarray(coeffs), np.asarray(descriptors)
        else:
            return np.asarray(descriptors)

    def atomic_shape_descriptors(
        self, l_max=5, radius=6.0, return_coefficients=False, with_property=None
    ) -> np.ndarray:
        """
        Calculate the shape descriptors[1,2] for all symmetry unique
        atoms in this crystal.

        Args:
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the molecular shape function.
            radius (float, optional): maximum distance (Angstroms) to include surroundings
                in the shape description
            with_property (str, optional): name of the surface property to include in the shape description
            return_coefficients (bool, optional): also return the spherical harmonic coefficients

        Returns:
            shape description vector

        References:
        ```
        [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
            https://dx.doi.org/10.1038/srep22204
        [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
            https://dx.doi.org/10.1002/anie.201906602
        ```
        """
        descriptors = []
        coeffs = []
        from chmpy.shape import SHT, stockholder_weight_descriptor

        sph = SHT(l_max)
        for surrounds in self.atomic_surroundings(radius=radius):
            n = surrounds["centre"]["element"]
            pos = surrounds["centre"]["cart_pos"]
            neighbour_els = surrounds["neighbours"]["element"]
            neighbour_pos = surrounds["neighbours"]["cart_pos"]

            ubound = Element[n].vdw_radius * 3 + 2.0
            desc = stockholder_weight_descriptor(
                sph,
                [n],
                [pos],
                neighbour_els,
                neighbour_pos,
                bounds=(0.15, ubound),
                coefficients=return_coefficients,
                with_property=with_property
            )
            if return_coefficients:
                descriptors.append(desc[1])
                coeffs.append(desc[0])
            else:
                descriptors.append(desc)
        if return_coefficients:
            return np.asarray(coeffs), np.asarray(descriptors)
        else:
            return np.asarray(descriptors)

    def atom_group_shape_descriptors(self, atoms, l_max=5, radius=6.0) -> np.ndarray:
        """Calculate the shape descriptors[1,2] for the given atomic
        group in this crystal.

        Args:
            atoms (Tuple): atoms to include in the as the 'inside' of the shape description.
            l_max (int, optional): maximum level of angular momenta to include in the spherical harmonic
                transform of the molecular shape function.
            radius (float, optional): maximum distance (Angstroms) to include surroundings
                in the shape description

        Returns:
            shape description vector

        References:
        ```
        [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
            https://dx.doi.org/10.1038/srep22204
        [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
            https://dx.doi.org/10.1002/anie.201906602
        ```
        """
        from chmpy.shape import SHT, stockholder_weight_descriptor

        sph = SHT(l_max)
        inside, outside = self.atom_group_surroundings(atoms, radius=radius)
        m = Molecule.from_arrays(*inside)
        c = np.array(m.centroid, dtype=np.float32)
        dists = np.linalg.norm(m.positions - c, axis=1)
        bounds = np.min(dists) / 2, np.max(dists) + 10.0
        return np.asarray(
            stockholder_weight_descriptor(
                sph, *inside, *outside, origin=c, bounds=bounds
            )
        )

    def shape_descriptors(self, kind="molecular", **kwargs):
        k = kind.lower()
        if k == "molecular":
            return self.molecular_shape_descriptors(**kwargs)
        elif k == "molecule":
            return self.molecule_shape_descriptors(**kwargs)
        elif k == "atomic":
            return self.atomic_shape_descriptors(**kwargs)
        elif k == "atom group":
            return self.atom_group_shape_descriptors(**kwargs)

    @property
    def site_labels(self):
        "array of labels for sites in the `asymmetric_unit`"
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

    @property
    def density(self):
        "Calculated density of this crystal structure in g/cm^3"
        if "density" in self.properties:
            return self.properties["density"]
        uc_mass = sum(Element[x].mass for x in self.unit_cell_atoms()["element"])
        uc_vol = self.unit_cell.volume()
        return uc_mass / uc_vol / 0.6022

    @classmethod
    def _ext_load_map(cls):
        return {
            ".cif": cls.from_cif_file,
            ".res": cls.from_shelx_file,
            ".vasp": cls.from_vasp_file,
            ".pdb": cls.from_pdb_file,
        }

    def _ext_save_map(self):
        return {".cif": self.to_cif_file, ".res": self.to_shelx_file}

    @classmethod
    def _fname_load_map(cls):
        return {"POSCAR": cls.from_vasp_file, "CONTCAR": cls.from_vasp_file}

    def _fname_save_map(self):
        return {"POSCAR": self.to_poscar_file, "CONTCAR": self.to_poscar_file}

    @classmethod
    def load(cls, filename, **kwargs) -> Union["Crystal", dict]:
        """
        Load a crystal structure from file (.res, .cif)

        Args:
            filename (str): the path to the crystal structure file

        Returns:
            the resulting crystal structure or dictionary of crystal structures
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

    @classmethod
    def from_vasp_string(cls, string, **kwargs):
        "Initialize a crystal structure from a VASP POSCAR string"
        from chmpy.fmt.vasp import parse_poscar

        vasp_data = parse_poscar(string)
        uc = UnitCell(vasp_data["direct"])
        sg = SpaceGroup(1)
        coords = vasp_data["positions"]
        if not vasp_data["coord_type"].startswith("d"):
            coords = uc.to_fractional(coords)
        asym = AsymmetricUnit(vasp_data["elements"], coords)
        return Crystal(uc, sg, asym, titl=vasp_data["name"])

    @classmethod
    def from_vasp_file(cls, filename, **kwargs):
        "Initialize a crystal structure from a VASP POSCAR file"
        return cls.from_vasp_string(Path(filename).read_text(), **kwargs)

    @classmethod
    def from_cif_data(cls, cif_data, titl=None):
        """Initialize a crystal structure from a dictionary
        of CIF data"""
        labels = cif_data.get("atom_site_label", None)
        symbols = cif_data.get("atom_site_type_symbol", None)
        if symbols is None:
            if labels is None:
                raise ValueError(
                    "Unable to determine elements in CIF, "
                    "need one of _atom_site_label or "
                    "_atom_site_type_symbol present"
                )
            elements = [Element[x] for x in labels]
        else:
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

        space_group = SpaceGroup(1)
        symop_data_names = (
            "symmetry_equiv_pos_as_xyz",
            "space_group_symop_operation_xyz",
        )
        number = space_group.international_tables_number
        for k in ("space_group_IT_number", "symmetry_Int_Tables_number"):
            if k in cif_data:
                number = cif_data[k]
                break

        for symop_data_block in symop_data_names:
            if symop_data_block in cif_data:
                symops = [
                    SymmetryOperation.from_string_code(x)
                    for x in cif_data[symop_data_block]
                ]
                try:
                    new_sg = SpaceGroup.from_symmetry_operations(symops)
                    space_group = new_sg
                except ValueError:
                    space_group.symmetry_operations = symops
                    symbol = cif_data.get("symmetry_space_group_name_H-M", "Unknown")
                    space_group.international_tables_number = number
                    space_group.symbol = symbol
                    space_group.full_symbol = symbol
                    LOG.warn(
                        "Initializing non-standard spacegroup setting %s, "
                        "some SG data may be missing",
                        symbol,
                    )
                break
        else:
            # fall back to international tables number
            space_group = SpaceGroup(number)

        return Crystal(unit_cell, space_group, asym, cif_data=cif_data, titl=titl)

    @classmethod
    def from_cif_file(cls, filename, data_block_name=None):
        """Initialize a crystal structure from a CIF file"""
        cif = Cif.from_file(filename)
        if data_block_name is not None:
            return cls.from_cif_data(cif.data[data_block_name], titl=data_block_name)

        crystals = {
            name: cls.from_cif_data(data, titl=name) for name, data in cif.data.items()
        }
        keys = list(crystals.keys())
        if len(keys) == 1:
            return crystals[keys[0]]
        return crystals

    @classmethod
    def from_pdb_file(cls, filename):
        from chmpy.fmt.pdb import Pdb
        pdb = Pdb.from_file(filename)
        uc = UnitCell.from_lengths_and_angles(
                [pdb.unit_cell["a"], pdb.unit_cell["b"], pdb.unit_cell["c"]],
                [pdb.unit_cell["alpha"], pdb.unit_cell["beta"], pdb.unit_cell["gamma"]],
                unit="degrees")
        pos_cart = np.c_[pdb.atoms["x"], pdb.atoms["y"], pdb.atoms["z"]]
        pos_frac = uc.to_fractional(pos_cart)
        elements = [Element.from_string(x) for x in pdb.atoms["element"]]
        labels = pdb.atoms["name"]
        asym = AsymmetricUnit(elements, pos_frac, labels=labels)
        sg = SpaceGroup.from_symbol(pdb.space_group)
        return Crystal(uc, sg, asym)

    @classmethod
    def from_cif_string(cls, file_content, **kwargs):
        data_block_name = kwargs.get("data_block_name", None)
        cif = Cif.from_string(file_content)
        if data_block_name is not None:
            return cls.from_cif_data(cif.data[data_block_name], titl=data_block_name)

        crystals = {
            name: cls.from_cif_data(data, titl=name) for name, data in cif.data.items()
        }
        keys = list(crystals.keys())
        if len(keys) == 1:
            return crystals[keys[0]]
        return crystals

    @classmethod
    def from_shelx_file(cls, filename, **kwargs):
        """Initialize a crystal structure from a shelx .res file"""
        p = Path(filename)
        titl = p.stem
        return cls.from_shelx_string(p.read_text(), titl=titl, **kwargs)

    @classmethod
    def from_shelx_string(cls, file_content, **kwargs):
        """Initialize a crystal structure from a shelx .res string"""
        from chmpy.fmt.shelx import parse_shelx_file_content

        shelx_dict = parse_shelx_file_content(file_content)
        asymmetric_unit = AsymmetricUnit.from_records(shelx_dict["ATOM"])
        space_group = SpaceGroup.from_symmetry_operations(
            shelx_dict["SYMM"], expand_latt=shelx_dict["LATT"]
        )
        unit_cell = UnitCell.from_lengths_and_angles(
            shelx_dict["CELL"]["lengths"], shelx_dict["CELL"]["angles"], unit="degrees"
        )
        return cls(unit_cell, space_group, asymmetric_unit, **kwargs)

    @classmethod
    def from_crystal17_opt_string(cls, string, **kwargs):
        from chmpy.fmt.crystal17 import load_crystal17_geometry_string

        data = load_crystal17_geometry_string(string)
        unit_cell = UnitCell(data["direct"])
        space_group = SpaceGroup.from_symmetry_operations(data["symmetry_operations"])
        asym = AsymmetricUnit(data["elements"], unit_cell.to_fractional(data["xyz"]))
        return Crystal(unit_cell, space_group, asym)

    @classmethod
    def from_crystal17_opt_file(cls, filename, **kwargs):
        p = Path(filename)
        titl = p.stem
        return cls.from_crystal17_opt_string(p.read_text(), titl=titl, **kwargs)

    @classmethod
    def from_molecule(cls, molecule, **kwargs):
        unit_cell = UnitCell.cubic(1000)

        asym = AsymmetricUnit(
            elements=molecule.elements, positions=unit_cell.to_fractional(molecule.positions),
            labels=molecule.labels
        )
        space_group = SpaceGroup(1)
        return cls(unit_cell, space_group, asym)

    @property
    def name(self) -> str:
        "synonym for titl"
        return self.titl

    @property
    def id(self) -> str:
        "synonym for titl"
        return self.titl

    @property
    def titl(self) -> str:
        if "titl" in self.properties:
            return self.properties["titl"]
        return self.asymmetric_unit.formula

    def to_cif_data(self, data_block_name=None) -> dict:
        "Convert this crystal structure to cif data dict"
        version = "1.0a1"
        if data_block_name is None:
            data_block_name = self.titl
        if "cif_data" in self.properties:
            cif_data = self.properties["cif_data"]
            cif_data[
                "audit_creation_method"
            ] = f"chmpy python library version {version}"
            cif_data["atom_site_fract_x"] = self.asymmetric_unit.positions[:, 0]
            cif_data["atom_site_fract_y"] = self.asymmetric_unit.positions[:, 1]
            cif_data["atom_site_fract_z"] = self.asymmetric_unit.positions[:, 2]
        else:
            cif_data = {
                "audit_creation_method": f"chmpy python library version {version}",
                "symmetry_equiv_pos_site_id": list(
                    range(1, len(self.symmetry_operations) + 1)
                ),
                "symmetry_equiv_pos_as_xyz": [str(x) for x in self.symmetry_operations],
                "cell_length_a": self.unit_cell.a,
                "cell_length_b": self.unit_cell.b,
                "cell_length_c": self.unit_cell.c,
                "cell_angle_alpha": self.unit_cell.alpha_deg,
                "cell_angle_beta": self.unit_cell.beta_deg,
                "cell_angle_gamma": self.unit_cell.gamma_deg,
                "atom_site_label": self.asymmetric_unit.labels,
                "atom_site_type_symbol": [
                    x.symbol for x in self.asymmetric_unit.elements
                ],
                "atom_site_fract_x": self.asymmetric_unit.positions[:, 0],
                "atom_site_fract_y": self.asymmetric_unit.positions[:, 1],
                "atom_site_fract_z": self.asymmetric_unit.positions[:, 2],
                "atom_site_occupancy": self.asymmetric_unit.properties.get(
                    "occupation", np.ones(len(self.asymmetric_unit))
                ),
            }
        return {data_block_name: cif_data}

    def structure_factors(self, **kwargs):
        from chmpy.crystal.sfac import structure_factors

        return structure_factors(self, **kwargs)

    def unique_reflections(self, **kwargs):
        from chmpy.crystal.sfac import reflections

        return reflections(self, **kwargs)

    def powder_pattern(self, **kwargs):
        from chmpy.crystal.sfac import powder_pattern
        from chmpy.crystal.powder import PowderPattern

        tt, f2 = powder_pattern(self, **kwargs)
        if not hasattr(self, "_have_warned_powder"):
            LOG.warn(
                "Warning -- pattern calculation is a work in progress, currently values may "
                "be incorrect for many systems. USE AT YOUR OWN RISK"
            )
            self._have_warned_powder = True
        return PowderPattern(tt, f2, **kwargs)

    def to_translational_symmetry(self, supercell=(1, 1, 1)) -> "Crystal":
        """
        Create a supercell of this crystal in space group P 1.

        Args:
            supercell (Tuple[int]): size of the supercell to be created

        Returns:
            Crystal object of a supercell in space group P 1
        """
        from itertools import product

        hmax, kmax, lmax = supercell
        a, b, c = self.unit_cell.lengths
        sc = UnitCell.from_lengths_and_angles(
            (hmax * a, kmax * b, lmax * c), self.unit_cell.angles
        )

        h = np.arange(hmax)
        k = np.arange(kmax)
        l = np.arange(lmax)
        molecules = []
        for q, r, s in product(h, k, l):
            for uc_mol in self.unit_cell_molecules():
                molecules.append(
                    uc_mol.translated(np.asarray([q, r, s]) @ self.unit_cell.lattice)
                )

        asym_pos = np.vstack([x.positions for x in molecules])
        asym_nums = np.hstack([x.atomic_numbers for x in molecules])
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], sc.to_fractional(asym_pos)
        )
        new_titl = self.titl + "_P1_supercell_{}_{}_{}".format(*supercell)
        new_crystal = Crystal(sc, SpaceGroup(1), asymmetric_unit, titl=new_titl)
        return new_crystal

    def to_cif_file(self, filename, **kwargs):
        "save this crystal to a CIF formatted file"
        cif_data = self.to_cif_data(**kwargs)
        return Cif(cif_data).to_file(filename)

    def to_cif_string(self, **kwargs):
        "save this crystal to a CIF formatted string"
        cif_data = self.to_cif_data(**kwargs)
        return Cif(cif_data).to_string()

    def to_poscar_string(self, **kwargs):
        "save this crystal to a VASP POSCAR formatted string"
        from chmpy.ext.vasp import poscar_string

        return poscar_string(self, name=self.titl)

    def to_poscar_file(self, filename, **kwargs):
        "save this crystal to a VASP POSCAR formatted file"
        Path(filename).write_text(self.to_poscar_string(**kwargs))

    def to_shelx_file(self, filename):
        """Write this crystal structure as a shelx .res formatted file"""
        Path(filename).write_text(self.to_shelx_string())

    def to_shelx_string(self, titl=None):
        """Represent this crystal structure as a shelx .res formatted string"""
        from chmpy.fmt.shelx import to_res_contents

        sfac = list(np.unique(self.site_atoms))
        atom_sfac = [sfac.index(x) + 1 for x in self.site_atoms]
        shelx_data = {
            "TITL": self.titl if titl is None else titl,
            "CELL": self.unit_cell.parameters,
            "SFAC": [Element[x].symbol for x in sfac],
            "SYMM": [
                str(s)
                for s in self.space_group.reduced_symmetry_operations()
                if not s.is_identity()
            ],
            "LATT": self.space_group.latt,
            "ATOM": [
                "{:3} {:3} {: 20.12f} {: 20.12f} {: 20.12f}".format(l, s, *pos)
                for l, s, pos in zip(
                    self.asymmetric_unit.labels, atom_sfac, self.site_positions
                )
            ],
        }
        return to_res_contents(shelx_data)

    def save(self, filename, **kwargs):
        """Save this crystal structure to file (.cif, .res, POSCAR)"""
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

    def choose_trigonal_lattice(self, choice="H"):
        """
        Change the choice of lattice for this crystal to either
        rhombohedral or hexagonal cell

        Args:
            choice (str, optional): The choice of the resulting lattice, either 'H' for hexagonal
                or 'R' for rhombohedral (default 'H').
        """
        if not self.space_group.has_hexagonal_rhombohedral_choices():
            raise ValueError("Invalid space group for choose_trigonal_lattice")
        if self.space_group.choice == choice:
            return
        cart_asym_pos = self.to_cartesian(self.asymmetric_unit.positions)
        assert choice in ("H", "R"), "Valid choices are H, R"
        if self.space_group.choice == "R":
            T = np.array(((-1, 1, 0), (1, 0, -1), (1, 1, 1)))
        else:
            T = 1 / 3 * np.array(((-1, 1, 1), (2, 1, 1), (-1, -2, 1)))
        new_uc = UnitCell(np.dot(T, self.unit_cell.direct))
        self.unit_cell = new_uc
        self.asymmetric_unit.positions = self.to_fractional(cart_asym_pos)
        self.space_group = SpaceGroup(
            self.space_group.international_tables_number, choice=choice
        )

    def as_P1(self) -> "Crystal":
        """Create a copy of this crystal in space group P 1, with the new
        asymmetric_unit consisting of self.unit_cell_molecules()"""
        return self.as_P1_supercell((1, 1, 1))

    def as_P1_supercell(self, size) -> "Crystal":
        """
        Create a supercell of this crystal in space group P 1.

        Args:
            size (Tuple[int]): size of the P 1 supercell to be created

        Returns:
            Crystal object of a supercell in space group P 1
        """
        import itertools as it

        umax, vmax, wmax = size
        a, b, c = self.unit_cell.lengths
        sc = UnitCell.from_lengths_and_angles(
            (umax * a, vmax * b, wmax * c), self.unit_cell.angles
        )

        u = np.arange(umax)
        v = np.arange(vmax)
        w = np.arange(wmax)
        sc_mols = []
        for q, r, s in it.product(u, v, w):
            for uc_mol in self.unit_cell_molecules():
                sc_mols.append(
                    uc_mol.translated(np.asarray([q, r, s]) @ self.unit_cell.lattice)
                )

        asym_pos = np.vstack([x.positions for x in sc_mols])
        asym_nums = np.hstack([x.atomic_numbers for x in sc_mols])
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], sc.to_fractional(asym_pos)
        )
        new_crystal = Crystal(sc, SpaceGroup(1), asymmetric_unit)
        new_crystal.properties["titl"] = self.titl + "-P1-{}-{}-{}".format(*size)
        return new_crystal

    def cartesian_symmetry_operations(self):
        """
        Create a list of symmetry operations (rotation, translation)
        for evaluation of transformations in cartesian space.

        The rotation matrices are stored to be used as np.dot(x, R),
        (i.e. post-multiplicaiton on row-major coordinates)

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: a list of (rotation, translation)
        """
        cart_symops = []
        d = self.unit_cell.direct
        i = self.unit_cell.inverse
        for symop in self.symmetry_operations:
            cart_symops.append(
                (
                    np.dot(d.T, np.dot(symop.rotation, i.T)).T,
                    self.to_cartesian(symop.translation),
                )
            )
        return cart_symops

    def symmetry_unique_dimers(self, radius=3.8, distance_method="nearest_atom"):
        """
        Calculate the information for all molecule
        pairs surrounding the symmetry_unique_molecules
        in this crystal within the given radius.

        Args:
            radius (float, optional): Maximum distance in Angstroms between any atom in the molecule
                and the resulting neighbouring atoms

        Returns:
            A dictionary of dimers (Molecule, elements, positions)
                where `elements` is an `np.ndarray` of atomic numbers,
                and `positions` is an `np.ndarray` of Cartesian atomic positions
        """
        from chmpy.core.dimer import Dimer
        from copy import deepcopy
        from collections import defaultdict

        hklmax = np.array([-np.inf, -np.inf, -np.inf])
        hklmin = np.array([np.inf, np.inf, np.inf])
        frac_radius = radius * 2 / np.array(self.unit_cell.lengths)

        for pos in self.asymmetric_unit.positions:
            hklmax = np.maximum(hklmax, np.ceil(frac_radius + pos))
            hklmin = np.minimum(hklmin, np.floor(pos - frac_radius))
        hklmin = np.minimum(hklmin, (-1, -1, -1))
        hklmax = np.maximum(hklmax, (1, 1, 1))

        hmax, kmax, lmax = hklmax.astype(int)
        hmin, kmin, lmin = hklmin.astype(int)

        shifts_frac = cartesian_product(
            np.arange(hmin, hmax), np.arange(kmin, kmax), np.arange(lmin, lmax)
        )

        shifts = self.to_cartesian(shifts_frac)
        LOG.debug(
            "Looking in %d neighbouring cells: %s : %s",
            len(shifts),
            hklmin.astype(int),
            hklmax.astype(int),
        )
        unique_dimers = []
        mol_dimers = []
        for mol_a in self.symmetry_unique_molecules():
            dimers_a = []
            for mol_b in self.unit_cell_molecules():
                for shift, shift_frac in zip(shifts, shifts_frac):
                    # shift_frac assumes the molecule is generated from the [0, 0, 0] cell, it's not
                    mol_bt = mol_b.translated(shift)
                    r = mol_a.distance_to(mol_bt, method=distance_method)
                    if r > 1e-1 and r < radius:
                        d = Dimer(
                            mol_a,
                            mol_bt,
                            separation=r,
                            transform_ab="calculate",
                            frac_shift=shift_frac,
                        )
                        for i, dimer in enumerate(unique_dimers):
                            if dimer.separation <= d.separation + 1e-3:
                                if d == dimer:
                                    dimers_a.append((i, d))
                                    break
                        else:
                            dimers_a.append((len(unique_dimers), d))
                            unique_dimers.append(d)
            mol_dimers.append(dimers_a)
        return unique_dimers, mol_dimers

    def nearest_neighbour_info(self, points, mol_idx=0, **kwargs):
        from scipy.spatial import cKDTree as KDTree
        from collections import namedtuple

        Neighbor = namedtuple("Neighbor", "asym_id generator_symop ab_symop separation")
        unique_dimers, mol_dimers = self.symmetry_unique_dimers(**kwargs)
        npos = []
        nidx = []
        dimers = mol_dimers[mol_idx]
        neighbour_info = []
        symm_string = lambda x: str(SymmetryOperation.from_integer_code(x[0]))
        for i, (unique_idx, d) in enumerate(dimers):
            npos.append(d.b.positions)
            nidx.append(np.ones(len(d.b), dtype=np.uint8) * i)
            neighbour_info.append(
                Neighbor(
                    d.b.properties["asym_mol_idx"],
                    symm_string(d.b.properties["generator_symop"]),
                    d.symm_str,
                    d.com_separation,
                )
            )
        npos = np.vstack(npos)
        nidx = np.hstack(nidx)
        tree = KDTree(npos)
        distances, idx = tree.query(points)
        return neighbour_info, nidx[idx]

    def normalize_hydrogen_bondlengths(self, bond_tolerance=0.4, **kwargs):
        BONDLENGTHS = {
            "C": 1.083,
            "N": 1.009,
            "O": 0.983,
            "B": 1.180,
        }
        nums = self.asymmetric_unit.atomic_numbers
        pos_cart = self.to_cartesian(self.asymmetric_unit.positions)
        H_idxs = np.where(nums == 1)[0]
        conn, t = self.unit_cell_connectivity(bond_tolerance=bond_tolerance, **kwargs)
        d = 0.0
        for key in conn.keys():
            for h in H_idxs:
                if h in key:
                    at = key[1 if key.index(h) == 0 else 0]
                    d = conn[key]
                    break
            else:
                continue
            el = str(Element[nums[at]])
            if el in BONDLENGTHS:
                v_xh = pos_cart[h, :] - pos_cart[at, :]
                norm = np.linalg.norm(v_xh)
                v_xh = BONDLENGTHS[el] * v_xh / norm
                pos_cart[h, :] = pos_cart[at, :] + v_xh
        self.asymmetric_unit.positions = self.to_fractional(pos_cart)

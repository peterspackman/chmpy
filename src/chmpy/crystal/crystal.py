import logging
from typing import Union

import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy.sparse import dok_matrix
from scipy.spatial import cKDTree as KDTree
from trimesh import Trimesh

from chmpy.core.element import Element
from chmpy.core.molecule import Molecule
from chmpy.util.num import cartesian_product

from .asymmetric_unit import AsymmetricUnit
from .space_group import SpaceGroup, SymmetryOperation
from .unit_cell import UnitCell

LOG = logging.getLogger(__name__)


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
    def symmetry_operations(self) -> list[SymmetryOperation]:
        "Symmetry operations belonging to the space group symmetry of this crystal."
        return self.space_group.symmetry_operations

    def to_cartesian(self, coords) -> np.ndarray:
        """
        Convert coordinates (row major) from fractional to
        Cartesian coordinates.

        Arguments:
            coords (np.ndarray): (N, 3) array of positions assumed to be in
                fractional coordinates

        Returns:
            (N, 3) array of positions transformed to Cartesian (orthogonal)
                coordinates by the unit cell of this crystal.
        """
        return self.unit_cell.to_cartesian(coords)

    def to_fractional(self, coords) -> np.ndarray:
        """
        Convert coordinates (row major) from Cartesian to
        fractional coordinates.

        Args:
            coords (np.ndarray): (N, 3) array of positions assumed to be
                in Cartesian (orthogonal) coordinates

        Returns:
            (N, 3) array of positions transformed to fractional coordinates
            by the unit cell of this crystal.
        """

        return self.unit_cell.to_fractional(coords)

    def to_reciprocal(self, coords) -> np.ndarray:
        """
        Convert coordinates (row major) from fractional to
        reciprocal space coordinates.

        Arguments:
            coords (np.ndarray): (N, 3) array of positions assumed to be in
                fractional coordinates

        Returns:
            (N, 3) array of positions transformed to reciprocal (orthogonal)
                coordinates by the unit cell of this crystal.
        """
        return self.unit_cell.to_reciprocal(coords)

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
            return self._unit_cell_atom_dict
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
        self._unit_cell_atom_dict = {
            "asym_atom": asym[mask],
            "frac_pos": translated[mask],
            "element": uc_nums[mask],
            "symop": sym[mask],
            "label": labels[mask],
            "occupation": occupation,
            "cart_pos": self.to_cartesian(translated[mask]),
        }
        return self._unit_cell_atom_dict

    def unit_cell_connectivity(
        self, tolerance=0.4, neighbouring_cells=1, **kwargs
    ) -> tuple:
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
            return self._uc_graph
        slab = self.slab(bounds=((-1, -1, -1), (1, 1, 1)))
        n_uc = slab["n_uc"]
        uc_pos = slab["frac_pos"][:n_uc]
        uc_nums = slab["element"][:n_uc]
        neighbour_pos = slab["frac_pos"][n_uc:]
        cart_uc_pos = self.to_cartesian(uc_pos)
        covalent_radii_dict = {
            x: Element.from_atomic_number(x).cov for x in np.unique(uc_nums)
        }
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

        self._uc_graph = uc_graph, properties
        return self._uc_graph

    def unit_cell_molecules(self, bond_tolerance=0.4, **kwargs) -> list[Molecule]:
        """
        Calculate the molecules for all sites in the unit cell,
        where the number of molecules will be equal to number of
        symmetry unique molecules times number of symmetry operations.

        Args:
            bond_tolerance (float, optional): Bonding tolerance
                (bonded if d < cov_a + cov_b + bond_tolerance)

        Returns:
            A list of all connected molecules in this crystal, which
            when translated by the unit cell would produce the full crystal.
            If the asymmetric is molecular, the list will be of length
            num_molecules_in_asymmetric_unit * num_symm_operations
        """

        if hasattr(self, "_unit_cell_molecules"):
            return self._unit_cell_molecules
        uc_graph, edge_cells = self.unit_cell_connectivity(
            tolerance=bond_tolerance, **kwargs
        )
        n_uc_mols, uc_mols = csgraph.connected_components(
            csgraph=uc_graph, directed=False, return_labels=True
        )
        uc_dict = self._unit_cell_atom_dict
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
        self._unit_cell_molecules = molecules
        return molecules

    def molecular_shell(
        self, mol_idx=0, radius=3.8, method="nearest_atom"
    ) -> list[Molecule]:
        """
        Calculate the neighbouring molecules around the molecule with index
        `mol_idx`, within the given `radius` using the specified `method`.

        Arguments:
            mol_idx (int, optional): The index (into `symmetry_unique_molecules`)
                of the central molecule for the shell
            radius (float, optional): The maximum distance (Angstroms) between
                the central molecule and the neighbours.
            method (str, optional): the method to use when determining inclusion
                of neighbours.

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

    def symmetry_unique_molecules(self, bond_tolerance=0.4, **kwargs) -> list[Molecule]:
        """
        Calculate a list of connected molecules which contain
        every site in the asymmetric_unit

        Populates the _symmetry_unique_molecules member, subsequent
        calls to this function will be a no-op.

        Args:
            bond_tolerance (float, optional): Bonding tolerance
                (bonded if d < cov_a + cov_b + bond_tolerance)

        Returns:
            List of all connected molecules in the asymmetric_unit of this
            crystal, i.e. the minimum list of connected molecules which contain
            all sites in the asymmetric unit.

            If the asymmetric is molecular, the list will be of length
            num_molecules_in_asymmetric_unit and the total number of atoms
            will be equal to the number of atoms in the asymmetric_unit
        """

        if hasattr(self, "_symmetry_unique_molecules"):
            return self._symmetry_unique_molecules
        uc_molecules = self.unit_cell_molecules(bond_tolerance=bond_tolerance, **kwargs)
        asym_atoms = np.zeros(len(self.asymmetric_unit), dtype=bool)
        molecules = []

        # sort by % of identity symop
        def order(x):
            return len(np.where(x.asym_symops == 16484)[0]) / len(x)

        for mol in sorted(uc_molecules, key=order, reverse=True):
            asym_atoms_in_g = np.unique(mol.properties["asymmetric_unit_atoms"])
            if np.all(asym_atoms[asym_atoms_in_g]):
                continue
            asym_atoms[asym_atoms_in_g] = True
            molecules.append(mol)
            if np.all(asym_atoms):
                break
        LOG.debug("%d symmetry unique molecules", len(molecules))
        self._symmetry_unique_molecules = molecules
        for i, mol in enumerate(molecules):
            mol.properties["asym_mol_idx"] = i

        ak = "asymmetric_unit_atoms"
        for mol in self.unit_cell_molecules():
            if "asym_mol_idx" in mol.properties:
                continue
            else:
                for asym_mol in molecules:
                    if len(mol) != len(asym_mol):
                        continue
                    if np.all(mol.properties[ak] == asym_mol.properties[ak]):
                        mol.properties["asym_mol_idx"] = asym_mol.properties[
                            "asym_mol_idx"
                        ]
                        break
                else:
                    LOG.warn(
                        "No equivalent asymmetric unit molecule found!?"
                        "-- this should not happen!"
                    )
        return molecules

    def slab(self, bounds=((-1, -1, -1), (1, 1, 1))) -> dict:
        """
        Calculate the atoms and associated information
        for a slab consisting of multiple unit cells.

        If unit cell atoms have not been calculated, this calculates
        their information and caches it.

        Args:
            bounds (Tuple, optional): Tuple of upper and lower corners (hkl)
                describing the bounds of the slab.

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
            radius (float): the maximum distance (Angstroms) from the origin
                for inclusion
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

    def atomic_surroundings(self, radius=6.0) -> list[dict]:
        """
        Calculate all atoms within the given `radius` of
        each atomic site in the asymmetric unit.

        Arguments:
            radius (float): the maximum distance (Angstroms) from the origin
                for inclusion

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
        for i, (n, pos) in enumerate(
            zip(self.asymmetric_unit.elements, cart_asym, strict=False)
        ):
            idxs = tree.query_ball_point(pos, radius)
            positions = slab["cart_pos"][idxs]
            elements = slab["element"][idxs]
            asym = slab["asym_atom"][idxs]
            d = np.linalg.norm(positions - pos, axis=1)
            keep = np.where(d > 1e-3)[0]
            results.append(
                {
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
                    },
                }
            )
        return results

    def atom_group_surroundings(self, atoms, radius=6.0) -> tuple:
        """
        Calculate all atoms within the given `radius` of the specified
        group of atoms in the asymetric unit.

        Arguments:
            radius (float): the maximum distance (Angstroms) from the origin
                for inclusion

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

    def molecule_environment(self, mol, radius=6.0, threshold=1e-3) -> tuple:
        """
        Calculate the atomic information for all
        atoms surrounding the given molecule in this crystal
        within the given radius. Atoms closer than `threshold`
        to any atom in the provided molecule will be excluded and
        considered part of the molecule.

        Args:
            mol (Molecule): the molecule whose environment to calculate
            radius (float, optional): Maximum distance in Angstroms between
                any atom in the molecule and the resulting neighbouring atoms
            threshold (float, optional): tolerance for detecting the neighbouring
                sites as part of the given molecule.

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

    def molecule_environments(self, radius=6.0, threshold=1e-3) -> list[tuple]:
        """
        Calculate the atomic information for all
        atoms surrounding each symmetry unique molecule
        in this crystal within the given radius.

        Args:
            radius (float, optional): Maximum distance in Angstroms between
                any atom in the molecule and the resulting neighbouring atoms
            threshold (float, optional): tolerance for detecting the neighbouring
                sites as part of the given molecule.

        Returns:
            A list of tuples of (Molecule, elements, positions)
            where `elements` is an `np.ndarray` of atomic numbers,
            and `positions` is an `np.ndarray` of Cartesian atomic positions
        """
        return [
            self.molecule_environment(x, radius=radius, threshold=threshold)
            for x in self.symmetry_unique_molecules()
        ]

    def functional_group_surroundings(self, radius=6.0, kind="carboxylic_acid") -> list:
        """
        Calculate the atomic information for all
        atoms surrounding each functional group in each symmetry unique molecule
        in this crystal within the given radius.

        Args:
            radius (float, optional): Maximum distance in Angstroms between
                any atom in the molecule and the resulting neighbouring atoms
            kind (str, optional): the functional group type

        Returns:
            A list of tuples of (func_el, func_pos, neigh_el, neigh_pos)
            where `func_el` and `neigh_el` are `np.ndarray` of atomic numbers,
            and `func_pos` and `neigh_pos` are `np.ndarray` of
            Cartesian atomic positions
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

    def promolecule_density_isosurfaces(self, **kwargs) -> list[Trimesh]:
        from .surface import promolecule_density_isosurfaces
        return promolecule_density_isosurfaces(self, **kwargs)

    def unit_cell_coordination_numbers(self) -> np.ndarray:
        """
        Calculate the coordination numbers for the unit cell atoms of this
        crystal using the EEQ method with periodic boundary conditions.

        Returns:
            an `ndarray` of coordination numbers for the unit cell atoms.
        """
        if hasattr(self, "_unit_cell_coordination_numbers"):
            return self._unit_cell_coordination_numbers

        from chmpy.crystal.eeq_pbc import calculate_coordination_numbers_crystal

        # Calculate coordination numbers with PBC
        cn = calculate_coordination_numbers_crystal(self)
        self._unit_cell_coordination_numbers = cn.astype(np.float32)

        return self._unit_cell_coordination_numbers

    def asymmetric_unit_coordination_numbers(self) -> np.ndarray:
        """
        Calculate the coordination numbers for the asymmetric unit of this
        crystal using the EEQ method with periodic boundary conditions.

        Returns:
            an `ndarray` of coordination numbers for the asymmetric unit atoms.
        """
        if hasattr(self, "_asymmetric_unit_coordination_numbers"):
            return self._asymmetric_unit_coordination_numbers

        # Get unit cell coordination numbers
        uc_cn = self.unit_cell_coordination_numbers()
        uc_atoms = self.unit_cell_atoms()

        # Map back to asymmetric unit
        asym_cn = np.empty(len(self.asymmetric_unit), dtype=np.float32)
        for i, cn in enumerate(uc_cn):
            asym_idx = uc_atoms["asym_atom"][i]
            asym_cn[asym_idx] = cn

        self._asymmetric_unit_coordination_numbers = asym_cn
        return asym_cn

    def unit_cell_partial_charges(self, method="eeq") -> np.ndarray:
        """
        Calculate the partial charges for the unit cell atoms of this
        crystal using the specified method with periodic boundary conditions.

        Args:
            method (str): Charge method to use ('eeq' or 'eem')

        Returns:
            an `ndarray` of partial charges for the unit cell atoms.
        """
        if hasattr(self, "_unit_cell_partial_charges"):
            return self._unit_cell_partial_charges

        method = method.lower()

        if method == "eeq":
            from chmpy.crystal.eeq_pbc import calculate_eeq_charges_pbc

            # Get unit cell atoms
            uc_atoms = self.unit_cell_atoms()
            positions = uc_atoms["cart_pos"]
            atomic_numbers = uc_atoms["element"]

            # Get cell vectors
            cell_vectors = self.unit_cell.lattice

            # Calculate net charge (usually 0 for crystals)
            charge = 0.0

            # Calculate charges with PBC
            charges = calculate_eeq_charges_pbc(
                atomic_numbers, positions, cell_vectors, charge
            )
        else:
            # Use molecular approach for EEM (less accurate for crystals)
            mols = self.unit_cell_molecules()
            charges = np.empty(len(self.unit_cell_atoms()["element"]), dtype=np.float32)

            # Set charge method for molecules
            for mol in mols:
                mol.properties["charge_method"] = method

            # Get charges from molecules
            for mol in mols:
                uc_indices = mol.properties.get("unit_cell_atoms", [])
                for i, charge in enumerate(mol.partial_charges):
                    if i < len(uc_indices):
                        charges[uc_indices[i]] = charge

        self._unit_cell_partial_charges = charges.astype(np.float32)
        return charges

    def asymmetric_unit_partial_charges(self, method="eeq") -> np.ndarray:
        """
        Calculate the partial charges for the asymmetric unit of this
        crystal using the specified method.

        Args:
            method (str): Charge method to use ('eeq' or 'eem')

        Returns:
            an `ndarray` of atomic partial charges.
        """
        if method.lower() == "eeq":
            # Get unit cell charges using EEQ with PBC
            uc_charges = self.unit_cell_partial_charges(method="eeq")
            uc_atoms = self.unit_cell_atoms()

            # Map back to asymmetric unit
            charges = np.empty(len(self.asymmetric_unit), dtype=np.float32)
            for i, charge in enumerate(uc_charges):
                asym_idx = uc_atoms["asym_atom"][i]
                charges[asym_idx] = charge

            return charges
        else:
            # Use the molecular approach for other methods
            mols = self.symmetry_unique_molecules()
            charges = np.empty(len(self.asymmetric_unit), dtype=np.float32)

            # Set charge method for molecules
            for mol in mols:
                mol.properties["charge_method"] = method

            for mol in mols:
                for idx, charge in zip(
                    mol.properties["asymmetric_unit_atoms"],
                    mol.partial_charges,
                    strict=False,
                ):
                    charges[idx] = charge

            return charges

    def void_surface(self, *args, **kwargs) -> Trimesh:
        from .surface import void_surface
        return void_surface(self, *args, **kwargs)

    def mesh_scene(self, **kwargs):
        from .surface import mesh_scene
        return mesh_scene(self, **kwargs)

    def hirshfeld_surfaces(self, **kwargs):
        "Alias for `self.stockholder_weight_isosurfaces`"
        return self.stockholder_weight_isosurfaces(**kwargs)

    def stockholder_weight_isosurfaces(self, kind="mol", **kwargs) -> list[Trimesh]:
        from .surface import stockholder_weight_isosurfaces
        return stockholder_weight_isosurfaces(self, kind=kind, **kwargs)

    def functional_group_shape_descriptors(self, l_max=5, radius=6.0, kind="carboxylic_acid") -> np.ndarray:
        from .shape_descriptors import functional_group_shape_descriptors
        return functional_group_shape_descriptors(self, l_max=l_max, radius=radius, kind=kind)

    def molecule_shape_descriptors(self, mol, l_max=5, radius=6.0, with_property=None) -> np.ndarray:
        from .shape_descriptors import molecule_shape_descriptors
        return molecule_shape_descriptors(self, mol, l_max=l_max, radius=radius, with_property=with_property)

    def molecular_shape_descriptors(self, l_max=5, radius=6.0, with_property=None, return_coefficients=False) -> np.ndarray:
        from .shape_descriptors import molecular_shape_descriptors
        return molecular_shape_descriptors(self, l_max=l_max, radius=radius, with_property=with_property, return_coefficients=return_coefficients)

    def atomic_shape_descriptors(self, l_max=5, radius=6.0, return_coefficients=False, with_property=None) -> np.ndarray:
        from .shape_descriptors import atomic_shape_descriptors
        return atomic_shape_descriptors(self, l_max=l_max, radius=radius, return_coefficients=return_coefficients, with_property=with_property)

    def atom_group_shape_descriptors(self, atoms, l_max=5, radius=6.0) -> np.ndarray:
        from .shape_descriptors import atom_group_shape_descriptors
        return atom_group_shape_descriptors(self, atoms, l_max=l_max, radius=radius)

    def shape_descriptors(self, kind="molecular", **kwargs):
        from .shape_descriptors import shape_descriptors
        return shape_descriptors(self, kind=kind, **kwargs)

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
        return f"<Crystal {self.asymmetric_unit.formula} {self.space_group.symbol}>"

    @property
    def density(self):
        "Calculated density of this crystal structure in g/cm^3"
        if "density" in self.properties:
            return self.properties["density"]
        uc_mass = sum(Element[x].mass for x in self.unit_cell_atoms()["element"])
        uc_vol = self.unit_cell.volume()
        return uc_mass / uc_vol / 0.6022

    @classmethod
    def load(cls, filename, **kwargs) -> Union["Crystal", dict]:
        from .io import load
        return load(filename, **kwargs)

    @classmethod
    def from_vasp_string(cls, string, **kwargs):
        from .io import from_vasp_string
        return from_vasp_string(string, **kwargs)

    @classmethod
    def from_vasp_file(cls, filename, **kwargs):
        from .io import from_vasp_file
        return from_vasp_file(filename, **kwargs)

    @classmethod
    def from_aims_string(cls, string, **kwargs):
        from .io import from_aims_string
        return from_aims_string(string, **kwargs)

    @classmethod
    def from_aims_file(cls, filename, **kwargs):
        from .io import from_aims_file
        return from_aims_file(filename, **kwargs)

    @classmethod
    def from_ase_atoms(cls, atoms, **kwargs):
        from .io import from_ase_atoms
        return from_ase_atoms(atoms, **kwargs)

    @classmethod
    def from_cif_data(cls, cif_data, titl=None):
        from .io import from_cif_data
        return from_cif_data(cif_data, titl=titl)

    @classmethod
    def _parse_hermann_mauguin_symbol(cls, hm_symbol, sg_number):
        from .io import _parse_hermann_mauguin_symbol
        return _parse_hermann_mauguin_symbol(hm_symbol, sg_number)

    @classmethod
    def from_cif_file(cls, filename, data_block_name=None):
        from .io import from_cif_file
        return from_cif_file(filename, data_block_name=data_block_name)

    @classmethod
    def from_pdb_file(cls, filename):
        from .io import from_pdb_file
        return from_pdb_file(filename)

    @classmethod
    def from_cif_string(cls, file_content, **kwargs):
        from .io import from_cif_string
        return from_cif_string(file_content, **kwargs)

    @classmethod
    def from_shelx_file(cls, filename, **kwargs):
        from .io import from_shelx_file
        return from_shelx_file(filename, **kwargs)

    @classmethod
    def from_shelx_string(cls, file_content, **kwargs):
        from .io import from_shelx_string
        return from_shelx_string(file_content, **kwargs)

    @classmethod
    def from_crystal17_opt_string(cls, string, **kwargs):
        from .io import from_crystal17_opt_string
        return from_crystal17_opt_string(string, **kwargs)

    @classmethod
    def from_crystal17_opt_file(cls, filename, **kwargs):
        from .io import from_crystal17_opt_file
        return from_crystal17_opt_file(filename, **kwargs)

    @classmethod
    def from_molecule(cls, molecule, **kwargs):
        from .io import from_molecule
        return from_molecule(molecule, **kwargs)

    @classmethod
    def from_gen_string(cls, contents, **kwargs):
        from .io import from_gen_string
        return from_gen_string(contents, **kwargs)

    @classmethod
    def from_gen_file(cls, filename, **kwargs):
        from .io import from_gen_file
        return from_gen_file(filename, **kwargs)

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

    def to_ase_atoms(self, **kwargs):
        from .io import to_ase_atoms
        return to_ase_atoms(self, **kwargs)

    def to_cif_data(self, data_block_name=None) -> dict:
        from .io import to_cif_data
        return to_cif_data(self, data_block_name=data_block_name)

    def structure_factors(self, **kwargs):
        from chmpy.crystal.sfac import structure_factors

        return structure_factors(self, **kwargs)

    def unique_reflections(self, **kwargs):
        from chmpy.crystal.sfac import reflections

        return reflections(self, **kwargs)

    def powder_pattern(self, **kwargs):
        from chmpy.crystal.powder import PowderPattern
        from chmpy.crystal.sfac import powder_pattern

        tt, f2 = powder_pattern(self, **kwargs)
        if not hasattr(self, "_have_warned_powder"):
            LOG.warn(
                "Warning -- pattern calculation is a work in progress, currently"
                "values may be incorrect for many systems. USE AT YOUR OWN RISK"
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
        from .io import to_cif_file
        return to_cif_file(self, filename, **kwargs)

    def to_cif_string(self, **kwargs):
        from .io import to_cif_string
        return to_cif_string(self, **kwargs)

    def to_poscar_string(self, **kwargs):
        from .io import to_poscar_string
        return to_poscar_string(self, **kwargs)

    def to_poscar_file(self, filename, **kwargs):
        from .io import to_poscar_file
        return to_poscar_file(self, filename, **kwargs)

    def to_shelx_file(self, filename):
        from .io import to_shelx_file
        return to_shelx_file(self, filename)

    def to_shelx_string(self, titl=None):
        from .io import to_shelx_string
        return to_shelx_string(self, titl=titl)

    def to_pdb_string(self, header=None):
        from .io import to_pdb_string
        return to_pdb_string(self, header=header)

    def to_pdb_file(self, filename, header=None):
        from .io import to_pdb_file
        return to_pdb_file(self, filename, header=header)

    def save(self, filename, **kwargs):
        from .io import save
        return save(self, filename, **kwargs)

    def enumerate_subgroups(self, max_index: int = 8) -> list:
        """Enumerate translationengleiche (t-) subgroups of this crystal's space group.

        Returns subgroups that preserve the same lattice but have fewer
        point operations. Each subgroup increases Z' by a factor equal
        to the subgroup index.

        Args:
            max_index: Maximum index [G:H] to consider (default 8)

        Returns:
            List of SubgroupResult, sorted by index then size
        """
        from .subgroup import SubgroupEnumerator

        enumerator = SubgroupEnumerator.from_space_group(self.space_group)
        return enumerator.enumerate_all(max_index=max_index)

    def _has_connected_asymmetric_unit(self) -> bool:
        """Check if each symmetry-unique molecule is connected.

        Returns True if the number of symmetry-unique molecules equals
        the expected Z' (= Z / |G|). When False, the asymmetric unit
        contains atoms from multiple molecules — typically because a
        molecule sits on a retained symmetry element (e.g. inversion).
        """
        n_uc_mols = len(self.unit_cell_molecules())
        n_symops = len(self.symmetry_operations)
        expected_z_prime = n_uc_mols / n_symops
        n_unique = len(self.symmetry_unique_molecules())
        return n_unique <= expected_z_prime

    def to_subgroup(
        self,
        target_z_prime: float | None = None,
        subgroup_index: int | None = None,
        subgroup_result=None,
        tolerance: float = 1e-4,
        reconnect: bool = True,
    ) -> "Crystal":
        """Create a new Crystal with reduced symmetry using a subgroup.

        The asymmetric unit is expanded to account for the reduced symmetry,
        while the unit cell remains the same (t-subgroup).

        Exactly one of target_z_prime, subgroup_index, or subgroup_result
        must be provided.

        Args:
            target_z_prime: Desired Z' value. The method finds a subgroup
                that achieves this Z'.
            subgroup_index: Index [G:H] of the desired subgroup. If multiple
                subgroups exist with this index, the first identified one
                is used.
            subgroup_result: A specific SubgroupResult to apply.
            tolerance: Tolerance for position comparison during expansion.
            reconnect: If True (default), prefer subgroups where the
                asymmetric unit forms connected molecules. When a molecule
                sits on a symmetry element (e.g. inversion center), this
                selects a subgroup that drops that element.

        Returns:
            New Crystal with the subgroup as space group and expanded
            asymmetric unit.

        Raises:
            ValueError: If no valid subgroup is found or arguments are invalid.
        """
        from .subgroup import (
            SubgroupEnumerator,
            expand_asymmetric_unit,
        )

        n_args = sum(x is not None for x in [target_z_prime, subgroup_index, subgroup_result])
        if n_args != 1:
            raise ValueError(
                "Exactly one of target_z_prime, subgroup_index, or "
                "subgroup_result must be provided"
            )

        enumerator = SubgroupEnumerator.from_space_group(self.space_group)

        def _build_crystal(result):
            new_asym = expand_asymmetric_unit(
                self.asymmetric_unit,
                self.symmetry_operations,
                result.symop_indices,
                enumerator.sg_table,
                tolerance=tolerance,
            )
            sub_symops = [self.symmetry_operations[i] for i in result.symop_indices]
            sg_number = result.space_group_number or 1
            try:
                new_sg = SpaceGroup.from_symmetry_operations(sub_symops)
            except ValueError:
                new_sg = SpaceGroup(sg_number)
                new_sg.symmetry_operations = sub_symops
            new_crystal = Crystal(self.unit_cell, new_sg, new_asym)
            new_crystal.properties = {
                "parent_space_group": self.space_group.symbol,
                "subgroup_index": result.index,
            }
            if result.space_group_symbol is not None:
                new_crystal.properties["subgroup_space_group"] = result.space_group_symbol
            if result.point_group_symbol is not None:
                new_crystal.properties["subgroup_point_group"] = result.point_group_symbol
            return new_crystal

        if subgroup_result is not None:
            return _build_crystal(subgroup_result)

        if target_z_prime is not None:
            # Z' = Z / |G|, where Z = number of molecules in the unit cell
            # and |G| = number of symmetry operations.
            n_uc_mols = len(self.unit_cell_molecules())
            n_symops = len(self.symmetry_operations)
            current_z_prime = n_uc_mols / n_symops
            candidates = enumerator.find_for_target_z_prime(
                current_z_prime, target_z_prime
            )
            if not candidates:
                raise ValueError(
                    f"No subgroup found that achieves Z' = {target_z_prime} "
                    f"(current Z' = {current_z_prime})"
                )
        else:
            candidates = enumerator.find_by_index(subgroup_index)
            if not candidates:
                raise ValueError(
                    f"No subgroup found with index {subgroup_index}"
                )

        # Sort candidates: prefer identified space groups
        candidates.sort(key=lambda c: (c.space_group_number is None, c.index))

        if not reconnect:
            return _build_crystal(candidates[0])

        # When reconnect=True, try each candidate and pick the first
        # one that produces a connected asymmetric unit (each symmetry-
        # unique molecule is a single connected fragment).  This avoids
        # subgroups whose retained symmetry (e.g. inversion) splits a
        # molecule that sits on that symmetry element.
        best = None
        for candidate in candidates:
            crystal = _build_crystal(candidate)
            if crystal._has_connected_asymmetric_unit():
                return crystal
            if best is None:
                best = crystal

        LOG.info(
            "No subgroup produces a fully connected asymmetric unit; "
            "the molecule likely sits on a retained symmetry element"
        )
        return best

    def to_standard_setting(self) -> "Crystal":
        """Transform this crystal to the standard ITA setting.

        After to_subgroup(), the symmetry operations may be in a
        non-standard setting (e.g. shifted origin). This method
        identifies the standard setting, computes the required origin
        shift and/or basis transformation, and returns a new Crystal
        with standard-setting symmetry operations.

        Returns:
            New Crystal in the standard ITA setting.

        Raises:
            ValueError: If the symmetry operations cannot be matched
                to any known standard setting.
        """
        from .subgroup import identify_standard_setting

        result = identify_standard_setting(self.symmetry_operations)
        if result is None:
            raise ValueError(
                "Could not identify a standard ITA setting for the "
                "current symmetry operations"
            )

        # No transform needed — already standard
        if result.basis_transform is None and result.origin_shift is None:
            new_sg = SpaceGroup(result.sg_number, choice=result.choice)
            new_asym = AsymmetricUnit(
                list(self.asymmetric_unit.elements),
                self.asymmetric_unit.positions.copy(),
                labels=list(self.asymmetric_unit.labels),
            )
            return Crystal(self.unit_cell, new_sg, new_asym)

        # Origin shift only (no basis transform) — most common case
        if result.basis_transform is None:
            positions = self.asymmetric_unit.positions - result.origin_shift
            positions = positions % 1.0
            new_sg = SpaceGroup(result.sg_number, choice=result.choice)
            new_asym = AsymmetricUnit(
                list(self.asymmetric_unit.elements),
                positions,
                labels=list(self.asymmetric_unit.labels),
            )
            new_crystal = Crystal(self.unit_cell, new_sg, new_asym)
            new_crystal.properties = dict(self.properties)
            return new_crystal

        # Basis transform present — need to rebuild the asymmetric unit
        return self._to_standard_setting_with_basis_transform(result)

    def detect_symmetry(self, tolerance: float = 0.01) -> "Crystal":
        """Detect the full symmetry of this crystal from atomic positions.

        Returns a new Crystal with the detected (possibly higher) symmetry.
        Uses native detection -- no spglib dependency.

        Args:
            tolerance: Cartesian distance tolerance in Angstroms.

        Returns:
            Crystal with detected space group, or self if no higher symmetry found.
        """
        from .subgroup import identify_standard_setting
        from .symmetry_finder import (
            find_asymmetric_unit_indices,
            find_symmetry_operations,
        )

        uc_dict = self.unit_cell_atoms()
        positions = uc_dict["frac_pos"]
        elements = uc_dict["element"]

        symops = find_symmetry_operations(
            self.unit_cell, positions, elements, atol=tolerance
        )

        if len(symops) <= len(self.symmetry_operations):
            return self

        result = identify_standard_setting(symops)
        if result is None:
            LOG.warning(
                "Found %d symmetry operations but could not identify space group",
                len(symops),
            )
            return self

        sg = SpaceGroup(result.sg_number, choice=result.choice)

        # Build asymmetric unit from equivalence classes
        asym_indices = find_asymmetric_unit_indices(
            positions, elements, symops, self.unit_cell.direct, atol=tolerance
        )

        asym = AsymmetricUnit(
            [Element.from_atomic_number(int(elements[i])) for i in asym_indices],
            positions[asym_indices],
            labels=uc_dict["label"][asym_indices],
        )
        return Crystal(self.unit_cell, sg, asym)

    def _to_standard_setting_with_basis_transform(self, result) -> "Crystal":
        """Handle to_standard_setting when a basis transform is needed.

        When the cell changes, the asymmetric unit must be reconstructed
        from the full unit cell content, because:
        1. The new cell may have different volume (det(P) != 1)
        2. Atoms on special positions in the old cell may be on general
           positions in the new cell, and vice versa

        The approach:
        1. Build the new cell from P @ old_direct
        2. Correctly transform symops using row-vector convention
        3. Identify standard setting for those symops (origin shift only)
        4. Tile old UC atoms into the new cell
        5. Find the asymmetric unit under the standard symops
        """
        from itertools import product as iproduct

        from scipy.spatial import cKDTree as KDTree

        from chmpy.core.element import Element

        from .subgroup import (
            _deduplicate_asymmetric_unit,
            identify_standard_setting,
        )
        from .symmetry_operation import SymmetryOperation

        P = result.basis_transform
        P_inv = np.linalg.inv(P)
        det_P_raw = abs(np.linalg.det(P))
        det_P = max(1, round(det_P_raw))  # For tiling range: at least 1

        # Build new unit cell
        new_direct = P @ self.unit_cell.direct
        new_uc = UnitCell(new_direct)

        # Correctly transform symops using row-vector convention:
        # R_new = P @ R_old @ P_inv (same for both conventions)
        # t_new = t_old @ P_inv (row-vector convention)
        correct_symops = []
        for s in self.symmetry_operations:
            R_new = P @ s.rotation @ P_inv
            t_new = s.translation @ P_inv
            correct_symops.append(
                SymmetryOperation(np.round(R_new).astype(float), t_new)
            )

        # Deduplicate transformed symops: centering translations become
        # lattice vectors under the primitive transform, producing
        # duplicate ops that differ only by integer translations.
        unique_symops = []
        seen_codes = set()
        for s in correct_symops:
            t_wrapped = s.translation % 1.0
            s_wrapped = SymmetryOperation(s.rotation, t_wrapped)
            code = s_wrapped.integer_code
            if code not in seen_codes:
                seen_codes.add(code)
                unique_symops.append(s_wrapped)
        correct_symops = unique_symops

        # Identify standard setting for the correctly-transformed symops
        # (should only need origin shift, no further basis transform)
        result2 = identify_standard_setting(correct_symops)
        if result2 is None:
            raise ValueError(
                "Could not identify standard setting after basis transform"
            )

        std_sg = SpaceGroup(result2.sg_number, choice=result2.choice)

        # Generate all unit cell atoms from the current crystal
        uc_dict = self.unit_cell_atoms()
        old_frac = uc_dict["frac_pos"]
        old_elems = uc_dict["element"]

        # Tile old UC atoms into the new (possibly larger/smaller) cell
        all_frac = []
        all_elems = []
        for shift in iproduct(range(-1, det_P + 1), repeat=3):
            shifted = old_frac + np.array(shift, dtype=float)
            cart = self.unit_cell.to_cartesian(shifted)
            frac_new = new_uc.to_fractional(cart)
            for i in range(len(frac_new)):
                f = frac_new[i]
                if np.all(f >= -1e-6) and np.all(f < 1.0 - 1e-6):
                    all_frac.append(f % 1.0)
                    all_elems.append(int(old_elems[i]))

        all_frac = np.array(all_frac)
        all_elems = np.array(all_elems)

        # Remove positional duplicates
        tree = KDTree(all_frac)
        dist = tree.sparse_distance_matrix(tree, max_distance=0.01)
        mask = np.ones(len(all_frac), dtype=bool)
        for (i, j), _ in dist.items():
            if i < j and all_elems[i] == all_elems[j]:
                mask[j] = False
        uc_frac = all_frac[mask]
        uc_elems = all_elems[mask]

        # Apply origin shift if needed
        if result2.origin_shift is not None:
            uc_frac = (uc_frac - result2.origin_shift) % 1.0

        # Find asymmetric unit under standard symops
        elem_list = [Element.from_atomic_number(e) for e in uc_elems]
        full_asym = AsymmetricUnit(elem_list, uc_frac)
        unique_idx, _ = _deduplicate_asymmetric_unit(
            full_asym, std_sg.symmetry_operations
        )

        final_elems = [elem_list[i] for i in unique_idx]
        final_pos = uc_frac[unique_idx]
        new_asym = AsymmetricUnit(final_elems, final_pos)

        new_crystal = Crystal(new_uc, std_sg, new_asym)
        new_crystal.properties = dict(self.properties)
        return new_crystal

    def choose_trigonal_lattice(self, choice="H"):
        """
        Change the choice of lattice for this crystal to either
        rhombohedral or hexagonal cell

        Args:
            choice (str, optional): The choice of the resulting lattice,
                either 'H' for hexagonal or 'R' for rhombohedral (default 'H').
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
            radius (float, optional): Maximum distance in Angstroms between any
                atom in the molecule and the resulting neighbouring atoms

        Returns:
            A dictionary of dimers (Molecule, elements, positions)
                where `elements` is an `np.ndarray` of atomic numbers,
                and `positions` is an `np.ndarray` of Cartesian atomic positions
        """
        from chmpy.core.dimer import Dimer

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
                for shift, shift_frac in zip(shifts, shifts_frac, strict=False):
                    # shift_frac assumes the molecule is generated from
                    # the [0, 0, 0] cell, it's not
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
        from collections import namedtuple

        from scipy.spatial import cKDTree as KDTree

        Neighbor = namedtuple("Neighbor", "asym_id generator_symop ab_symop separation")
        unique_dimers, mol_dimers = self.symmetry_unique_dimers(**kwargs)
        npos = []
        nidx = []
        dimers = mol_dimers[mol_idx]
        neighbour_info = []

        def symm_string(x):
            return str(SymmetryOperation.from_integer_code(x[0]))

        for i, (_unique_idx, d) in enumerate(dimers):
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
        for key in conn.keys():
            for h in H_idxs:
                if h in key:
                    at = key[1 if key.index(h) == 0 else 0]
                    conn[key]
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

    def assign_atom_types(self, force_field="UFF", **kwargs):
        from .force_field import assign_atom_types
        return assign_atom_types(self, force_field=force_field, **kwargs)

    def get_atom_types(self, force_field="UFF", use_cached=True, **kwargs):
        from .force_field import get_atom_types
        return get_atom_types(self, force_field=force_field, use_cached=use_cached, **kwargs)

    def get_ff_parameters(self, force_field="UFF", use_cached=True, **kwargs):
        from .force_field import get_ff_parameters
        return get_ff_parameters(self, force_field=force_field, use_cached=use_cached, **kwargs)

    def get_unique_atom_types(self, force_field="UFF", use_cached=True, **kwargs):
        from .force_field import get_unique_atom_types
        return get_unique_atom_types(self, force_field=force_field, use_cached=use_cached, **kwargs)

    def get_lj_parameters_array(self, force_field="UFF", use_cached=True, **kwargs):
        from .force_field import get_lj_parameters_array
        return get_lj_parameters_array(self, force_field=force_field, use_cached=use_cached, **kwargs)

    def export_lammps_data(self, filename, force_field="UFF", **kwargs):
        from .force_field import export_lammps_data
        return export_lammps_data(self, filename, force_field=force_field, **kwargs)

    def export_raspa_files(self, force_field="UFF", output_dir=".", **kwargs):
        from .force_field import export_raspa_files
        return export_raspa_files(self, force_field=force_field, output_dir=output_dir, **kwargs)

    def atom_typing_summary(self, force_field="UFF", **kwargs):
        from .force_field import atom_typing_summary
        return atom_typing_summary(self, force_field=force_field, **kwargs)

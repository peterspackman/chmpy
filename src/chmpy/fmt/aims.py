"""FHI-aims format readers and writers.

This module handles FHI-aims geometry.in, control.in, and aims.out files
for crystals and molecules.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chmpy.core import Molecule
from chmpy.core.element import Element

LOG = logging.getLogger(__name__)


def crystal_to_geometry_string(crystal, use_cartesian=False):
    """Convert a Crystal object to an FHI-aims geometry.in string.

    Args:
        crystal: Crystal object to convert
        use_cartesian (bool, optional): If True, write atom positions in
            Cartesian coordinates. If False (default), use fractional coordinates.

    Returns:
        str: FHI-aims geometry.in format string

    Examples:
        >>> from chmpy import Crystal
        >>> crystal = Crystal.load("structure.cif")
        >>> geom_str = crystal_to_geometry_string(crystal)
    """
    lines = []

    # Write lattice vectors
    lattice = crystal.unit_cell.lattice
    for i in range(3):
        vec = lattice[i]
        lines.append(f"lattice_vector {vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")

    lines.append("")  # Blank line after lattice vectors

    # Get unit cell atoms
    uc_atoms = crystal.unit_cell_atoms()
    elements = [Element[x] for x in uc_atoms["element"]]

    if use_cartesian:
        # Convert fractional to Cartesian
        frac_pos = uc_atoms["frac_pos"]
        cart_pos = crystal.unit_cell.to_cartesian(frac_pos)
        for pos, el in zip(cart_pos, elements, strict=True):
            lines.append(f"atom {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {el.symbol}")
    else:
        # Use fractional coordinates
        frac_pos = uc_atoms["frac_pos"]
        for pos, el in zip(frac_pos, elements, strict=True):
            lines.append(f"atom_frac {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {el.symbol}")

    lines.append("")  # Trailing blank line
    return "\n".join(lines)


def molecule_to_geometry_string(molecule):
    """Convert a Molecule object to an FHI-aims geometry.in string.

    Args:
        molecule: Molecule object to convert

    Returns:
        str: FHI-aims geometry.in format string (no lattice vectors for molecules)

    Examples:
        >>> from chmpy import Molecule
        >>> mol = Molecule.load("water.xyz")
        >>> geom_str = molecule_to_geometry_string(mol)
    """
    lines = []

    # For molecules, just write atom positions in Cartesian coordinates
    for pos, el in zip(molecule.positions, molecule.elements, strict=True):
        lines.append(f"atom {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {el.symbol}")

    lines.append("")  # Trailing blank line
    return "\n".join(lines)


def to_geometry_string(obj, **kwargs):
    """Convert a Crystal or Molecule object to FHI-aims geometry.in format.

    Args:
        obj: Crystal or Molecule object
        **kwargs: Additional keyword arguments passed to the appropriate conversion function

    Returns:
        str: FHI-aims geometry.in format string
    """
    if isinstance(obj, Molecule):
        return molecule_to_geometry_string(obj)
    else:
        return crystal_to_geometry_string(obj, **kwargs)


def parse_geometry_string(contents):
    """Parse an FHI-aims geometry.in file.

    Args:
        contents (str): Contents of the geometry.in file

    Returns:
        dict: Dictionary containing:
            - lattice: (3, 3) array of lattice vectors (if periodic)
            - elements: List of Element objects
            - positions: (N, 3) array of atomic positions
            - fractional: bool indicating if positions are fractional

    Examples:
        >>> from pathlib import Path
        >>> geom_str = Path("geometry.in").read_text()
        >>> data = parse_geometry_string(geom_str)
    """
    lines = contents.splitlines()

    lattice_vectors = []
    elements = []
    positions = []
    fractional = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if not tokens:
            continue

        keyword = tokens[0]

        if keyword == "lattice_vector":
            # Parse lattice vector: lattice_vector x y z
            if len(tokens) < 4:
                LOG.warning("Incomplete lattice_vector line: %s", line)
                continue
            vec = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            lattice_vectors.append(vec)

        elif keyword == "atom":
            # Parse Cartesian atom: atom x y z element
            if len(tokens) < 5:
                LOG.warning("Incomplete atom line: %s", line)
                continue
            pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            el = Element.from_string(tokens[4])
            positions.append(pos)
            elements.append(el)

        elif keyword == "atom_frac":
            # Parse fractional atom: atom_frac x y z element
            if len(tokens) < 5:
                LOG.warning("Incomplete atom_frac line: %s", line)
                continue
            pos = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            el = Element.from_string(tokens[4])
            positions.append(pos)
            elements.append(el)
            fractional = True

    result = {
        "elements": elements,
        "positions": np.array(positions),
        "fractional": fractional,
    }

    if lattice_vectors:
        result["lattice"] = np.array(lattice_vectors)

    LOG.debug("Parsed %d atoms, %d lattice vectors", len(elements), len(lattice_vectors))

    return result


def geometry_string_to_molecule(contents):
    """Convert an FHI-aims geometry.in string to a Molecule object.

    Args:
        contents (str): Contents of the geometry.in file

    Returns:
        Molecule: A new Molecule object

    Raises:
        ValueError: If the geometry contains lattice vectors (use geometry_string_to_crystal instead)

    Examples:
        >>> from pathlib import Path
        >>> geom_str = Path("geometry.in").read_text()
        >>> mol = geometry_string_to_molecule(geom_str)
    """
    data = parse_geometry_string(contents)

    if "lattice" in data:
        raise ValueError(
            "Geometry contains lattice vectors, indicating a periodic system. "
            "Use geometry_string_to_crystal() instead."
        )

    return Molecule(data["elements"], data["positions"])


def geometry_file_to_molecule(filename):
    """Load a Molecule from an FHI-aims geometry.in file.

    Args:
        filename (str or Path): Path to the geometry.in file

    Returns:
        Molecule: A new Molecule object
    """
    return geometry_string_to_molecule(Path(filename).read_text())


def geometry_string_to_crystal(contents):
    """Convert an FHI-aims geometry.in string to a Crystal object.

    Args:
        contents (str): Contents of the geometry.in file

    Returns:
        Crystal: A new Crystal object

    Raises:
        ValueError: If the geometry does not contain lattice vectors

    Examples:
        >>> from pathlib import Path
        >>> geom_str = Path("geometry.in").read_text()
        >>> crystal = geometry_string_to_crystal(geom_str)
    """
    # Import here to avoid circular imports
    from chmpy.crystal import Crystal

    return Crystal.from_aims_string(contents)


def geometry_file_to_crystal(filename):
    """Load a Crystal from an FHI-aims geometry.in file.

    Args:
        filename (str or Path): Path to the geometry.in file

    Returns:
        Crystal: A new Crystal object
    """
    return geometry_string_to_crystal(Path(filename).read_text())


def generate_control_in(
    crystal_or_molecule,
    species_defaults_dir,
    basis="light",
    xc="pbe",
    k_grid=None,
    relax_geometry=False,
    relax_unit_cell=False,
    output_options=None,
    extra_keywords=None,
):
    """Generate an FHI-aims control.in file.

    Args:
        crystal_or_molecule: Crystal or Molecule object
        species_defaults_dir (str or Path): Path to species_defaults directory
            (e.g., "~/src/FHIaims-250822/species_defaults")
        basis (str, optional): Basis set tier (default: "light")
            Options: "light", "lightdense", "intermediate", "tight", "really_tight", etc.
        xc (str, optional): Exchange-correlation functional (default: "pbe")
            Common options: "pbe", "pbe0", "b86bpbe-25", "hse06", etc.
        k_grid (tuple or list, optional): k-point grid for periodic systems (e.g., (4, 4, 4))
            If None, will use defaults based on system size
        relax_geometry (bool or dict, optional): Enable geometry relaxation
            If True, uses default settings. If dict, can specify method and convergence
        relax_unit_cell (bool or str, optional): Enable unit cell relaxation
            If True, uses "full". Can also be "shape" or specific constraints
        output_options (list, optional): List of output keywords (e.g., ["hirshfeld", "mulliken"])
        extra_keywords (dict, optional): Additional control.in keywords as key-value pairs

    Returns:
        str: FHI-aims control.in format string

    Examples:
        >>> from chmpy import Crystal
        >>> crystal = Crystal.load("structure.cif")
        >>> control_str = generate_control_in(
        ...     crystal,
        ...     species_defaults_dir="~/src/FHIaims-250822/species_defaults",
        ...     basis="light",
        ...     xc="pbe",
        ...     k_grid=(4, 4, 4),
        ...     relax_geometry=True,
        ...     relax_unit_cell="full"
        ... )
    """
    species_dir = Path(species_defaults_dir).expanduser()

    # Determine default version (try defaults_2020 first)
    defaults_version = "defaults_2020"
    if not (species_dir / defaults_version / basis).exists():
        # Fall back to other versions
        for version in ["defaults_2010", "defaults_next"]:
            if (species_dir / version / basis).exists():
                defaults_version = version
                break
        else:
            raise ValueError(
                f"Could not find basis '{basis}' in species_defaults directory: {species_dir}"
            )

    basis_dir = species_dir / defaults_version / basis

    # Get unique elements in the structure
    if isinstance(crystal_or_molecule, Molecule):
        elements = set(crystal_or_molecule.elements)
        is_periodic = False
    else:
        uc_atoms = crystal_or_molecule.unit_cell_atoms()
        elements = {Element[x] for x in uc_atoms["element"]}
        is_periodic = True

    # Start building control.in
    lines = []
    lines.append("#" + "=" * 79)
    lines.append("# FHI-aims control.in file")
    lines.append("# Generated by chmpy")
    lines.append(f"# Basis set: {basis} ({defaults_version})")
    lines.append("#" + "=" * 79)

    # XC functional
    lines.append(f"xc                                 {xc}")

    # Output options
    if output_options:
        for opt in output_options:
            lines.append(f"output                             {opt}")

    # Geometry relaxation
    if relax_geometry:
        if isinstance(relax_geometry, dict):
            method = relax_geometry.get("method", "trm")
            conv = relax_geometry.get("convergence", 0.005)
            lines.append(f"relax_geometry {method}                 {conv}")
        else:
            lines.append("relax_geometry trm                 0.005")

    # Unit cell relaxation
    if relax_unit_cell:
        if isinstance(relax_unit_cell, str):
            lines.append(f"relax_unit_cell                    {relax_unit_cell}")
        else:
            lines.append("relax_unit_cell                    full")

    # k-point grid for periodic systems
    if is_periodic:
        if k_grid is not None:
            if len(k_grid) == 3:
                lines.append(f"k_grid                             {k_grid[0]} {k_grid[1]} {k_grid[2]}")
            else:
                raise ValueError("k_grid must be a tuple/list of 3 integers")
        else:
            # Simple default: use (2, 2, 2) for periodic systems
            lines.append("k_grid                             2 2 2")

    # Extra keywords
    if extra_keywords:
        for key, value in extra_keywords.items():
            if value is True:
                lines.append(f"{key}")
            elif value is not False and value is not None:
                lines.append(f"{key:30s} {value}")

    lines.append("#" + "=" * 79)
    lines.append("")

    # Add species defaults
    for element in sorted(elements):
        species_file = basis_dir / f"{element.atomic_number:02d}_{element.symbol}_default"
        if not species_file.exists():
            LOG.warning(f"Species file not found: {species_file}")
            LOG.warning(f"Skipping species {element.symbol}")
            continue

        # Read and append species definition
        species_content = species_file.read_text()
        lines.append(species_content)

    return "\n".join(lines)


@dataclass
class OptimizationStep:
    """Data from a single geometry optimization step.

    Attributes:
        step_number: The optimization step number (0 = initial)
        energy: Total energy in eV
        max_force: Maximum force component in eV/Angstrom
        forces_on_atoms: Norm of forces on atoms in eV/Angstrom
        forces_on_lattice: Norm of forces on lattice vectors in eV/Angstrom^3 (if periodic)
        lattice: (3, 3) array of lattice vectors in Angstrom (if periodic)
        elements: List of Element objects
        positions: (N, 3) array of atomic positions in Angstrom (Cartesian)
    """

    step_number: int
    energy: float
    max_force: float | None = None
    forces_on_atoms: float | None = None
    forces_on_lattice: float | None = None
    lattice: np.ndarray | None = None
    elements: list | None = None
    positions: np.ndarray | None = None

    def to_molecule(self):
        """Convert this step to a Molecule object."""
        if self.elements is None or self.positions is None:
            raise ValueError("No atomic structure available for this step")
        return Molecule(self.elements, self.positions)

    def to_crystal(self):
        """Convert this step to a Crystal object."""
        if self.lattice is None:
            raise ValueError("No lattice vectors available - this is not a periodic system")
        if self.elements is None or self.positions is None:
            raise ValueError("No atomic structure available for this step")

        from chmpy.crystal import Crystal
        from chmpy.crystal.asymmetric_unit import AsymmetricUnit
        from chmpy.crystal.space_group import SpaceGroup
        from chmpy.crystal.unit_cell import UnitCell

        uc = UnitCell(self.lattice)
        sg = SpaceGroup(1)
        frac_coords = uc.to_fractional(self.positions)
        asym = AsymmetricUnit(self.elements, frac_coords)
        return Crystal(uc, sg, asym)


@dataclass
class AimsOutput:
    """Parser for FHI-aims output files (aims.out).

    This class parses FHI-aims output files and extracts key information
    including energies, forces, and optimization trajectories.

    Attributes:
        filename: Path to the aims.out file
        converged: Whether the calculation converged
        is_optimization: Whether this was a geometry optimization
        is_periodic: Whether this is a periodic calculation
        final_energy: Final total energy in eV
        n_atoms: Number of atoms
        n_steps: Number of optimization steps (0 for single-point)
        steps: List of OptimizationStep objects for optimization runs
        final_structure: The final atomic structure (OptimizationStep)

    Examples:
        >>> output = AimsOutput.from_file("aims.out")
        >>> print(f"Final energy: {output.final_energy:.6f} eV")
        >>> if output.is_optimization:
        ...     print(f"Converged in {output.n_steps} steps")
        ...     for step in output.steps:
        ...         print(f"  Step {step.step_number}: E = {step.energy:.6f} eV")
        >>> final_mol = output.final_structure.to_molecule()
    """

    filename: str | None = None
    converged: bool = False
    is_optimization: bool = False
    is_periodic: bool = False
    final_energy: float | None = None
    n_atoms: int = 0
    n_steps: int = 0
    steps: list = field(default_factory=list)
    final_structure: OptimizationStep | None = None

    @classmethod
    def from_file(cls, filename):
        """Parse an FHI-aims output file.

        Args:
            filename (str or Path): Path to the aims.out file

        Returns:
            AimsOutput: Parsed output data
        """
        return cls.from_string(Path(filename).read_text(), filename=str(filename))

    @classmethod
    def from_string(cls, contents, filename=None):
        """Parse FHI-aims output from a string.

        Args:
            contents (str): Contents of the aims.out file
            filename (str, optional): Original filename for reference

        Returns:
            AimsOutput: Parsed output data
        """
        output = cls(filename=filename)
        output._parse(contents)
        return output

    def _parse(self, contents):
        """Parse the aims.out file contents."""
        lines = contents.splitlines()

        # Patterns for extracting data
        energy_pattern = re.compile(
            r"\|\s*Total energy uncorrected\s*:\s*(-?\d+\.\d+E[+-]\d+)\s*eV"
        )
        converged_pattern = re.compile(r"Present geometry is converged")
        opt_step_pattern = re.compile(r"Geometry optimization: Attempting to predict")
        n_atoms_pattern = re.compile(r"\|\s*Number of atoms\s*:\s*(\d+)")
        max_force_pattern = re.compile(
            r"Maximum force component is\s+(-?\d+\.\d+E[+-]\d+)\s*eV/A"
        )
        forces_atoms_pattern = re.compile(
            r"\|\| Forces on atoms\s+\|\|\s*=\s*(-?\d+\.\d+E[+-]\d+)\s*eV/A"
        )
        forces_lattice_pattern = re.compile(
            r"\|\| Forces on lattice\s+\|\|\s*=\s*(-?\d+\.\d+E[+-]\d+)\s*eV/A"
        )
        lattice_vector_pattern = re.compile(
            r"lattice_vector\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
        )
        atom_pattern = re.compile(
            r"atom\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\w+)"
        )

        # Track state
        energies = []
        current_step = 0
        in_structure_block = False
        structure_type = None  # "updated" or "final"
        current_lattice = []
        current_elements = []
        current_positions = []
        current_max_force = None
        current_forces_atoms = None
        current_forces_lattice = None
        pending_energy = None

        for line in lines:
            # Check for number of atoms
            match = n_atoms_pattern.search(line)
            if match:
                self.n_atoms = int(match.group(1))

            # Check for lattice vectors (indicates periodic system)
            if "lattice_vector" in line and "k_grid" not in line:
                self.is_periodic = True

            # Check for energy
            match = energy_pattern.search(line)
            if match:
                energy = float(match.group(1))
                energies.append(energy)
                pending_energy = energy

            # Check for optimization step
            if opt_step_pattern.search(line):
                self.is_optimization = True

            # Check for convergence
            if converged_pattern.search(line):
                self.converged = True

            # Check for max force
            match = max_force_pattern.search(line)
            if match:
                current_max_force = float(match.group(1))

            # Check for forces on atoms
            match = forces_atoms_pattern.search(line)
            if match:
                current_forces_atoms = float(match.group(1))

            # Check for forces on lattice
            match = forces_lattice_pattern.search(line)
            if match:
                current_forces_lattice = float(match.group(1))

            # Check for structure blocks
            if "Updated atomic structure:" in line:
                in_structure_block = True
                structure_type = "updated"
                current_lattice = []
                current_elements = []
                current_positions = []
                continue

            if "Final atomic structure:" in line:
                in_structure_block = True
                structure_type = "final"
                current_lattice = []
                current_elements = []
                current_positions = []
                continue

            # Parse structure block
            if in_structure_block:
                # Check for lattice vector
                match = lattice_vector_pattern.search(line)
                if match:
                    vec = [float(match.group(j)) for j in range(1, 4)]
                    current_lattice.append(vec)
                    continue

                # Check for atom
                match = atom_pattern.search(line)
                if match:
                    pos = [float(match.group(j)) for j in range(1, 4)]
                    el = Element.from_string(match.group(4))
                    current_positions.append(pos)
                    current_elements.append(el)
                    continue

                # Check if we've finished the structure block
                # Structure block ends with a line of dashes or empty after atoms
                if line.strip().startswith("---") and current_elements:
                    in_structure_block = False
                    step = OptimizationStep(
                        step_number=current_step,
                        energy=pending_energy if pending_energy else (energies[-1] if energies else None),
                        max_force=current_max_force,
                        forces_on_atoms=current_forces_atoms,
                        forces_on_lattice=current_forces_lattice,
                        lattice=np.array(current_lattice) if current_lattice else None,
                        elements=current_elements,
                        positions=np.array(current_positions),
                    )

                    if structure_type == "updated":
                        self.steps.append(step)
                        current_step += 1
                    elif structure_type == "final":
                        self.final_structure = step

                    # Reset force tracking for next step
                    current_max_force = None
                    current_forces_atoms = None
                    current_forces_lattice = None

        # Set final energy
        if energies:
            self.final_energy = energies[-1]

        # Set n_steps
        self.n_steps = len(self.steps)

        # If no steps were parsed but we have a final structure, this is a single-point
        if not self.steps and self.final_structure:
            self.n_steps = 0

    def get_trajectory(self):
        """Get the optimization trajectory as a list of Molecule or Crystal objects.

        Returns:
            list: List of Molecule or Crystal objects for each optimization step
        """
        trajectory = []
        for step in self.steps:
            if self.is_periodic:
                trajectory.append(step.to_crystal())
            else:
                trajectory.append(step.to_molecule())

        # Add final structure if available
        if self.final_structure:
            if self.is_periodic:
                trajectory.append(self.final_structure.to_crystal())
            else:
                trajectory.append(self.final_structure.to_molecule())

        return trajectory

    def get_energies(self):
        """Get all energies from the optimization trajectory.

        Returns:
            np.ndarray: Array of energies in eV for each step
        """
        energies = [step.energy for step in self.steps if step.energy is not None]
        if self.final_structure and self.final_structure.energy is not None:
            energies.append(self.final_structure.energy)
        return np.array(energies)

    def get_max_forces(self):
        """Get maximum force components from the optimization trajectory.

        Returns:
            np.ndarray: Array of max force components in eV/Angstrom
        """
        forces = [step.max_force for step in self.steps if step.max_force is not None]
        if self.final_structure and self.final_structure.max_force is not None:
            forces.append(self.final_structure.max_force)
        return np.array(forces)

    def to_xyz_trajectory(self, filename):
        """Write the optimization trajectory to an XYZ file.

        Args:
            filename (str or Path): Output XYZ file path
        """
        trajectory = self.get_trajectory()
        if not trajectory:
            raise ValueError("No trajectory data available")

        lines = []
        for i, struct in enumerate(trajectory):
            if self.is_periodic:
                # For crystals, get positions from the structure
                mol = struct.asymmetric_unit.to_molecule(struct.unit_cell)
            else:
                mol = struct

            n_atoms = len(mol.elements)
            lines.append(str(n_atoms))

            # Get energy if available
            if i < len(self.steps):
                energy = self.steps[i].energy
            elif self.final_structure:
                energy = self.final_structure.energy
            else:
                energy = None

            comment = f"Step {i}"
            if energy is not None:
                comment += f" E={energy:.8f} eV"
            lines.append(comment)

            for el, pos in zip(mol.elements, mol.positions, strict=True):
                lines.append(f"{el.symbol:2s} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f}")

        Path(filename).write_text("\n".join(lines) + "\n")

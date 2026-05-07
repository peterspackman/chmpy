"""Complete example for FHI-aims input generation.

This example shows how to:
1. Load or create a crystal structure
2. Generate both geometry.in and control.in files
3. Set up a complete FHI-aims calculation
"""

from pathlib import Path

from chmpy import Crystal
from chmpy.fmt.aims import crystal_to_geometry_string, generate_control_in


def setup_aims_calculation(
    crystal,
    output_dir,
    species_defaults_dir,
    basis="light",
    xc="pbe",
    k_grid=None,
    relax_geometry=False,
    relax_unit_cell=False,
    extra_keywords=None,
):
    """Set up a complete FHI-aims calculation.

    Args:
        crystal: Crystal object
        output_dir: Directory to write files
        species_defaults_dir: Path to FHI-aims species_defaults
        basis: Basis set (default: "light")
        xc: XC functional (default: "pbe")
        k_grid: k-point grid tuple (default: auto)
        relax_geometry: Enable geometry relaxation (default: False)
        relax_unit_cell: Enable cell relaxation (default: False)
        extra_keywords: Optional dict of additional control.in keywords.

    Returns:
        tuple: (geometry_path, control_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    geom_str = crystal_to_geometry_string(crystal, use_cartesian=False)
    geom_path = output_dir / "geometry.in"
    geom_path.write_text(geom_str)

    control_str = generate_control_in(
        crystal,
        species_defaults_dir=species_defaults_dir,
        basis=basis,
        xc=xc,
        k_grid=k_grid,
        relax_geometry=relax_geometry,
        relax_unit_cell=relax_unit_cell,
        output_options=["hirshfeld"],
        extra_keywords=extra_keywords,
    )
    control_path = output_dir / "control.in"
    control_path.write_text(control_str)

    return geom_path, control_path


def main():
    # Configuration
    species_defaults = "~/src/FHIaims-250822/species_defaults"

    # Example 1: Single-point calculation
    print("=" * 80)
    print("Example 1: Single-point energy calculation")
    print("=" * 80)

    crystal = Crystal.load("src/chmpy/tests/test_files/acetic_acid.cif")
    geom_path, control_path = setup_aims_calculation(
        crystal,
        output_dir="aims_sp",
        species_defaults_dir=species_defaults,
        basis="light",
        xc="pbe",
        k_grid=(4, 4, 4),
    )
    print(f"Created: {geom_path}")
    print(f"Created: {control_path}")
    print()

    # Example 2: Geometry optimization
    print("=" * 80)
    print("Example 2: Geometry optimization (fixed cell)")
    print("=" * 80)

    geom_path, control_path = setup_aims_calculation(
        crystal,
        output_dir="aims_relax",
        species_defaults_dir=species_defaults,
        basis="light",
        xc="pbe",
        k_grid=(4, 4, 4),
        relax_geometry=True,
    )
    print(f"Created: {geom_path}")
    print(f"Created: {control_path}")
    print()

    # Example 3: Full optimization (geometry + cell) with dispersion correction
    print("=" * 80)
    print("Example 3: Full optimization with dispersion correction")
    print("=" * 80)

    output_dir = Path("aims_opt_full")
    geom_path, control_path = setup_aims_calculation(
        crystal,
        output_dir=output_dir,
        species_defaults_dir=species_defaults,
        basis="lightdense",
        xc="b86bpbe-25",
        k_grid=(4, 6, 4),
        relax_geometry={"method": "trm", "convergence": 0.005},
        relax_unit_cell="full",
        extra_keywords={
            "xdm": "0.69125110 1.57470830",
            "output_level": "MD_light",
        },
    )
    print(f"Created complete calculation in: {output_dir}")
    print(f"  - {geom_path}")
    print(f"  - {control_path}")
    print()
    print("To run:")
    print(f"  cd {output_dir}")
    print("  aims > aims.out")

    print()
    print("=" * 80)
    print("Setup complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

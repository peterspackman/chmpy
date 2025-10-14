"""Example script demonstrating filtered fingerprint calculations.

This example shows how to:
1. Load a crystal structure
2. Generate Hirshfeld surfaces
3. Compute filtered fingerprints for specific element pairs
4. Create publication-quality plots
"""

import matplotlib.pyplot as plt

from chmpy import Crystal
from chmpy.crystal.fingerprint import (
    filtered_histogram_by_elements,
    fingerprint_histogram,
    plot_filtered_histogram,
    plot_fingerprint_histogram,
)


def main():
    # Load the crystal structure (adjust path as needed)
    crystal = Crystal.load("./src/chmpy/tests/test_files/acetic_acid.cif")
    print(f"Loaded crystal: {crystal}")

    # Generate Hirshfeld surface for the first molecule
    # Using lower resolution for faster computation
    surfaces = crystal.hirshfeld_surfaces(separation=0.2, radius=3.8, kind="mol")
    mesh = surfaces[0]
    print(f"Generated surface with {len(mesh.vertices)} vertices")

    # Compute the full fingerprint histogram
    full_hist = fingerprint_histogram(mesh, bins=200)

    # Example 1: Filter for C...H contacts (including inverse H...C)
    ch_hist = filtered_histogram_by_elements(mesh, "C", "H", bins=200, include_inverse=True)

    # Example 2: Filter for O...H contacts (including inverse H...O)
    oh_hist = filtered_histogram_by_elements(mesh, "O", "H", bins=200, include_inverse=True)

    # Example 3: Filter for H...H contacts (symmetric, so inverse doesn't matter)
    hh_hist = filtered_histogram_by_elements(mesh, "H", "H", bins=200)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot the full fingerprint
    plot_fingerprint_histogram(full_hist, ax=axes[0, 0])
    axes[0, 0].set_title("Full Fingerprint", fontsize=12, fontweight="bold")

    # Plot filtered fingerprints
    plot_filtered_histogram(ch_hist, full_hist, ax=axes[0, 1])
    axes[0, 1].set_title("C...H Contacts (both directions)", fontsize=12, fontweight="bold")

    plot_filtered_histogram(oh_hist, full_hist, ax=axes[1, 0])
    axes[1, 0].set_title("O...H Contacts (both directions)", fontsize=12, fontweight="bold")

    plot_filtered_histogram(hh_hist, full_hist, ax=axes[1, 1])
    axes[1, 1].set_title("H...H Contacts", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("filtered_fingerprints.png", dpi=300, bbox_inches="tight")
    print("Saved figure to filtered_fingerprints.png")
    plt.show()


if __name__ == "__main__":
    main()

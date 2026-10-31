"""Regressions for shape and surface bugs that gave plausible wrong numbers."""

import numpy as np
import pytest

from chmpy.shape.shape_descriptors import make_N_invariants
from chmpy.shape.sht import SHT


def _band_limited(theta, phi):
    return (
        1
        + 0.3 * np.cos(theta)
        + 0.5 * np.sin(theta) * np.cos(phi)
        + 0.2 * np.sin(theta) ** 2 * np.sin(2 * phi)
        - 0.4 * np.sin(theta) * np.cos(theta) * np.sin(phi)
    )


def test_evaluate_at_points_matches_the_function_off_the_grid():
    sht = SHT(6)
    theta, phi = sht.grid
    real = sht.analysis(_band_limited(theta, phi))
    complex_values = _band_limited(theta, phi) + 1j * np.sin(theta) * np.sin(phi)
    full = sht.analysis(complex_values.astype(complex))
    rng = np.random.default_rng(3)
    for t, p in zip(
        rng.uniform(0, np.pi, 20), rng.uniform(0, 2 * np.pi, 20), strict=True
    ):
        expected = _band_limited(t, p)
        assert sht.evaluate_at_points(real, t, p) == pytest.approx(expected)
        assert sht.evaluate_at_points(full, t, p) == pytest.approx(
            expected + 1j * np.sin(t) * np.sin(p)
        )


def test_n_invariants_take_one_l_at_a_time():
    coefficients = np.zeros(9, dtype=complex)
    coefficients[1:4] = (0.3, 0.4j, 1.2)  # l = 1 only
    invariants = make_N_invariants(coefficients)
    expected_l1 = np.sqrt(0.3**2 + 0.4**2 + 1.2**2)
    np.testing.assert_allclose(invariants, [0.0, expected_l1, 0.0])


def test_reconstructed_surface_is_consistently_wound():
    pytest.importorskip("trimesh")
    from chmpy.shape.reconstruct import reconstructed_surface

    sht = SHT(4)
    theta, phi = sht.grid
    mesh = reconstructed_surface(sht.analysis(1.0 + 0.0 * theta))
    assert mesh.is_winding_consistent
    assert mesh.volume > 0


def test_promolecule_isosurface_lies_on_the_isovalue():
    pytest.importorskip("trimesh")
    from chmpy.interpolate.density import PromoleculeDensity
    from chmpy.surface import promolecule_density_isosurface

    # deliberately spread differently along x, y and z
    positions = np.array(
        [[0, 0, 0], [1.5, 0, 0], [1.5, 1.2, 0.3], [-0.5, -0.4, 2.0]]
    ) + np.array([3.0, 7.0, -4.0])
    density = PromoleculeDensity((np.array([6, 6, 8, 1]), positions))
    mesh = promolecule_density_isosurface(density, isovalue=0.002, sep=0.2)
    rho = density.rho(np.asarray(mesh.vertices, dtype=np.float32))
    np.testing.assert_allclose(np.median(rho), 0.002, rtol=0.05)
    np.testing.assert_allclose(
        mesh.vertices.mean(axis=0), positions.mean(axis=0), atol=0.3
    )


def test_cube_isosurface_on_an_oblique_grid(tmp_path):
    pytest.importorskip("trimesh")
    from chmpy.fmt.cube import CubeData

    bohr = 0.529177210903
    basis = np.array([[0.2, 0, 0], [0.05, 0.2, 0], [0, 0.03, 0.25]])  # angstrom
    origin = np.array([-1.0, -2.0, 0.5])
    centre = np.array([2.0, 1.5, 4.0])
    n = (40, 45, 35)
    index = np.stack(
        np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1
    ).reshape(-1, 3)
    values = np.exp(-np.sum((index @ basis + origin - centre) ** 2, axis=1) / 0.5)
    lines = ["title", "subtitle", f"1 {' '.join(str(x / bohr) for x in origin)}"]
    lines += [
        f"{k} {' '.join(str(x / bohr) for x in b)}"
        for k, b in zip(n, basis, strict=True)
    ]
    lines.append(f"1 0.0 {' '.join(str(x / bohr) for x in centre)}")
    lines += [
        " ".join(f"{v:.6e}" for v in values[i : i + 6])
        for i in range(0, len(values), 6)
    ]
    path = tmp_path / "gaussian.cube"
    path.write_text("\n".join(lines) + "\n")

    mesh = CubeData(path).isosurface(0.5)
    radius = np.linalg.norm(mesh.vertices - centre, axis=1)
    np.testing.assert_allclose(radius, np.sqrt(-0.5 * np.log(0.5)), atol=0.02)
    assert mesh.volume > 0


def test_property_colours_span_the_colour_map():
    pytest.importorskip("matplotlib")
    from chmpy.util.color import property_to_color

    colours = property_to_color(np.array([-1.0, 0.0, 2.0, 5.0, 10.0]), cmap="viridis")
    assert len({tuple(c) for c in np.round(colours, 6)}) == 5


def test_cartesian_to_spherical_inverts_spherical_to_cartesian():
    from chmpy.util.num import cartesian_to_spherical, spherical_to_cartesian

    xyz = np.array([[0.3, -1.2, 0.7], [1.0, 2.0, -3.0]])
    np.testing.assert_allclose(spherical_to_cartesian(cartesian_to_spherical(xyz)), xyz)


def test_periodic_coordination_numbers_count_every_image():
    from chmpy import Crystal
    from chmpy.crystal.eeq_pbc import calculate_coordination_numbers_crystal

    from .. import TEST_FILES

    crystal = Crystal.load(TEST_FILES["acetic_acid.cif"])
    cn = calculate_coordination_numbers_crystal(crystal)
    hydrogens = cn[crystal.unit_cell_atoms()["element"] == 1]
    np.testing.assert_allclose(hydrogens, 1.0, atol=0.05)

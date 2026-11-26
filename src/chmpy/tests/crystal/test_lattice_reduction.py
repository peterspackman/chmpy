import unittest

import numpy as np

from chmpy import Crystal
from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.crystal.lattice_reduction import niggli_reduce, selling_reduce

from .. import TEST_FILES

TRICLINIC = UnitCell.triclinic(5.1, 6.3, 7.7, 81.0, 97.0, 104.0, unit="degrees")
# a P-1 cell from a structure search whose basis had collapsed toward a plane
COLLAPSED = UnitCell.triclinic(
    14.2404, 13.6622, 43.7246, 93.410, 34.087, 60.430, unit="degrees"
)
FCC_PRIMITIVE = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def random_unimodular(rng, steps=6):
    u = np.eye(3, dtype=int)
    for _ in range(steps):
        e = np.eye(3, dtype=int)
        i, j = rng.choice(3, 2, replace=False)
        e[i, j] = rng.integers(-4, 5)
        u = e @ u
    return u


def niggli_form(vectors):
    g = vectors @ vectors.T
    return np.array([g[0, 0], g[1, 1], g[2, 2], 2 * g[1, 2], 2 * g[0, 2], 2 * g[0, 1]])


def selling_scalars(vectors):
    b = np.vstack([vectors, -vectors.sum(axis=0)])
    return (b @ b.T)[np.triu_indices(4, 1)]


class SharedPropertiesTestCase(unittest.TestCase):
    def test_reductions_preserve_the_lattice(self):
        direct = COLLAPSED.direct
        for reduce in (niggli_reduce, selling_reduce):
            r = reduce(direct)
            self.assertEqual(abs(round(np.linalg.det(r.transform))), 1)
            np.testing.assert_allclose(r.transform @ direct, r.vectors, atol=1e-9)
            self.assertGreater(np.linalg.det(r.vectors), 0)

    def test_fractional_coordinates_follow_the_transform(self):
        direct = COLLAPSED.direct
        frac = np.array([[0.31, -0.72, 0.15], [0.5, 0.5, 0.0]])
        for reduce in (niggli_reduce, selling_reduce):
            r = reduce(direct)
            frac2 = frac @ np.linalg.inv(r.transform)
            np.testing.assert_allclose(frac2 @ r.vectors, frac @ direct, atol=1e-9)

    def test_dependent_vectors_are_rejected(self):
        flat = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0]])
        for reduce in (niggli_reduce, selling_reduce):
            with self.assertRaises(ValueError):
                reduce(flat)


class NiggliTestCase(unittest.TestCase):
    def assert_niggli_reduced(self, vectors, eps=1e-6):
        A, B, C, xi, eta, zeta = niggli_form(vectors)
        self.assertLessEqual(A, B + eps)
        self.assertLessEqual(B, C + eps)
        self.assertLessEqual(abs(xi), B + eps)
        self.assertLessEqual(abs(eta), A + eps)
        self.assertLessEqual(abs(zeta), A + eps)
        off = np.array([xi, eta, zeta])
        self.assertTrue(np.all(off > -eps) and np.all(off > eps) or np.all(off <= eps))
        self.assertGreaterEqual(xi + eta + zeta + A + B, -eps)

    def test_reduced_cell_is_left_alone(self):
        cell = UnitCell.triclinic(10.0, 11.0, 12.0, 88.0, 89.0, 87.0, unit="degrees")
        np.testing.assert_array_equal(niggli_reduce(cell.direct).transform, np.eye(3))

    def test_fcc(self):
        rng = np.random.default_rng(1)
        r = niggli_reduce(random_unimodular(rng) @ FCC_PRIMITIVE)
        np.testing.assert_allclose(
            niggli_form(r.vectors), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], atol=1e-12
        )

    def test_any_basis_gives_the_same_cell(self):
        rng = np.random.default_rng(12830)
        expected = niggli_form(niggli_reduce(TRICLINIC.direct).vectors)
        for _ in range(50):
            r = niggli_reduce(random_unimodular(rng) @ TRICLINIC.direct)
            self.assert_niggli_reduced(r.vectors)
            np.testing.assert_allclose(niggli_form(r.vectors), expected, atol=1e-9)

    def test_collapsed_cell(self):
        r = niggli_reduce(COLLAPSED.direct)
        self.assert_niggli_reduced(r.vectors)
        reduced = UnitCell(r.vectors)
        self.assertLess(reduced.c, COLLAPSED.c)
        np.testing.assert_allclose(reduced.volume(), COLLAPSED.volume(), rtol=1e-10)

    def test_idempotent(self):
        once = niggli_reduce(COLLAPSED.direct).vectors
        np.testing.assert_allclose(niggli_reduce(once).vectors, once, atol=1e-9)

    def test_unit_cell_method(self):
        reduced = COLLAPSED.niggli_reduced()
        np.testing.assert_allclose(
            reduced.direct, niggli_reduce(COLLAPSED.direct).vectors
        )


class SellingTestCase(unittest.TestCase):
    def test_scalars_are_non_positive(self):
        rng = np.random.default_rng(7)
        for cell in (TRICLINIC, COLLAPSED):
            for _ in range(20):
                r = selling_reduce(random_unimodular(rng) @ cell.direct)
                self.assertLessEqual(selling_scalars(r.vectors).max(), 1e-6)

    def test_any_basis_gives_the_same_scalars(self):
        rng = np.random.default_rng(3)
        expected = np.sort(selling_scalars(selling_reduce(TRICLINIC.direct).vectors))
        for _ in range(50):
            r = selling_reduce(random_unimodular(rng) @ TRICLINIC.direct)
            np.testing.assert_allclose(
                np.sort(selling_scalars(r.vectors)), expected, atol=1e-9
            )

    def test_fcc(self):
        # fcc: two perpendicular pairs among the four vectors, the rest -a^2/4
        rng = np.random.default_rng(2)
        r = selling_reduce(random_unimodular(rng) @ FCC_PRIMITIVE)
        np.testing.assert_allclose(
            np.sort(selling_scalars(r.vectors)), [-0.25] * 4 + [0, 0], atol=1e-12
        )

    def test_vectors_ordered_by_length(self):
        lengths = np.linalg.norm(selling_reduce(COLLAPSED.direct).vectors, axis=1)
        self.assertTrue(np.all(np.diff(lengths) >= -1e-12))

    def test_unit_cell_method(self):
        reduced = COLLAPSED.selling_reduced()
        np.testing.assert_allclose(
            reduced.direct, selling_reduce(COLLAPSED.direct).vectors
        )


def orbit(crystal):
    "Cartesian images of the asymmetric unit under every operation"
    frac = crystal.space_group.apply_all_symops(crystal.asymmetric_unit.positions)[1]
    return crystal.to_cartesian(frac)


def same_structure(crystal, reduced, tol=1e-6):
    """Whether two descriptions hold the same atoms, comparing modulo the
    lattice of `reduced`, which contains that of `crystal`."""
    a = reduced.to_fractional(orbit(crystal))
    b = reduced.to_fractional(orbit(reduced))

    def covered(x, y):
        d = x[:, None, :] - y[None, :, :]
        d -= np.round(d)
        return np.abs(d).max(axis=2).min(axis=1).max() < tol

    return covered(a, b) and covered(b, a)


def general_crystal(cell, space_group):
    asym = AsymmetricUnit(
        [Element["C"], Element["O"]], [[0.11, 0.23, 0.37], [0.31, 0.07, 0.19]]
    )
    return Crystal(cell, space_group, asym)


C2C = general_crystal(
    UnitCell.monoclinic(9.1, 7.3, 11.7, 104.0, unit="degrees"), SpaceGroup(15)
)


class CrystalReductionTestCase(unittest.TestCase):
    def test_keeps_structure_and_symmetry(self):
        acetic = Crystal.load(TEST_FILES["acetic_acid.cif"])
        for method in ("niggli_reduced", "selling_reduced"):
            reduced = getattr(acetic, method)()
            self.assertTrue(same_structure(acetic, reduced), method)
            self.assertEqual(reduced.space_group.international_tables_number, 33)
            self.assertEqual(len(reduced.space_group), 4)

    def test_reduced_cell_is_reduced(self):
        reduced = C2C.niggli_reduced()
        np.testing.assert_allclose(
            reduced.unit_cell.direct, niggli_reduce(reduced.unit_cell.direct).vectors
        )

    def test_centred_crystal_becomes_primitive(self):
        reduced = C2C.niggli_reduced()
        self.assertTrue(same_structure(C2C, reduced))
        self.assertEqual(len(reduced.space_group), 4)
        self.assertEqual(reduced.space_group.international_tables_number, 15)
        self.assertEqual(reduced.space_group.centering, "primitive")
        np.testing.assert_allclose(
            reduced.unit_cell.volume(), C2C.unit_cell.volume() / 2, rtol=1e-10
        )

    def test_basis_transform_is_recorded(self):
        reduced = C2C.niggli_reduced()
        P = reduced.properties["basis_transform"]
        np.testing.assert_allclose(P @ C2C.unit_cell.direct, reduced.unit_cell.direct)
        # the asymmetric unit moves as one, by a lattice vector
        shift = reduced.asymmetric_unit.positions - (
            C2C.asymmetric_unit.positions @ np.linalg.inv(P)
        )
        np.testing.assert_allclose(shift, np.round(shift), atol=1e-9)
        np.testing.assert_allclose(shift, shift[:1].repeat(len(shift), axis=0))

    def test_rhombohedral_cif(self):
        r3c = Crystal.load(TEST_FILES["r3c_example.cif"])
        reduced = r3c.niggli_reduced()
        self.assertTrue(same_structure(r3c, reduced))
        self.assertEqual(len(reduced.space_group), 6)
        self.assertEqual(reduced.properties["cif_data"]["cell_formula_units_Z"], 6)

    def test_selling_keeps_hexagonal_symmetry(self):
        cell = UnitCell.hexagonal(5.0, 8.0, unit="degrees")
        crystal = general_crystal(cell, SpaceGroup(194))
        reduced = crystal.selling_reduced()
        self.assertEqual(len(reduced.space_group), 24)
        self.assertTrue(same_structure(crystal, reduced))

    def test_cif_round_trip_keeps_untabulated_setting(self):
        reduced = C2C.niggli_reduced()
        self.assertFalse(reduced.space_group.tabulated)
        text = reduced.to_cif_string()
        back = Crystal.from_cif_string(text)
        self.assertEqual(
            {s.integer_code for s in back.symmetry_operations},
            {s.integer_code for s in reduced.symmetry_operations},
        )
        self.assertEqual(back.space_group.international_tables_number, 15)
        hall = next(x for x in text.splitlines() if "space_group_name_Hall" in x)
        self.assertIn("?", hall)

    def test_symbol_only_formats_fall_back_to_p1(self):
        from chmpy.fmt.gulp import crystal_to_gulp_input

        reduced = C2C.niggli_reduced()
        text = crystal_to_gulp_input(reduced)
        self.assertIn("P 1", text)
        n_sites = len(reduced.unit_cell_atoms()["frac_pos"])
        self.assertEqual(len(reduced.as_P1().asymmetric_unit), n_sites)

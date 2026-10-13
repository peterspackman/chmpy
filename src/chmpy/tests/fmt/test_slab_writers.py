import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chmpy import Crystal, Slab
from chmpy.calc.adapters.gulp import available as gulp_available
from chmpy.core.element import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell
from chmpy.fmt.aims import parse_geometry_string
from chmpy.fmt.vasp import parse_poscar
from chmpy.fmt.xyz_file import parse_xyz_string

GULP = shutil.which("gulp")
HAVE_GULP = gulp_available()
NACL_BUCK = """buck
Na core Cl core 1170.41 0.3150 0.0 0.0 12.0
Cl core Cl core 1227.20 0.3214 14.53 0.0 12.0"""
BULK_NACL = f"""single
cell
5.64 5.64 5.64 90 90 90
frac
Na core 0 0 0 1.0
Cl core 0.5 0.5 0.5 -1.0
space
225
{NACL_BUCK}
"""


def rocksalt_slab(layers=4):
    nacl = Crystal(
        UnitCell.cubic(5.64),
        SpaceGroup(225),
        AsymmetricUnit(
            [Element["Na"], Element["Cl"]], np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        ),
    )
    return Slab.from_crystal(nacl, (1, 0, 0), layers=layers, vacuum=10.0)


class SlabWriterTestCase(unittest.TestCase):
    def setUp(self):
        self.slab = rocksalt_slab()
        self.fixed = self.slab.fixed_mask(2)

    def test_aims_round_trip(self):
        contents = self.slab.to_aims_geometry_string(fixed=self.fixed)
        parsed = Crystal.from_aims_string(contents)
        np.testing.assert_allclose(parsed.unit_cell.direct, self.slab.cell, atol=1e-6)
        self.assertEqual(contents.count("constrain_relaxation"), self.fixed.sum())
        self.assertEqual(
            len(parse_geometry_string(contents)["elements"]), len(self.slab)
        )

    def test_poscar_round_trip_with_selective_dynamics(self):
        parsed = parse_poscar(self.slab.to_poscar_string(fixed=self.fixed))
        np.testing.assert_allclose(parsed["direct"], self.slab.cell, atol=1e-8)
        self.assertEqual(len(parsed["positions"]), len(self.slab))
        flags = parsed["selective_dynamics"]
        self.assertEqual((~flags[:, 0]).sum(), self.fixed.sum())

    def test_extxyz(self):
        contents = self.slab.to_extxyz_string()
        elements, positions = parse_xyz_string(contents)
        np.testing.assert_allclose(positions, self.slab.positions, atol=1e-6)
        header = contents.splitlines()[1]
        self.assertIn("MillerH=1 MillerK=0 MillerL=0", header)
        self.assertIn("PlaneGroup=p4mm", header)

    def test_gulp_regions(self):
        contents = self.slab.to_gulp_input(keywords=["single"], fixed=self.fixed)
        self.assertIn("sfractional region 1", contents)
        self.assertIn("sfractional region 2", contents)
        self.assertIn("svectors", contents.splitlines())

    def test_save_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("geometry.in", "slab.gin", "POSCAR", "slab.xyz", "slab.cif"):
                self.slab.save(Path(tmp, name))
                self.assertTrue(Path(tmp, name).read_text())

    @unittest.skipUnless(HAVE_GULP, "needs GULP")
    def test_gulp_rocksalt_100_surface_energy(self):
        bulk = run_gulp(BULK_NACL)
        energies = {}
        for layers in (4, 6):
            slab = rocksalt_slab(layers)
            contents = slab.to_gulp_input(
                keywords=["single"],
                charges=[1.0, -1.0],
                additional_keywords={"": NACL_BUCK},
            )
            energies[layers] = (run_gulp(contents) - layers * bulk) / (2 * slab.area)
        joules_per_m2 = 16.0218
        self.assertGreater(energies[4] * joules_per_m2, 0.1)
        self.assertLess(energies[4] * joules_per_m2, 0.4)
        self.assertAlmostEqual(energies[4], energies[6], places=3)

    @unittest.skipUnless(HAVE_GULP, "needs GULP")
    def test_gulp_region_2_surface_energy_matches_two_sided(self):
        bulk = run_gulp(BULK_NACL)
        slab = rocksalt_slab(6)
        two_sided = (
            run_gulp(
                slab.to_gulp_input(
                    keywords=["single"],
                    charges=[1.0, -1.0],
                    additional_keywords={"": NACL_BUCK},
                )
            )
            - 6 * bulk
        ) / (2 * slab.area)
        contents = slab.to_gulp_input(
            keywords=["single"],
            fixed=slab.fixed_mask(3),
            charges=[1.0, -1.0],
            bulk_energy=bulk,
            additional_keywords={"": NACL_BUCK},
        )
        out = subprocess.run(
            [GULP], input=contents, capture_output=True, text=True, timeout=60
        ).stdout
        gamma = float(
            re.search(r"Surface energy \(region 1\)\s+=\s+(\S+)", out).group(1)
        )
        self.assertAlmostEqual(gamma, two_sided * 16.0218, places=2)

    @unittest.skipUnless(HAVE_GULP, "needs GULP")
    def test_oriented_bulk_has_the_bulk_energy(self):
        from chmpy.fmt.gulp import crystal_to_gulp_input

        bulk = run_gulp(BULK_NACL)
        for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)]:
            slab = Slab.from_crystal(rocksalt_slab().crystal, hkl, layers=2)
            oriented = slab.oriented_bulk()
            charges = [
                1.0 if n == 11 else -1.0
                for n in oriented.asymmetric_unit.atomic_numbers
            ]
            contents = crystal_to_gulp_input(
                oriented,
                keywords=["single"],
                charges=charges,
                additional_keywords={"": NACL_BUCK},
            )
            n_formula = len(oriented.asymmetric_unit) // 2
            self.assertAlmostEqual(
                run_gulp(contents) / n_formula, bulk, places=5, msg=hkl
            )


def run_gulp(contents) -> float:
    "Total energy (eV) of the cell in a GULP single point."
    out = subprocess.run(
        [GULP], input=contents, capture_output=True, text=True, timeout=60
    ).stdout
    match = re.search(r"Primitive unit cell\s+=\s+(\S+) eV", out) or re.search(
        r"Total lattice energy\s+=\s+(\S+) eV", out
    )
    return float(match.group(1))


class SlabReaderTestCase(unittest.TestCase):
    def setUp(self):
        self.slab = rocksalt_slab()
        self.slab.fixed = self.slab.fixed_mask(1)

    def assert_round_trip(self, name, metadata=True):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, name)
            self.slab.save(path)
            loaded = Slab.load(path)
        self.assertEqual(len(loaded), len(self.slab))
        np.testing.assert_allclose(loaded.cell[:2], self.slab.cell[:2], atol=1e-6)
        np.testing.assert_array_equal(
            np.sort(loaded.elements), np.sort(self.slab.elements)
        )
        self.assertEqual(loaded.fixed.sum(), self.slab.fixed.sum())
        if metadata:
            self.assertEqual(loaded.hkl, (1, 0, 0))
            self.assertEqual(loaded.layers, 4)
        return loaded

    def test_aims(self):
        loaded = self.assert_round_trip("geometry.in")
        np.testing.assert_allclose(loaded.positions, self.slab.positions, atol=1e-6)

    def test_poscar(self):
        self.assert_round_trip("POSCAR")

    def test_extxyz(self):
        loaded = self.assert_round_trip("slab.xyz")
        np.testing.assert_array_equal(loaded.layer, self.slab.layer)
        np.testing.assert_array_equal(loaded.fragment, self.slab.fragment)

    def test_gulp(self):
        loaded = self.assert_round_trip("slab.gin")
        self.assertAlmostEqual(loaded.thickness, 3 * 2.82, places=4)

    def test_loaded_slab_symmetry_from_atoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "geometry.in")
            rocksalt_slab(layers=3).save(path)
            loaded = Slab.load(path)
        self.assertIsNone(loaded.crystal)
        self.assertEqual(loaded.plane_group.symbol, "p4mm")
        self.assertTrue(loaded.is_symmetric)
        self.assertEqual(loaded.space_group().sg_number, 123)

    def test_slab_from_arrays_is_reoriented(self):
        cell = np.array([[0.0, 0.0, 4.0], [4.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        slab = Slab(cell, ["Cu"], [[0.0, 10.0, 0.0]])
        np.testing.assert_allclose(slab.cell, np.diag([4.0, 4.0, 20.0]), atol=1e-12)
        np.testing.assert_allclose(slab.positions, [[0.0, 0.0, 10.0]], atol=1e-12)

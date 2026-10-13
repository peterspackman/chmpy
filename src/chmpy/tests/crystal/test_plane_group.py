import unittest
from fractions import Fraction as F

import numpy as np

from chmpy.crystal.plane_group import closure, identify_plane_group

H = F(1, 2)
CENTRING = ((1, 0), (0, 1)), (H, H)
TWO = ((-1, 0), (0, -1)), (0, 0)
MX = ((-1, 0), (0, 1)), (0, 0)
FOUR = ((0, -1), (1, 0)), (0, 0)
THREE = ((0, -1), (1, -1)), (0, 0)
SIX = ((1, -1), (1, 0)), (0, 0)

SQUARE = np.eye(2)
RECTANGULAR = np.diag([9.0, 25.0])
HEXAGONAL = np.array([[1.0, -0.5], [-0.5, 1.0]])
OBLIQUE = np.array([[9.0, 2.0], [2.0, 16.0]])

GENERATORS = {
    "p1": ([], OBLIQUE),
    "p2": ([TWO], OBLIQUE),
    "pm": ([MX], RECTANGULAR),
    "pg": ([(MX[0], (0, H))], RECTANGULAR),
    "cm": ([MX, CENTRING], RECTANGULAR),
    "p2mm": ([TWO, MX], RECTANGULAR),
    "p2mg": ([TWO, (MX[0], (H, 0))], RECTANGULAR),
    "p2gg": ([TWO, (MX[0], (H, H))], RECTANGULAR),
    "c2mm": ([TWO, MX, CENTRING], RECTANGULAR),
    "p4": ([FOUR], SQUARE),
    "p4mm": ([FOUR, MX], SQUARE),
    "p4gm": ([FOUR, (MX[0], (H, H))], SQUARE),
    "p3": ([THREE], HEXAGONAL),
    "p3m1": ([THREE, (((0, -1), (-1, 0)), (0, 0))], HEXAGONAL),
    "p31m": ([THREE, (((0, 1), (1, 0)), (0, 0))], HEXAGONAL),
    "p6": ([SIX], HEXAGONAL),
    "p6mm": ([SIX, (((0, -1), (-1, 0)), (0, 0))], HEXAGONAL),
}

ORDERS = {
    "p1": 1, "p2": 2, "pm": 2, "pg": 2, "cm": 4, "p2mm": 4, "p2mg": 4,
    "p2gg": 4, "c2mm": 8, "p4": 4, "p4mm": 8, "p4gm": 8, "p3": 3,
    "p3m1": 6, "p31m": 6, "p6": 6, "p6mm": 12,
}  # fmt: skip


def transform(ops, p, origin):
    "Re-express ops in the basis x = P x' + origin."
    p = np.array(p, dtype=object)
    p_inv = np.array([[p[1, 1], -p[0, 1]], [-p[1, 0], p[0, 0]]], dtype=object) * (
        1 if p[0, 0] * p[1, 1] - p[0, 1] * p[1, 0] == 1 else -1
    )
    origin = np.array([F(x) for x in origin], dtype=object)
    result = []
    for a, t in ops:
        a = np.array(a, dtype=object)
        t = np.array(t, dtype=object)
        a_new = p_inv.dot(a).dot(p)
        t_new = p_inv.dot(a.dot(origin) + t - origin)
        result.append((a_new.astype(int), tuple(t_new)))
    return result


class PlaneGroupTestCase(unittest.TestCase):
    def test_group_orders(self):
        for symbol, (generators, _) in GENERATORS.items():
            self.assertEqual(len(closure(generators)), ORDERS[symbol], symbol)

    def test_standard_settings(self):
        for number, (symbol, (generators, metric)) in enumerate(GENERATORS.items()):
            group = identify_plane_group(generators, metric)
            self.assertEqual(group.symbol, symbol)
            self.assertEqual(group.number, number + 1)

    def test_other_bases_and_origins(self):
        bases = [
            ((1, 1), (0, 1)),
            ((2, 1), (1, 1)),
            ((0, 1), (-1, 0)),
            ((1, -2), (1, -1)),
        ]
        origins = [(0, 0), (F(1, 3), F(1, 4)), (F(1, 2), F(1, 6))]
        for symbol, (generators, metric) in GENERATORS.items():
            ops = closure(generators)
            for p in bases:
                for origin in origins:
                    new_metric = np.array(p).T @ metric @ np.array(p)
                    group = identify_plane_group(transform(ops, p, origin), new_metric)
                    self.assertEqual(group.symbol, symbol, f"{p} {origin}")

    def test_centred_lattice_label(self):
        group = identify_plane_group(GENERATORS["c2mm"][0], RECTANGULAR)
        self.assertEqual(group.lattice, "centred rectangular")
        self.assertEqual(group.point_group, "2mm")

    def test_extra_translations_fold_into_lattice(self):
        half_a = ((1, 0), (0, 1)), (H, 0)
        group = identify_plane_group([half_a, TWO, MX], RECTANGULAR)
        self.assertEqual(group.symbol, "p2mm")

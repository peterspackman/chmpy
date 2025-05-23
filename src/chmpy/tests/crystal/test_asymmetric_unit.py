import unittest

import numpy as np

from chmpy import Element
from chmpy.crystal import AsymmetricUnit

_ICE_II_LABELS = (
    "O1",
    "H1",
    "H2",
    "O2",
    "H3",
    "H4",
    "O3",
    "H5",
    "H6",
    "O4",
    "H7",
    "H8",
    "O5",
    "H9",
    "H10",
    "O6",
    "H11",
    "H12",
    "O7",
    "H13",
    "H14",
    "O8",
    "H15",
    "H16",
    "O9",
    "H17",
    "H18",
    "O10",
    "H19",
    "H20",
    "O11",
    "H21",
    "H22",
    "O12",
    "H23",
    "H24",
)
_ICE_II_ELEMENTS = [Element[x] for x in _ICE_II_LABELS]

_ICE_II_POSITIONS = np.array(
    [
        0.273328954083,
        0.026479033257,
        0.855073668062,
        0.152000330304,
        0.043488909374,
        0.793595454907,
        0.420775085827,
        0.191165194485,
        0.996362203192,
        0.144924657237,
        0.726669877048,
        0.973520141937,
        0.206402797363,
        0.847998481439,
        0.956510183901,
        0.003636687868,
        0.579223433079,
        0.808833958746,
        0.026477491142,
        0.855072949204,
        0.273328854276,
        0.043487719387,
        0.793594459529,
        0.152000553858,
        0.191163388489,
        0.996362120061,
        0.420774953988,
        0.726670757782,
        0.973520932681,
        0.144926297633,
        0.847999275418,
        0.956510882297,
        0.206404294889,
        0.579224602173,
        0.808834869258,
        0.003637530197,
        0.855073412561,
        0.273329597478,
        0.026478702027,
        0.793594909621,
        0.152000771295,
        0.043488316376,
        0.996362312075,
        0.420775512814,
        0.191164826329,
        0.973520717390,
        0.144925628579,
        0.726671054509,
        0.956510600982,
        0.206403626100,
        0.847999547813,
        0.808834607385,
        0.003637609551,
        0.579224562315,
        0.477029330652,
        0.749805220756,
        0.331717174202,
        0.402360172390,
        0.720795433576,
        0.401054786853,
        0.368036378343,
        0.742284933413,
        0.207434128329,
        0.668282055550,
        0.522969467265,
        0.250193622013,
        0.598945169999,
        0.597639203188,
        0.279204514235,
        0.792565160978,
        0.631962548905,
        0.257714022497,
        0.749805496250,
        0.331717033025,
        0.477029827575,
        0.720795009402,
        0.401054437437,
        0.402360618546,
        0.742284706875,
        0.207433751728,
        0.368036342085,
        0.522969071341,
        0.250193392512,
        0.668282780114,
        0.597638176364,
        0.279203622225,
        0.598945231951,
        0.631962932785,
        0.257715003205,
        0.792566578018,
        0.331715381178,
        0.477028907327,
        0.749804544234,
        0.401053887354,
        0.402360576463,
        0.720795552111,
        0.207432480540,
        0.368035542438,
        0.742284142147,
        0.250193225247,
        0.668282913065,
        0.522970147212,
        0.279203658434,
        0.598945325854,
        0.597639149965,
        0.257715011998,
        0.792566781760,
        0.631964289620,
    ]
).reshape(-1, 3)


def ice_ii_asym():
    return AsymmetricUnit(_ICE_II_ELEMENTS, _ICE_II_POSITIONS, labels=_ICE_II_LABELS)


class AsymmetricUnitTestCase(unittest.TestCase):
    def test_asymmetric_unit_constructor(self):
        asym = ice_ii_asym()
        self.assertTrue(len(asym) == 36)
        asym_generated_labels = AsymmetricUnit(_ICE_II_ELEMENTS, _ICE_II_POSITIONS)
        self.assertTrue(all(asym_generated_labels.labels == asym.labels))

    def test_repr(self):
        asym = ice_ii_asym()
        self.assertTrue(asym.__repr__() == "<H24O12>")

    def test_from_records(self):
        records = [{"label": "H1", "element": "H", "position": (0, 0, 0)}]
        asym = AsymmetricUnit.from_records(records)
        self.assertTrue(len(asym) == 1)

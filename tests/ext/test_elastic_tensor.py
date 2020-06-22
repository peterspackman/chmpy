from chmpy import Molecule
from chmpy.ext.elastic_tensor import ElasticTensor
import unittest
import numpy as np

TENSOR = """
    48.137	11.411	12.783	0.000	-3.654	0.000
    11.411	34.968	14.749	0.000	-0.094	0.000
    12.783	14.749	26.015	0.000	-4.528	0.000
    0.000	0.000	0.000	14.545	0.000	0.006
    -3.654	-0.094	-4.528	0.000	10.771	0.000
    0.000	0.000	0.000	0.006	0.000	11.947
"""


class ElasticTensorTestCase(unittest.TestCase):
    def setUp(self):
        self.elastic = ElasticTensor.from_string(TENSOR)

    def test_youngs_modulus(self):
        ym = self.elastic.youngs_modulus([[0, 0, 1]])
        self.assertAlmostEqual(ym[0], 16.95754579335107)

    def test_linear_compressibility(self):
        lc = self.elastic.linear_compressibility([[0, 0, 1]])
        self.assertAlmostEqual(lc[0], 28.214314777188065)

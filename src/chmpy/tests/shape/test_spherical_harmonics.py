import math
import unittest

from chmpy.shape.spherical_harmonics import SphericalHarmonics


class TestSphericalHarmonics(unittest.TestCase):
    def setUp(self):
        self.sph = SphericalHarmonics(4, True)

    def cmp(self, a, b, delta=1e-12):
        self.assertAlmostEqual(a.real, b.real, delta=delta)
        self.assertAlmostEqual(a.imag, b.imag, delta=delta)

    def test_angular(self):
        eval_result = self.sph.single_angular(math.pi / 4, math.pi / 4)
        self.cmp(eval_result[0], complex(0.28209479177387814, 0.0))
        self.cmp(eval_result[1], complex(0.17274707473566775, -0.17274707473566772))
        self.cmp(eval_result[2], complex(0.34549414947133544, 0))
        self.cmp(eval_result[3], complex(-0.17274707473566775, -0.17274707473566772))
        self.cmp(eval_result[4], complex(1.1826236627522426e-17, -0.19313710101159473))
        self.cmp(eval_result[5], complex(0.27313710764801974, -0.2731371076480197))
        self.cmp(eval_result[6], complex(0.15769578262626002, 0))
        self.cmp(eval_result[7], complex(-0.2731371076480198, -0.27313710764801974))
        self.cmp(eval_result[8], complex(1.1826236627522428e-17, 0.19313710101159476))
        self.cmp(eval_result[9], complex(-0.104305955908196, -0.10430595590819601))
        self.cmp(eval_result[10], complex(2.212486281755292e-17, -0.3613264303300692))
        self.cmp(eval_result[11], complex(0.24238513808561293, -0.24238513808561288))
        self.cmp(eval_result[12], complex(-0.13193775767639848, 0))
        self.cmp(eval_result[13], complex(-0.24238513808561296, -0.24238513808561293))
        self.cmp(eval_result[14], complex(2.2124862817552912e-17, 0.3613264303300691))
        self.cmp(eval_result[15], complex(0.104305955908196, -0.10430595590819601))
        self.cmp(
            eval_result[16], complex(-0.11063317311124561, -1.3548656133020197e-17)
        )
        self.cmp(eval_result[17], complex(-0.22126634622249125, -0.2212663462224913))
        self.cmp(eval_result[18], complex(2.5604553376501068e-17, -0.4181540897233056))
        self.cmp(eval_result[19], complex(0.08363081794466115, -0.08363081794466114))
        self.cmp(eval_result[20], complex(-0.34380302747441394, 0))
        self.cmp(eval_result[21], complex(-0.08363081794466115, -0.08363081794466114))
        self.cmp(eval_result[22], complex(2.5604553376501068e-17, 0.4181540897233056))
        self.cmp(eval_result[23], complex(0.22126634622249125, -0.2212663462224913))
        self.cmp(eval_result[24], complex(-0.11063317311124561, 1.3548656133020197e-17))

    def test_cartesian(self):
        pos = [2.0, 0.0, 1.0]
        theta = 0.0
        phi = 0.0
        eval_result = self.sph.single_cartesian(*pos)
        eval_ang = self.sph.single_angular(theta, phi)
        for i in range(25):
            self.cmp(eval_result[i], eval_ang[i])

import numpy as np

class AssocLegendre:
    def __init__(self, lm):
        self.lmax = lm
        self.a, self.b = self._compute_ab(self.lmax)
        self.cache = np.zeros((lm + 1, lm + 1))
        self.norm = np.zeros((lm + 1, lm + 1))

    @staticmethod
    def _amm(m):
        a = 1.0
        for k in range(1, np.abs(m) + 1):
            a *= (2 * k + 1) / (2 * k)
        return np.sqrt( a / (4 * np.pi))


    @staticmethod
    def _amn(m, n):
        return np.sqrt((4 * n * n - 1) / (n * n - m * m))

    @staticmethod
    def _bmn(m, n):
        return - np.sqrt(
            (2 * n + 1) * ((n - 1)*(n - 1) - m * m) / 
            ((2 * n- 3) * (n * n - m * m))
        )

    @staticmethod
    def _compute_ab(lmax):
        a = np.empty((lmax + 1, lmax + 1))
        b = np.empty((lmax + 1, lmax + 1))
        for m in range(0, lmax + 1):
            a[m, m] = AssocLegendre._amm(m)
            for l in range(np.abs(m) + 1, lmax + 1):
                a[l, m] = AssocLegendre._amn(m, l)
                b[l, m] = AssocLegendre._bmn(m, l)
        return a, b

    def evaluate_batch(self, x, result=None):
        if result is None:
            result = np.zeros((self.lmax + 1) * (self.lmax + 2) // 2)
        idx = 0
        for m in range(0, self.lmax + 1):
            for l in range(m, self.lmax + 1):
                if(l == m):
                    result[idx] = self.a[l, m] * (1 - x * x) ** (0.5 * m)
                elif (l == (m + 1)):
                    result[idx] = self.a[l, m] * x * self.cache[l - 1, m]
                else:
                    result[idx] = (
                        self.a[l, m] * x * self.cache[l - 1, m] +
                        self.b[l, m] * self.cache[l - 2, m]
                    )
                self.cache[l, m] = result[idx]
                idx += 1
        return result

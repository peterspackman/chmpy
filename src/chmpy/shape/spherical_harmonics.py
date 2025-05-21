import numpy as np

from ._sht import AssocLegendre


class SphericalHarmonics:
    def __init__(self, lm, phase=True):
        self.lmax = lm
        self.phase = phase
        self.plm = AssocLegendre(lm)
        self.plm_work_array = np.empty(self.nplm(), dtype=np.float64)

    def idx_c(self, l, m):
        return l * (l + 1) + m

    def nlm(self):
        "the number of complex SHT coefficients"
        return (self.lmax + 1) * (self.lmax + 1)

    def nplm(self):
        "the number of real SHT coefficients (i.e. legendre polynomial terms)"
        return (self.lmax + 1) * (self.lmax + 2) // 2

    def single_angular(self, theta, phi, result=None):
        if result is None:
            result = np.empty(self.nlm(), dtype=np.complex128)

        ct = np.cos(theta)
        self.plm.evaluate_batch(ct, result=self.plm_work_array)

        plm_idx = 0
        for l in range(0, self.lmax + 1):
            offset = l * (l + 1)
            result[offset] = self.plm_work_array[plm_idx]
            plm_idx += 1

        c = np.exp(phi * 1j)
        cm = c
        for m in range(1, self.lmax + 1):
            sign = 1
            if self.phase and (m & 1):
                sign = -1
            for l in range(m, self.lmax + 1):
                l_offset = l * (l + 1)
                rr = cm
                ii = np.conj(rr)
                rr = sign * self.plm_work_array[plm_idx] * rr
                ii = sign * self.plm_work_array[plm_idx] * ii
                if m & 1:
                    ii = -ii
                result[l_offset - m] = ii
                result[l_offset + m] = rr
                plm_idx += 1
            cm *= c
        return result

    def batch_angular(self, pos, result=None):
        if result is None:
            result = np.empty((pos.shape[0], self.nlm()), dtype=np.complex128)

        for i in range(pos.shape[0]):
            result[i, :] = self.single_angular(*pos[i], result=result[i, :])
        return result

    def single_cartesian(self, x, y, z, result=None):
        if result is None:
            result = np.empty(self.nlm(), dtype=np.complex128)
        pass
        epsilon = 1e-12
        ct = z
        self.plm.evaluate_batch(ct, result=self.plm_work_array)

        st = 0.0
        if abs(1.0 - ct) > epsilon:
            st = np.sqrt(1.0 - ct * ct)

        plm_idx = 0
        for l in range(0, self.lmax + 1):
            l_offset = l * (l + 1)
            result[l_offset] = self.plm_work_array[plm_idx]
            plm_idx += 1

        c = 0j
        if st > epsilon:
            c = complex(x / st, y / st)
        cm = c
        for m in range(1, self.lmax + 1):
            sign = 1
            if self.phase and (m & 1):
                sign = -1
            for l in range(m, self.lmax + 1):
                l_offset = l * (l + 1)
                rr = sign * cm * self.plm_work_array[plm_idx]
                ii = sign * np.conj(cm) * self.plm_work_array[plm_idx]
                if m & 1:
                    ii = -ii
                result[l_offset - m] = ii
                result[l_offset + m] = rr
                plm_idx += 1
            cm *= c
        return result

    def batch_cartesian(self, pos, result=None):
        if result is None:
            result = np.empty((pos.shape[0], self.nlm()), dtype=np.complex128)
        pass

    def __eval__(self, *parameters, result=None, cartesian=False):
        pass

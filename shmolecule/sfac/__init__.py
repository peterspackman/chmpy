from . import atomic_form_factors
from . import _sfac
import numpy as np


UNIQUE_REFLECTION_MULTIPLICITY = {
    ("triclinic", "-1"): 2,
    ("monoclinic", "2/m"): 4,
    ("orthorhombic", "mmm"): 8,
    ("tetragonal", "4/m"): 8,
    ("tetragonal", "4/mmm"): 16,
    ("cubic", "m-3"): 24,
    ("cubic", "m-3m"): 48,
}


UNIQUE_REFLECTION_TYPES = {
    ("triclinic", "-1", "*"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 0, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 1, 0), ((-1, 0, 0), (0, 1, 0), (0, 0, -1))),
        ((0, 1, -1), ((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    ),
    ("monoclinic", "2/m", "aface"): (
        ((0, 0, 0), ((0, 1, 0), (1, 0, 0), (0, 0, 1))),
        ((0, -1, 1), ((0, -1, 0), (1, 0, 0), (0, 0, 1))),
    ),
    ("monoclinic", "2/m", "bface"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 0, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("monoclinic", "2/m", "cface"): (
        ((0, 0, 0), ((1, 0, 0), (0, 0, 1), (0, 1, 0))),
        ((-1, 1, 0), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)))
    ),
    ("orthorhombic", "mmm", "*"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("tetragonal", "4/mmm", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("tetragonal", "4/m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 2, 0), ((1, 1, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("hexagonal", "6/mmm", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("hexagonal", "6/m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 2, 0), ((0, 1, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("hexagonal", "-3m1", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((0, 1, 1), ((0, 1, 0), (1, 1, 0), (0, 0, 1)))
    ),
    ("hexagonal", "-31m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 1, -1), ((1, 0, 0), (1, 1, 0), (0, 0, -1))),
    ),
    ("hexagonal", "-3m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 0, -1), (1, 1, 1))),
        ((1, 1, 0), ((1, 0, -1), (0, 0, -1), (1, 1, 1)))
    ),
    ("hexagonal", "-3", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 0, -1), (1, 1, 1))),
        ((1, 1, 0), ((1, 0, -1), (0, 0, -1), (1, 1, 1))),
        ((0, -1, -2), ((1, 0, 0), (1, 0,-1), (-1, -1, -1))),
        ((1, 0, -2),  ((1, 0, -1), (0, 0,-1), (-1, -1, -1))),
    ),
    ("cubic", "m3m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),
    ),
    ("cubic", "m3", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),
        ((1, 2, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1)))
    )
}

def reflections(crystal, wavelength=0.7, size=10):
    h_max, k_max, l_max = size, size, size
    recip = crystal.unit_cell.reciprocal_lattice.copy()
    system = crystal.space_group.crystal_system
    centering = "*" if system != "monoclinic" else crystal.space_group.centering
    laue_class = crystal.space_group.laue_class
 
    apexes, bases = zip(*UNIQUE_REFLECTION_TYPES[(system, laue_class, centering)])
    h_range = np.r_[0:h_max + 1]
    k_range = np.r_[0:k_max + 1]
    l_range = np.r_[0:l_max + 1]
    sections = []
    for a, b in zip(apexes, bases):
        v1 = np.r_[b[0][0], b[0][1], b[0][2]]
        v2 = np.r_[b[1][0], b[1][1], b[1][2]]
        v3 = np.r_[b[2][0], b[2][1], b[2][2]]
        hkl = v1 * h_range[:, None]
        hkl = hkl + (v2 * k_range[:, None])[:, None]
        hkl = hkl + (v3 * l_range[:, None, None])[:, None, None]
        hkl = hkl.reshape(-1, 3)
        hkl += a
        sections.append(hkl)
    hkl = np.vstack(sections)
    q = np.dot(hkl, recip)
    q_mag = np.linalg.norm(q, axis=1)
    mask = q_mag < (1/wavelength)
    return q[mask], q_mag[mask], hkl[mask]


def powder_pattern(crystal, wavelength=0.7, two_theta_range=(5, 50)):
    hkl, q, q_mag, sfac = calculate_structure_factors(crystal, wavelength=wavelength)
    f2 = np.abs(sfac * sfac)
    theta = np.arcsin(wavelength * q_mag / (4 * np.pi))
    two_theta = 2 * theta
    l, u = np.radians(two_theta_range)
    l = max(l, 1e-3)
    mask = (two_theta <= u) & (two_theta >= l)
    t = theta[mask]
    tt = two_theta[mask]
    i = f2[mask]
    costt = np.cos(tt)
    sintt = np.sin(tt)
    scaled_intensities = i * (1 + costt * costt) / (8 * sintt * sintt * np.cos(t))
    return np.degrees(tt), scaled_intensities


def calculate_structure_factors(crystal, wavelength=0.7):
    q, q_mag, hkl = reflections(crystal, wavelength=wavelength)
    uc = crystal.unit_cell_atoms()
    asym_sfac_idx = [
        _sfac.get_form_factor_index(el.symbol)
        for el in crystal.asymmetric_unit.elements
    ]
    indices = [asym_sfac_idx[i] for i in uc["asym_atom"]]
    N = len(indices)
    unique_indices = set(asym_sfac_idx)
    q2 = q_mag * q_mag
    sintl = q2 / (16 * np.pi * np.pi)
    scattering_factors = {
        i: _sfac.scattering_factors(i, sintl)
        for i in unique_indices
    }
    frac_pos = uc["frac_pos"]
    exp_fac = np.zeros(hkl.shape[0], dtype=np.complex128)
    exp_fac.imag = 2 * np.pi
    sfac = np.zeros_like(exp_fac)
    for i, idx in enumerate(indices):
        fj = scattering_factors[idx]
        pos = frac_pos[i]
        sfac += fj * np.exp(exp_fac * np.dot(hkl, pos))
    return hkl, q, q_mag, sfac

#   for (size_t atom = 0; atom < uc_elements.size(); atom++) {
#       const auto el = uc_elements[atom];
#       const auto& fj = fjs[el];
#       const auto& pos = xt::view(uc_positions, atom, xt::all());
#       auto fj_dw = fj * xt::exp(-U[atom] * q2);
#       F_calc.add_contribution_from(pos, fj_dw);
#        np.exp
        
#        -2i * M_PI * xt::linalg::dot(m_hkl, pos_frac);


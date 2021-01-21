from . import _sfac
import numpy as np
from collections import namedtuple


StructureFactors = namedtuple("StructureFactors", "hkl q q_mag values normalization")
Reflections = namedtuple("Reflection", "q q_mag hkl")

LAMBDA_Cu = 1.54059

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
    ("-1", "*"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 0, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 1, 0), ((-1, 0, 0), (0, 1, 0), (0, 0, -1))),
        ((0, 1, -1), ((1, 0, 0), (0, 1, 0), (0, 0, -1))),
    ),
    ("2/m", "aface"): (
        ((0, 0, 0), ((0, 1, 0), (1, 0, 0), (0, 0, 1))),
        ((0, -1, 1), ((0, -1, 0), (1, 0, 0), (0, 0, 1))),
    ),
    ("2/m", "bface"): (
        ((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ((-1, 0, 1), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("2/m", "cface"): (
        ((0, 0, 0), ((1, 0, 0), (0, 0, 1), (0, 1, 0))),
        ((-1, 1, 0), ((-1, 0, 0), (0, 0, 1), (0, 1, 0))),
    ),
    ("mmm", "*"): (((0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, 1))),),
    ("4/mmm", "*"): (((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),),
    ("4/m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 2, 0), ((1, 1, 0), (0, 1, 0), (0, 0, 1))),
    ),
    ("6/mmm", "*"): (((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),),
    ("6/m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 2, 0), ((0, 1, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("-3m1", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((0, 1, 1), ((0, 1, 0), (1, 1, 0), (0, 0, 1))),
    ),
    ("-31m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (0, 0, 1))),
        ((1, 1, -1), ((1, 0, 0), (1, 1, 0), (0, 0, -1))),
    ),
    ("-3m", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 0, -1), (1, 1, 1))),
        ((1, 1, 0), ((1, 0, -1), (0, 0, -1), (1, 1, 1))),
    ),
    ("-3", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 0, -1), (1, 1, 1))),
        ((1, 1, 0), ((1, 0, -1), (0, 0, -1), (1, 1, 1))),
        ((0, -1, -2), ((1, 0, 0), (1, 0, -1), (-1, -1, -1))),
        ((1, 0, -2), ((1, 0, -1), (0, 0, -1), (-1, -1, -1))),
    ),
    ("m3m", "*"): (((0, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),),
    ("m3", "*"): (
        ((0, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1))),
        ((1, 2, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1))),
    ),
}


def hklmax(uc, dmin):
    return (np.array(uc.lengths) / dmin).astype(int)


def reflections(crystal, wavelength=LAMBDA_Cu, dmin=LAMBDA_Cu / 2, sort=True):
    """Calculate the unique reflections contributing to
    diffraction for the given crystal, along with the q
    vector at the provided wavelength"""
    h_max, k_max, l_max = hklmax(crystal.unit_cell, dmin)
    recip = crystal.unit_cell.reciprocal_lattice.copy()
    system = crystal.space_group.crystal_system
    centering = (
        "*" if system != "monoclinic" else crystal.space_group.choice[0] + "face"
    )
    laue_class = crystal.space_group.laue_class
    if (
        crystal.space_group.has_hexagonal_rhombohedral_choices()
        and crystal.unit_cell.is_rhombohedral
    ):
        raise NotImplementedError("Rhombohedral crystals not currently supported")

    #   apexes, bases = zip(*UNIQUE_REFLECTION_TYPES[(laue_class, centering)])
    #   h_range = np.r_[0: h_max + 1]
    #   k_range = np.r_[0: k_max + 1]
    #   l_range = np.r_[0: l_max + 1]
    #   sections = []
    #   for a, b in zip(apexes, bases):
    #       v1 = np.r_[b[0][0], b[0][1], b[0][2]]
    #       v2 = np.r_[b[1][0], b[1][1], b[1][2]]
    #       v3 = np.r_[b[2][0], b[2][1], b[2][2]]
    #       hkl = v1 * h_range[:, None]
    #       hkl = hkl + (v2 * k_range[:, None])[:, None]
    #       hkl = hkl + (v3 * l_range[:, None, None])[:, None, None]
    #       hkl = hkl.reshape(-1, 3)
    #       hkl += a
    #       sections.append(hkl)
    #   hkl = np.vstack(sections)
    h, k, l = np.mgrid[-h_max:h_max, -k_max:k_max, -l_max:l_max].astype(np.int32)
    hkl = np.c_[h.ravel(), k.ravel(), l.ravel()]
    G = hkl @ recip
    q = np.linalg.norm(G, axis=1)
    mask = q <= (2 / wavelength)
    if sort:
        G = G[mask]
        q = q[mask]
        hkl = hkl[mask]
        order = np.argsort(q)
        return Reflections(G[order], q[order], hkl[order])

    return Reflections(G[mask], q[mask], hkl[mask])


def powder_pattern(crystal, wavelength=LAMBDA_Cu, two_theta_range=(5, 50)):
    """Calculate the powder pattern the given crystal at a given wavelength
    over a given range of 2-theta"""
    sfac = structure_factors(crystal, wavelength=wavelength)
    hkl, G, q, sfac, norm = sfac
    f2 = np.abs(sfac)
    f2 = f2 * f2 / norm
    theta = np.arcsin(wavelength * q / 2)
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


def structure_factors(crystal, wavelength=LAMBDA_Cu):
    """Calculate the structure factors for a given crystal at the provided
    wavelength"""
    G, q, hkl = reflections(crystal, wavelength=wavelength)
    facs = _sfac.calculate_unique_plane_factors(hkl, q)
    keep = facs > 0
    facs = facs[keep]
    G = G[keep]
    q = q[keep]
    hkl = hkl[keep]
    uc = crystal.unit_cell_atoms()
    asym_sfac_idx = [
        _sfac.get_form_factor_index(el.symbol)
        for el in crystal.asymmetric_unit.elements
    ]
    indices = [asym_sfac_idx[i] for i in uc["asym_atom"]]
    N = len(indices)
    unique_indices = set(asym_sfac_idx)
    sintl = q / 2
    sintl2 = sintl * sintl
    scattering_factors = {
        i: _sfac.scattering_factors(i, sintl2) for i in unique_indices
    }
    frac_pos = uc["frac_pos"]
    exp_fac = np.zeros(hkl.shape[0], dtype=np.complex128)
    exp_fac.imag = 2 * np.pi
    sfac = np.zeros_like(exp_fac)
    normalization = np.zeros_like(q)
    for i, idx in enumerate(indices):
        fj = scattering_factors[idx]
        pos = frac_pos[i]
        hkl_dot_pos = hkl @ pos
        sfac += fj * np.exp(exp_fac * hkl_dot_pos)
        normalization += fj * fj
    f000 = sfac[0]
    sfac *= facs
    return StructureFactors(hkl, G, q, sfac, normalization)

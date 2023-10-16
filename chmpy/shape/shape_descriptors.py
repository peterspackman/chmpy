from chmpy import StockholderWeight, PromoleculeDensity
from chmpy.shape._sht import expand_coeffs_to_full
from chmpy.interpolate._density import (
    sphere_stockholder_radii,
    sphere_promolecule_radii,
)
from ._invariants import p_invariants_c, p_invariants_r
import logging
import numpy as np

LOG = logging.getLogger(__name__)
_HAVE_WARNED_ABOUT_LMAX_P = False


def make_N_invariants(coefficients) -> np.ndarray:
    """
    Construct the `N` type invariants from SHT coefficients.
    If coefficients is of length n, the size of the result will be sqrt(n)

    Arguments:
        coefficients (np.ndarray): the set of spherical harmonic coefficients

    Returns:
        np.ndarray the `N` type rotational invariants based on these coefficients
    """
    size = int(np.sqrt(len(coefficients)))
    invariants = np.empty(shape=(size), dtype=np.float64)
    for i in range(0, size):
        lower, upper = i ** 2, (i + 1) ** 2
        invariants[i] = np.sum(
            coefficients[lower : upper + 1]
            * np.conj(coefficients[lower : upper + 1])
        ).real
    return np.sqrt(invariants)


def make_invariants(l_max, coefficients, kinds="NP") -> np.ndarray:
    """
    Construct the `N` and/or `P` type invariants from SHT coefficients.

    Arguments:
        l_max (int): the maximum angular momentum of the coefficients
        coefficients (np.ndarray): the set of spherical harmonic coefficients
        kinds (str, optional): which kinds of invariants to include

    Returns:
        np.ndarray the `N` and/or `P` type rotational invariants based on these coefficients
    """

    global _HAVE_WARNED_ABOUT_LMAX_P
    invariants = []
    if "N" in kinds:
        invariants.append(make_N_invariants(coefficients))
    if "P" in kinds:
        # Because we only have factorial precision (double precision)
        # in our clebsch implementation up to 70! l_max for P type
        # invariants is restricted to <= 23
        # TODO use a better clebsch gordan coefficients implementation
        # e.g. that in https://github.com/GXhelsinki/Clebsch-Gordan-Coefficients-
        pfunc = p_invariants_c
        MAX_L_MAX = 23
        if l_max > MAX_L_MAX:
            if not _HAVE_WARNED_ABOUT_LMAX_P:
                LOG.warn(
                    f"P type invariants only supported up to l_max = {MAX_L_MAX}: "
                    "will only using N type invariants beyond that."
                )
                _HAVE_WARNED_ABOUT_LMAX_P = True
            c = coefficients[: MAX_L_MAX * MAX_L_MAX]
            invariants.append(pfunc(c))
        else:
            invariants.append(pfunc(coefficients))
    return np.hstack(invariants)


def _compute_property_in_j_channel(sht, r, property_function, origin=None):
    x, y, z = sht.grid_cartesian
    xyz = np.c_[x.flatten(), y.flatten(), z.flatten()] * r.flatten()[:, np.newaxis]
    if origin is not None:
        xyz += origin
    prop_values = property_function(xyz)
    r_cplx = np.empty(r.shape, dtype=np.complex128)
    r_cplx.real = r
    r_cplx.imag = prop_values.reshape(r.shape)
    return r_cplx

def stockholder_weight_descriptor(sht, n_i, p_i, n_e, p_e, **kwargs):
    """
    Calculate the 'stockholder weight' shape descriptors based on the
    Hirshfeld weight i.e. ratio of electron density from the 'interior'
    to the total electron density.

    Args:
        sht (SHT): the spherical harmonic transform object handle
        n_i (np.ndarray): atomic numbers of the interior atoms
        p_i (np.ndarray): Cartesian coordinates of the interior atoms
        n_e (np.ndarray): atomic numbers of the exterior atoms
        p_e (np.ndarray): Cartesian coordinates of the exterior atoms

    Keyword Args:
        isovalue (float): change the Hirshfeld weight value (default 0.5)
        background (float): include an optional 'background' electron
            density (default 0.0)
        with_property (str): calculate the combined shape + surface
            property descriptor using the specified property on the
            surface (e.g. d_norm, esp)
        bounds (Tuple): modify the lower/upper bounds on the search for
            the isovalue (default 0.1, 20.0)
        coefficients (bool): also return the coefficients of the SHT
        origin (np.ndarray): specify the center of the surface
            (default is the geometric centroid of the interior atoms)
        kinds (str): the kinds of invariants to calculate (default 'NP')
    Returns:
        np.ndarray: the rotation invariant descriptors of the Hirshfeld surface shape
    """
    isovalue = kwargs.get("isovalue", 0.5)
    background = kwargs.get("background", 0.0)
    property_function = kwargs.get("with_property", None)
    r_min, r_max = kwargs.get("bounds", (0.1, 20.0))
    o = kwargs.get("origin", np.mean(p_i, axis=0, dtype=np.float32))
    s = StockholderWeight.from_arrays(n_i, p_i, n_e, p_e, background=background)
    g = np.empty((sht.grid[0].size, 3), dtype=np.float32)
    x, y, z = sht.grid_cartesian
    g[:, 0] = x.flatten()
    g[:, 1] = y.flatten()
    g[:, 2] = z.flatten()

    r = sphere_stockholder_radii(s.s, o, g, r_min, r_max, 1e-7, 30, isovalue).reshape(sht.grid[0].shape)
    if np.any(r < 0):
        raise ValueError(
            f"Unable to find isovalue {isovalue:.2f} in all directions for bounds ({r_min:.2f}, {r_max:.2f})"
        )
    real = True
    if property_function is not None:
        if property_function == "d_norm":
            property_function = lambda x: s.d_norm(x)[3]
        elif property_function == "esp":
            from chmpy import Molecule

            els = s.dens_a.elements
            pos = s.dens_a.positions
            property_function = Molecule.from_arrays(
                s.dens_a.elements, s.dens_a.positions
            ).electrostatic_potential
        r = _compute_property_in_j_channel(sht, r, property_function, origin=o)
        real = False
    l_max = sht.lmax
    coeffs = sht.analysis(r)

    coeff4inv = expand_coeffs_to_full(l_max, coeffs) if real else coeffs
    invariants = make_invariants(
        l_max, coeff4inv, kinds=kwargs.get("kinds", "NP")
    )

    if kwargs.get("coefficients", False):
        return coeffs, invariants
    return invariants


def promolecule_density_descriptor(sht, n_i, p_i, **kwargs):
    """
    Calculate the shape description of the promolecule density isosurface.

    Args:
        sht (SHT): the spherical harmonic transform object handle
        n_i (np.ndarray): atomic numbers of the atoms
        p_i (np.ndarray): Cartesian coordinates of the atoms
        **kwargs: keyword arguments for optional settings.
    Keyword Args:
        isovalue (float): change the Hirshfeld weight value (default 0.5)
        with_property (str): calculate the combined shape + surface
            property descriptor using the specified property on the
            surface (e.g. d_norm, esp)
        bounds (Tuple): modify the lower/upper bounds on the search for
            the isovalue (default 0.1, 20.0)
        coefficients (bool): also return the coefficients of the SHT
        origin (np.ndarray): specify the center of the surface
            (default is the geometric centroid of the atoms)
        kinds (str): the kinds of invariants to calculate (default 'NP')
    Returns:
        np.ndarray: the rotation invariant descriptors of the promolecule surface shape
    """
    isovalue = kwargs.get("isovalue", 0.0002)
    property_function = kwargs.get("with_property", None)
    r_min, r_max = kwargs.get("bounds", (0.4, 20.0))
    pro = PromoleculeDensity((n_i, p_i))
    g = np.empty((sht.grid[0].size, 3), dtype=np.float32)
    x, y, z = sht.grid_cartesian
    g[:, 0] = x.flatten()
    g[:, 1] = y.flatten()
    g[:, 2] = z.flatten()

    o = kwargs.get("origin", np.mean(p_i, axis=0, dtype=np.float32))
    r = sphere_promolecule_radii(pro.dens, o, g, r_min, r_max, 1e-12, 30, isovalue).reshape(sht.grid[0].shape)
    if np.any(r < 0):
        raise ValueError(
            f"Unable to find isovalue {isovalue:.2f} in all directions for bounds ({r_min:.2f}, {r_max:.2f})"
        )
    real = True
    if property_function is not None:
        if property_function == "d_norm":
            property_function = lambda x: pro.d_norm(x)[1]
        elif property_function == "esp":
            from chmpy import Molecule

            els = pro.elements
            pos = pro.positions
            property_function = Molecule.from_arrays(els, pos).electrostatic_potential
        r = _compute_property_in_j_channel(sht, r, property_function)
        real = False
    l_max = sht.lmax
    coeffs = sht.analysis(r)
    coeff4inv = expand_coeffs_to_full(l_max, coeffs) if real else coeffs
    invariants = make_invariants(
        l_max, coeff4inv, kinds=kwargs.get("kinds", "NP")
    )

    if kwargs.get("coefficients", False):
        return coeffs, invariants
    return invariants

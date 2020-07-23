from chmpy import StockholderWeight, PromoleculeDensity
from chmpy.util.num import spherical_to_cartesian
from scipy.optimize import minimize_scalar
from chmpy.interpolate._density import (
    sphere_stockholder_radii,
    sphere_promolecule_radii,
)
from ._invariants import p_invariants_c, p_invariants_r
import logging
import numpy as np

LOG = logging.getLogger(__name__)
_HAVE_WARNED_ABOUT_LMAX_P = False


def make_N_invariants(coefficients, real=True) -> np.ndarray:
    """
    Construct the `N` type invariants from SHT coefficients.
    If coefficients is of length n, the size of the result will be sqrt(n)

    Arguments:
        coefficients (np.ndarray): the set of spherical harmonic coefficients
        real (bool, optional): whether to assume the coefficients are from a
            real SHT (true) or a complex SHT (false)

    Returns:
        np.ndarray the `N` type rotational invariants based on these coefficients
    """
    if real:
        # n = (l_max +2)(l_max+1)/2
        n = len(coefficients)
        size = int((-3 + np.sqrt(8 * n + 1)) // 2) + 1
        lower = 0
        invariants = np.empty(shape=(size), dtype=np.float64)
        for i in range(0, size):
            x = i + 1
            upper = lower + x
            invariants[i] = np.sum(
                coefficients[lower : upper + 1]
                * np.conj(coefficients[lower : upper + 1])
            ).real
            lower += x
    else:
        size = int(np.sqrt(len(coefficients)))
        invariants = np.empty(shape=(size), dtype=np.float64)
        for i in range(0, size):
            lower, upper = i ** 2, (i + 1) ** 2
            invariants[i] = np.sum(
                coefficients[lower : upper + 1]
                * np.conj(coefficients[lower : upper + 1])
            ).real
    return np.sqrt(invariants)


def make_invariants(l_max, coefficients, kinds="NP", real=True) -> np.ndarray:
    """
    Construct the `N` and/or `P` type invariants from SHT coefficients.

    Arguments:
        l_max (int): the maximum angular momentum of the coefficients
        coefficients (np.ndarray): the set of spherical harmonic coefficients
        kinds (str, optional): which kinds of invariants to include
        real (bool, optional): whether to assume the coefficients are from a
            real SHT (true) or a complex SHT (false)

    Returns:
        np.ndarray the `N` and/or `P` type rotational invariants based on these coefficients
    """

    global _HAVE_WARNED_ABOUT_LMAX_P
    invariants = []
    if "N" in kinds:
        invariants.append(make_N_invariants(coefficients, real=real))
    if "P" in kinds:
        # Because we only have factorial precision (double precision)
        # in our clebsch implementation up to 70! l_max for P type
        # invariants is restricted to <= 23
        # TODO use a better clebsch gordan coefficients implementation
        # e.g. that in https://github.com/GXhelsinki/Clebsch-Gordan-Coefficients-
        pfunc = p_invariants_r if real else p_invariants_c
        MAX_L_MAX = 23
        if l_max > MAX_L_MAX:
            if not _HAVE_WARNED_ABOUT_LMAX_P:
                LOG.warn(
                    f"P type invariants only supported up to l_max = {MAX_L_MAX}: "
                    "will only using N type invariants beyond that."
                )
                _HAVE_WARNED_ABOUT_LMAX_P = True
            if real:
                c = coefficients[: ((MAX_L_MAX + 2)* (MAX_L_MAX + 1)) // 2]
            else:
                c = coefficients[:(MAX_L_MAX*MAX_L_MAX)]
            invariants.append(pfunc(c))
        else:
            invariants.append(pfunc(coefficients))
    return np.hstack(invariants)


def stockholder_weight_descriptor(sht, n_i, p_i, n_e, p_e, **kwargs):
    """
    Calculate the 'stockholder weight' shape descriptors based on the
    Hirshfeld weight i.e. ratio of electron density from the 'interior'
    to the total electron density.

    Parameters:
        sht (SHT): the spherical harmonic transform object handle
        n_i (np.ndarray): atomic numbers of the interior atoms
        p_i (np.ndarray): Cartesian coordinates of the interior atoms
        n_e (np.ndarray): atomic numbers of the exterior atoms
        p_e (np.ndarray): Cartesian coordinates of the exterior atoms
        kwargs (dict): keyword arguments for optional settings.
            Options include:
            ```
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
            ```
    Returns:
        np.ndarray: the rotation invariant descriptors of the Hirshfeld surface shape
    """
    isovalue = kwargs.get("isovalue", 0.5)
    background = kwargs.get("background", 0.0)
    property_function = kwargs.get("with_property", None)
    r_min, r_max = kwargs.get("bounds", (0.1, 20.0))
    s = StockholderWeight.from_arrays(n_i, p_i, n_e, p_e, background=background)
    g = np.empty(sht.grid.shape, dtype=np.float32)
    g[:, :] = sht.grid[:, :]
    o = kwargs.get("origin", np.mean(p_i, axis=0, dtype=np.float32))
    r = sphere_stockholder_radii(s.s, o, g, r_min, r_max, 1e-7, 30, isovalue)
    real = True
    if property_function is not None:
        if property_function == "d_norm":
            property_function = s.d_norm
        elif property_function == "esp":
            from chmpy import Molecule

            els = s.dens_a.elements
            pos = s.dens_a.positions
            property_function = Molecule.from_arrays(
                s.dens_a.elements, s.dens_a.positions
            ).electrostatic_potential
        xyz = sht.grid_cartesian * r[:, np.newaxis]
        prop_values = property_function(xyz)
        r_cplx = np.empty(r.shape, dtype=np.complex128)
        r_cplx.real = r
        r_cplx.imag = prop_values
        r = r_cplx
        real = False
    l_max = sht.l_max
    coeffs = sht.analyse(r)
    invariants = make_invariants(l_max, coeffs, kinds=kwargs.get("kinds", "NP"), real=real)
    if kwargs.get("coefficients", False):
        return coeffs, invariants
    return invariants


def promolecule_density_descriptor(sht, n_i, p_i, **kwargs):
    """
    Calculate the shape description of the promolecule density isosurface.

    Parameters:
        sht (SHT): the spherical harmonic transform object handle
        n_i (np.ndarray): atomic numbers of the atoms
        p_i (np.ndarray): Cartesian coordinates of the atoms
        kwargs (dict): keyword arguments for optional settings.
            Options include:
            ```
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
            ```
    Returns:
        np.ndarray: the rotation invariant descriptors of the promolecule surface shape
    """
    isovalue = kwargs.get("isovalue", 0.0002)
    property_function = kwargs.get("with_property", None)
    r_min, r_max = kwargs.get("bounds", (0.4, 20.0))
    pro = PromoleculeDensity((n_i, p_i))
    g = np.empty(sht.grid.shape, dtype=np.float32)
    g[:, :] = sht.grid[:, :]
    o = kwargs.get("origin", np.mean(p_i, axis=0, dtype=np.float32))
    r = sphere_promolecule_radii(pro.dens, o, g, r_min, r_max, 1e-7, 30, isovalue)
    real = True
    if property_function is not None:
        if property_function == "d_norm":
            property_function = lambda x: pro.d_norm(x)[1]
        elif property_function == "esp":
            from chmpy import Molecule

            els = pro.elements
            pos = pro.positions
            property_function = Molecule.from_arrays(els, pos).electrostatic_potential
        xyz = sht.grid_cartesian * r[:, np.newaxis]
        prop_values = property_function(xyz)
        r_cplx = np.empty(r.shape, dtype=np.complex128)
        r_cplx.real = r
        r_cplx.imag = prop_values
        r = r_cplx
        real = False
    l_max = sht.l_max
    coeffs = sht.analyse(r)
    invariants = make_invariants(l_max, coeffs, kinds=kwargs.get("kinds", "NP"), real=real)
    if kwargs.get("coefficients", False):
        return coeffs, invariants
    return invariants

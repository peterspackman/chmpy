"""Shape descriptor calculations for Crystal structures."""

import numpy as np

from chmpy.core.element import Element
from chmpy.core.molecule import Molecule


def functional_group_shape_descriptors(
    crystal, l_max=5, radius=6.0, kind="carboxylic_acid"
) -> np.ndarray:
    """
    Calculate the shape descriptors `[1,2]` for the all atoms in
    the functional group given for all symmetry unique molecules in this crystal.

    Args:
        crystal: Crystal instance
        l_max (int, optional): maximum level of angular momenta to include
            in the spherical harmonic transform of the molecular shape function.
            (default: 5)
        radius (float, optional): maximum distance (Angstroms) of neighbouring
            atoms to include in stockholder weight calculation (default: 5)
        kind (str, optional): Identifier for the functional group
            type (default: 'carboxylic_acid')

    Returns:
        shape description vector

    References:
    ```
    [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
        https://dx.doi.org/10.1038/srep22204
    [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
        https://dx.doi.org/10.1002/anie.201906602
    ```
    """
    descriptors = []
    from chmpy.shape import SHT, stockholder_weight_descriptor

    sph = SHT(l_max)
    for (
        in_els,
        in_pos,
        neighbour_els,
        neighbour_pos,
    ) in crystal.functional_group_surroundings(kind=kind, radius=radius):
        masses = np.asarray([Element[x].mass for x in in_els])
        c = np.sum(in_pos * masses[:, np.newaxis] / np.sum(masses), axis=0).astype(
            np.float32
        )
        dists = np.linalg.norm(in_pos - c, axis=1)
        bounds = np.min(dists) / 2, np.max(dists) + 10.0
        descriptors.append(
            stockholder_weight_descriptor(
                sph,
                in_els,
                in_pos,
                neighbour_els,
                neighbour_pos,
                origin=c,
                bounds=bounds,
            )
        )
    return np.asarray(descriptors)


def molecule_shape_descriptors(
    crystal, mol, l_max=5, radius=6.0, with_property=None
) -> np.ndarray:
    """
    Calculate the molecular shape descriptors `[1,2]` for
    the provided molecule in the crystal.

    Args:
        crystal: Crystal instance
        mol: Molecule instance
        l_max (int, optional): maximum level of angular momenta to include
            in the spherical harmonic
            transform of the molecular shape function.
        radius (float, optional): maximum distance (Angstroms) to include
            surroundings in the shape description
        with_property (str, optional): name of the surface property to include
            in the shape description

    Returns:
        shape description vector

    References:
    ```
    [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
        https://dx.doi.org/10.1038/srep22204
    [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
        https://dx.doi.org/10.1002/anie.201906602
    ```
    """
    from chmpy.shape import SHT, stockholder_weight_descriptor

    sph = SHT(l_max)
    mol, neighbour_els, neighbour_pos = crystal.molecule_environment(
        mol, radius=radius
    )
    c = np.array(mol.centroid, dtype=np.float32)
    dists = np.linalg.norm(mol.positions - c, axis=1)
    bounds = np.min(dists) / 2, np.max(dists) + 10.0
    return stockholder_weight_descriptor(
        sph,
        mol.atomic_numbers,
        mol.positions,
        neighbour_els,
        neighbour_pos,
        origin=c,
        bounds=bounds,
        with_property=with_property,
    )


def molecular_shape_descriptors(
    crystal, l_max=5, radius=6.0, with_property=None, return_coefficients=False
) -> np.ndarray:
    """
    Calculate the molecular shape descriptors[1,2] for all symmetry unique
    molecules in this crystal.

    Args:
        crystal: Crystal instance
        l_max (int, optional): maximum level of angular momenta to include
            in the spherical harmonic transform of the molecular shape function.
        radius (float, optional): maximum distance (Angstroms) to include
            surroundings in the shape description
        with_property (str, optional): name of the surface property to include
            in the shape description
        return_coefficients (bool, optional): also return the spherical
            harmonic coefficients

    Returns:
        shape description vector

    References:
    ```
    [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
        https://dx.doi.org/10.1038/srep22204
    [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
        https://dx.doi.org/10.1002/anie.201906602
    ```
    """
    descriptors = []
    coeffs = []
    from chmpy.shape import SHT, stockholder_weight_descriptor

    sph = SHT(l_max)
    for mol, neighbour_els, neighbour_pos in crystal.molecule_environments(
        radius=radius
    ):
        c = np.array(mol.centroid, dtype=np.float32)
        dists = np.linalg.norm(mol.positions - c, axis=1)
        bounds = np.min(dists) / 2, np.max(dists) + 10.0
        descriptor = stockholder_weight_descriptor(
            sph,
            mol.atomic_numbers,
            mol.positions,
            neighbour_els,
            neighbour_pos,
            origin=c,
            bounds=bounds,
            with_property=with_property,
            coefficients=return_coefficients,
        )

        if return_coefficients:
            coeffs.append(descriptor[0])
            descriptors.append(descriptor[1])
        else:
            descriptors.append(descriptor)
    if return_coefficients:
        return np.asarray(coeffs), np.asarray(descriptors)
    else:
        return np.asarray(descriptors)


def atomic_shape_descriptors(
    crystal, l_max=5, radius=6.0, return_coefficients=False, with_property=None
) -> np.ndarray:
    """
    Calculate the shape descriptors[1,2] for all symmetry unique
    atoms in this crystal.

    Args:
        crystal: Crystal instance
        l_max (int, optional): maximum level of angular momenta to include
            in the spherical harmonic transform of the molecular shape function.
        radius (float, optional): maximum distance (Angstroms) to include
            surroundings in the shape description
        with_property (str, optional): name of the surface property to include
            in the shape description
        return_coefficients (bool, optional): also return the spherical
            harmonic coefficients

    Returns:
        shape description vector

    References:
    ```
    [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
        https://dx.doi.org/10.1038/srep22204
    [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
        https://dx.doi.org/10.1002/anie.201906602
    ```
    """
    descriptors = []
    coeffs = []
    from chmpy.shape import SHT, stockholder_weight_descriptor

    sph = SHT(l_max)
    for surrounds in crystal.atomic_surroundings(radius=radius):
        n = surrounds["centre"]["element"]
        pos = surrounds["centre"]["cart_pos"]
        neighbour_els = surrounds["neighbours"]["element"]
        neighbour_pos = surrounds["neighbours"]["cart_pos"]

        ubound = Element[n].vdw_radius * 3 + 2.0
        desc = stockholder_weight_descriptor(
            sph,
            [n],
            [pos],
            neighbour_els,
            neighbour_pos,
            bounds=(0.15, ubound),
            coefficients=return_coefficients,
            with_property=with_property,
        )
        if return_coefficients:
            descriptors.append(desc[1])
            coeffs.append(desc[0])
        else:
            descriptors.append(desc)
    if return_coefficients:
        return np.asarray(coeffs), np.asarray(descriptors)
    else:
        return np.asarray(descriptors)


def atom_group_shape_descriptors(crystal, atoms, l_max=5, radius=6.0) -> np.ndarray:
    """Calculate the shape descriptors[1,2] for the given atomic
    group in this crystal.

    Args:
        crystal: Crystal instance
        atoms (Tuple): atoms to include in the as the 'inside'
            of the shape description.
        l_max (int, optional): maximum level of angular momenta to include
            in the spherical harmonic transform of the molecular shape function.
        radius (float, optional): maximum distance (Angstroms) to include
            surroundings in the shape description

    Returns:
        shape description vector

    References:
    ```
    [1] PR Spackman et al. Sci. Rep. 6, 22204 (2016)
        https://dx.doi.org/10.1038/srep22204
    [2] PR Spackman et al. Angew. Chem. 58 (47), 16780-16784 (2019)
        https://dx.doi.org/10.1002/anie.201906602
    ```
    """
    from chmpy.shape import SHT, stockholder_weight_descriptor

    sph = SHT(l_max)
    inside, outside = crystal.atom_group_surroundings(atoms, radius=radius)
    m = Molecule.from_arrays(*inside)
    c = np.array(m.centroid, dtype=np.float32)
    dists = np.linalg.norm(m.positions - c, axis=1)
    bounds = np.min(dists) / 2, np.max(dists) + 10.0
    return np.asarray(
        stockholder_weight_descriptor(
            sph, *inside, *outside, origin=c, bounds=bounds
        )
    )


def shape_descriptors(crystal, kind="molecular", **kwargs):
    k = kind.lower()
    if k == "molecular":
        return molecular_shape_descriptors(crystal, **kwargs)
    elif k == "molecule":
        return molecule_shape_descriptors(crystal, **kwargs)
    elif k == "atomic":
        return atomic_shape_descriptors(crystal, **kwargs)
    elif k == "atom group":
        return atom_group_shape_descriptors(crystal, **kwargs)

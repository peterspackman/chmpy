def wolf_sum(crystal, cutoff=16.0, eta=0.2, charges=None):
    """
    Compute the Coulomb interaction via Wolf sum and damped Coulomb potential
    using point charges.

    Arguments:
        crystal (Crystal): the crystal for which to compute the Wolf sum.
        cutoff (float, optional): the cutoff radius (in Angstroms) for which to compute the
            neighbouring charges (default=16)
        eta (float, optional): the eta parameter (1/Angstroms), if unsure just leave this at
            its default (default=0.2)
        charges (array_like, optional): charges of the atoms in the asymmetric unit, if not
            provided then they will be 'guessed' using the EEM method on the isolated molecules

        Returns:
            the total electrostatic energy of the asymmetric unit in the provided crystal
            (Hartrees)
    """
    import numpy as np
    from chmpy.util.unit import ANGSTROM_TO_BOHR
    from scipy.special import erfc

    if charges is None:
        charges = np.empty(len(crystal.asym))
        for mol in crystal.symmetry_unique_molecules():
            pq = mol.partial_charges
            charges[mol.properties["asymmetric_unit_atoms"]] = pq
    else:
        charges = np.array(charges)

    # convert to Bohr (i.e. perform the calculation in au)
    eta /= ANGSTROM_TO_BOHR
    rc = cutoff * ANGSTROM_TO_BOHR
    trc = erfc(eta * rc) / rc
    sqrt_pi = np.sqrt(np.pi)

    self_term = 0
    pair_term = 0

    for surrounds in crystal.atomic_surroundings(radius=cutoff):  # angstroms here
        i = surrounds["centre"]["asym_atom"]
        qi = charges[i]
        self_term += qi * qi
        qj = charges[surrounds["neighbours"]["asym_atom"]]
        rij = ANGSTROM_TO_BOHR * surrounds["neighbours"]["distance"]
        pair_term += np.sum(qi * qj * (erfc(eta * rij) / rij - trc))

    self_term *= 0.5 * trc + eta / sqrt_pi

    return 0.5 * pair_term - self_term

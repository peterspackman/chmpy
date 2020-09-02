import numpy as np
import logging
from chmpy.exe import Gaussian, Tonto
from chmpy.templates import load_template
from chmpy.util.num import kabsch_rotation_matrix
from pathlib import Path

G09_SCF = load_template("gaussian_scf")
TONTO_PAIR_ENERGY = load_template("tonto_pair_energy")
LOG = logging.getLogger(__name__)

CE_SCALE_FACTORS = {
    "CE-B3LYP": (1.057, 0.740, 0.871, 0.618),
    "CE-HF": (1.019, 0.651, 0.901, 0.811),
}


def tonto_pair_energy(wavefunctions, transforms, na, nb):
    from tempfile import TemporaryDirectory

    (rot_a, shift_a), (rot_b, shift_b) = transforms
    with TemporaryDirectory() as working_directory:

        fchk_a = Path(working_directory, "mol_a.fchk")
        fchk_b = Path(working_directory, "mol_b.fchk")
        fchk_a.write_text(wavefunctions[0])
        fchk_b.write_text(wavefunctions[1])
        input_file = TONTO_PAIR_ENERGY.render(
            name="test",
            idxs_a=f"1 ... {na}",
            idxs_b=f"{na + 1} ... {na + nb}",
            rot_a=" ".join(str(x) for x in rot_a.T.ravel()),
            shift_a=" ".join(str(x) for x in shift_a),
            rot_b=" ".join(str(x) for x in rot_b.T.ravel()),
            shift_b=" ".join(str(x) for x in shift_b),
            fchk_a=fchk_a,
            fchk_b=fchk_b,
        )
        LOG.debug("Tonto input:\n%s", input_file)

        t = Tonto(input_file, working_directory=working_directory)
        t.run()
        fchk_a.unlink()
        fchk_b.unlink()
    return t.stdout_contents


def parse_tonto_interaction_energies_stdout(stdout_contents):
    coul = 0.0
    pol = 0.0
    disp = 0.0
    rep = 0.0
    for line in stdout_contents.splitlines():
        # coulomb
        if "Delta E_coul (kJ/mol)" in line:
            coul = float(line.strip().split()[-1])
        elif "Polarization energy (kJ/mol)" in line:
            pol = float(line.strip().split()[-1])
        elif "Grimme06 dispersion energy (kJ/mol)" in line:
            disp = float(line.strip().split()[-1])
        elif "Delta E_exch-rep (kJ/mol)" in line:
            rep = float(line.strip().split()[-1])
    return coul, pol, disp, rep


def interaction_energies(c, model="CE-B3LYP", radius=3.8, nthreads=1):
    from tqdm import tqdm
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from chmpy import Molecule
    from chmpy.crystal.symmetry_operation import SymmetryOperation
    from chmpy.util.num import kabsch_rotation_matrix

    LOG.debug("Calculating %s model interaction_energies for %s", model, c)
    mols = c.symmetry_unique_molecules()

    for i, mol in tqdm(
        enumerate(mols), desc="Calculating wavefunctions", total=len(mols)
    ):
        if "fchk_contents" not in mol.properties:
            LOG.debug("Calculating wavefunction for %s", mol)
            mol_at_origin = mol.translated(-mol.centroid)
            input_file = G09_SCF.render(
                link0={"chk": f"mol_{i}.chk"},
                method="B3LYP",
                basis="6-31G(d,p)",
                route_commands="6d 10f NoSymm",
                title=f"{mol} wavefunction for chmpy {model} interaction energy",
                charge=0,
                multiplicity=1,
                geometry=mol_at_origin.to_xyz_string(header=False),
            )
            LOG.warn("G09 input:\n%s", input_file)
            g09 = Gaussian(input_file, run_formchk=f"mol_{i}.chk")
            g09.run()
            mol.properties["fchk_contents"] = g09.fchk_contents

    dimers, asym_pair_ids = c.symmetry_unique_dimers(radius=radius)

    stdout_contents = []
    args = []

    for d in dimers:
        LOG.debug("Calculating pair energy for %s", d)
        asym_a = mols[d.a.properties["asym_mol_idx"]]
        asym_b = mols[d.b.properties["asym_mol_idx"]]
        wfn_a = asym_a.properties["fchk_contents"]
        wfn_b = asym_b.properties["fchk_contents"]
        wavefunctions = (wfn_a, wfn_b)
        mol_fchk_a = Molecule.from_fchk_string(wfn_a)
        mol_fchk_b = Molecule.from_fchk_string(wfn_b)
        symop = SymmetryOperation.from_integer_code(
            d.b.properties["generator_symop"][0]
        )
        shift_a = d.a.centroid
        rot_a = kabsch_rotation_matrix(mol_fchk_a.positions, d.a.positions)
        rot_b = kabsch_rotation_matrix(mol_fchk_b.positions, d.b.positions)
        shift_b = d.b.centroid
        shift_b = d.b.centroid
        LOG.debug("\nrot_a:\n%s\nrot_b:\n%s", rot_a, rot_b)
        LOG.debug("shift_a: %s, shift_b: %s", shift_a, shift_b)
        transforms = ((rot_a, shift_a), (rot_b, shift_b))
        na = len(asym_a)
        nb = len(asym_b)
        args.append((wavefunctions, transforms, na, nb))

    with ThreadPoolExecutor(nthreads) as e:
        stdout_contents = list(
            tqdm(
                e.map(lambda x: tonto_pair_energy(*x), args),
                desc="Calculating pair energies",
                total=len(args),
            )
        )

    energies = []
    for stdout in stdout_contents:
        energies.append(parse_tonto_interaction_energies_stdout(stdout))

    df = pd.DataFrame(energies, columns=["E_coul", "E_pol", "E_disp", "E_rep"])
    scale_factors = CE_SCALE_FACTORS[model]
    df["mol_A"] = [d.a.properties["asym_mol_idx"] for d in dimers]
    df["mol_B"] = [d.b.properties["asym_mol_idx"] for d in dimers]
    df["d_closest"] = [d.closest_separation for d in dimers]
    df["d_centroid"] = [d.centroid_separation for d in dimers]
    df["d_com"] = [d.com_separation for d in dimers]
    df["symm"] = [d.transform_string() for d in dimers]
    df["E_tot"] = (
        scale_factors[0] * df["E_coul"]
        + scale_factors[1] * df["E_pol"]
        + scale_factors[2] * df["E_disp"]
        + scale_factors[3] * df["E_rep"]
    )
    return df

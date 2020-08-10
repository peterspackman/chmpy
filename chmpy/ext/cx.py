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
    fchk_a = Path("mol_a.fchk")
    fchk_b = Path("mol_b.fchk")
    fchk_a.write_text(wavefunctions[0])
    fchk_b.write_text(wavefunctions[1])
    (rot_a, shift_a), (rot_b, shift_b) = transforms
    input_file = TONTO_PAIR_ENERGY.render(
        name="test",
        idxs_a=f"1 ... {na}",
        idxs_b=f"{na + 1} ... {na + nb}",
        rot_a=" ".join(str(x) for x in rot_a.ravel()), shift_a=" ".join(str(x) for x in shift_a),
        rot_b=" ".join(str(x) for x in rot_b.ravel()), shift_b=" ".join(str(x) for x in shift_b),
        fchk_a=fchk_a, fchk_b=fchk_b,
    )
    LOG.debug("Tonto input:\n%s", input_file)
    t = Tonto(input_file)
    t.run()
    fchk_a.unlink()
    fchk_b.unlink()
    Path("/home/uniwa/staff2/staff/00087762/linux/stdin").write_text(input_file)
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


def interaction_energies(c, model="CE-B3LYP", radius=3.8):
    from tqdm import tqdm
    import pandas as pd
    LOG.debug("Calculating %s model interaction_energies for %s", model, c)
    mols = c.symmetry_unique_molecules()

    for i, mol in tqdm(enumerate(mols), desc="Calculating wavefunctions", total=len(mols)):
        if "fchk_contents" not in mol.properties:
            LOG.debug("Calculating wavefunction for %s", mol)
            input_file = G09_SCF.render(
                link0={"chk": f"mol_{i}.chk"},
                method="B3LYP", basis="6-31G(d,p)",
                route_commands="NoSymm 6d 10f",
                title=f"{mol} wavefunction for chmpy {model} interaction energy",
                charge=0, multiplicity=1,
                geometry=mol.to_xyz_string(header=False),
            )
            LOG.debug("G09 input:\n%s", input_file)
            g09 = Gaussian(input_file, run_formchk=f"mol_{i}.chk")
            g09.run()
            mol.properties["fchk_contents"] = g09.fchk_contents
    
    dimers, asym_pair_ids = c.symmetry_unique_dimers()

    stdout_contents = []
    for d in tqdm(dimers, "Calculating tonto pair energies", total=len(dimers)):
        LOG.debug("Calculating pair energy for %s", d)
        asym_a = mols[d.a.properties["asym_mol_idx"]]
        asym_b = mols[d.b.properties["asym_mol_idx"]]
        wfn_a = asym_a.properties["fchk_contents"]
        wfn_b = asym_b.properties["fchk_contents"] 
        wavefunctions = (wfn_a, wfn_b)
        shift_a = np.zeros(3)
        rot_a = np.eye(3)
        seitz = d.seitz_b
        rot_b = np.dot(c.uc.direct.T,  np.dot(seitz[:3, :3], c.uc.inverse.T))
        shift_b = c.to_cartesian(seitz[:3, 3])
        LOG.debug("\nrot_a:\n%s\nrot_b:\n%s", rot_a, rot_b)
        LOG.debug("shift_a: %s, shift_b: %s", shift_a, shift_b)
        transforms = ((rot_a, shift_a), (rot_b, shift_b))
        na = len(asym_a)
        nb = len(asym_b)
        contents = tonto_pair_energy(wavefunctions, transforms, na, nb)
        stdout_contents.append(contents)

    energies = []
    for stdout in stdout_contents:
        energies.append(parse_tonto_interaction_energies_stdout(stdout))

    df = pd.DataFrame(
        energies, columns=["E_coul", "E_pol", "E_disp", "E_rep"]
    )
    scale_factors = CE_SCALE_FACTORS[model]
    df["d"] = [d.separation for d in dimers]
    df["symm"] = [d.transform_string() for d in dimers]
    df["E_tot"] = (
        scale_factors[0]  * df["E_coul"] +
        scale_factors[1]  * df["E_pol"] +
        scale_factors[2]  * df["E_disp"] +
        scale_factors[3]  * df["E_rep"]
    )
    return df

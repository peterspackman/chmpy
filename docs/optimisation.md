# Optimisation with an ASE calculator

`chmpy.opt` relaxes molecules and crystals with a trust-region optimiser, and
can compute elastic constants and force constants on top. It does not need ASE,
but most machine-learned potentials ship as ASE calculators. You can use any of
them by wrapping it once:

``` python
from chmpy.calc import Calculator

# any ase.calculators.calculator.Calculator works here
from mace.calculators import mace_mp
calc = Calculator.from_ase(mace_mp(model="medium", default_dtype="float64"))
```

To try the examples below without installing a model, use ASE's built-in
effective-medium potential: `from ase.calculators.emt import EMT` and
`calc = Calculator.from_ase(EMT())`. EMT covers a handful of metals (Al, Cu,
Ag, Au, Ni, Pd, Pt) and gives only rough results for H, C, N and O.

For PET-MAD and the other [UPET](https://github.com/lab-cosmo/upet) models
there is also a native route that skips ASE entirely. It takes the same
`dtype` and `device` options:

``` python
from chmpy.calc.adapters.metatomic import upet

calc = upet("pet-mad-s", dtype="float64")   # or Calculator.from_ase(UPETCalculator(...))
```

Use `float64` if you plan to compute elastic constants or phonons: both
take differences of forces, and float32 noise is amplified by them.

The ASE wrapper keeps one `ase.Atoms` alive and writes each new geometry into it,
rather than copying a structure on every call. Results are cached, so asking
for the same geometry twice costs nothing.

## Molecules

``` python
from chmpy import Molecule
from chmpy.opt import relax

water = Molecule.load("water.xyz")
result = relax(water, calc, fmax=0.01)
result
# <Relaxation converged in 6 steps (7 evaluations), E=1.879225 eV, fmax=0.009192>

relaxed = result.structure   # a Molecule, like the input
relaxed.save("water_relaxed.xyz")
```

`fmax` is the largest force in eV/Å. `relax` takes the same arguments whatever
kind of structure you pass. `progress=True` prints one line per step (see
[Progress](#progress)), and `fixed=` takes a boolean mask of atoms to hold
still.

Some models need a charge or a spin as input. Pass them with `info`, and they
are copied into `atoms.info` on every call:

``` python
result = relax(water, calc, info={"charge": 0, "spin": 1})
```

## Crystals

By default a `Crystal` is relaxed **inside its space group**. The degrees of
freedom are the asymmetric unit plus the cell strains the point group allows,
so the symmetry holds exactly at every step rather than being restored
afterwards. This needs about `1/|G|` as many variables as a P1 relaxation, and
noisy forces from an MLIP cannot push the structure into a lower symmetry.

``` python
import numpy as np
from chmpy import Crystal
from chmpy.core import Element
from chmpy.crystal import AsymmetricUnit, SpaceGroup, UnitCell

copper = Crystal(
    UnitCell.from_lengths_and_angles((3.7, 3.7, 3.7), np.radians((90, 90, 90))),
    SpaceGroup(225),
    AsymmetricUnit([Element["Cu"]], np.array([[0.0, 0.0, 0.0]])),
)
result = relax(copper, calc, fmax=1e-4, smax=1e-3)
result
# <Relaxation converged in 3 steps (4 evaluations), E=-0.028146 eV, fmax=9.562e-15, smax=0.0006596>
result.structure
# <Crystal Cu Fm-3m>
result.structure.unit_cell.lengths
# [3.5898, 3.5898, 3.5898]
```

`smax` is the largest stress component in GPa. Forces and stress are checked
separately, each in its own units.

Structures loaded from files work the same way:

``` python
crystal = Crystal.load("acetic_acid.cif")

relax(crystal, calc)                     # atoms and cell, in Pna2_1
relax(crystal, calc, cell=False)         # atoms only, at fixed cell
relax(crystal, calc, pressure=1.0)       # minimise the enthalpy at 1 GPa
relax(crystal, calc, symmetry=False)     # P1: every atom, all six strains
relax(crystal, calc, stages="two-stage") # fixed cell first, then everything

result = relax(crystal, calc)
result.structure.save("acetic_acid_relaxed.cif")  # still in Pna2_1
```

If a structure has no freedoms at all, `relax` raises rather than spending a
calculation to find that out. Examples are fcc copper at fixed cell, or any
crystal whose atoms all sit on fixed special positions.

### Starting from an `ase.Atoms`

An `Atoms` object carries no space group, so it is relaxed in P1. Convert it to
a `Crystal` first if you want symmetry; alternatively, wrap it as a `System`:

``` python
from chmpy.calc import System

system = System.from_ase(atoms)
result = relax(system, calc)        # atoms, plus the cell if periodic
relaxed_atoms = result.structure.to_ase()
```

## Elastic constants and phonons

Both start from a relaxed structure and use its symmetry to reduce the number
of calculations:

``` python
from chmpy.opt import elastic_tensor
from chmpy.vib import force_constants

relaxed = relax(copper, calc, fmax=1e-4, smax=1e-3).structure

elastic = elastic_tensor(relaxed, calc)
elastic
# <ElasticResult 3 independent constants, 2 strains, 5 evaluations, ...>
elastic.c_voigt[:, 0]
# [172.6, 115.5, 115.5, 0, 0, 0]   GPa
elastic.tensor.is_stable()
# True

constants = force_constants(relaxed, calc, cutoff=6.0)
constants.frequencies([0.5, 0.5, 0.5], units="thz")
```

For a cubic crystal, `elastic_tensor` applies two strains rather than six.
The responses to the others are rotations of these two. At each strained
geometry, the atoms relax within the subgroup that the strain leaves intact.
`force_constants` displaces one atom per symmetry orbit, and only along the
directions its site symmetry does not already determine. Pass `symmetry=False`
to either function to do every calculation explicitly, which is a useful
cross-check.

For a noisy model, `elastic_tensor(..., strain="auto")` keeps increasing the
strain until the tensor's own noise estimate is acceptable. The result reports
`noise_fraction`, and it logs a warning if the tensor should not be trusted.

## Lattice energies

``` python
result = calc.lattice_energy(Crystal.load("acetic_acid.cif"))
```

This relaxes the crystal and each symmetry-unique molecule, then takes the
difference per molecule. See `chmpy.opt.lattice_energy`.

## Progress

`relax`, `elastic_tensor`, `force_constants` and `lattice_energy` all take a
`progress` argument. It is silent by default; `progress=True` prints:

``` python
calc.lattice_energy(crystal, progress=True)
# [1/2] relaxing the crystal (<Crystal C8H9NO2 P2_1/n>, Z=4)
#   relax <SymmetryAdapted P2_1/n: 60 atomic + 4 cell dof for 80 atoms>
#        1 +    E=   -485.71...  fmax=  0.29... smax=  0.41...  delta=...
#        ...
# [1/2] crystal: <Relaxation converged in 41 steps ...>
# [2/2] relaxing molecule 0 (C8H9NO2)
#   ...
```

Pass a callable to get a `chmpy.opt.progress.Progress` for every event
instead. Each event has `task`, `stage` and `message` fields, an `index` out
of a `total` when the count is known, and a `depth` with the `parents` it is
nested inside. Events from inside a relaxation also carry the optimiser's
`step`. Use these to drive a progress bar without parsing text:

``` python
from tqdm import tqdm

bar = None

def show(event):
    global bar
    if event.depth == 0 and event.total and not event.done:
        if bar is None:
            bar = tqdm(total=event.total)
        bar.set_description(event.stage)
        bar.update(1)

elastic_tensor(relaxed, calc, progress=show)
```

## Going the other way

A chmpy calculator can be handed to code that expects an ASE one:

``` python
from ase.optimize import BFGS

atoms = relaxed.to_ase_atoms()
atoms.calc = calc.as_ase()
BFGS(atoms).run(fmax=0.01)
```

## Complete examples

Two command-line scripts in `examples/` put this together with MACE-Polar:

* `examples/lattice_energy.py structure.cif [--json] [-d out/]` relaxes a
  molecular crystal and its molecules, and reports the lattice energy with
  its interaction and strain parts.
* `examples/elastic_tensor.py structure.cif [--strain auto] [--json] [-d out/]`
  relaxes the crystal, then computes the relaxed-atom elastic tensor, its
  averaged moduli, and the Young's modulus, linear compressibility and sound
  velocities along each cell axis.

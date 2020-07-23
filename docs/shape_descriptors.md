# Shape descriptors

## What they are


## How to calculate shape descriptors

### Molecules in crystals

``` python
from chmpy import Crystal
c = Crystal.load("tests/test_files/acetic_acid.cif")
desc = c.molecular_shape_descriptors()
```

### Atoms in crystals
Likewise, atomic shape descriptors can be conveniently
calculated directly from the `Crystal` object:

``` python
from chmpy import Crystal
c = Crystal.load("tests/test_files/acetic_acid.cif")
desc = c.atomic_shape_descriptors()
```

### Isolated molecules

Hirshfeld surfaces typically only have a sensible definition
in a crystal (or at least in a environment where the molecule
is not isolated). As such, the more sensible descriptor to
utilise may be one of the **Promolecule density isosurface**.

This can be readily calculated using the `Molecule` object:

``` python
from chmpy import Molecule
m = Molecule.load("tests/test_files/water.xyz")
desc = m.shape_descriptors()

# use EEM calculated charges to describe the shape and the ESP
# up to maximum angular momentum 16
desc_with_esp = m.shape_descriptors(l_max=16, with_property="esp")
```

However, another useful descriptor of atomic environments
is a Hirshfeld-type descriptor in a molecule, where in order to
'close' the exterior of the surface we introduce a `background`
density, as follows:

``` python
from chmpy import Molecule
m = Molecule.load("tests/test_files/water.xyz")
# with the default background density
desc = m.atomic_shape_descriptors()

# or with, a larger background density, contracting the atoms
desc = m.atomic_shape_descriptors(background=0.0001)
```

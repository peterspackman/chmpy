# Parallelization

`chmpy` is pure Python and gets its parallelism from threads. numpy and
scipy release the GIL while they work, so the threads run side by side.

## The `workers` argument

Functions that can use several cores take a `workers` argument, which means
the same thing it does in scipy: the number of threads to use, or `-1` for one
per CPU. `-1` is the default.

| Function | Parallel part |
| --- | --- |
| `chmpy.calc.nearest_neighbors` | the KD-tree search |
| `chmpy.descriptors.adaptive_cna`, `interval_cna`, `fixed_cna` | the neighbour search, then the analysis in chunks of atoms |
| `chmpy.descriptors.cna.neighbor_bonds`, `cna_signatures`, `classify` | chunks of atoms |

The answer does not depend on `workers`. Pass `workers=1` when you run several
analyses side by side yourself, for example one per trajectory frame across a
process pool, so that each one stays on a single core:

```python
from chmpy.descriptors import adaptive_cna

types = adaptive_cna(positions, cell, workers=1)
```

## BLAS threads

numpy hands matrix products to a BLAS library, which keeps its own pool of
threads. Most of `chmpy`'s products are too small for that to matter, but if
it competes with your own parallelism, limit it with the environment variable
for your BLAS, set before Python starts:

```bash
export OMP_NUM_THREADS=1         # OpenBLAS, as shipped in numpy's wheels
export VECLIB_MAXIMUM_THREADS=1  # Apple Accelerate
```

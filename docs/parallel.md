# Parallelization

Some of the `Cython` code in `chmpy` makes use of OpenMP
parallelism. If this is interfering with your own parallelism
at a higher level, or you simply wish to modify how many cores
the code should make use of, consider setting the environment variable
`OMP_NUM_THREADS` to the desired number of threads.

``` bash
export OMP_NUM_THREADS=1
```

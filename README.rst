shmolecule
-----------

A library for Hirshfeld & promolecule surfaces, spherical harmonic shape
descriptors and more...

Installation
^^^^^^^^^^^^

Basic installation can be done through the python package manager ``pip``::

    pip install cython numpy
    pip install git+https://github.com/peterspackman/shmolecule.git

If you wish to utilise the spherical harmonic shape descriptors, you will
need to install the wonderful SHTns library <https://nschaeff.bitbucket.io/shtns/>
library. Be sure to cite the SHTns library if you use shape descriptors in any
publication. There is no up to date package for shtns in pypi or conda, but on
unix systems the installation is straightforward.

Features
^^^^^^^^
While the library is intended to be flexible and make it easy to build
complex pipelines or properties, the following is a brief summary of 
intended features:

* Load crystal structures from ``.cif`` or ``.res`` files.
* Evaluate promolecule and procrystal electron densities.
* Easily generate Hirshfeld or promolecule isosurfaces and associated properties.
* Easily generate spherical harmonic shape descriptors for atoms, molecules, or molecular fragments.
* Efficiently calculate crystal slabs

It should also serve as a simple, easy to read library for learning
how to represent crystal structures, molecules etc. and evaluate
scientifically relevant information quickly and efficiently using
python.

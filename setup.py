import setuptools
from setuptools.extension import Extension as Ext
from numpy.distutils.core import setup, Extension as NumpyExt
from numpy import get_include
from Cython.Build import cythonize

ext_modules = [
    NumpyExt("shmolecule.linterp", sources=["shmolecule/linterp.c"], language="c")
]

ext_modules_cython = cythonize(
    [
        Ext(
            "shmolecule._density",
            sources=["shmolecule/_density.pyx"],
            include_dirs=[get_include()],
        ),
        Ext(
            "shmolecule._invariants",
            sources=["shmolecule/_invariants.pyx"],
            include_dirs=[get_include()],
        ),
        Ext(
            "shmolecule.mc._mc_lewiner",
            sources=["shmolecule/mc/_mc_lewiner.pyx"],
            include_dirs=[get_include()],
        ),
    ]
)

ext_modules += ext_modules_cython

setup(
    name="shmolecule",
    version="0.2a1",
    description="Promolecule and Hirshfeld surfaces using python",
    url="https://github.com/peterspackman/shmolecule",
    keywords=["chemistry", "molecule", "crystal", "electron density", "isosurface"],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=["shmolecule"],
    package_data={"shmolecule": ["*.npz"]},
    ext_modules=ext_modules,
    install_requires=["numpy", "scipy", "trimesh", "shtns"],
    zip_safe=True,
)

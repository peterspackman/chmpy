import setuptools
from numpy.distutils.core import setup, Extension as NumpyExt
from numpy import get_include

ext_modules = [
    NumpyExt(
        "promolecule.interp",
        sources=["promolecule/interp.f90"],
        language="f90",
    ),
] 

setup(
    name="promolecule",
    version="0.1a1",
    description="Promolecule and Hirshfeld surfaces using python",
    url="https://github.com/peterspackman/promolecule-python",
    keywords=[
        "chemistry",
        "molecule",
        "crystal",
        "electron density",
        "isosurface"
    ],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=["promolecule",],
    package_data={
        "promolecule": ["*.npz"],
    },
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "scipy",
    ],
    zip_safe=True,
)

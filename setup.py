import setuptools
from setuptools.extension import Extension as Ext
from numpy.distutils.core import setup, Extension as NumpyExt
from numpy import get_include
from Cython.Build import cythonize
import sys

_EXTRA_COMPILE_ARGS = ["-fopenmp"]
_EXTRA_LINK_ARGS = ["-fopenmp"]

if sys.platform == "darwin":
    _EXTRA_COMPILE_ARGS = []
    _EXTRA_LINK_ARGS = []

ext_modules = [
    NumpyExt(
        "chmpy.interpolate.linterp",
        sources=["chmpy/interpolate/linterp.c"],
        language="c",
    )
]

ext_modules_cython = cythonize(
    [
        Ext(
            "chmpy.interpolate._density",
            sources=["chmpy/interpolate/_density.pyx"],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.shape._invariants",
            sources=["chmpy/shape/_invariants.pyx"],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.crystal.sfac._sfac",
            sources=["chmpy/crystal/sfac/_sfac.pyx"],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.mc._mc_lewiner",
            sources=["chmpy/mc/_mc_lewiner.pyx"],
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.sampling._lds",
            sources=["chmpy/sampling/_lds.pyx"],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.sampling._sobol",
            sources=["chmpy/sampling/_sobol.pyx"],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            include_dirs=[get_include()],
        ),
    ]
)

ext_modules += ext_modules_cython

setup(
    name="chmpy",
    version="1.0a2",
    description="Molecules, crystals, promolecule and Hirshfeld surfaces using python",
    url="https://github.com/peterspackman/chmpy",
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
    packages=["chmpy"],
    package_data={
        "chmpy.ints": ["*.npz"],
        "chmpy.interpolate": ["*.npz"],
        "chmpy.subgraphs": ["*.gt"],
        "chmpy.templates": ["*.jinja2"],
        "chmpy.subgraphs": ["*.gt"],
        "chmpy.sampling": ["*.npz"],
    },
    ext_modules=ext_modules,
    entry_points={"console_scripts": ["chmpy-convert= chmpy.cmd.convert:main",]},
    install_requires=["numpy", "scipy", "trimesh", "matplotlib", "seaborn"],
    extras_require={"sht": ["shtns"], "graph": ["graph_tool"],},
    zip_safe=True,
)

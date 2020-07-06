import setuptools
from setuptools.extension import Extension as Ext
from numpy.distutils.core import setup, Extension as NumpyExt
from numpy import get_include
from Cython.Build import cythonize

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
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.shape._invariants",
            sources=["chmpy/shape/_invariants.pyx"],
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.crystal.sfac._sfac",
            sources=["chmpy/crystal/sfac/_sfac.pyx"],
            language="c++",
            include_dirs=[get_include()],
        ),
        Ext(
            "chmpy.mc._mc_lewiner",
            sources=["chmpy/mc/_mc_lewiner.pyx"],
            include_dirs=[get_include()],
        ),
    ]
)

ext_modules += ext_modules_cython

setup(
    name="chmpy",
    version="1.0a1",
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
    },
    ext_modules=ext_modules,
    entry_points={"console_scripts": ["chmpy-convert= chmpy.cmd.convert:main",]},
    install_requires=["numpy", "scipy", "trimesh", "matplotlib"],
    extras_require={"sht": ["shtns"], "graph": ["graph_tool"],},
    zip_safe=True,
)

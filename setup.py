from setuptools import Extension, setup

extension_modules = [
    Extension(
        "chmpy.interpolate._linterp",
        sources=["chmpy/interpolate/_linterp.c"],
        language="c",
    ),
    Extension(
        "chmpy.interpolate._density",
        sources=["chmpy/interpolate/_density.pyx"],
    ),
    Extension(
        "chmpy.shape._invariants",
        sources=["chmpy/shape/_invariants.pyx"],
    ),
    Extension(
        "chmpy.shape._sht",
        sources=["chmpy/shape/_sht.pyx"],
    ),
    Extension(
        "chmpy.crystal.sfac._sfac",
        sources=["chmpy/crystal/sfac/_sfac.pyx"],
    ),
    Extension(
        "chmpy.mc._mc_lewiner",
        sources=["chmpy/mc/_mc_lewiner.pyx"],
    ),
    Extension(
        "chmpy.sampling._lds",
        sources=["chmpy/sampling/_lds.pyx"],
    ),
    Extension(
        "chmpy.sampling._sobol",
        sources=["chmpy/sampling/_sobol.pyx"],
    ),
]

setup(
    ext_modules=extension_modules,
)

from setuptools import Extension, setup
import numpy

np_defines = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
np_includes = [numpy.get_include()]

extension_modules = [
    Extension(
        "chmpy.interpolate._density",
        sources=["src/chmpy/interpolate/_density.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
    Extension(
        "chmpy.shape._invariants",
        sources=["src/chmpy/shape/_invariants.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
    Extension(
        "chmpy.shape._sht",
        sources=["src/chmpy/shape/_sht.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
    Extension(
        "chmpy.crystal.sfac._sfac",
        sources=["src/chmpy/crystal/sfac/_sfac.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
    Extension(
        "chmpy.mc._mc_lewiner",
        sources=["src/chmpy/mc/_mc_lewiner.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
    Extension(
        "chmpy.sampling._lds",
        sources=["src/chmpy/sampling/_lds.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
    Extension(
        "chmpy.sampling._sobol",
        sources=["src/chmpy/sampling/_sobol.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
]

setup(
    ext_modules=extension_modules,
)

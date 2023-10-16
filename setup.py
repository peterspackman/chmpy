from setuptools import Extension, setup

np_defines = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

extension_modules = [
    Extension(
        "chmpy.interpolate._density",
        sources=["chmpy/interpolate/_density.pyx"],
        define_macros=np_defines,
    ),
    Extension(
        "chmpy.shape._invariants",
        sources=["chmpy/shape/_invariants.pyx"],
        define_macros=np_defines,
    ),
    Extension(
        "chmpy.shape._sht",
        sources=["chmpy/shape/_sht.pyx"],
        define_macros=np_defines,
    ),
    Extension(
        "chmpy.crystal.sfac._sfac",
        sources=["chmpy/crystal/sfac/_sfac.pyx"],
        define_macros=np_defines,
    ),
    Extension(
        "chmpy.mc._mc_lewiner",
        sources=["chmpy/mc/_mc_lewiner.pyx"],
        define_macros=np_defines,
    ),
    Extension(
        "chmpy.sampling._lds",
        sources=["chmpy/sampling/_lds.pyx"],
        define_macros=np_defines,
    ),
    Extension(
        "chmpy.sampling._sobol",
        sources=["chmpy/sampling/_sobol.pyx"],
        define_macros=np_defines,
    ),
]

setup(
    ext_modules=extension_modules,
)

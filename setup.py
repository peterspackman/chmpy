import numpy
from setuptools import Extension, setup

np_defines = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
np_includes = [numpy.get_include()]

extension_modules = [
    Extension(
        "chmpy.interpolate._density",
        sources=["src/chmpy/interpolate/_density.pyx"],
        define_macros=np_defines,
        include_dirs=np_includes,
    ),
]

setup(
    ext_modules=extension_modules,
)

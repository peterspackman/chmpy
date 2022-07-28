import setuptools
from setuptools import Extension, Distribution
from typing import List
from Cython.Build import cythonize
from Cython.Distutils.build_ext import new_build_ext as cython_build
from numpy import get_include
from pathlib import Path
import sys

BUILD_DIR = Path("cython_build")

_EXTRA_COMPILE_ARGS = ["-fopenmp"]
_EXTRA_LINK_ARGS = ["-fopenmp"]

if sys.platform == "darwin":
    _EXTRA_COMPILE_ARGS = []
    _EXTRA_LINK_ARGS = []

def get_extension_modules() -> List[Extension]:
    ext_modules = [
        Extension(
            "chmpy.interpolate._linterp",
            sources=["chmpy/interpolate/_linterp.c"],
            extra_compile_args=_EXTRA_COMPILE_ARGS,
            extra_link_args=_EXTRA_LINK_ARGS,
            language="c",
            include_dirs=[get_include()],
        )
    ]

    ext_modules_cython = cythonize(
        [
            Extension(
                "chmpy.interpolate._density",
                sources=["chmpy/interpolate/_density.pyx"],
                extra_compile_args=_EXTRA_COMPILE_ARGS,
                extra_link_args=_EXTRA_LINK_ARGS,
                include_dirs=[get_include()],
            ),
            Extension(
                "chmpy.shape._invariants",
                sources=["chmpy/shape/_invariants.pyx"],
                extra_compile_args=_EXTRA_COMPILE_ARGS,
                extra_link_args=_EXTRA_LINK_ARGS,
                include_dirs=[get_include()],
            ),
            Extension(
                "chmpy.shape._sht",
                sources=["chmpy/shape/_sht.pyx"],
                extra_compile_args=_EXTRA_COMPILE_ARGS,
                extra_link_args=_EXTRA_LINK_ARGS,
                include_dirs=[get_include()],
            ),
            Extension(
                "chmpy.crystal.sfac._sfac",
                sources=["chmpy/crystal/sfac/_sfac.pyx"],
                extra_compile_args=_EXTRA_COMPILE_ARGS,
                extra_link_args=_EXTRA_LINK_ARGS,
                include_dirs=[get_include()],
            ),
            Extension(
                "chmpy.mc._mc_lewiner",
                sources=["chmpy/mc/_mc_lewiner.pyx"],
                include_dirs=[get_include()],
            ),
            Extension(
                "chmpy.sampling._lds",
                sources=["chmpy/sampling/_lds.pyx"],
                extra_compile_args=_EXTRA_COMPILE_ARGS,
                extra_link_args=_EXTRA_LINK_ARGS,
                include_dirs=[get_include()],
            ),
            Extension(
                "chmpy.sampling._sobol",
                sources=["chmpy/sampling/_sobol.pyx"],
                extra_compile_args=_EXTRA_COMPILE_ARGS,
                extra_link_args=_EXTRA_LINK_ARGS,
                include_dirs=[get_include()],
            ),
        ]
    )

    ext_modules += ext_modules_cython
    return ext_modules

def cythonize_helper(extension_modules: List[Extension]) -> List[Extension]:
    """Cythonize all Python extensions"""
    import multiprocessing

    return cythonize(
        module_list=extension_modules,

        # Don't build in source tree (this leaves behind .c files)
        build_dir=BUILD_DIR,

        # Don't generate an .html output file. Would contain source.
        annotate=False,

        # Parallelize our build
        nthreads=multiprocessing.cpu_count() * 2,

        # Tell Cython we're using Python 3. Becomes default in Cython 3
        compiler_directives={"language_level": "3"},

        # (Optional) Always rebuild, even if files untouched
        force=True,
    )

extension_modules = cythonize_helper(get_extension_modules())
distribution = Distribution({
    "ext_modules": extension_modules,
    "cmdclass": {
        "build_ext": cython_build,
    },
})

# Grab the build_ext command and copy all files back to source dir.
# Done so Poetry grabs the files during the next step in its build.
distribution.run_command("build_ext")
build_ext_cmd = distribution.get_command_obj("build_ext")
build_ext_cmd.copy_extensions_to_source()

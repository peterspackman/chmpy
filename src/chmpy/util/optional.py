"""Optional dependencies, and saying clearly what to install when one is missing.

Plotting and mesh handling are a small part of what chmpy does, but they carry
most of its install weight, so they are optional extras rather than hard
requirements. Everything that needs them imports through here, which keeps the
failure a sentence telling you what to install rather than a bare
`ModuleNotFoundError` raised somewhere deep in a call stack.
"""

from __future__ import annotations

import importlib

#: which extra provides each optional module
EXTRAS = {
    "matplotlib": "plots",
    "trimesh": "mesh",
    "ase": "ase",
}


def require(module, purpose=None):
    """Import an optional module, or explain how to install it.

    Args:
        module: the module to import, e.g. "matplotlib.pyplot"
        purpose: what the caller wanted it for, used in the error message

    Returns:
        the imported module

    Raises:
        ImportError: with the pip command that would fix it
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        top = module.split(".")[0]
        extra = EXTRAS.get(top)
        wanted = f"{purpose} needs" if purpose else "this needs"
        if extra:
            detail = (
                f"an optional dependency of chmpy. "
                f"Install it with:  pip install 'chmpy[{extra}]'"
            )
        else:
            # no extra provides it, so naming one would send the reader off to
            # install something that does not exist
            detail = f"which chmpy does not depend on. Install it with:  pip install {top}"
        raise ImportError(f"{wanted} {module}, {detail}") from exc


def pyplot(purpose="plotting"):
    """`matplotlib.pyplot`, or a message explaining how to install it."""
    return require("matplotlib.pyplot", purpose)


def trimesh(purpose="mesh handling"):
    """`trimesh`, or a message explaining how to install it."""
    return require("trimesh", purpose)


def have(module):
    """Whether an optional module can be imported, without raising."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True

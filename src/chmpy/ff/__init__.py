"""
Force field parameter assignment module.
"""

from .params import (
    assign_fit_lj_type,
    assign_uff_type_from_coordination,
    crystal_lj_params,
    crystal_uff_params,
    get_lj_parameters,
    get_uff_parameters,
    load_lj_params,
    molecule_lj_params,
    molecule_uff_params,
    print_lj_summary,
    print_uff_summary,
)

__all__ = [
    # New primary functions
    "get_lj_parameters",
    "print_lj_summary",
    "crystal_lj_params",
    "molecule_lj_params",
    "assign_fit_lj_type",
    # Backwards compatibility
    "get_uff_parameters",
    "assign_uff_type_from_coordination",
    "print_uff_summary",
    "crystal_uff_params",
    "molecule_uff_params",
    # Utilities
    "load_lj_params",
]

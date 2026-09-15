"""
HADES
=====

High-throughput Analysis for the Design of Energetic Systems.
"""

# Provide convenient module aliases
from . import isodesmic as eof
from . import det_v_p as detonation

# Functions that currently exist
from .oxygen_balance import (
    atom_counts_from_smiles,
    oxygen_balance,
)

from .isodesmic import (
    calculate_hf,
    calculate_hr,
    filter_and_average,
)

from .det_v_p import (
    calc_gas_products,
    calc_phi,
    det_p,
    det_v,
)

from .properties import calculate_properties

__version__ = "1.0.0"

__all__ = [
    # Existing entries...
    "eof",
    "detonation",
    "atom_counts_from_smiles",
    "oxygen_balance",
    "calculate_hf",
    "calculate_hr",
    "filter_and_average",
    "calc_gas_products",
    "calc_phi",
    "det_v",
    "det_p",

    # High-level API
    "calculate_properties",
]

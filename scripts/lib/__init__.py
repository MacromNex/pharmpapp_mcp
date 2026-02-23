# scripts/lib/__init__.py
"""
Shared library for PharmPapp MCP scripts.

This package provides common utilities for cyclic peptide analysis.
"""

__version__ = "1.0.0"
__author__ = "PharmPapp MCP Implementation"

from .io import load_molecules, save_output
from .molecules import (
    parse_smiles,
    validate_cyclic_peptide,
    generate_3d_conformer,
    save_molecule
)
from .validation import (
    validate_input_file,
    check_required_columns,
    validate_smiles_list
)

__all__ = [
    'load_molecules',
    'save_output',
    'parse_smiles',
    'validate_cyclic_peptide',
    'generate_3d_conformer',
    'save_molecule',
    'validate_input_file',
    'check_required_columns',
    'validate_smiles_list'
]
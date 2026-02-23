"""
Input validation utilities for PharmPapp MCP scripts.

Functions to validate inputs and check data quality.
"""

from pathlib import Path
from typing import Union, Optional, List, Tuple
import pandas as pd
from rdkit import Chem


def validate_input_file(file_path: Union[str, Path], required_format: Optional[str] = None) -> bool:
    """
    Validate that input file exists and has correct format.

    Args:
        file_path: Path to file
        required_format: Expected file extension (optional)

    Returns:
        True if file is valid

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is incorrect
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if required_format:
        if not file_path.suffix.lower() == required_format.lower():
            raise ValueError(f"Expected {required_format} file, got {file_path.suffix}")

    return True


def check_required_columns(data: pd.DataFrame, required_columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check if DataFrame has required columns.

    Args:
        data: Input DataFrame
        required_columns: List of required column names

    Returns:
        Tuple of (found_columns, missing_columns)
    """
    available_columns = list(data.columns)
    found_columns = [col for col in required_columns if col in available_columns]
    missing_columns = [col for col in required_columns if col not in available_columns]

    return found_columns, missing_columns


def validate_smiles_list(smiles_list: List[str], strict: bool = False) -> Tuple[List[str], List[str]]:
    """
    Validate a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        strict: If True, raise error on any invalid SMILES

    Returns:
        Tuple of (valid_smiles, invalid_smiles)

    Raises:
        ValueError: If strict=True and any SMILES is invalid
    """
    valid_smiles = []
    invalid_smiles = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            invalid_smiles.append(smiles)

    if strict and invalid_smiles:
        raise ValueError(f"Invalid SMILES found: {invalid_smiles[:5]}{'...' if len(invalid_smiles) > 5 else ''}")

    return valid_smiles, invalid_smiles


def validate_descriptor_data(data: pd.DataFrame, min_descriptors: int = 10) -> dict:
    """
    Validate molecular descriptor data.

    Args:
        data: DataFrame with descriptor data
        min_descriptors: Minimum number of numeric descriptors required

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'numeric_columns': [],
        'non_numeric_columns': [],
        'missing_data_columns': [],
        'constant_columns': []
    }

    # Check for numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns.tolist()

    results['numeric_columns'] = numeric_columns
    results['non_numeric_columns'] = non_numeric_columns

    if len(numeric_columns) < min_descriptors:
        results['valid'] = False
        results['issues'].append(f"Insufficient numeric descriptors: {len(numeric_columns)} < {min_descriptors}")

    # Check for missing data
    missing_data_columns = []
    for col in numeric_columns:
        if data[col].isnull().any():
            missing_data_columns.append(col)

    results['missing_data_columns'] = missing_data_columns

    if missing_data_columns:
        results['issues'].append(f"Columns with missing data: {len(missing_data_columns)}")

    # Check for constant columns
    constant_columns = []
    for col in numeric_columns:
        if data[col].nunique() <= 1:
            constant_columns.append(col)

    results['constant_columns'] = constant_columns

    if constant_columns:
        results['issues'].append(f"Constant columns found: {len(constant_columns)}")

    return results


def validate_prediction_input(data: pd.DataFrame, model_descriptors: List[str]) -> dict:
    """
    Validate that input data is suitable for model prediction.

    Args:
        data: Input DataFrame
        model_descriptors: List of descriptors required by model

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'available_descriptors': [],
        'missing_descriptors': [],
        'coverage': 0.0
    }

    # Check descriptor availability
    available_descriptors = [desc for desc in model_descriptors if desc in data.columns]
    missing_descriptors = [desc for desc in model_descriptors if desc not in data.columns]

    results['available_descriptors'] = available_descriptors
    results['missing_descriptors'] = missing_descriptors
    results['coverage'] = len(available_descriptors) / len(model_descriptors)

    if results['coverage'] < 0.8:  # Require at least 80% descriptor coverage
        results['valid'] = False
        results['issues'].append(f"Low descriptor coverage: {results['coverage']:.1%}")

    # Check data quality for available descriptors
    descriptor_data = data[available_descriptors]

    # Check for missing values
    missing_counts = descriptor_data.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0].index.tolist()

    if columns_with_missing:
        results['issues'].append(f"Missing values in {len(columns_with_missing)} descriptor columns")

    # Check for infinite values
    infinite_counts = descriptor_data.replace([float('inf'), float('-inf')], float('nan')).isnull().sum() - missing_counts
    columns_with_infinite = infinite_counts[infinite_counts > 0].index.tolist()

    if columns_with_infinite:
        results['issues'].append(f"Infinite values in {len(columns_with_infinite)} descriptor columns")

    return results


def suggest_data_fixes(validation_results: dict) -> List[str]:
    """
    Suggest fixes for data validation issues.

    Args:
        validation_results: Results from validation functions

    Returns:
        List of suggested fixes
    """
    suggestions = []

    if 'missing_descriptors' in validation_results and validation_results['missing_descriptors']:
        suggestions.append("Calculate missing descriptors using RDKit or remove incompatible models")

    if 'missing_data_columns' in validation_results and validation_results['missing_data_columns']:
        suggestions.append("Fill missing values with mean/median or remove rows with missing data")

    if 'constant_columns' in validation_results and validation_results['constant_columns']:
        suggestions.append("Remove constant columns as they provide no information")

    if 'coverage' in validation_results and validation_results['coverage'] < 0.8:
        suggestions.append("Use a different model or calculate additional required descriptors")

    if not validation_results.get('valid', True):
        suggestions.append("Review data preprocessing and descriptor calculation steps")

    return suggestions
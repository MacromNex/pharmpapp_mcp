"""
I/O utilities for PharmPapp MCP scripts.

Simplified file loading and saving functions extracted from use cases.
"""

from pathlib import Path
from typing import Union, Optional, List, Any
import pandas as pd
from rdkit import Chem


def load_molecules(input_file: Union[str, Path], file_format: str = 'auto') -> List[Chem.Mol]:
    """
    Load molecules from input file.

    Args:
        input_file: Path to input file
        file_format: File format ('auto', '.smi', '.sdf', '.mol')

    Returns:
        List of RDKit molecules

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is unsupported
    """
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if file_format == 'auto':
        file_format = input_file.suffix.lower()

    molecules = []

    if file_format in ['.smi', '.smiles']:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    smiles = parts[0]
                    mol_id = parts[1] if len(parts) > 1 else f"mol_{line_num}"

                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        mol.SetProp("_Name", mol_id)
                        mol.SetProp("SMILES", smiles)
                        molecules.append(mol)

    elif file_format in ['.sdf', '.mol']:
        supplier = Chem.SDMolSupplier(str(input_file))
        for i, mol in enumerate(supplier):
            if mol:
                if not mol.HasProp("_Name"):
                    mol.SetProp("_Name", f"mol_{i+1}")
                molecules.append(mol)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return molecules


def save_output(data: Any, file_path: Union[str, Path], format_type: str = 'auto') -> None:
    """
    Save output data to file.

    Args:
        data: Data to save (DataFrame, list, dict)
        file_path: Output file path
        format_type: Output format ('auto', 'csv', 'excel', 'json')

    Raises:
        ValueError: If format is unsupported
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type == 'auto':
        format_type = file_path.suffix.lower()

    if isinstance(data, pd.DataFrame):
        if format_type in ['.csv', 'csv']:
            data.to_csv(file_path, index=False)
        elif format_type in ['.xlsx', '.xls', 'excel']:
            data.to_excel(file_path, index=False)
        elif format_type in ['.json', 'json']:
            data.to_json(file_path, indent=2)
        else:
            # Default to CSV for DataFrames
            data.to_csv(file_path, index=False)

    elif isinstance(data, (list, dict)):
        import json
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    else:
        # Try to convert to string and save
        with open(file_path, 'w') as f:
            f.write(str(data))


def load_data_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.

    Args:
        file_path: Path to data file

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def create_demo_smiles(output_path: Union[str, Path]) -> Path:
    """
    Create demo SMILES file with cyclic peptides.

    Args:
        output_path: Path where to save demo file

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    demo_smiles = [
        "C1C[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC3=CC=C(C=C3)O)CC(C)C\tcyclic_peptide_1",
        "C[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O)CC4=CNC5=CC=CC=C54\tcyclic_peptide_2",
        "CC[C@H](C)[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)C)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O\tcyclic_peptide_3"
    ]

    with open(output_path, 'w') as f:
        f.write('\n'.join(demo_smiles))

    return output_path
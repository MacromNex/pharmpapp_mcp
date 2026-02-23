"""
Molecular manipulation utilities for PharmPapp MCP scripts.

RDKit-based functions for working with cyclic peptides.
"""

from pathlib import Path
from typing import Union, Optional, List
from rdkit import Chem
from rdkit.Chem import AllChem


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse SMILES string to RDKit molecule.

    Args:
        smiles: SMILES string

    Returns:
        RDKit molecule or None if parsing failed
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol


def validate_cyclic_peptide(mol: Chem.Mol) -> bool:
    """
    Check if molecule is a cyclic peptide.

    Args:
        mol: RDKit molecule

    Returns:
        True if molecule appears to be a cyclic peptide
    """
    if mol is None:
        return False

    # Check for ring structure
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return False

    # Check for peptide bonds (C-N bonds)
    has_amide_bonds = False
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        # Look for amide bonds: C(=O)-N
        if (begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N'):
            # Check if carbon is part of carbonyl
            for neighbor in begin_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(
                    begin_atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                    has_amide_bonds = True
                    break

        if has_amide_bonds:
            break

    return has_amide_bonds


def generate_3d_conformer(mol: Chem.Mol, num_conformers: int = 1) -> Chem.Mol:
    """
    Generate 3D conformer(s) for a molecule.

    Args:
        mol: RDKit molecule
        num_conformers: Number of conformers to generate

    Returns:
        Molecule with 3D conformer(s)
    """
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Embed conformers
    result = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=42)

    if result == -1:
        # Fallback to basic embedding
        AllChem.EmbedMolecule(mol, randomSeed=42)

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except:
        # Fallback to UFF if MMFF fails
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol)
        except:
            pass  # Continue without optimization

    return mol


def save_molecule(mol: Chem.Mol, file_path: Union[str, Path], format_type: str = "pdb") -> None:
    """
    Save molecule to file in specified format.

    Args:
        mol: RDKit molecule
        file_path: Output file path
        format_type: Output format ('pdb', 'sdf', 'smi')

    Raises:
        ValueError: If format is unsupported
    """
    if mol is None:
        raise ValueError("Cannot save None molecule")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    format_type = format_type.lower()

    if format_type == "pdb":
        pdb_block = Chem.MolToPDBBlock(mol)
        with open(file_path, 'w') as f:
            f.write(pdb_block)

    elif format_type == "sdf":
        writer = Chem.SDWriter(str(file_path))
        writer.write(mol)
        writer.close()

    elif format_type in ["smi", "smiles"]:
        smiles = Chem.MolToSmiles(mol)
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "molecule"
        with open(file_path, 'w') as f:
            f.write(f"{smiles}\t{mol_name}\n")

    else:
        raise ValueError(f"Unsupported format: {format_type}")


def calculate_basic_properties(mol: Chem.Mol) -> dict:
    """
    Calculate basic molecular properties.

    Args:
        mol: RDKit molecule

    Returns:
        Dictionary of basic molecular properties
    """
    if mol is None:
        return {}

    from rdkit.Chem import Descriptors

    properties = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_hbd': Descriptors.NumHDonors(mol),
        'num_hba': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_rings': Descriptors.RingCount(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
    }

    return properties


def standardize_molecule(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Standardize molecule (remove salts, neutralize, etc.).

    Args:
        mol: RDKit molecule

    Returns:
        Standardized molecule or None
    """
    if mol is None:
        return None

    try:
        # Remove salts (take largest fragment)
        mol = Chem.rdMolStandardize.FragmentParent(mol)

        # Neutralize charges
        neutralizer = Chem.rdMolStandardize.Uncharger()
        mol = neutralizer.uncharge(mol)

        # Normalize
        normalizer = Chem.rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        return mol

    except:
        # Return original molecule if standardization fails
        return mol
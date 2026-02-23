#!/usr/bin/env python3
"""
Script: calculate_descriptors.py
Description: Calculate molecular descriptors for cyclic peptides using RDKit

Original Use Case: examples/use_case_2_calculate_descriptors.py
Dependencies Removed: None (already minimal)

Usage:
    python scripts/calculate_descriptors.py --input <input_file> --output <output_file>

Example:
    python scripts/calculate_descriptors.py --input examples/data/sample.smi --output results/descriptors.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import logging
import warnings

# Essential scientific packages
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, GraphDescriptors
from rdkit.Chem import Fragments, rdmolops
from rdkit.Chem.QED import qed
from rdkit.Chem.FilterCatalog import *

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "include_3d": False,
    "include_fingerprints": False,
    "max_conformers": 1,
    "optimize_geometry": False,
    "descriptor_types": ["basic", "topological", "constitutional", "physicochemical"]
}

# ==============================================================================
# Core Class (extracted and simplified from use case)
# ==============================================================================
class CyclicPeptideDescriptorCalculator:
    """
    Calculate molecular descriptors for cyclic peptides using RDKit.

    Simplified from examples/use_case_2_calculate_descriptors.py
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize descriptor calculator."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self._init_filters()

    def _init_filters(self):
        """Initialize rule-based descriptor filters."""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.filter_catalog = FilterCatalog(params)
        except Exception:
            self.filter_catalog = None
            self.logger.warning("Could not initialize filter catalog")

    def parse_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Parse SMILES string to RDKit molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.logger.warning(f"Could not parse SMILES: {smiles}")
            return None
        return mol

    def validate_cyclic_peptide(self, mol: Chem.Mol) -> bool:
        """Validate that molecule is a cyclic peptide."""
        # Check for ring structure
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            return False

        # Check for peptide bonds (C-N bonds)
        for bond in mol.GetBonds():
            if (bond.GetBeginAtom().GetSymbol() == 'C' and
                bond.GetEndAtom().GetSymbol() == 'N'):
                return True
        return False

    def load_molecules(self, input_file: Union[str, Path], file_format: str = 'auto') -> List[Chem.Mol]:
        """Load molecules from input file."""
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

                        mol = self.parse_smiles(smiles)
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

        self.logger.info(f"Loaded {len(molecules)} molecules from {input_file}")
        return molecules

    def calculate_descriptors(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Calculate comprehensive molecular descriptors."""
        descriptors = {}

        # Basic molecular properties
        descriptors.update({
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHBD': Descriptors.NumHDonors(mol),
            'NumHBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'RingCount': Descriptors.RingCount(mol),
            # 'FractionCsp3': Descriptors.FractionCsp3(mol),  # Not available in this RDKit version
            'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'NumAtoms': mol.GetNumAtoms(),
            'NumBonds': mol.GetNumBonds(),
        })

        # Constitutional descriptors
        descriptors.update({
            'MolMR': Descriptors.MolMR(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
            'Chi0v': Descriptors.Chi0v(mol),
            'Chi1v': Descriptors.Chi1v(mol),
            'Chi2v': Descriptors.Chi2v(mol),
            'Chi3v': Descriptors.Chi3v(mol),
            'Chi4v': Descriptors.Chi4v(mol),
            'Chi0n': Descriptors.Chi0n(mol),
            'Chi1n': Descriptors.Chi1n(mol),
            'Chi2n': Descriptors.Chi2n(mol),
            'Chi3n': Descriptors.Chi3n(mol),
            'Chi4n': Descriptors.Chi4n(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'Kappa3': Descriptors.Kappa3(mol),
        })

        # Topological descriptors
        descriptors.update({
            'BertzCT': Descriptors.BertzCT(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'Ipc': Descriptors.Ipc(mol),
        })

        # Electronic descriptors
        try:
            descriptors.update({
                'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
                'MinPartialCharge': Descriptors.MinPartialCharge(mol),
                'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol),
                'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(mol),
            })
        except:
            # Skip if partial charges can't be calculated
            pass

        # Surface area descriptors
        descriptors.update({
            'LabuteASA': Descriptors.LabuteASA(mol),
            'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
            'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
            'PEOE_VSA3': Descriptors.PEOE_VSA3(mol),
            'PEOE_VSA4': Descriptors.PEOE_VSA4(mol),
            'PEOE_VSA5': Descriptors.PEOE_VSA5(mol),
            'PEOE_VSA6': Descriptors.PEOE_VSA6(mol),
            'PEOE_VSA7': Descriptors.PEOE_VSA7(mol),
            'PEOE_VSA8': Descriptors.PEOE_VSA8(mol),
            'PEOE_VSA9': Descriptors.PEOE_VSA9(mol),
            'PEOE_VSA10': Descriptors.PEOE_VSA10(mol),
            'PEOE_VSA11': Descriptors.PEOE_VSA11(mol),
            'PEOE_VSA12': Descriptors.PEOE_VSA12(mol),
            'PEOE_VSA13': Descriptors.PEOE_VSA13(mol),
            'PEOE_VSA14': Descriptors.PEOE_VSA14(mol),
        })

        # MOE-like descriptors
        descriptors.update({
            'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
            'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
            'SlogP_VSA3': Descriptors.SlogP_VSA3(mol),
            'SlogP_VSA4': Descriptors.SlogP_VSA4(mol),
            'SlogP_VSA5': Descriptors.SlogP_VSA5(mol),
            'SlogP_VSA6': Descriptors.SlogP_VSA6(mol),
            'SlogP_VSA7': Descriptors.SlogP_VSA7(mol),
            'SlogP_VSA8': Descriptors.SlogP_VSA8(mol),
            'SlogP_VSA9': Descriptors.SlogP_VSA9(mol),
            'SlogP_VSA10': Descriptors.SlogP_VSA10(mol),
            'SlogP_VSA11': Descriptors.SlogP_VSA11(mol),
            'SlogP_VSA12': Descriptors.SlogP_VSA12(mol),
        })

        # Fragment counts
        descriptors.update({
            'fr_NH0': Fragments.fr_NH0(mol),
            'fr_NH1': Fragments.fr_NH1(mol),
            'fr_NH2': Fragments.fr_NH2(mol),
            'fr_Ar_N': Fragments.fr_Ar_N(mol),
            'fr_Ar_NH': Fragments.fr_Ar_NH(mol),
            'fr_Ar_OH': Fragments.fr_Ar_OH(mol),
            'fr_ArN': Fragments.fr_ArN(mol),
            'fr_C_O': Fragments.fr_C_O(mol),
            'fr_C_O_noCOO': Fragments.fr_C_O_noCOO(mol),
            'fr_COO': Fragments.fr_COO(mol),
            'fr_COO2': Fragments.fr_COO2(mol),
            'fr_N_O': Fragments.fr_N_O(mol),
            'fr_Ndealkylation1': Fragments.fr_Ndealkylation1(mol),
            'fr_Ndealkylation2': Fragments.fr_Ndealkylation2(mol),
            'fr_Nhpyrrole': Fragments.fr_Nhpyrrole(mol),
            'fr_SH': Fragments.fr_SH(mol),
            'fr_aldehyde': Fragments.fr_aldehyde(mol),
            'fr_alkyl_carbamate': Fragments.fr_alkyl_carbamate(mol),
            'fr_alkyl_halide': Fragments.fr_alkyl_halide(mol),
            'fr_allylic_oxid': Fragments.fr_allylic_oxid(mol),
            'fr_amide': Fragments.fr_amide(mol),
            'fr_amidine': Fragments.fr_amidine(mol),
            'fr_aniline': Fragments.fr_aniline(mol),
            'fr_aryl_methyl': Fragments.fr_aryl_methyl(mol),
            'fr_azide': Fragments.fr_azide(mol),
            'fr_azo': Fragments.fr_azo(mol),
            'fr_barbitur': Fragments.fr_barbitur(mol),
            'fr_benzene': Fragments.fr_benzene(mol),
            'fr_benzodiazepine': Fragments.fr_benzodiazepine(mol),
            'fr_bicyclic': Fragments.fr_bicyclic(mol),
            'fr_diazo': Fragments.fr_diazo(mol),
            'fr_dihydropyridine': Fragments.fr_dihydropyridine(mol),
            'fr_epoxide': Fragments.fr_epoxide(mol),
            'fr_ester': Fragments.fr_ester(mol),
            'fr_ether': Fragments.fr_ether(mol),
            'fr_furan': Fragments.fr_furan(mol),
            'fr_guanido': Fragments.fr_guanido(mol),
            'fr_halogen': Fragments.fr_halogen(mol),
            'fr_hdrzine': Fragments.fr_hdrzine(mol),
            'fr_hdrzone': Fragments.fr_hdrzone(mol),
            'fr_imidazole': Fragments.fr_imidazole(mol),
            'fr_imide': Fragments.fr_imide(mol),
            'fr_isocyan': Fragments.fr_isocyan(mol),
            'fr_isothiocyan': Fragments.fr_isothiocyan(mol),
            'fr_ketone': Fragments.fr_ketone(mol),
            'fr_ketone_Topliss': Fragments.fr_ketone_Topliss(mol),
            'fr_lactam': Fragments.fr_lactam(mol),
            'fr_lactone': Fragments.fr_lactone(mol),
            'fr_methoxy': Fragments.fr_methoxy(mol),
            'fr_morpholine': Fragments.fr_morpholine(mol),
            'fr_nitrile': Fragments.fr_nitrile(mol),
            'fr_nitro': Fragments.fr_nitro(mol),
            'fr_nitro_arom': Fragments.fr_nitro_arom(mol),
            'fr_nitro_arom_nonortho': Fragments.fr_nitro_arom_nonortho(mol),
            'fr_nitroso': Fragments.fr_nitroso(mol),
            'fr_oxazole': Fragments.fr_oxazole(mol),
            'fr_oxime': Fragments.fr_oxime(mol),
            'fr_para_hydroxylation': Fragments.fr_para_hydroxylation(mol),
            'fr_phenol': Fragments.fr_phenol(mol),
            'fr_phenol_noOrthoHbond': Fragments.fr_phenol_noOrthoHbond(mol),
            'fr_phos_acid': Fragments.fr_phos_acid(mol),
            'fr_phos_ester': Fragments.fr_phos_ester(mol),
            'fr_piperdine': Fragments.fr_piperdine(mol),
            'fr_piperzine': Fragments.fr_piperzine(mol),
            'fr_priamide': Fragments.fr_priamide(mol),
            'fr_prisulfonamd': Fragments.fr_prisulfonamd(mol),
            'fr_pyridine': Fragments.fr_pyridine(mol),
            'fr_quatN': Fragments.fr_quatN(mol),
            'fr_sulfide': Fragments.fr_sulfide(mol),
            'fr_sulfonamd': Fragments.fr_sulfonamd(mol),
            'fr_sulfone': Fragments.fr_sulfone(mol),
            'fr_term_acetylene': Fragments.fr_term_acetylene(mol),
            'fr_tetrazole': Fragments.fr_tetrazole(mol),
            'fr_thiazole': Fragments.fr_thiazole(mol),
            'fr_thiocyan': Fragments.fr_thiocyan(mol),
            'fr_thiophene': Fragments.fr_thiophene(mol),
            'fr_unbrch_alkane': Fragments.fr_unbrch_alkane(mol),
            'fr_urea': Fragments.fr_urea(mol),
        })

        # Drug-likeness descriptors
        try:
            descriptors['QED'] = qed(mol)
        except:
            descriptors['QED'] = None

        # Rule violations
        descriptors.update({
            'lipinski_violations': self._lipinski_violations(mol),
            'oprea_violations': self._oprea_violations(mol),
        })

        # PAINS filters
        if self.filter_catalog:
            try:
                matches = self.filter_catalog.HasMatch(mol)
                descriptors['pains_alerts'] = int(matches)
            except:
                descriptors['pains_alerts'] = 0
        else:
            descriptors['pains_alerts'] = 0

        return descriptors

    def _lipinski_violations(self, mol: Chem.Mol) -> int:
        """Calculate Lipinski rule violations."""
        violations = 0

        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1

        return violations

    def _oprea_violations(self, mol: Chem.Mol) -> int:
        """Calculate Oprea rule violations."""
        violations = 0

        if Descriptors.MolWt(mol) < 200 or Descriptors.MolWt(mol) > 600:
            violations += 1
        if Descriptors.MolLogP(mol) < 2 or Descriptors.MolLogP(mol) > 6:
            violations += 1
        if Descriptors.NumRotatableBonds(mol) > 8:
            violations += 1
        if Descriptors.RingCount(mol) > 4:
            violations += 1

        return violations

    def process_molecules(self, molecules: List[Chem.Mol]) -> pd.DataFrame:
        """Process a list of molecules and calculate descriptors."""
        results = []

        for i, mol in enumerate(molecules):
            try:
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i+1}"
                smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(mol)

                descriptors = self.calculate_descriptors(mol)
                descriptors['Molecule_ID'] = mol_name
                descriptors['SMILES'] = smiles
                descriptors['Is_Cyclic_Peptide'] = self.validate_cyclic_peptide(mol)

                results.append(descriptors)

                self.logger.info(f"Calculated descriptors for molecule {i+1}/{len(molecules)}: {mol_name}")

            except Exception as e:
                self.logger.error(f"Error processing molecule {i+1}: {str(e)}")
                continue

        df = pd.DataFrame(results)

        # Reorder columns to put ID and SMILES first
        id_cols = ['Molecule_ID', 'SMILES', 'Is_Cyclic_Peptide']
        other_cols = [col for col in df.columns if col not in id_cols]
        df = df[id_cols + other_cols]

        return df

    def save_output(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save output DataFrame to file."""
        if file_path.suffix.lower() == '.csv':
            df.to_csv(file_path, index=False)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        else:
            # Default to CSV
            df.to_csv(file_path, index=False)

        self.logger.info(f"Saved {len(df)} descriptors to {file_path}")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_calculate_descriptors(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate molecular descriptors for cyclic peptides.

    Args:
        input_file: Path to input file (SMILES or SDF)
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with calculated descriptors
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_calculate_descriptors("input.smi", "output.csv")
        >>> print(f"Calculated {len(result['result'])} descriptors")
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize calculator
    calculator = CyclicPeptideDescriptorCalculator(config)

    # Load molecules
    molecules = calculator.load_molecules(input_file)

    if not molecules:
        raise ValueError("No valid molecules found in input file")

    # Calculate descriptors
    descriptors_df = calculator.process_molecules(molecules)

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calculator.save_output(descriptors_df, output_path)

    return {
        "result": descriptors_df,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "molecules_processed": len(molecules),
            "descriptors_calculated": len(descriptors_df.columns) - 3,  # Exclude ID, SMILES, Is_Cyclic_Peptide
            "config": config
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i',
                       help='Input file path (SMILES, SDF)')
    parser.add_argument('--output', '-o',
                       help='Output file path (CSV or Excel)')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo data')

    args = parser.parse_args()

    # Validate arguments
    if not args.demo and not args.input:
        parser.error("--input is required unless --demo is used")

    # Handle demo mode
    if args.demo:
        # Create demo SMILES file if it doesn't exist
        demo_smiles_path = Path("examples/data/demo_cyclic_peptides.smi")
        if not demo_smiles_path.exists():
            demo_smiles_path.parent.mkdir(parents=True, exist_ok=True)
            demo_smiles = [
                "C1C[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC3=CC=C(C=C3)O)CC(C)C\tcyclic_peptide_1",
                "C[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O)CC4=CNC5=CC=CC=C54\tcyclic_peptide_2",
                "CC[C@H](C)[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)C)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O\tcyclic_peptide_3"
            ]
            with open(demo_smiles_path, 'w') as f:
                f.write('\n'.join(demo_smiles))

        args.input = str(demo_smiles_path)
        if not args.output:
            args.output = "demo_descriptors.csv"

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    try:
        # Run calculation
        result = run_calculate_descriptors(
            input_file=args.input,
            output_file=args.output,
            config=config
        )

        # Print summary
        print(f"✅ Success: Processed {result['metadata']['molecules_processed']} molecules")
        print(f"📊 Calculated {result['metadata']['descriptors_calculated']} descriptors")
        if result['output_file']:
            print(f"💾 Output saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == '__main__':
    main()
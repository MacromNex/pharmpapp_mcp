# PharmPapp MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported (rdkit, numpy, pandas, sklearn)
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `calculate_descriptors.py` | Calculate molecular descriptors from SMILES | No | `configs/calculate_descriptors_config.json` |
| `predict_permeability.py` | Predict permeability using trained models | No | `configs/predict_permeability_config.json` |
| `batch_analysis.py` | Batch analysis and model comparison | No | `configs/batch_analysis_config.json` |

## Installation

Ensure you have the environment set up:

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env
```

## Usage

### 1. Calculate Molecular Descriptors

```bash
# Basic usage
python scripts/calculate_descriptors.py --input examples/data/demo_cyclic_peptides.smi --output results/descriptors.csv

# With demo data
python scripts/calculate_descriptors.py --demo

# With custom config
python scripts/calculate_descriptors.py --input molecules.smi --output descriptors.csv --config configs/custom_config.json
```

**Input Formats:**
- SMILES files (`.smi`, `.smiles`)
- SDF files (`.sdf`)

**Output:**
- CSV file with calculated descriptors
- 144+ molecular descriptors per molecule
- Includes validation flags for cyclic peptides

### 2. Predict Permeability

```bash
# Basic usage
python scripts/predict_permeability.py --input descriptors.csv --model caco2_c --output predictions.csv

# With demo data
python scripts/predict_permeability.py --demo

# Different models
python scripts/predict_permeability.py --input descriptors.csv --model rrck_c --output rrck_predictions.csv
```

**Available Models:**
- `rrck_c`: RRCK-C (SVR, 103 descriptors)
- `pampa_c`: PAMPA-C (RandomForest, 71 descriptors)
- `caco2_l`: Caco2-L (RandomForest, 82 descriptors)
- `caco2_c`: Caco2-C (RandomForest, 79 descriptors)
- `caco2_a`: Caco2-A (RandomForest, 92 descriptors)

**Input:**
- CSV file with molecular descriptors
- Optional target values for training/validation

**Output:**
- CSV file with predictions
- Model performance metrics if targets available

### 3. Batch Analysis

```bash
# Analyze all models
python scripts/batch_analysis.py --input descriptors.csv --output results/batch_analysis/

# Specific models only
python scripts/batch_analysis.py --input descriptors.csv --output results/batch/ --models caco2_c rrck_c

# Skip plots
python scripts/batch_analysis.py --input descriptors.csv --output results/batch/ --no-plots
```

**Output:**
- Summary statistics for all models
- Individual prediction files
- Correlation plots (if matplotlib available)
- Analysis metadata

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving utilities
- `molecules.py`: RDKit molecular manipulation
- `validation.py`: Input validation and quality checks

## Configuration

Each script can use JSON configuration files from `configs/`:

- `default_config.json`: General settings
- `calculate_descriptors_config.json`: Descriptor calculation settings
- `predict_permeability_config.json`: Model prediction settings
- `batch_analysis_config.json`: Batch analysis settings

Example configuration override:
```bash
python scripts/calculate_descriptors.py --input molecules.smi --config configs/custom.json
```

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from scripts.calculate_descriptors import run_calculate_descriptors
from scripts.predict_permeability import run_predict_permeability
from scripts.batch_analysis import run_batch_analysis

# In MCP tool:
@mcp.tool()
def calculate_cyclic_peptide_descriptors(input_file: str, output_file: str = None):
    return run_calculate_descriptors(input_file, output_file)

@mcp.tool()
def predict_cyclic_peptide_permeability(input_file: str, model: str, output_file: str = None):
    return run_predict_permeability(input_file, model, output_file)

@mcp.tool()
def analyze_permeability_models(input_file: str, output_dir: str):
    return run_batch_analysis(input_file, output_dir)
```

## Dependencies

**Essential (always required):**
- `numpy`
- `pandas`
- `rdkit`
- `scikit-learn`
- `joblib`

**Optional (graceful fallback):**
- `matplotlib` (for plots in batch analysis)
- `seaborn` (for correlation plots)
- `openpyxl` (for Excel file support)

## Troubleshooting

### Common Issues

1. **Missing Descriptors Warning**: Some models require specific descriptors that may not be calculated. Use descriptor calculator first or use models with better descriptor compatibility.

2. **RDKit Version Compatibility**: Some descriptor names may vary between RDKit versions. The scripts include fallbacks for common issues.

3. **Small Dataset Performance**: Models may show poor performance (negative R²) with very small training sets (< 20 molecules). This is expected behavior.

4. **Feature Mismatch in Batch Analysis**: Different models require different descriptor sets. This is a known limitation - use individual models for datasets with limited descriptors.

### Performance Tips

- Use `--demo` flags to test functionality quickly
- Start with descriptor calculation, then use those descriptors for predictions
- For large datasets, consider using `n_jobs=-1` in configuration files
- Batch analysis works best with datasets containing all required descriptors

## Error Handling

All scripts include comprehensive error handling:
- File not found errors with clear messages
- Missing descriptor warnings with suggestions
- Graceful fallbacks for optional dependencies
- Detailed logging for troubleshooting

## Examples

See `examples/` directory for:
- Sample input data files
- Example descriptor outputs
- Model prediction results
- Batch analysis reports

For more examples and detailed usage, refer to the original use cases in `examples/use_case_*.py`.
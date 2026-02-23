# PharmPapp MCP

> MCP tools for cyclic peptide computational analysis - Permeability prediction and molecular descriptor calculation

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP (Model Context Protocol) server provides computational tools for cyclic peptide analysis based on the PharmPapp framework. It offers both synchronous and asynchronous APIs for molecular property calculation, permeability prediction, and batch analysis workflows commonly used in drug discovery pipelines.

### Features
- **Molecular descriptor calculation** for cyclic peptides using RDKit
- **Permeability prediction** using 5 PharmPapp models (RRCK-C, PAMPA-C, Caco2-L/C/A)
- **Batch analysis** with multi-model comparison and consensus predictions
- **Job management** for long-running computations with real-time monitoring
- **Seamless LLM integration** through standardized MCP protocol

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   └── server.py           # MCP server with 13 tools
├── scripts/
│   ├── calculate_descriptors.py  # Molecular descriptor calculation
│   ├── predict_permeability.py   # Permeability prediction
│   ├── batch_analysis.py         # Multi-model comparison
│   └── lib/                      # Shared utilities (18 functions)
├── examples/
│   └── data/               # Demo data
│       ├── demo_cyclic_peptides.smi     # Sample cyclic peptide SMILES
│       ├── example for Caco2_A.csv      # Model training data (10 molecules)
│       ├── example for Caco2_C.csv      # Model training data (10 molecules)
│       ├── example for Caco2_L.csv      # Model training data (10 molecules)
│       ├── example for PAMPA-C.csv      # Model training data (10 molecules)
│       └── example for RRCK-C.csv       # Model training data (10 molecules)
├── configs/                # Configuration files
│   ├── calculate_descriptors_config.json
│   ├── predict_permeability_config.json
│   ├── batch_analysis_config.json
│   └── default_config.json
└── repo/                   # Original PharmPapp repository
```

---

## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

### Manual Setup (Advanced)

For manual installation or customization, follow these steps.

#### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- RDKit (installed automatically via conda-forge)

#### Create Environment

Please follow the procedure from `reports/step3_environment.md`. The recommended workflow is shown below:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pharmpapp_mcp

# Check for package manager preference
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi
echo "Using package manager: $PKG_MGR"

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install scientific computing dependencies
mamba run -p ./env pip install pandas numpy scikit-learn matplotlib seaborn tqdm loguru click

# Install RDKit from conda-forge (essential for molecular work)
mamba install -p ./env -c conda-forge rdkit -y

# Install MCP dependencies
mamba run -p ./env pip install --force-reinstall --no-cache-dir fastmcp
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/calculate_descriptors.py` | Calculate molecular descriptors from SMILES/SDF | See below |
| `scripts/predict_permeability.py` | Predict permeability using PharmPapp models | See below |
| `scripts/batch_analysis.py` | Batch analysis and model comparison | See below |

### Script Examples

#### Calculate Molecular Descriptors

```bash
# Activate environment
mamba activate ./env

# Run with demo data
python scripts/calculate_descriptors.py --demo

# Process custom SMILES file
python scripts/calculate_descriptors.py \
  --input molecules.smi \
  --output descriptors.csv

# Include additional descriptor types
python scripts/calculate_descriptors.py \
  --input examples/data/demo_cyclic_peptides.smi \
  --output full_descriptors.csv \
  --config configs/calculate_descriptors_config.json
```

**Parameters:**
- `--input, -i`: Input SMILES (.smi) or SDF (.sdf) file (required)
- `--output, -o`: Output CSV file path (default: auto-generated)
- `--config`: JSON configuration file (default: basic descriptors)
- `--demo`: Use built-in demo data (3 cyclic peptides)

#### Predict Permeability

```bash
python scripts/predict_permeability.py \
  --input descriptors.csv \
  --model caco2_c \
  --output predictions.csv

# Available models: rrck_c, pampa_c, caco2_l, caco2_c, caco2_a
python scripts/predict_permeability.py \
  --input examples/data/"example for Caco2_C.csv" \
  --model caco2_c \
  --demo
```

#### Batch Analysis

```bash
python scripts/batch_analysis.py \
  --input descriptors.csv \
  --output batch_results/ \
  --models caco2_c rrck_c pampa_c
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
mamba run -p ./env fastmcp install src/server.py --name cycpep-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pharmpapp_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pharmpapp_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What cyclic peptide tools are available from cycpep-tools?
```

#### Property Calculation (Fast)
```
Calculate molecular descriptors for examples/data/demo_cyclic_peptides.smi and save to descriptors.csv
```

#### Permeability Prediction
```
Predict permeability using the caco2_c model for @examples/data/"example for Caco2_C.csv"
```

#### Structure Validation
```
Validate the input file @examples/data/demo_cyclic_peptides.smi and tell me what's in it
```

#### Batch Processing (Submit API)
```
Submit a batch analysis job for @descriptors.csv using models caco2_c and rrck_c, save results to batch_results/
```

#### Job Management
```
Check the status of job abc12345
Get the logs for job abc12345
List all my jobs
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/demo_cyclic_peptides.smi` | Reference demo SMILES file |
| `@configs/calculate_descriptors_config.json` | Reference config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pharmpapp_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pharmpapp_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What cyclic peptide tools are available?
> Calculate descriptors for the demo cyclic peptides
> Submit a batch analysis for multiple permeability models
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculate_cyclic_peptide_descriptors` | Calculate molecular descriptors | `input_file`, `output_file`, `include_3d`, `include_fingerprints` |
| `predict_cyclic_peptide_permeability` | Predict permeability with single model | `input_file`, `model`, `output_file`, `demo` |
| `get_available_models` | List available permeability models | None |
| `validate_input_file` | Validate SMILES/SDF file | `input_file` |
| `get_example_data` | List available example data | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_batch_analysis` | Multi-model comparison analysis | `input_file`, `output_dir`, `models`, `create_plots` |
| `submit_large_descriptor_calculation` | Large-scale descriptor calculation | `input_file`, `output_file`, `include_3d`, `include_fingerprints` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and runtime |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs (tail 50) |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs with status filter |
| `cleanup_old_jobs` | Remove old completed jobs |

---

## Examples

### Example 1: Quick Descriptor Calculation

**Goal:** Calculate basic molecular properties for cyclic peptides

**Using Script:**
```bash
python scripts/calculate_descriptors.py \
  --input examples/data/demo_cyclic_peptides.smi \
  --output demo_descriptors.csv
```

**Using MCP (in Claude Code):**
```
Calculate molecular descriptors for @examples/data/demo_cyclic_peptides.smi and save to demo_descriptors.csv
```

**Expected Output:**
- CSV file with 144 molecular descriptors per molecule
- Descriptors include: MolWt, LogP, TPSA, NumHBD, NumHBA, etc.
- Processing time: ~3 seconds for 3 molecules

### Example 2: Permeability Prediction

**Goal:** Predict cyclic peptide permeability using PharmPapp models

**Using Script:**
```bash
python scripts/predict_permeability.py \
  --input "examples/data/example for Caco2_C.csv" \
  --model caco2_c \
  --output caco2_predictions.csv
```

**Using MCP (in Claude Code):**
```
Predict permeability using caco2_c model for the example Caco2_C data and save results
```

**Expected Output:**
- Predicted permeability values with confidence scores
- Model performance metrics (R², RMSE)
- Processing time: 1-3 seconds for 10 molecules

### Example 3: Multi-Model Comparison

**Goal:** Compare predictions across different permeability models

**Using MCP (in Claude Code):**
```
Submit a batch analysis for @examples/data/"example for Caco2_C.csv" using all available models. Save results to batch_comparison/

Then check the job status and show me the results when complete.
```

**Workflow:**
1. Job submitted with unique job_id
2. Monitor progress with `get_job_status`
3. Retrieve results with `get_job_result`
4. View detailed logs with `get_job_log`

### Example 4: End-to-End Pipeline

**Goal:** Complete workflow from SMILES to multi-model predictions

**Using MCP (in Claude Code):**
```
1. First, calculate descriptors for @examples/data/demo_cyclic_peptides.smi
2. Then submit a batch analysis using the calculated descriptors with models caco2_c and rrck_c
3. Monitor the job and show me a summary when complete
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Molecules | Use With |
|------|-------------|-----------|----------|
| `demo_cyclic_peptides.smi` | Sample cyclic peptide SMILES | 3 | descriptor calculation |
| `example for Caco2_A.csv` | Caco2-A training data | 10 | permeability prediction |
| `example for Caco2_C.csv` | Caco2-C training data | 10 | permeability prediction |
| `example for Caco2_L.csv` | Caco2-L training data | 10 | permeability prediction |
| `example for PAMPA-C.csv` | PAMPA-C training data | 10 | permeability prediction |
| `example for RRCK-C.csv` | RRCK-C training data | 10 | permeability prediction |

### Sample Cyclic Peptide SMILES

The demo file contains 3 representative cyclic peptides:
```
C1C[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC3=CC=C(C=C3)O)CC(C)C
C[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O)CC4=CNC5=CC=CC=C54
CC[C@H](C)[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)C)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O
```

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `calculate_descriptors_config.json` | Descriptor calculation settings | `include_3d`, `include_fingerprints`, `descriptor_types` |
| `predict_permeability_config.json` | Model prediction settings | `test_size`, `cross_validation`, `scale_features` |
| `batch_analysis_config.json` | Batch analysis settings | `models_to_test`, `create_plots` |
| `default_config.json` | General settings | `random_state`, `n_jobs` |

### Configuration Example

```json
{
  "processing": {
    "include_3d": false,
    "include_fingerprints": false,
    "optimize_geometry": false
  },
  "descriptor_types": ["basic", "topological", "constitutional", "physicochemical"],
  "quality_control": {
    "validate_cyclic_peptide": true,
    "calculate_lipinski_violations": true
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
mamba run -p ./env pip install pandas numpy scikit-learn matplotlib seaborn tqdm loguru click
mamba install -p ./env -c conda-forge rdkit -y
mamba run -p ./env pip install --force-reinstall --no-cache-dir fastmcp
```

**Problem:** RDKit import errors
```bash
# Install RDKit from conda-forge (essential)
mamba install -p ./env -c conda-forge rdkit -y

# Test RDKit installation
mamba run -p ./env python -c "from rdkit import Chem; print('RDKit working!')"
```

**Problem:** Import errors
```bash
# Verify all dependencies
mamba run -p ./env python -c "
from src.server import mcp
from rdkit import Chem
import pandas
import fastmcp
print('All dependencies working!')
"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove cycpep-tools
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Invalid SMILES error
```
Ensure your SMILES string is valid for cyclic peptides. Use the cyclo() notation
or ensure ring closure is properly specified with numbers (e.g., C1...C1).
```

**Problem:** Tools not working
```bash
# Test server directly
mamba run -p ./env python -c "
from src.server import mcp
print('Available tools:', list(mcp.list_tools().keys()))
print('Total tools:', len(mcp.list_tools()))
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory and logs
ls -la jobs/
```

In Claude Code:
```
Get the log for job <job_id> with 100 lines to see what happened
```

**Problem:** Descriptor compatibility error**
```
Error: Too many missing descriptors. Available: 63/94
```

**Solution**: This occurs when trying to predict permeability with descriptors that don't match the model requirements. Use the example CSV files for compatible data, or enhance descriptor calculation to include MOE-style descriptors.

**Problem:** Out of memory errors
```
Use submit_large_descriptor_calculation for datasets >1000 molecules
Process in smaller batches for very large datasets
```

### Common Model Issues

**Problem:** "PAMPA model requires 77 descriptors, found 0"

This is a known limitation. The descriptor calculation produces RDKit descriptors while the models expect MOE2D descriptors. Solutions:
1. Use the provided example CSV files which have compatible descriptors
2. Wait for enhanced descriptor mapping in future versions

### Debugging

In Claude Code:
```
# Check job status
Check the status of job <job_id>

# View recent logs
Get the log for job <job_id> with 50 lines

# List all jobs
List all my jobs and their statuses

# Validate input data
Validate the input file @examples/data/demo_cyclic_peptides.smi
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test basic functionality
python scripts/calculate_descriptors.py --demo
```

### Starting Dev Server

```bash
# Run MCP server in development mode
mamba run -p ./env fastmcp dev src/server.py

# Test server connectivity
mamba run -p ./env python -c "from src.server import mcp; print('Server ready')"
```

---

## Performance Guidelines

### When to Use Sync vs Submit API

**Use Sync API when:**
- Dataset < 1000 molecules
- Only basic 2D descriptors needed
- Interactive analysis requiring immediate feedback
- Quick validation and exploration

**Use Submit API when:**
- Dataset > 1000 molecules
- Including 3D descriptors or fingerprints (much slower)
- Batch processing multiple models
- Long-running analyses that may take >10 minutes

### Resource Usage

| Operation | Memory | CPU | Time (1000 molecules) |
|-----------|--------|-----|----------------------|
| Basic descriptors | ~100MB | Low | ~2 minutes |
| 3D descriptors | ~500MB | High | ~15 minutes |
| Fingerprints | ~200MB | Medium | ~5 minutes |
| Model prediction | ~50MB | Low | ~1 minute |
| Batch analysis | ~300MB | Medium | ~10 minutes |

---

## License

Based on [PharmPapp](https://github.com/Fraunhofer-ITMP/PharmPapp) - Fraunhofer ITMP implementation

## Credits

This MCP server implements Python equivalents of the KNIME workflows from the original PharmPapp repository, providing programmatic access to cyclic peptide permeability prediction models through modern LLM interfaces.

---

## API Reference Summary

### Job Management (6 tools)
- `get_job_status` - Check job progress and runtime
- `get_job_result` - Get completed job results
- `get_job_log` - View execution logs
- `cancel_job` - Cancel running job
- `list_jobs` - List jobs with status filter
- `cleanup_old_jobs` - Remove old completed jobs

### Synchronous Analysis (2 tools)
- `calculate_cyclic_peptide_descriptors` - Fast descriptor calculation
- `predict_cyclic_peptide_permeability` - Single model permeability prediction

### Asynchronous Processing (2 tools)
- `submit_batch_analysis` - Multi-model comparison
- `submit_large_descriptor_calculation` - Large-scale descriptor calculation

### Utilities (3 tools)
- `get_available_models` - Model information and requirements
- `validate_input_file` - File format validation
- `get_example_data` - Available demo data information

**Total**: 13 MCP tools covering the complete cyclic peptide computational analysis workflow.
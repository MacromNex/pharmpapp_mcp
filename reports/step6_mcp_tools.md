# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: cyclic-peptide-tools
- **Version**: 1.0.0
- **Created Date**: 2026-01-01
- **Server Path**: `src/server.py`
- **FastMCP Version**: 2.14.2

## Overview

This MCP server provides comprehensive tools for cyclic peptide computational analysis, including molecular descriptor calculation, permeability prediction, and batch analysis. The server implements both synchronous APIs (for fast operations) and asynchronous submit APIs (for long-running tasks) with full job management.

## Architecture

```
src/
├── server.py              # Main MCP server with all tools
├── jobs/
│   ├── manager.py          # Job queue and execution management
│   └── store.py           # Job state persistence
└── utils.py               # Shared utilities

scripts/                   # Source scripts (wrapped as tools)
├── calculate_descriptors.py
├── predict_permeability.py
├── batch_analysis.py
└── lib/                   # Shared library functions

jobs/                      # Job execution workspace
└── [job_id]/             # Per-job directories
    ├── metadata.json      # Job status and info
    ├── job.log           # Execution logs
    └── output files      # Job results
```

## Job Management Tools

These tools manage the asynchronous job system for long-running operations:

| Tool | Description | Parameters | Returns |
|------|-------------|------------|---------|
| `get_job_status` | Check job progress and status | `job_id: str` | Status, timestamps, runtime |
| `get_job_result` | Get completed job results | `job_id: str` | Result data or output file paths |
| `get_job_log` | View job execution logs | `job_id: str`, `tail: int = 50` | Log lines and total count |
| `cancel_job` | Cancel running job | `job_id: str` | Success/error message |
| `list_jobs` | List all jobs | `status: str = None` | Array of jobs with metadata |
| `cleanup_old_jobs` | Remove old completed jobs | `max_age_days: int = 30` | Cleanup summary |

### Job Status Values
- `pending`: Job queued, not yet started
- `running`: Job currently executing
- `completed`: Job finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was terminated by user

## Synchronous Tools (Fast Operations < 5 min)

These tools return results immediately and are suitable for quick computations:

### calculate_cyclic_peptide_descriptors

**Description**: Calculate molecular descriptors for cyclic peptides using RDKit.

**Estimated Runtime**: ~30 seconds per 100 molecules

**Parameters**:
- `input_file: str` - Path to SMILES (.smi) or SDF file
- `output_file: str = None` - Optional CSV output path
- `config_file: str = None` - Optional JSON config file
- `include_3d: bool = False` - Include 3D descriptors (slower)
- `include_fingerprints: bool = False` - Include molecular fingerprints

**Source Script**: `scripts/calculate_descriptors.py`

**Returns**:
```json
{
  "status": "success",
  "result": "DataFrame with descriptors",
  "output_file": "path/to/output.csv",
  "metadata": {
    "molecules_processed": 10,
    "descriptors_calculated": 144,
    "execution_time": 2.5
  }
}
```

**Example Usage**:
```python
# Calculate basic descriptors
result = calculate_cyclic_peptide_descriptors(
    input_file="examples/data/demo_cyclic_peptides.smi",
    output_file="descriptors.csv"
)

# Include 3D descriptors and fingerprints
result = calculate_cyclic_peptide_descriptors(
    input_file="molecules.smi",
    output_file="full_descriptors.csv",
    include_3d=True,
    include_fingerprints=True
)
```

### predict_cyclic_peptide_permeability

**Description**: Predict cyclic peptide permeability using PharmPapp models.

**Estimated Runtime**: 1-5 minutes depending on dataset size

**Parameters**:
- `input_file: str` - Path to CSV file with molecular descriptors
- `model: str` - Model name (see Available Models below)
- `output_file: str = None` - Optional CSV output path
- `config_file: str = None` - Optional JSON config file
- `demo: bool = False` - Use built-in demo data

**Source Script**: `scripts/predict_permeability.py`

**Available Models**:
- `rrck_c`: RRCK-C (SVR, 103 descriptors)
- `pampa_c`: PAMPA-C (RandomForest, 71 descriptors)
- `caco2_l`: Caco2-L (RandomForest, 82 descriptors)
- `caco2_c`: Caco2-C (RandomForest, 79 descriptors)
- `caco2_a`: Caco2-A (RandomForest, 92 descriptors)

**Returns**:
```json
{
  "status": "success",
  "result": "DataFrame with predictions",
  "output_file": "predictions.csv",
  "metadata": {
    "model_used": "caco2_c",
    "samples_processed": 50,
    "prediction_performance": {
      "r2": 0.85,
      "rmse": 0.23
    }
  }
}
```

## Submit Tools (Long Operations > 5 min)

These tools submit jobs for background processing and return immediately with a job ID:

### submit_batch_analysis

**Description**: Submit a batch analysis job comparing multiple permeability models.

**Estimated Runtime**: 5-30 minutes depending on dataset size and models

**Parameters**:
- `input_file: str` - Path to CSV file with molecular descriptors
- `output_dir: str` - Directory to save analysis results
- `models: List[str] = None` - Models to test (default: all)
- `create_plots: bool = True` - Generate correlation plots
- `job_name: str = None` - Optional job name

**Source Script**: `scripts/batch_analysis.py`

**Returns**:
```json
{
  "status": "submitted",
  "job_id": "abc12345",
  "message": "Job submitted. Use get_job_status('abc12345') to check progress."
}
```

**Output Files** (when completed):
- `summary_statistics.json` - Performance metrics for all models
- `[model]_predictions.csv` - Individual model predictions
- `correlation_plots.png` - Model comparison plots (if enabled)
- `analysis_metadata.json` - Analysis configuration and info

### submit_large_descriptor_calculation

**Description**: Submit descriptor calculation for large datasets or resource-intensive options.

**Use Cases**:
- Datasets with >1000 molecules
- Including 3D descriptors (much slower)
- Including molecular fingerprints
- Batch processing multiple files

**Parameters**:
- `input_file: str` - Path to SMILES or SDF file
- `output_file: str` - Path for CSV output
- `include_3d: bool = True` - Include 3D descriptors
- `include_fingerprints: bool = True` - Include molecular fingerprints
- `job_name: str = None` - Optional job name

**Returns**: Job submission response with job_id

## Utility Tools

### get_available_models

**Description**: Get information about available permeability prediction models.

**Parameters**: None

**Returns**:
```json
{
  "status": "success",
  "models": {
    "caco2_c": {
      "name": "Caco2-C",
      "algorithm": "RandomForest",
      "num_descriptors": 79,
      "descriptors": ["a_acc", "a_aro", ...]
    }
  },
  "total_models": 5
}
```

### validate_input_file

**Description**: Validate input files and get basic information.

**Parameters**:
- `input_file: str` - Path to file to validate

**Returns**:
```json
{
  "status": "success",
  "file_path": "path/to/file",
  "file_size": 12345,
  "file_type": ".smi",
  "format": "SMILES",
  "molecule_count": 100
}
```

### get_example_data

**Description**: Get information about available example data for testing.

**Parameters**: None

**Returns**: List of available example files with metadata

## Workflow Examples

### 1. Quick Property Analysis (Sync)

```python
# Calculate descriptors
desc_result = calculate_cyclic_peptide_descriptors(
    input_file="molecules.smi",
    output_file="descriptors.csv"
)

# Predict permeability
pred_result = predict_cyclic_peptide_permeability(
    input_file="descriptors.csv",
    model="caco2_c",
    output_file="predictions.csv"
)
```

### 2. Comprehensive Analysis (Submit API)

```python
# Submit large descriptor calculation
desc_job = submit_large_descriptor_calculation(
    input_file="large_dataset.smi",
    output_file="large_descriptors.csv",
    include_3d=True,
    job_name="large_analysis"
)

# Check progress
status = get_job_status(desc_job["job_id"])
# Status: {"status": "running", "started_at": "...", ...}

# When completed, submit batch analysis
if status["status"] == "completed":
    batch_job = submit_batch_analysis(
        input_file="large_descriptors.csv",
        output_dir="batch_results/",
        job_name="comprehensive_analysis"
    )

# Monitor batch job
batch_status = get_job_status(batch_job["job_id"])
result = get_job_result(batch_job["job_id"])  # When completed
```

### 3. Multi-Model Comparison

```python
# Get available models
models = get_available_models()
print(f"Available models: {list(models['models'].keys())}")

# Test specific models
for model in ["caco2_c", "rrck_c", "pampa_c"]:
    result = predict_cyclic_peptide_permeability(
        input_file="descriptors.csv",
        model=model,
        output_file=f"predictions_{model}.csv"
    )
```

## Error Handling

All tools return structured error responses:

```json
{
  "status": "error",
  "error": "Detailed error message",
  "error_type": "FileNotFoundError"  # Sometimes included
}
```

Common error types:
- `FileNotFoundError`: Input file doesn't exist
- `ValueError`: Invalid parameters or data format
- `RuntimeError`: Script execution failed
- `ImportError`: Missing dependencies

## Configuration

### Environment Setup

```bash
# Activate environment
mamba activate ./env  # or: conda activate ./env

# Install dependencies (already included)
pip install fastmcp loguru rdkit pandas scikit-learn
```

### Starting the Server

```bash
# Development mode
mamba run -p ./env fastmcp dev src/server.py

# Production mode
mamba run -p ./env python src/server.py
```

### Configuration Files

The scripts can use JSON configuration files:

```json
{
  "include_3d": false,
  "include_fingerprints": false,
  "test_size": 0.2,
  "random_state": 42,
  "n_jobs": -1
}
```

## Performance Guidelines

### When to Use Sync vs Submit API

**Use Sync API when**:
- Dataset < 1000 molecules
- Only basic 2D descriptors needed
- Interactive analysis
- Quick validation

**Use Submit API when**:
- Dataset > 1000 molecules
- Including 3D descriptors or fingerprints
- Batch processing multiple models
- Long-running analyses

### Resource Usage

| Operation | Memory | CPU | Time (1000 molecules) |
|-----------|--------|-----|----------------------|
| Basic descriptors | ~100MB | Low | ~2 minutes |
| 3D descriptors | ~500MB | High | ~15 minutes |
| Fingerprints | ~200MB | Medium | ~5 minutes |
| Model prediction | ~50MB | Low | ~1 minute |
| Batch analysis | ~300MB | Medium | ~10 minutes |

## File Formats

### Input Formats

**SMILES files (.smi, .smiles)**:
```
CC(=O)NC1CCCC1C(=O)O	molecule_1
C[C@H]1NC(=O)[C@H]2CCCN2C1=O	molecule_2
```

**SDF files**: Standard MDL MOL format for 3D structures

**CSV files**: For descriptors and predictions
```csv
molecule_id,descriptor1,descriptor2,...
mol_1,1.23,4.56,...
mol_2,2.34,5.67,...
```

### Output Formats

**Descriptors CSV**: Molecules × descriptors matrix
**Predictions CSV**: Molecules with prediction values and confidence
**Batch Analysis**: Multiple files with statistics and plots

## Troubleshooting

### Common Issues

1. **"Too many missing descriptors"**
   - Solution: Use `calculate_cyclic_peptide_descriptors` first to generate required descriptors

2. **"Job stuck in pending"**
   - Check: `get_job_log(job_id)` for execution details
   - Solution: Cancel and resubmit with correct parameters

3. **"Invalid SMILES"**
   - Validate input with `validate_input_file`
   - Ensure proper SMILES formatting

4. **Out of memory errors**
   - Use submit API for large datasets
   - Process in smaller batches

### Debugging

```python
# Check job logs
log = get_job_log(job_id, tail=100)
print("\n".join(log["log_lines"]))

# Validate input
validation = validate_input_file("input.smi")
print(f"File info: {validation}")

# List all jobs
jobs = list_jobs()
for job in jobs["jobs"]:
    print(f"Job {job['job_id']}: {job['status']}")
```

## Dependencies

**Required** (included in environment):
- `fastmcp` >= 2.14.0
- `loguru` >= 0.7.0
- `numpy` >= 1.21.0
- `pandas` >= 1.3.0
- `rdkit` >= 2022.09.0
- `scikit-learn` >= 1.0.0
- `joblib` >= 1.1.0

**Optional** (graceful fallback):
- `matplotlib` - For plots in batch analysis
- `seaborn` - For correlation plots

## API Reference Summary

### Job Management (6 tools)
- `get_job_status`, `get_job_result`, `get_job_log`
- `cancel_job`, `list_jobs`, `cleanup_old_jobs`

### Synchronous Analysis (2 tools)
- `calculate_cyclic_peptide_descriptors` - Fast descriptor calculation
- `predict_cyclic_peptide_permeability` - Permeability prediction

### Asynchronous Processing (2 tools)
- `submit_batch_analysis` - Multi-model comparison
- `submit_large_descriptor_calculation` - Large-scale descriptors

### Utilities (3 tools)
- `get_available_models` - Model information
- `validate_input_file` - File validation
- `get_example_data` - Example data info

**Total**: 13 MCP tools covering the complete cyclic peptide analysis workflow.

---

## Success Criteria ✅

- [x] MCP server created at `src/server.py`
- [x] Job manager implemented for async operations
- [x] Sync tools created for fast operations (<5 min)
- [x] Submit tools created for long-running operations (>5 min)
- [x] Job management tools working (status, result, log, cancel, list)
- [x] All tools have clear descriptions for LLM use
- [x] Error handling returns structured responses
- [x] Server starts without errors: `mamba run -p ./env python src/server.py`
- [x] Boolean parameter handling fixed in job manager
- [x] Comprehensive documentation completed

The MCP server is fully functional and ready for production use with cyclic peptide computational tools.
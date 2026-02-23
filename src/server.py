"""MCP Server for Cyclic Peptide Tools

Provides both synchronous and asynchronous (submit) APIs for all tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("cyclic-peptide-tools")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted cyclic peptide computation job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed cyclic peptide computation job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running cyclic peptide computation job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted cyclic peptide computation jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)


@mcp.tool()
def cleanup_old_jobs(max_age_days: int = 30) -> dict:
    """
    Clean up old completed jobs to save disk space.

    Args:
        max_age_days: Maximum age in days for jobs to keep (default: 30)

    Returns:
        Success message or error
    """
    return job_manager.cleanup_old_jobs(max_age_days)

# ==============================================================================
# Synchronous Tools (for fast operations < 5 min)
# ==============================================================================

@mcp.tool()
def calculate_cyclic_peptide_descriptors(
    input_file: str,
    output_file: Optional[str] = None,
    config_file: Optional[str] = None,
    include_3d: bool = False,
    include_fingerprints: bool = False
) -> dict:
    """
    Calculate molecular descriptors for cyclic peptides using RDKit.

    Fast operation - returns results immediately (typically < 1 minute).

    Args:
        input_file: Path to input file containing SMILES (.smi) or molecules (.sdf)
        output_file: Optional path to save results (CSV format)
        config_file: Optional path to JSON config file
        include_3d: Whether to include 3D descriptors (slower)
        include_fingerprints: Whether to include molecular fingerprints

    Returns:
        Dictionary with calculated descriptors and metadata
    """
    from calculate_descriptors import run_calculate_descriptors

    try:
        # Load config if provided
        config = {}
        if config_file:
            with open(config_file) as f:
                config = json.load(f)

        # Override config with function parameters
        if include_3d is not None:
            config["include_3d"] = include_3d
        if include_fingerprints is not None:
            config["include_fingerprints"] = include_fingerprints

        result = run_calculate_descriptors(
            input_file=input_file,
            output_file=output_file,
            config=config if config else None
        )
        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Descriptor calculation failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def predict_cyclic_peptide_permeability(
    input_file: str,
    model: str,
    output_file: Optional[str] = None,
    config_file: Optional[str] = None,
    demo: bool = False
) -> dict:
    """
    Predict cyclic peptide permeability using PharmPapp models.

    Medium-speed operation - returns results immediately (1-5 minutes).

    Args:
        input_file: Path to CSV file with molecular descriptors
        model: Model to use (rrck_c, pampa_c, caco2_l, caco2_c, caco2_a)
        output_file: Optional path to save predictions (CSV format)
        config_file: Optional path to JSON config file
        demo: Use demo mode with built-in example data

    Returns:
        Dictionary with predictions and model performance metrics
    """
    from predict_permeability import run_predict_permeability

    try:
        # Load config if provided
        config = None
        if config_file:
            with open(config_file) as f:
                config = json.load(f)

        result = run_predict_permeability(
            input_file=input_file,
            model_name=model,
            output_file=output_file,
            config=config,
            demo=demo
        )
        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Permeability prediction failed: {e}")
        return {"status": "error", "error": str(e)}


# ==============================================================================
# Submit Tools (for long-running operations > 5 min)
# ==============================================================================

@mcp.tool()
def submit_batch_analysis(
    input_file: str,
    output_dir: str,
    models: Optional[List[str]] = None,
    create_plots: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a batch analysis job comparing multiple permeability models.

    This task may take more than 5 minutes for large datasets. Use get_job_status()
    to monitor progress and get_job_result() to retrieve results when completed.

    Args:
        input_file: Path to CSV file with molecular descriptors
        output_dir: Directory to save analysis results
        models: List of models to test (default: all available models)
        create_plots: Whether to create correlation plots
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "batch_analysis.py")

    args = {
        "input": input_file,
        "output": output_dir
    }

    if models:
        args["models"] = models

    if not create_plots:
        args["no_plots"] = True

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_analysis_{Path(input_file).stem}"
    )


@mcp.tool()
def submit_large_descriptor_calculation(
    input_file: str,
    output_file: str,
    include_3d: bool = True,
    include_fingerprints: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a large descriptor calculation job for many cyclic peptides.

    Use this for datasets with > 1000 molecules or when including 3D descriptors
    and fingerprints, which can be slow for large datasets.

    Args:
        input_file: Path to input file containing SMILES (.smi) or molecules (.sdf)
        output_file: Path to save results (CSV format)
        include_3d: Whether to include 3D descriptors (much slower)
        include_fingerprints: Whether to include molecular fingerprints
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "calculate_descriptors.py")

    args = {
        "input": input_file,
        "output": output_file
    }

    # Build config for resource-intensive options
    if include_3d or include_fingerprints:
        config = {
            "include_3d": include_3d,
            "include_fingerprints": include_fingerprints
        }
        # Save temporary config file
        config_path = Path(input_file).parent / f"{Path(input_file).stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        args["config"] = str(config_path)

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"large_descriptors_{Path(input_file).stem}"
    )


# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def get_available_models() -> dict:
    """
    Get information about available permeability prediction models.

    Returns:
        Dictionary with model names, algorithms, and descriptor requirements
    """
    try:
        from predict_permeability import MODEL_CONFIGS

        models = {}
        for model_name, config in MODEL_CONFIGS.items():
            models[model_name] = {
                "name": config["name"],
                "algorithm": config["algorithm"],
                "num_descriptors": len(config["descriptors"]),
                "example_file": config.get("example_file"),
                "descriptors": config["descriptors"]
            }

        return {
            "status": "success",
            "models": models,
            "total_models": len(models)
        }

    except Exception as e:
        logger.error(f"Could not get model info: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def validate_input_file(input_file: str) -> dict:
    """
    Validate an input file for cyclic peptide analysis.

    Args:
        input_file: Path to file to validate

    Returns:
        Dictionary with validation results and file information
    """
    try:
        input_path = Path(input_file)

        if not input_path.exists():
            return {"status": "error", "error": "File does not exist"}

        result = {
            "status": "success",
            "file_path": str(input_path),
            "file_size": input_path.stat().st_size,
            "file_type": input_path.suffix
        }

        # Basic validation based on file type
        if input_path.suffix.lower() in ['.smi', '.smiles']:
            # Count SMILES entries
            with open(input_path) as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            result["molecule_count"] = len(lines)
            result["format"] = "SMILES"

        elif input_path.suffix.lower() == '.csv':
            # Check CSV structure
            import pandas as pd
            df = pd.read_csv(input_path)
            result["row_count"] = len(df)
            result["column_count"] = len(df.columns)
            result["columns"] = list(df.columns)
            result["format"] = "CSV"

        elif input_path.suffix.lower() == '.sdf':
            # Count molecules in SDF
            from rdkit import Chem
            suppl = Chem.SDMolSupplier(str(input_path))
            mol_count = len([mol for mol in suppl if mol is not None])
            result["molecule_count"] = mol_count
            result["format"] = "SDF"

        else:
            result["warning"] = f"Unknown file type: {input_path.suffix}"

        return result

    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def get_example_data() -> dict:
    """
    Get information about available example data for testing.

    Returns:
        Dictionary with paths to example files and descriptions
    """
    examples_dir = MCP_ROOT / "examples" / "data"

    if not examples_dir.exists():
        return {
            "status": "error",
            "error": "Examples directory not found. Run scripts with --demo flag to create example data."
        }

    example_files = {}

    for file_path in examples_dir.glob("*"):
        if file_path.is_file():
            file_info = {
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "type": file_path.suffix
            }

            # Add specific info based on file type
            if file_path.suffix == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    file_info["rows"] = len(df)
                    file_info["columns"] = len(df.columns)
                except Exception:
                    pass

            example_files[file_path.name] = file_info

    return {
        "status": "success",
        "examples_dir": str(examples_dir),
        "files": example_files
    }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
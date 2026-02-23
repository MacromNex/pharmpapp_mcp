"""Shared utilities for the MCP server."""

from pathlib import Path
import sys

# Add scripts directory to Python path
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR / "lib"))

def get_script_path(script_name: str) -> Path:
    """Get full path to a script."""
    return SCRIPTS_DIR / script_name

def get_examples_dir() -> Path:
    """Get examples directory path."""
    return MCP_ROOT / "examples"

def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists and return Path object."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
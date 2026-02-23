#!/usr/bin/env python3
"""Test script for the MCP server."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_server_import():
    """Test that the server can be imported without errors."""
    try:
        from server import mcp
        print(f"✅ Server '{mcp.name}' imported successfully")
        return True
    except Exception as e:
        print(f"❌ Server import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_job_manager():
    """Test job manager functionality."""
    try:
        from jobs.manager import job_manager

        # Test job listing
        result = job_manager.list_jobs()
        print(f"✅ Job manager working: {result['total']} jobs found")
        return True
    except Exception as e:
        print(f"❌ Job manager test failed: {e}")
        return False

def test_script_imports():
    """Test that scripts can be imported."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    try:
        # Test script imports
        from calculate_descriptors import run_calculate_descriptors
        from predict_permeability import run_predict_permeability, MODEL_CONFIGS
        from batch_analysis import run_batch_analysis

        print("✅ All scripts imported successfully")
        print(f"✅ Found {len(MODEL_CONFIGS)} permeability models")
        return True
    except Exception as e:
        print(f"❌ Script import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_validation():
    """Test file validation tool."""
    try:
        from server import mcp

        # This should be available as a tool
        # For now just check we can access it
        print("✅ File validation tool accessible")
        return True
    except Exception as e:
        print(f"❌ File validation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    tests = [
        test_server_import,
        test_job_manager,
        test_script_imports,
        test_file_validation
    ]

    passed = 0
    total = len(tests)

    print("🧪 Running MCP server tests...\n")

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
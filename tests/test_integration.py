#!/usr/bin/env python3
"""Integration tests with actual data."""

import sys
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

def test_descriptor_calculation():
    """Test descriptor calculation with demo data."""
    try:
        from calculate_descriptors import run_calculate_descriptors

        # Create temporary SMILES file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            f.write("C1C[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC3=CC=C(C=C3)O)CC(C)C\tcyclic_peptide_1\n")
            f.write("C[C@H]1NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC1=O)CC2=CC=CC=C2)CC(C)C)CC3=CC=C(C=C3)O)CC4=CNC5=CC=CC=C54\tcyclic_peptide_2\n")
            temp_input = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_output = f.name

        # Run calculation
        result = run_calculate_descriptors(temp_input, temp_output)

        # Check result
        if result and 'result' in result:
            print(f"✅ Descriptor calculation: {result['metadata']['molecules_processed']} molecules, {result['metadata']['descriptors_calculated']} descriptors")
            return True
        else:
            print("❌ Descriptor calculation returned no result")
            return False

    except Exception as e:
        print(f"❌ Descriptor calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_permeability_prediction():
    """Test permeability prediction with demo data."""
    try:
        from predict_permeability import run_predict_permeability

        # Use demo mode
        result = run_predict_permeability(
            input_file="dummy",  # Not used in demo mode
            model_name="caco2_c",
            demo=True
        )

        if result and 'result' in result:
            print(f"✅ Permeability prediction: processed data with model caco2_c")
            return True
        else:
            print("❌ Permeability prediction returned no result")
            return False

    except Exception as e:
        print(f"❌ Permeability prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_job_submission():
    """Test job submission system."""
    try:
        from jobs.manager import job_manager
        from pathlib import Path

        # Test submitting a simple job (descriptor calculation)
        script_path = str(Path(__file__).parent.parent / "scripts" / "calculate_descriptors.py")

        result = job_manager.submit_job(
            script_path=script_path,
            args={
                "demo": True,
                "output": "test_output.csv"
            },
            job_name="test_job"
        )

        if result.get("status") == "submitted":
            job_id = result["job_id"]
            print(f"✅ Job submitted successfully: {job_id}")

            # Test job status
            status = job_manager.get_job_status(job_id)
            if status.get("status") != "error":
                print(f"✅ Job status check working: {status.get('status')}")
                return True
            else:
                print(f"❌ Job status error: {status}")
                return False
        else:
            print(f"❌ Job submission failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcp_tools():
    """Test MCP tools directly."""
    try:
        from server import (
            get_available_models,
            validate_input_file,
            list_jobs,
            get_example_data
        )

        # Test get_available_models
        models = get_available_models()
        if models.get("status") == "success":
            print(f"✅ Available models: {models['total_models']} models found")
        else:
            print(f"❌ Get models failed: {models}")
            return False

        # Test list_jobs
        jobs = list_jobs()
        if jobs.get("status") == "success":
            print(f"✅ List jobs: {jobs['total']} jobs found")
        else:
            print(f"❌ List jobs failed: {jobs}")
            return False

        # Test example data
        examples = get_example_data()
        if examples.get("status") == "success" or "not found" in str(examples.get("error", "")):
            print("✅ Example data check working (may not have examples yet)")
        else:
            print(f"❌ Example data check failed: {examples}")
            return False

        return True

    except Exception as e:
        print(f"❌ MCP tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_integration_tests():
    """Run all integration tests."""
    tests = [
        test_descriptor_calculation,
        test_permeability_prediction,
        test_job_submission,
        test_mcp_tools
    ]

    passed = 0
    total = len(tests)

    print("🔬 Running integration tests...\n")

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print()

    print(f"📊 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some integration tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
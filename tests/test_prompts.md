# Step 7: MCP Integration Test Prompts

This file contains all test prompts for validating the Cyclic Peptide MCP server integration with Claude Code and other LLM clients.

## Tool Discovery Tests

### Prompt 1: List All Available Tools
"What MCP tools are available from cycpep-tools? Give me a brief description of each."

**Expected Response:** Should list all 13 tools:
- Job management tools (6): get_job_status, get_job_result, get_job_log, cancel_job, list_jobs, cleanup_old_jobs
- Sync calculation tools (2): calculate_cyclic_peptide_descriptors, predict_cyclic_peptide_permeability
- Submit API tools (2): submit_batch_analysis, submit_large_descriptor_calculation
- Utility tools (3): get_available_models, validate_input_file, get_example_data

### Prompt 2: Tool Details
"Explain how to use the calculate_cyclic_peptide_descriptors tool, including all parameters and return values."

**Expected Response:** Should provide detailed documentation about the tool's parameters, usage, and output format.

## Sync Tool Tests

### Prompt 3: Basic Property Calculation
"Use the validate_input_file tool to check if this SMILES file exists: examples/data/cyclic_peptides.smi"

**Expected Response:** Should validate the file and return file information including molecule count and format.

### Prompt 4: Calculate Descriptors for Demo Data
"Use calculate_cyclic_peptide_descriptors on the example SMILES file examples/data/cyclic_peptides.smi. Save results to test_descriptors.csv"

**Expected Response:** Should calculate descriptors and return success with calculated descriptor counts and output file path.

### Prompt 5: Model Information Query
"Use get_available_models to show me what permeability prediction models are available."

**Expected Response:** Should list all 5 available models: rrck_c, pampa_c, caco2_l, caco2_c, caco2_a with their algorithms and descriptor requirements.

### Prompt 6: Predict Permeability
"Use predict_cyclic_peptide_permeability to predict PAMPA permeability for the descriptor file test_descriptors.csv. Use the pampa_c model."

**Expected Response:** Should predict permeability values and return predictions with model performance metrics.

### Prompt 7: Error Handling Test
"Try to calculate descriptors for a non-existent file: fake_file.smi"

**Expected Response:** Should return an error message about file not found.

### Prompt 8: Get Example Data
"Use get_example_data to show me what example files are available for testing."

**Expected Response:** Should list all files in examples/data/ with their sizes and types.

## Submit API Tests (Long-Running Tasks)

### Prompt 9: Submit Large Descriptor Calculation
"Submit a large descriptor calculation job using submit_large_descriptor_calculation for examples/data/cyclic_peptides.smi with 3D descriptors and fingerprints enabled. Save to large_descriptors.csv"

**Expected Response:** Should return job submission confirmation with job_id.

### Prompt 10: Check Job Status
"Check the status of job [JOB_ID_FROM_PROMPT_9] using get_job_status."

**Expected Response:** Should return job status (pending/running/completed) with timestamps.

### Prompt 11: Get Job Logs
"Show me the last 20 lines of logs for job [JOB_ID_FROM_PROMPT_9] using get_job_log."

**Expected Response:** Should return recent log entries from the job execution.

### Prompt 12: Get Job Results
"When job [JOB_ID_FROM_PROMPT_9] is completed, get its results using get_job_result."

**Expected Response:** Should return job results with output file information and row counts.

### Prompt 13: Submit Batch Analysis
"Submit a batch analysis job using submit_batch_analysis for the descriptor file test_descriptors.csv. Save to batch_results/ and test all available models."

**Expected Response:** Should return batch job submission confirmation with job_id.

### Prompt 14: List All Jobs
"Use list_jobs to show me all submitted jobs and their current status."

**Expected Response:** Should list all jobs with their IDs, names, status, and submission times.

### Prompt 15: Cancel Job (if applicable)
"Cancel any running job using cancel_job."

**Expected Response:** Should successfully cancel the job if it's running, or return error if not cancellable.

## Batch Processing Tests

### Prompt 16: Multiple Model Testing
"I want to test multiple permeability models on the same data. First calculate descriptors for examples/data/cyclic_peptides.smi, then predict using all 5 available models (rrck_c, pampa_c, caco2_l, caco2_c, caco2_a)."

**Expected Response:** Should execute descriptor calculation followed by multiple prediction runs, showing results for each model.

### Prompt 17: Batch Job Status Check
"Check the status of the batch analysis job [BATCH_JOB_ID] and show me the results when it's complete."

**Expected Response:** Should monitor batch job and display comprehensive results when finished.

## End-to-End Real-World Scenarios

### Prompt 18: Complete Drug Discovery Pipeline
"I have a set of cyclic peptide SMILES in examples/data/cyclic_peptides.smi. Run a complete analysis:
1. Validate the input file
2. Calculate molecular descriptors with 3D features
3. Predict permeability using the PAMPA model
4. Show me compounds with molecular weight < 800 and predicted permeability > 0.5"

**Expected Response:** Should execute full pipeline and filter results based on criteria.

### Prompt 19: Comparative Model Analysis
"For the cyclic peptides in examples/data/cyclic_peptides.smi:
1. Calculate descriptors
2. Run predictions with both Caco-2 models (caco2_c and caco2_a)
3. Compare the predictions and identify any significant differences"

**Expected Response:** Should run comparative analysis and highlight differences between model predictions.

### Prompt 20: Virtual Screening Workflow
"Perform a virtual screening workflow:
1. Calculate descriptors for examples/data/cyclic_peptides.smi including fingerprints
2. Predict RRCK permeability
3. Identify the top 5 compounds with highest predicted permeability
4. Show their SMILES, molecular weight, and permeability scores"

**Expected Response:** Should complete screening and rank compounds by permeability.

### Prompt 21: Error Recovery Test
"Submit a descriptor calculation job for a large file that doesn't exist, then check its status and logs to see how errors are handled."

**Expected Response:** Should demonstrate error handling in the job system with clear error messages.

### Prompt 22: Job Management Workflow
"Show me how to manage multiple jobs:
1. Submit 2 different jobs simultaneously
2. List all jobs
3. Check status of both
4. Cancel one of them if it's still running
5. Get results from the completed job"

**Expected Response:** Should demonstrate complete job lifecycle management.

## Performance and Edge Case Tests

### Prompt 23: Large Dataset Test (if available)
"If we have a larger SMILES file, test the performance by calculating descriptors with all features enabled and measuring execution time."

**Expected Response:** Should handle larger datasets appropriately, either through sync or submit API.

### Prompt 24: Invalid SMILES Handling
"Create a test file with invalid SMILES strings and see how the descriptor calculation handles them."

**Expected Response:** Should gracefully handle invalid SMILES with appropriate error messages.

### Prompt 25: Resource Cleanup Test
"Use cleanup_old_jobs to clean up jobs older than 1 day and verify the cleanup works correctly."

**Expected Response:** Should successfully clean up old jobs and report the number of jobs removed.

## Final Validation Test

### Prompt 26: Complete System Test
"Demonstrate the full capabilities of the cyclic peptide MCP tools by:
1. Listing all available tools
2. Getting available models
3. Validating example data
4. Running both sync and async operations
5. Managing job lifecycle
6. Showing results interpretation

This should prove that all components are working correctly for a new user."

**Expected Response:** Should execute comprehensive demonstration showing all major features working correctly.

## Success Criteria for Each Test

### Sync Tools (Prompts 1-8)
- ✅ Response time < 30 seconds
- ✅ Structured JSON/dict responses
- ✅ Clear error messages for invalid inputs
- ✅ Correct file validation and processing

### Submit API (Prompts 9-15)
- ✅ Job submission returns valid job_id
- ✅ Status tracking works correctly
- ✅ Log retrieval functions properly
- ✅ Results can be retrieved when completed
- ✅ Job cancellation works for running jobs

### Batch Processing (Prompts 16-17)
- ✅ Multiple operations execute correctly
- ✅ Results from different models are comparable
- ✅ Batch jobs complete successfully

### Real-World Scenarios (Prompts 18-22)
- ✅ Complete workflows execute end-to-end
- ✅ Data filtering and analysis work correctly
- ✅ Error handling is robust
- ✅ Job management demonstrates full lifecycle

### Performance Tests (Prompts 23-25)
- ✅ Large datasets are handled appropriately
- ✅ Invalid data is handled gracefully
- ✅ Resource cleanup functions correctly

### Integration Test (Prompt 26)
- ✅ All tools are accessible and functional
- ✅ System demonstrates production readiness
- ✅ User experience is smooth and intuitive
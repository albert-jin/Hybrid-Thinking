#!/bin/bash

# --- 1. Environment Setup ---
# Replace with your specific conda environment path
export PATH="[YOUR_CONDA_ENV_PATH]/bin:${PATH}"
# Set your API Key if required by evaluators like LiveCodeBench
export OPENAI_API_KEY=""

# --- 2. Path Configuration ---
# Define your working directory and results directory
WORK_DIR=""
RESULTS_DIR=""

# Change to the working directory
cd "$WORK_DIR" || { echo "Directory not found: $WORK_DIR"; exit 1; }

# --- 3. Execution ---
echo "--- Starting Code Dataset Re-evaluation Process ---"
echo "Scanning Directory: $RESULTS_DIR"

# Execute the standalone Python re-evaluation script
python standalone_code_reeval.py --results_dir "$RESULTS_DIR"

echo "--- Re-evaluation Completed ---"
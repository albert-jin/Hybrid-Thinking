#!/bin/bash

# --- 1. Environment Setup ---
# Replace with your specific conda environment path
export PATH="[YOUR_CONDA_ENV_PATH]/bin:${PATH}"
# Set your API Key (required by some LiveCodeBench evaluators)
export OPENAI_API_KEY=""

# --- 2. Directory Configuration ---
# Define your main working directory
WORK_DIR=""
# Define the directory where result files are located
RESULTS_DIR=""

# Navigate to the working directory
cd "$WORK_DIR" || { echo "Directory not found: $WORK_DIR"; exit 1; }

# --- 3. Execution Block ---
echo "==============================================="
echo "Evaluating LiveCodeBench using local library"
echo "Local library path: ${WORK_DIR}/LiveCodeBench_pkg"
echo "Scanning results directory: $RESULTS_DIR"
echo "==============================================="

# Execute the local evaluation script
python local_lcb_eval.py --scan_dir "$RESULTS_DIR"
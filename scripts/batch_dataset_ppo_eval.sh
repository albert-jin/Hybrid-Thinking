#!/bin/bash

# --- 1. Environment Setup ---
# Replace with your virtual environment path
export PATH="[YOUR_CONDA_ENV_PATH]/bin:${PATH}"
# Set your API Key here
export OPENAI_API_KEY=""

# --- 2. Path and Parameter Configuration ---
# Base paths for script and datasets
EVAL_SCRIPT_PATH=""
DATASETS_BASE_DIR=""

# Model and weight paths
MODEL_PATH=""
# Path to the PPO checkpoint to be verified
PPO_CHECKPOINT_PATH=""

# Automatically define the directory for logs based on checkpoint path
CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")

# Evaluation mode and range
FORCE_MODE=""        # e.g., "ppo"
START_IDX=0
END_IDX=0            # Set a large number to cover all samples
EVAL_BATCH_SIZE=0

# Resource configuration
NUM_GPUS=0
MAX_RUNNING_REQUESTS=0
MEM_FRAC=0.0

# Output directory settings
OUTPUT_BASE_DIR=""
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)

# --- 3. Dataset List ---
# Add dataset filenames (without .json extension) to the array below
DATASETS_TO_EVAL=(
    ""
)

# --- 4. Evaluation Function ---
run_evaluation() {
    local DATASET_NAME=$1
    local EVAL_DATASET_PATH="${DATASETS_BASE_DIR}/${DATASET_NAME}.json"

    # Define the log file path using an absolute path
    local LOG_FILE_PATH="${CHECKPOINT_DIR}/eval_log_${DATASET_NAME}_MODE_${FORCE_MODE}_on_${CHECKPOINT_NAME}.log"

    echo "=========================================================================="
    echo "--- [START] Evaluating Dataset: $DATASET_NAME (Mode: $FORCE_MODE) ---"
    echo "Checkpoint: $PPO_CHECKPOINT_PATH"
    echo "Log path: $LOG_FILE_PATH"
    echo "=========================================================================="

    # Run the evaluation script and log both stdout and stderr
    (
        echo "Model Path: $MODEL_PATH"
        echo "Agent Path: $PPO_CHECKPOINT_PATH"
        echo "Dataset Path: $EVAL_DATASET_PATH"
        echo "Mode: $FORCE_MODE"
        echo "--------------------------"

        python -u "$EVAL_SCRIPT_PATH" \
            --model_name "$MODEL_PATH" \
            --num_gpus $NUM_GPUS \
            --max_running_requests $MAX_RUNNING_REQUESTS \
            --mem_fraction_static $MEM_FRAC \
            --log_level "info" \
            --disable_overlap_schedule \
            --enable_soft_thinking \
            --max_topk 10 \
            --dataset_path "$EVAL_DATASET_PATH" \
            --batch_size $EVAL_BATCH_SIZE \
            --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
            --api_base "" \
            --api_key "$OPENAI_API_KEY" \
            --judge_model_name "" \
            --use_llm_judge \
            --max_generated_tokens 32768 \
            --temperature 0.6 \
            --top_p 0.95 \
            --top_k 30 \
            --force_mode "$FORCE_MODE" \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_BASE_DIR
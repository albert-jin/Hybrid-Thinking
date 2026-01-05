#!/bin/bash

# --- 1. Environment Setup ---
# Replace with your virtual environment path
export PATH="[YOUR_CONDA_ENV_PATH]/bin:${PATH}"
# Ensure API Key is set
export OPENAI_API_KEY=""

# --- 2. Path and Parameter Configuration ---
# Base paths for script and datasets
EVAL_SCRIPT_PATH=""
DATASETS_BASE_DIR=""

# Model and checkpoint paths
MODEL_PATH=""
# PPO checkpoint path to be verified
PPO_CHECKPOINT_PATH=""

# Define CHECKPOINT_DIR as an absolute path
CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")

# Evaluation mode and range
FORCE_MODE=""        # e.g., "ppo"
START_IDX=0
END_IDX=100000       # Set a large number to cover all samples
EVAL_BATCH_SIZE=64

# Resource configuration
NUM_GPUS=4
MAX_RUNNING_REQUESTS=32
MEM_FRAC=0.8

# Output directory
OUTPUT_BASE_DIR=""
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)

# --- 3. Define the list of datasets to evaluate ---
# Fill in your dataset names (without .json extension)
DATASETS_TO_EVAL=(
    ""
)

# --- 4. Define evaluation function (encapsulates loop logic) ---
run_evaluation() {
    local DATASET_NAME=$1
    local EVAL_DATASET_PATH="${DATASETS_BASE_DIR}/${DATASET_NAME}.json"

    # Define log file path
    local LOG_FILE_PATH="${CHECKPOINT_DIR}/eval_log_${DATASET_NAME}_MODE_${FORCE_MODE}_on_${CHECKPOINT_NAME}.log"

    echo "=========================================================================="
    echo "--- [START] Evaluating Dataset: $DATASET_NAME (Mode: $FORCE_MODE) ---"
    echo "Using Checkpoint: $PPO_CHECKPOINT_PATH"
    echo "Log saved to: $LOG_FILE_PATH"
    echo "=========================================================================="

    # Execute evaluation script and pipe output to both console and log file
    (
        echo "Model Path: $MODEL_PATH"
        echo "Agent Path: $PPO_CHECKPOINT_PATH"
        echo "Dataset: $EVAL_DATASET_PATH"
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
            --output_dir "$OUTPUT_BASE_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX

        echo "--- Evaluation script finished (Dataset: $DATASET_NAME) ---"
    ) 2>&1 | tee "$LOG_FILE_PATH"
}

# --- 5. Loop through all datasets ---
for dataset in "${DATASETS_TO_EVAL[@]}"; do
    if [ -n "$dataset" ]; then
        run_evaluation "$dataset"
    fi
done

echo "--- All PPO mode evaluations completed ---"
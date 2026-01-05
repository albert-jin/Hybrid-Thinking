#!/bin/bash

# --- 1. Environment Setup ---
# Replace with your specific conda environment path
export PATH="[YOUR_CONDA_ENV_PATH]/bin:${PATH}"
# Set your API Key
export OPENAI_API_KEY=""

# --- 2. Core Parameter Configuration ---
# Model paths and identifiers
MODEL_PATH=""
MODEL_ID_SCOPE=""

# Resource configuration
NUM_GPUS=0
MAX_RUNNING_REQUESTS=0
MEM_FRAC=0.0

# Training configuration
TRAIN_DATASET_PATH=""
WRONG_QUESTION_SET_PATH=""
WRONG_QUESTION_PROB=0.0

TRAIN_BATCH_SIZE=0
NUM_STEPS=0

# Save and Evaluation intervals
SAVE_DIR="ppo_checkpoints/[PROJECT_NAME]_$(date +%Y%m%d_%H%M%S)"
SAVE_INTERVAL=0
EVAL_INTERVAL=0

# --- Training Log Parameters ---
LOG_TRAIN_RESULTS="--log_train_results" # Enable logging
TRAIN_LOG_INTERVAL=0                    # Interval for saving logs and printing accuracy

# Define log file path
LOG_FILE_PATH="${SAVE_DIR}.log"

# --- 3. Execution Block ---
(
    echo "--- Starting PPO Controller Step-Based Training ---"
    echo "Real-time logs: $LOG_FILE_PATH"
    echo "Detailed results: ${SAVE_DIR}/train_results_log.jsonl"
    echo "Model Path: $MODEL_PATH"
    echo "Save Directory: $SAVE_DIR"
    echo "--------------------------------------------------"

    python -u train_ppo_controller.py \
        --model_name "$MODEL_PATH" \
        --model_id_scope "$MODEL_ID_SCOPE" \
        --num_gpus $NUM_GPUS \
        --max_running_requests $MAX_RUNNING_REQUESTS \
        --mem_fraction_static $MEM_FRAC \
        --log_level "info" \
        \
        --disable_overlap_schedule \
        --enable_soft_thinking \
        --max_topk 10 \
        \
        --train_dataset "" \
        --dataset_path "$TRAIN_DATASET_PATH" \
        --eval_dataset_path "" \
        \
        --num_steps $NUM_STEPS \
        --wrong_question_set_path "$WRONG_QUESTION_SET_PATH" \
        --wrong_question_prob $WRONG_QUESTION_PROB \
        --eval_interval $EVAL_INTERVAL \
        \
        --batch_size $TRAIN_BATCH_SIZE \
        --save_dir "$SAVE_DIR" \
        --save_interval $SAVE_INTERVAL \
        \
        $LOG_TRAIN_RESULTS \
        --train_log_interval $TRAIN_LOG_INTERVAL \
        \
        --api_base "" \
        --api_key "$OPENAI_API_KEY" \
        --judge_model_name "" \
        --use_llm_judge \
        \
        --max_generated_tokens 1024 \
        --temperature 0.6 \
        --top_p 0.95 \
        --think_end_str "</think>"

    echo "--- PPO Training Script Execution Finished ---"
) 2>&1 | tee "$LOG_FILE_PATH"
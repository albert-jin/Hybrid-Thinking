#!/bin/bash

# --- 1. 环境设置 ---
export PATH="/root/.conda/envs/st/bin:${PATH}"
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9"

# --- 2. 核心参数配置 ---
MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
MODEL_ID_SCOPE="qwen/QwQ-32B"

# 资源配置
NUM_GPUS=4
MAX_RUNNING_REQUESTS=64
MEM_FRAC=0.8

# 训练配置
TRAIN_DATASET_PATH="./datasets/train_gsm8k.json"
WRONG_QUESTION_SET_PATH="./datasets/wrong_questions.json"
WRONG_QUESTION_PROB=0.3

TRAIN_BATCH_SIZE=64
NUM_STEPS=20000

SAVE_DIR="ppo_checkpoints/gsm8k_controller_$(date +%Y%m%d_%H%M%S)"
SAVE_INTERVAL=100
EVAL_INTERVAL=500

# <--- 新增：训练日志参数 ---
LOG_TRAIN_RESULTS="--log_train_results" # <-- 开启日志
TRAIN_LOG_INTERVAL=100                  # <-- 每 100 步保存一次日志并打印训练准确率
# <--- 新增结束 ---

# 日志文件
LOG_FILE_PATH="${SAVE_DIR}.log"
(
    echo "--- 启动 PPO 控制器 Step-Based 训练 ---"
    echo "日志将实时保存到: $LOG_FILE_PATH"
    echo "详细训练结果将保存到: ${SAVE_DIR}/train_results_log.jsonl" # <--- 新增提示
    echo "模型路径: $MODEL_PATH"
    echo "保存目录: $SAVE_DIR"
    echo "--------------------------"

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
        --train_dataset "train_gsm8k" \
        --dataset_path "$TRAIN_DATASET_PATH" \
        --eval_dataset_path "/root/shared-nvme/gj/Hybrid-Thinking/datasets/gsm8k.json" \
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
        # <--- 新增的参数 ---
        $LOG_TRAIN_RESULTS \
        --train_log_interval $TRAIN_LOG_INTERVAL \
        # <--- 新增结束 ---
        \
        --api_base "https://api.deepseek.com/v1" \
        --api_key "$OPENAI_API_KEY" \
        --judge_model_name "deepseek-chat" \
        --use_llm_judge \
        \
        --max_generated_tokens 1024 \
        --temperature 0.6 \
        --top_p 0.95 \
        --think_end_str "</think>"

    echo "--- PPO 训练脚本执行完毕 ---"
) 2>&1 | tee "$LOG_FILE_PATH"
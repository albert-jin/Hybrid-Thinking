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

# --- 训练配置 (已更新为 Step-based 逻辑) ---
TRAIN_DATASET_PATH="./datasets/train_gsm8k.json"           # (M) 主训练集
WRONG_QUESTION_SET_PATH="./datasets/wrong_questions.json"  # (N) 错题本 (请确保此文件存在, 否则将只使用 M)
WRONG_QUESTION_PROB=0.3                                    # (K) 30% 的概率从 N 中抽取

TRAIN_BATCH_SIZE=64
NUM_STEPS=5000                                             # (新) 总共训练 5000 个批次
# NUM_EPOCHS=3                                             # (旧) 已被 NUM_STEPS 替代

SAVE_DIR="ppo_checkpoints/gsm8k_controller_$(date +%Y%m%d_%H%M%S)"
SAVE_INTERVAL=100                                          # 每 100 步保存一次
EVAL_INTERVAL=500                                          # 每 500 步验证一次
# --- 更新结束 ---

# 日志文件 (与之前相同)
LOG_FILE_PATH="${SAVE_DIR}.log"
(
    echo "--- 启动 PPO 控制器 Step-Based 训练 ---"
    echo "日志将实时保存到: $LOG_FILE_PATH"
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
        # <--- 新增的参数 ---
        --num_steps $NUM_STEPS \
        --wrong_question_set_path "$WRONG_QUESTION_SET_PATH" \
        --wrong_question_prob $WRONG_QUESTION_PROB \
        --eval_interval $EVAL_INTERVAL \
        # <--- 新增结束 ---
        \
        --batch_size $TRAIN_BATCH_SIZE \
        --save_dir "$SAVE_DIR" \
        --save_interval $SAVE_INTERVAL \
        \
        --api_base "https://api.deepseek.com/v1" \
        --api_key "$OPENAI_API_KEY" \
        --judge_model_name "deepseek-chat" \
        \
        --max_generated_tokens 1024 \
        --temperature 0.6 \
        --top_p 0.95 \
        --think_end_str "</think>"

    echo "--- PPO 训练脚本执行完毕 ---"
) 2>&1 | tee "$LOG_FILE_PATH"
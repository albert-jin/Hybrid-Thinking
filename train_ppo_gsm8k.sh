#!/bin/bash

# --- 1. 环境设置 ---
# 激活 Conda 环境
export PATH="/root/.conda/envs/st/bin:${PATH}"
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9"

# --- 2. 核心参数配置 ---
# (请根据需要修改这些变量)

# 模型路径 (使用您 QwQ-32B 的路径)
MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
MODEL_ID_SCOPE="qwen/QwQ-32B"

# 资源配置
NUM_GPUS=4
MAX_RUNNING_REQUESTS=64 # (PPO 训练) 最大并发请求数
MEM_FRAC=0.8

# 训练配置
TRAIN_DATASET_PATH="./datasets/train_gsm8k.json"
TRAIN_BATCH_SIZE=64  # PPO 训练的批大小
NUM_EPOCHS=3       # 训练轮数
SAVE_DIR="ppo_checkpoints/gsm8k_controller_$(date +%Y%m%d_%H%M%S)"
SAVE_INTERVAL=50   # 每 50 个训练步骤保存一次 Agent 权重
LOG_FILE_PATH="${SAVE_DIR}.log"
# --- 3. 执行 PPO 训练脚本 ---

echo "--- 启动 PPO 控制器训练 ---"
echo "日志将实时保存到: $LOG_FILE_PATH"

# <--- 修改：使用 ( ... ) 2>&1 | tee 捕获所有输出 ---
# 这样会同时在屏幕上显示，并写入 $LOG_FILE_PATH 文件
(
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
    --num_epochs $NUM_EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --save_dir "$SAVE_DIR" \
    --save_interval $SAVE_INTERVAL \
    \
    --api_base "https://api.deepseek.com/v1" \
    --api_key "$API_KEY" \
    --judge_model_name "deepseek-chat" \
    \
    --max_generated_tokens 1024 \
    --temperature 0.6 \
    --top_p 0.95 \
    --think_end_str "</think>"

echo "--- PPO 训练脚本执行完毕 ---"
) 2>&1 | tee "$LOG_FILE_PATH"
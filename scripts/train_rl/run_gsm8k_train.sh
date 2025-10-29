#!/bin/bash

# 脚本: run_gsm8k_train.sh
# 目的: 启动 Hybrid-Thinking 项目的 RL 策略选择器训练。
# 位置: /workspace/Hybrid-Thinking/scripts/train_rl/

# --- 配置 ---

# 设置脚本出错时立即退出
set -e
set -u
set -o pipefail

# 项目根目录 (相对于此脚本的位置)
PROJECT_ROOT="/workspace/Hybrid-Thinking"

# 模型和数据集路径 (根据您的提供)
MODEL_DIR="/workspace/downloaded_models/Qwen/QwQ-32B"
DATASET_PATH="${PROJECT_ROOT}/datasets/train_gsm8k.json"
DATASET_NAME="gsm8k" # 用于评估器映射

# 输出目录 (用于保存检查点和日志)
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints_rl_gsm8k_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${CHECKPOINT_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# GPU 配置
NUM_GPUS=1 # 使用多少个 GPU 进行 Tensor Parallelism (tp_size)

# 训练超参数 (可以根据需要调整)
TOTAL_STEPS=100000      # 总训练步数 (解决的问题数量)
SAVE_INTERVAL=5000     # 每隔多少步保存一次检查点
LOG_INTERVAL=100       # 每隔多少步记录一次日志
TRAIN_INTERVAL=4       # 每解决多少个问题训练一次 DQN
BATCH_SIZE=32          # DQN 训练的批次大小
LEARNING_RATE=1e-5     # DQN 学习率
STATE_MAX_LEN=35       # RL 状态的最大序列长度

# SGLang 生成参数
MAX_GENERATED_TOKENS=2048
TEMPERATURE=0.6
TOP_P=0.95

# 恢复训练 (如果需要，取消注释并设置路径)
# LOAD_CHECKPOINT_PATH="/path/to/your/checkpoint.pt"
LOAD_CHECKPOINT_ARG=""
# if [ -n "${LOAD_CHECKPOINT_PATH}" ]; then
#   LOAD_CHECKPOINT_ARG="--load_checkpoint ${LOAD_CHECKPOINT_PATH}"
# fi

# 随机种子
RANDOM_SEED=42

# --- 准备 ---

# 创建输出目录
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"

# 切换到项目根目录执行 Python 脚本
cd "${PROJECT_ROOT}" || exit 1

echo "======================================================"
echo "Starting RL Training for Hybrid Thinking"
echo "Model Path: ${MODEL_DIR}"
echo "Dataset Path: ${DATASET_PATH}"
echo "Dataset Name: ${DATASET_NAME}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Log File: ${LOG_FILE}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Total Steps: ${TOTAL_STEPS}"
echo "======================================================"

# --- 执行训练 ---

# 使用 tee 将输出同时打印到控制台和日志文件
# 注意：确保 train_rl_strategy.py 在 ${PROJECT_ROOT} 下
python train_rl_strategy.py \
    --model_path "${MODEL_DIR}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset "${DATASET_NAME}" \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --total_train_steps ${TOTAL_STEPS} \
    --save_interval_steps ${SAVE_INTERVAL} \
    --log_interval_steps ${LOG_INTERVAL} \
    --train_interval_steps ${TRAIN_INTERVAL} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --state_max_len ${STATE_MAX_LEN} \
    --enable_soft_thinking_global \
    --max_generated_tokens ${MAX_GENERATED_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --random_seed ${RANDOM_SEED} \
    ${LOAD_CHECKPOINT_ARG} \
    2>&1 | tee "${LOG_FILE}"

echo "======================================================"
echo "Training finished."
echo "Checkpoints saved in: ${CHECKPOINT_DIR}"
echo "Logs saved in: ${LOG_FILE}"
echo "======================================================"

# 可选：切换回原来的目录 (如果需要)
# cd - > /dev/null
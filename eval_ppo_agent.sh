#!/bin/bash

# --- 1. 环境设置 ---
export PATH="/root/.conda/envs/st/bin:${PATH}"
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9" # 用于 matheval 评判

# --- 2. 核心参数配置 ---
# (请根据需要修改这些变量)

# 模型路径
MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"

# !! 关键 !!
# !! 修改为您训练好的 Agent Checkpoint 文件的 *确切路径* !!
PPO_CHECKPOINT_PATH="/root/shared-nvme/gj/Hybrid-Thinking/ppo_checkpoints/gsm8k_controller_20251103_003927/ppo_agent_step_1250.pth" # <--- 示例路径，请修改为您真实的 .pth 文件

# 验证集
EVAL_DATASET_PATH="/root/shared-nvme/gj/Hybrid-Thinking/datasets/gsm8k_10.json"
EVAL_BATCH_SIZE=64 # 评估时使用的批次大小

# <--- 新增：结果保存目录和数据范围 --->
OUTPUT_DIR="eval_results"  # 结果将保存在 ./eval_results/gsm8k/ 目录中
START_IDX=0                # 从验证集的第 0 个样本开始
END_IDX=10                 # 到第 10 个样本结束 (您可以设置为 10000 来运行全部)
# <--- 新增结束 --->

# 资源配置
NUM_GPUS=4
MAX_RUNNING_REQUESTS=64
MEM_FRAC=0.8

# <--- 新增：定义日志文件路径 (与之前相同) ---
CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)
LOG_FILE_PATH="${CHECKPOINT_DIR}/eval_log_on_${CHECKPOINT_NAME}_${START_IDX}_${END_IDX}.log"
# <--- 新增结束 ---

# --- 3. 执行 PPO 评估脚本 (并保存日志) ---

echo "--- 启动 PPO Agent 评估 ---"
echo "日志将实时保存到: $LOG_FILE_PATH"
echo "详细JSON结果将保存到: $OUTPUT_DIR"

# <--- 修改：使用 ( ... ) 2>&1 | tee 捕获所有输出 ---
(
    echo "模型路径: $MODEL_PATH"
    echo "Agent 路径: $PPO_CHECKPOINT_PATH"
    echo "验证集: $EVAL_DATASET_PATH"
    echo "--------------------------"

    python -u eval_ppo_agent.py \
        --model_name "$MODEL_PATH" \
        --num_gpus $NUM_GPUS \
        --max_running_requests $MAX_RUNNING_REQUESTS \
        --mem_fraction_static $MEM_FRAC \
        --log_level "info" \
        \
        --disable_overlap_schedule \
        --enable_soft_thinking \
        --max_topk 10 \
        \
        --dataset_path "$EVAL_DATASET_PATH" \
        --batch_size $EVAL_BATCH_SIZE \
        \
        --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
        \
        --api_base "https://api.deepseek.com/v1" \
        --api_key "$OPENAI_API_KEY" \
        --judge_model_name "deepseek-chat" \
        \
        --max_generated_tokens 1024 \
        --temperature 0.6 \
        --top_p 0.95 \
        --force_mode hard  \
        --think_end_str "</think>" \
        --output_dir "$OUTPUT_DIR" \
        --start_idx $START_IDX \
        --end_idx $END_IDX

    echo "--- PPO 评估脚本执行完毕 ---"
) 2>&1 | tee "$LOG_FILE_PATH"
# <--- 修改结束 ---
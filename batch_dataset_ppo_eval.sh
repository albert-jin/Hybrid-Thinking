#!/bin/bash

# --- 1. 环境设置 ---
export PATH="/root/.conda/envs/st/bin:${PATH}"
# 确保设置 API Key
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9"

# --- 2. 路径和参数配置 ---
# 假设脚本运行在 Hybrid-Thinking 目录下，且 eval_ppo_agent.py 在同一目录
EVAL_SCRIPT_PATH="./eval_ppo_agent.py"
DATASETS_BASE_DIR="/root/shared-nvme/gj/Hybrid-Thinking/datasets"

# 模型和权重路径
MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
# !! 核心 !! 使用您需要验证的 PPO 权重块路径
PPO_CHECKPOINT_PATH="/root/shared-nvme/gj/Hybrid-Thinking/ppo_checkpoints/gsm8k_controller_20251105_041949/ppo_agent_step_13500.pth"

# === !!! 关键修正：添加这一行来定义 CHECKPOINT_DIR !!! ===
CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")
# ===============================================================

# 评估模式和范围
FORCE_MODE="ppo"   # 指定使用 PPO Agent 模式
START_IDX=0        # 从数据集开头开始
END_IDX=100000     # 结束索引（设置一个大数，以覆盖所有样本）
EVAL_BATCH_SIZE=64

# 资源配置
NUM_GPUS=4
MAX_RUNNING_REQUESTS=32
MEM_FRAC=0.8

# 输出目录 (结果将保存在 ./eval_results/[dataset_name]/ 目录下)
OUTPUT_BASE_DIR="./eval_results"
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)

# --- 3. 定义要评估的数据集列表 ---
# 使用您的文件列表中的对应 JSON 文件名（不含 .json）
DATASETS_TO_EVAL=(
    "math500"
    "aime2024"
    "gsm8k"
    "gpqa_diamond"
)

# --- 4. 定义评估函数 (封装循环逻辑) ---
run_evaluation() {
    local DATASET_NAME=$1
    local EVAL_DATASET_PATH="${DATASETS_BASE_DIR}/${DATASET_NAME}.json"

    # 现在 $CHECKPOINT_DIR 是一个完整的绝对路径
    local LOG_FILE_PATH="${CHECKPOINT_DIR}/eval_log_${DATASET_NAME}_MODE_${FORCE_MODE}_on_${CHECKPOINT_NAME}.log"

    echo "=========================================================================="
    echo "--- [START] 评估数据集： $DATASET_NAME (模式: $FORCE_MODE) ---"
    echo "使用权重: $PPO_CHECKPOINT_PATH"
    echo "日志将保存到: $LOG_FILE_PATH" # <-- 这次将打印出完整的绝对路径
    echo "=========================================================================="

    # 执行评估脚本，并将实时日志输出到控制台和文件
    (
        echo "模型路径: $MODEL_PATH"
        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "验证集: $EVAL_DATASET_PATH"
        echo "评估模式: $FORCE_MODE"
        echo "--------------------------"

        python -u "$EVAL_SCRIPT_PATH" \
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
            --use_llm_judge \
            \
            --max_generated_tokens 32768 \
            --temperature 0.6 \
            --top_p 0.95 \
            --force_mode "$FORCE_MODE" \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_BASE_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX

        echo "--- PPO 评估脚本执行完毕 (Dataset: $DATASET_NAME) ---"
    ) 2>&1 | tee "$LOG_FILE_PATH"
}

# --- 5. 循环执行所有数据集 ---
for dataset in "${DATASETS_TO_EVAL[@]}"; do
    run_evaluation "$dataset"
done

echo "--- 所有数据集的 PPO 模式评估已完成 ---"
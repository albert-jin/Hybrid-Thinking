#!/bin/bash

# --- 1. 环境设置 ---
export PATH="/root/.conda/envs/st/bin:${PATH}"
# 确保在脚本开始时设置一次 API Key
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9"

# --- 2. 核心参数配置 ---
MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
# 假设您只需要评估一个固定的 PPO 权重块来对比三种模式
PPO_CHECKPOINT_PATH="/root/shared-nvme/gj/Hybrid-Thinking/ppo_checkpoints/gsm8k_controller_20251103_003927/ppo_agent_step_1250.pth"
EVAL_DATASET_PATH="/root/shared-nvme/gj/Hybrid-Thinking/datasets/gsm8k_10.json"
EVAL_BATCH_SIZE=64

# 评估范围
START_IDX=0
END_IDX=10000

# 资源配置 (不变)
NUM_GPUS=4
MAX_RUNNING_REQUESTS=64
MEM_FRAC=0.8

# 日志和结果的基础路径
OUTPUT_DIR="eval_results"
CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)

# 定义要运行的模式列表
FORCE_MODES=("hard" "soft" "ppo")

# --- 3. 定义评估函数 (封装了重复的逻辑) ---
run_evaluation() {
    local MODE=$1

    # 为每种模式创建独特的日志文件路径
    local LOG_FILE_PATH="${CHECKPOINT_DIR}/eval_log_mode_${MODE}_on_${CHECKPOINT_NAME}_${START_IDX}_${END_IDX}.log"

    echo "====================================================="
    echo "--- 启动评估：MODE = $MODE ---"
    echo "日志将保存到: $LOG_FILE_PATH"
    echo "====================================================="

    # 使用 tee -a 实现追加模式，但在这里每次循环都使用独立的日志文件，所以覆盖是安全的
    (
        echo "模型路径: $MODEL_PATH"
        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "验证集: $EVAL_DATASET_PATH"
        echo "评估模式: $MODE"
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
            --use_llm_judge \
            \
            --max_generated_tokens 32768 \
            --temperature 0.6 \
            --top_p 0.95 \
            --force_mode "$MODE" \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX

        echo "--- PPO 评估脚本执行完毕 (Mode: $MODE) ---"
    ) 2>&1 | tee "$LOG_FILE_PATH"
}

# --- 4. 循环执行所有模式 ---
for mode in "${FORCE_MODES[@]}"; do
    run_evaluation "$mode"
done

echo "--- 所有三种模式的评估已完成 ---"
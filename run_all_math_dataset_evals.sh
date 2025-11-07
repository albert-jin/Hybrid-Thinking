#!/bin/bash

# --- 1. 环境设置 (与 eval 脚本相同) ---
export PATH="/root/.conda/envs/st/bin:${PATH}"
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9" # 用于 matheval 评判

# --- 2. 核心参数配置 (公共部分) ---
# (这些参数在所有评估中保持不变)

MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
PPO_CHECKPOINT_PATH="/root/shared-nvme/gj/Hybrid-Thinking/ppo_checkpoints/gsm8k_controller_20251105_041949/ppo_agent_step_13500.pth"

# 基础数据集目录 (来自您的 ls 输出)
DATASET_DIR="/root/shared-nvme/gj/Hybrid-Thinking/datasets"

# 评估参数
EVAL_BATCH_SIZE=64
OUTPUT_DIR="eval_results"  # Python 脚本会自动创建子目录 (例如: eval_results/gsm8k/)
START_IDX=0
END_IDX=10000              # 运行所有样本 (10000 足够大)

# 资源配置
NUM_GPUS=4
MAX_RUNNING_REQUESTS=64
MEM_FRAC=0.8

# PPO Agent 的基本信息 (用于日志命名)
CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)

# --- 3. 定义要串行评估的数据集 ---
# (根据您 ls 输出中的文件名)
DATASETS_TO_RUN=(
    "gsm8k"
    "math500"
    "aime2024"
    "gpqa_diamond"
)

# --- 4. 串行执行评估 ---

echo "====== 启动批量串行评估 ======"
echo "将评估 Agent: $PPO_CHECKPOINT_PATH"
echo "将依次运行以下数据集: ${DATASETS_TO_RUN[*]}"
echo "================================="

# 循环遍历定义好的数据集列表
for DATASET_NAME in "${DATASETS_TO_RUN[@]}"; do

    # 1. 设置特定于数据集的路径
    EVAL_DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}.json"

    # 2. 检查文件是否存在
    if [ ! -f "$EVAL_DATASET_PATH" ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! 警告: 未找到数据集 $EVAL_DATASET_PATH"
        echo "!! 跳过 [${DATASET_NAME}] 的评估。"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        continue
    fi

    # 3. <--- 关键：为每个数据集创建 *唯一* 的日志文件 ---
    # 日志名现在包含数据集名称，以防覆盖
    LOG_FILE_PATH="${CHECKPOINT_DIR}/eval_log_on_${CHECKPOINT_NAME}_DATASET_${DATASET_NAME}_${START_IDX}_${END_IDX}.log"

    # 4. 打印状态
    echo ""
    echo "----------------------------------------------------"
    echo "--- 正在启动 [${DATASET_NAME}] 的评估 ---"
    echo "--- 验证集: $EVAL_DATASET_PATH"
    echo "--- 日志将保存到: $LOG_FILE_PATH"
    echo "----------------------------------------------------"

    # 5. 执行 PPO 评估脚本 (与原脚本中的命令一致)
    # 使用 ( ... ) 2>&1 | tee 捕获所有输出到日志
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
            --use_llm_judge \
            \
            --max_generated_tokens 1024 \
            --temperature 0.6 \
            --top_p 0.95 \
            \
            # <-- 注意：这里使用了 'ppo' 模式 -->
            # 这将使用您训练好的 PPO Agent 来做决策 (这符合 eval_ppo_agent.py 的默认行为和目的)。
            # 您原始脚本中的 'hard' 模式将强制使用 Hard Thinking，而不评估 Agent 的决策。
            --force_mode ppo  \
            \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX

        echo "--- [${DATASET_NAME}] 评估脚本执行完毕 ---"
    ) 2>&1 | tee "$LOG_FILE_PATH"

    echo "--- [${DATASET_NAME}] 评估完成。日志已保存。"

done

echo "----------------------------------------------------"
echo "====== 批量串行评估全部完成 ======"
echo "所有结果（JSON 和日志）已保存。"
echo "JSON 结果位于: $OUTPUT_DIR/ (按数据集分子目录)"
echo "日志文件位于: $CHECKPOINT_DIR/"
echo "----------------------------------------------------"
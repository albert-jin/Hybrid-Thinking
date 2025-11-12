#!/bin/bash

# =======================================================
# 批量串行评估 *代码* PPO Agent 脚本
#
# (已更新: livecodebench 将在最后单独执行, 以防网络失败)
#
# 此脚本将为每个代码数据集执行一个 *两阶段* 评估：
# 1. 阶段 1 (GENERATE): 加载 PPO Agent 并生成所有代码。
# 2. 阶段 2 (RE-EVAL):  不加载模型，调用评估器 (humanevaleval/mbppeval)
#                     来安全地评判代码。
# =======================================================

# --- 1. 环境设置 ---
export PATH="/root/.conda/envs/st/bin:${PATH}"

# --- 2. 核心参数配置 (公共部分) ---
MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
PPO_CHECKPOINT_PATH="/root/shared-nvme/gj/Hybrid-Thinking/ppo_checkpoints/gsm8k_controller_20251105_041949/ppo_agent_step_13500.pth"

DATASET_DIR="/root/shared-nvme/gj/Hybrid-Thinking/datasets"
EVAL_BATCH_SIZE=64
NUM_SAMPLES=1
OUTPUT_DIR="eval_results"
START_IDX=0
END_IDX=10000

NUM_GPUS=4
MAX_RUNNING_REQUESTS=64
MEM_FRAC=0.8

CHECKPOINT_DIR=$(dirname "$PPO_CHECKPOINT_PATH")
CHECKPOINT_NAME=$(basename "$PPO_CHECKPOINT_PATH" .pth)

# --- 3. 定义要串行评估的 *代码* 数据集 ---
# (注意：livecodebench 将被单独处理)
DATASETS_TO_RUN_FIRST=(
    "humaneval"
    "mbpp"
)
DATASET_TO_RUN_LAST="livecodebench"


# =======================================================
# 4. 串行执行评估 (阶段 A: 本地数据集)
# =======================================================

echo "====== 启动 *代码* 批量串行评估 (两阶段) ======"
echo "将评估 Agent: $PPO_CHECKPOINT_PATH"
echo "将首先运行本地数据集: ${DATASETS_TO_RUN_FIRST[*]}"
echo "================================="

# 循环遍历 *非* livecodebench 的数据集
for DATASET_NAME in "${DATASETS_TO_RUN_FIRST[@]}"; do

    EVAL_DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}.json"

    if [ ! -f "$EVAL_DATASET_PATH" ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! 警告: 未找到数据集 $EVAL_DATASET_PATH"
        echo "!! 跳过 [${DATASET_NAME}] 的评估。"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        continue
    fi

    # -----------------------------------------------
    # 阶段 1: GENERATE (生成)
    # -----------------------------------------------
    LOG_FILE_PATH_GENERATE="${CHECKPOINT_DIR}/eval_log_on_${CHECKPOINT_NAME}_DATASET_${DATASET_NAME}_STAGE_1_GENERATE.log"

    echo ""
    echo "----------------------------------------------------"
    echo "--- 正在启动 [${DATASET_NAME}] :: 阶段 1 (GENERATE) ---"
    echo "--- 验证集: $EVAL_DATASET_PATH"
    echo "--- 日志将保存到: $LOG_FILE_PATH_GENERATE"
    echo "----------------------------------------------------"

    (
        echo "模型路径: $MODEL_PATH"
        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "验证集: $EVAL_DATASET_PATH"
        echo "模式: PPO (默认)"
        echo "阶段: 1 (GENERATE)"
        echo "--------------------------"

        python -u eval_code_ppo_agent.py \
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
            --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
            --force_mode ppo \
            \
            --dataset_path "$EVAL_DATASET_PATH" \
            --batch_size $EVAL_BATCH_SIZE \
            --num_samples $NUM_SAMPLES \
            \
            --max_generated_tokens 1024 \
            --temperature 0.6 \
            --top_p 0.95 \
            --repetition_penalty 1.0 \
            \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX

    ) 2>&1 | tee "$LOG_FILE_PATH_GENERATE"

    echo "--- [${DATASET_NAME}] 阶段 1 (GENERATE) 完成。---"


    # -----------------------------------------------
    # 阶段 2: RE-EVAL (评判)
    # -----------------------------------------------
    LOG_FILE_PATH_REEVAL="${CHECKPOINT_DIR}/eval_log_on_${CHECKPOINT_NAME}_DATASET_${DATASET_NAME}_STAGE_2_REEVAL.log"

    echo ""
    echo "----------------------------------------------------"
    echo "--- 正在启动 [${DATASET_NAME}] :: 阶段 2 (RE-EVAL) ---"
    echo "--- 日志将保存到: $LOG_FILE_PATH_REEVAL"
    echo "----------------------------------------------------"

    (
        echo "模型路径: $MODEL_PATH"
        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "验证集: $EVAL_DATASET_PATH"
        echo "模式: PPO (默认)"
        echo "阶段: 2 (RE-EVAL)"
        echo "--------------------------"

        python -u eval_code_ppo_agent.py \
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
            --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
            --force_mode ppo \
            \
            --dataset_path "$EVAL_DATASET_PATH" \
            --batch_size $EVAL_BATCH_SIZE \
            --num_samples $NUM_SAMPLES \
            \
            --max_generated_tokens 1024 \
            --temperature 0.6 \
            --top_p 0.95 \
            --repetition_penalty 1.0 \
            \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX \
            \
            --reeval

    ) 2>&1 | tee "$LOG_FILE_PATH_REEVAL"

    echo "--- [${DATASET_NAME}] 阶段 2 (RE-EVAL) 完成。---"

done # 结束本地数据集循环

echo "================================="
echo "=== 本地数据集 (humaneval, mbpp) 评估完成 ==="
echo "================================="


# =======================================================
# 5. 串行执行评估 (阶段 B: LiveCodeBench - 最后执行)
# =======================================================

echo ""
echo "====== 正在启动 [${DATASET_TO_RUN_LAST}] (最后执行) ======"
DATASET_NAME="$DATASET_TO_RUN_LAST"
EVAL_DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}.json"

if [ ! -f "$EVAL_DATASET_PATH" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! 警告: 未找到数据集 $EVAL_DATASET_PATH"
    echo "!! 跳过 [${DATASET_NAME}] 的评估。"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
else
    # -----------------------------------------------
    # 阶段 1: GENERATE (LiveCodeBench)
    # -----------------------------------------------
    LOG_FILE_PATH_GENERATE="${CHECKPOINT_DIR}/eval_log_on_${CHECKPOINT_NAME}_DATASET_${DATASET_NAME}_STAGE_1_GENERATE.log"

    echo ""
    echo "----------------------------------------------------"
    echo "--- 正在启动 [${DATASET_NAME}] :: 阶段 1 (GENERATE) ---"
    echo "--- 验证集: $EVAL_DATASET_PATH"
    echo "--- 日志将保存到: $LOG_FILE_PATH_GENERATE"
    echo "----------------------------------------------------"

    (
        echo "模型路径: $MODEL_PATH"
        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "验证集: $EVAL_DATASET_PATH"
        echo "模式: PPO (默认)"
        echo "阶段: 1 (GENERATE)"
        echo "--------------------------"

        python -u eval_code_ppo_agent.py \
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
            --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
            --force_mode ppo \
            \
            --dataset_path "$EVAL_DATASET_PATH" \
            --batch_size $EVAL_BATCH_SIZE \
            --num_samples $NUM_SAMPLES \
            \
            --max_generated_tokens 1024 \
            --temperature 0.6 \
            --top_p 0.95 \
            --repetition_penalty 1.0 \
            \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX

    ) 2>&1 | tee "$LOG_FILE_PATH_GENERATE"

    echo "--- [${DATASET_NAME}] 阶段 1 (GENERATE) 完成。---"


    # -----------------------------------------------
    # 阶段 2: RE-EVAL (LiveCodeBench)
    # -----------------------------------------------
    LOG_FILE_PATH_REEVAL="${CHECKPOINT_DIR}/eval_log_on_${CHECKPOINT_NAME}_DATASET_${DATASET_NAME}_STAGE_2_REEVAL.log"

    echo ""
    echo "----------------------------------------------------"
    echo "--- Gj 正在启动 [${DATASET_NAME}] :: 阶段 2 (RE-EVAL) ---"
    echo "--- (此阶段可能需要网络访问) ---"
    echo "--- 日志将保存到: $LOG_FILE_PATH_REEVAL"
    echo "----------------------------------------------------"

    (
        echo "模型路径: $MODEL_PATH"
        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "验证集: $EVAL_DATASET_PATH"
        echo "模式: PPO (默认)"
        echo "阶段: 2 (RE-EVAL)"
        echo "--------------------------"

        python -u eval_code_ppo_agent.py \
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
            --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
            --force_mode ppo \
            \
            --dataset_path "$EVAL_DATASET_PATH" \
            --batch_size $EVAL_BATCH_SIZE \
            --num_samples $NUM_SAMPLES \
            \
            --max_generated_tokens 1024 \
            --temperature 0.6 \
            --top_p 0.95 \
            --repetition_penalty 1.0 \
            \
            --think_end_str "</think>" \
            --output_dir "$OUTPUT_DIR" \
            --start_idx $START_IDX \
            --end_idx $END_IDX \
            \
            --reeval

    ) 2>&1 | tee "$LOG_FILE_PATH_REEVAL"

    echo "--- [${DATASET_NAME}] 阶段 2 (RE-EVAL) 完成。---"
fi

echo "----------------------------------------------------"
echo "====== *代码* 批量串行评估全部完成 ======"
echo "所有结果（JSON 和日志）已保存。"
echo "JSON 结果位于: $OUTPUT_DIR/ (按数据集分子目录)"
echo "日志文件位于: $CHECKPOINT_DIR/"
echo "----------------------------------------------------"
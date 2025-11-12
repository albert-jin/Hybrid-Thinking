#!/bin/bash

# =======================================================
# 仅重跑 *代码* PPO Agent 评估的 *阶段 2 (RE-EVAL)*
#
# 此脚本假设阶段 1 的 JSON 文件已经存在。
# 它将调用 eval_code_ppo_agent.py (需要已打上 --reeval_input_file 补丁)
# 来执行沙盒评判。
# =======================================================

# --- 1. 环境设置 ---
export PATH="/root/.conda/envs/st/bin:${PATH}"

# --- 2. 核心参数配置 (公共部分) ---
# (这些参数在评判阶段是必需的, 以便评估器 (如 LCB) 能正确加载配置)

MODEL_PATH="/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B"
PPO_CHECKPOINT_PATH="/root/shared-nvme/gj/Hybrid-Thinking/ppo_checkpoints/gsm8k_controller_20251105_041949/ppo_agent_step_13500.pth"
DATASET_DIR="/root/shared-nvme/gj/Hybrid-Thinking/datasets"
OUTPUT_DIR="eval_results"
NUM_SAMPLES=1 # 必须与生成时 SAMPLES_1 匹配

# --- 3. 定义要重跑评判的文件 (!!! 您提供的路径 !!!) ---
FILES_TO_REEVAL=(
    # [Dataset Name] [Path to Step 1 JSON file]
#    "humaneval /root/shared-nvme/gj/Hybrid-Thinking/eval_results/humaneval/eval_CODE_ppo_agent_step_13500_on_humaneval_MODE_PPO_REEVAL_GENERATE_SAMPLES_1_20251111_050245_results.json"
#    "mbpp /root/shared-nvme/gj/Hybrid-Thinking/eval_results/mbpp/eval_CODE_ppo_agent_step_13500_on_mbpp_MODE_PPO_REEVAL_GENERATE_SAMPLES_1_20251111_051056_results.json"
    "livecodebench /root/shared-nvme/gj/Hybrid-Thinking/eval_results/livecodebench/eval_CODE_ppo_agent_step_13500_on_livecodebench_MODE_PPO_REEVAL_GENERATE_SAMPLES_1_20251111_052230_results.json"
)

# --- 4. (可选) LCB 离线缓存设置 ---
# (如果 LCB 失败是因为网络, 请确保此路径正确)
LCB_CACHE_PATH="/root/shared-nvme/gj/Hybrid-Thinking/lcb_eval_data"


# --- 5. 串行执行重评判 ---

echo "====== 启动 *代码* 批量重评判 (仅 阶段 2) ======"
echo "将评估 Agent: $PPO_CHECKPOINT_PATH"
echo "================================="

# 循环遍历文件列表
for entry in "${FILES_TO_REEVAL[@]}"; do
    # 解析条目
    read -r DATASET_NAME INPUT_JSON_FILE <<< "$entry"

    # 对应的原始数据集文件 (eval_code_ppo_agent.py 仍然需要这个)
    EVAL_DATASET_PATH="${DATASET_DIR}/${DATASET_NAME}.json"

    # 创建日志文件
    LOG_FILE_PATH_REEVAL="${INPUT_JSON_FILE}.STEP2_RERUN.log"

    echo ""
    echo "----------------------------------------------------"
    echo "--- 正在重跑 [${DATASET_NAME}] :: 阶段 2 (RE-EVAL) ---"
    echo "--- 输入文件: $INPUT_JSON_FILE"
    echo "--- 日志将保存到: $LOG_FILE_PATH_REEVAL"
    echo "----------------------------------------------------"

    (
        # 检查是否为 LCB 并设置离线缓存
        if [ "$DATASET_NAME" == "livecodebench" ]; then
            echo "--- (!!!) 检测到 LiveCodeBench, 启用离线缓存 (!!!) ---"

            # (!!!) 关键修改：直接指向包含 'livecodebench___code_generation_lite' 的父目录
            export HF_DATASETS_CACHE="/root/shared-nvme/gj/Hybrid-Thinking"
            export HF_DATASETS_OFFLINE=1

            echo "环境变量设置: HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
            echo "环境变量设置: HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
        else
            # 确保 LCB 的设置不会污染其他数据集
            unset HF_DATASETS_CACHE
            unset HF_DATASETS_OFFLINE
        fi

        echo "Agent 路径: $PPO_CHECKPOINT_PATH"
        echo "模式: PPO (默认)"
        echo "阶段: 2 (RE-EVAL)"
        echo "--------------------------"

        # 运行评判
        python -u eval_code_ppo_agent.py \
            --model_name "$MODEL_PATH" \
            --ppo_agent_checkpoint_path "$PPO_CHECKPOINT_PATH" \
            --dataset_path "$EVAL_DATASET_PATH" \
            --num_samples $NUM_SAMPLES \
            --output_dir "$OUTPUT_DIR" \
            --force_mode ppo \
            \
            --log_level "info" \
            --enable_soft_thinking \
            --max_topk 10 \
            \
            --reeval \
            --reeval_input_file "$INPUT_JSON_FILE"

    ) 2>&1 | tee "$LOG_FILE_PATH_REEVAL"

    echo "--- [${DATASET_NAME}] 阶段 2 (RE-EVAL) 重跑完成。---"

done # 结束数据集循环

echo "----------------------------------------------------"
echo "====== *代码* 批量重评判全部完成 ======"
echo "所有新的结果（JSON 和日志）已保存在原始结果目录中。"
echo "----------------------------------------------------"
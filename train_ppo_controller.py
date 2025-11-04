#!/usr/bin/env python
"""
train_ppo_controller.py (混合采样 Step-based 训练版)

主训练脚本，用于"遥控"嵌入在 sglang 调度器内部的 PPO Agent。
它负责：
1. 加载 sglang 引擎 (引擎会自动加载 PPO Agent)。
2. 加载主训练集 M 和错题本 N。
3. 【新】按固定的 Step 步数进行训练。
4. 【新】在每一步中，按 K 概率混合 M (顺序) 和 N (随机) 的数据来构建批次。
5. 【新】每隔 N 步运行一次验证。
"""
import os
import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
import shutil
import sys
import random # (PPO): 用于打乱数据集
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client, AIMEEvaluator
from modelscope.hub.snapshot_download import snapshot_download
import asyncio
import matheval
import torch
import uvloop
from sglang.utils import get_exception_traceback

# (PPO): 我们只训练数学数据集
MATH_DATASETS = ["math500","aime2024","aime2025","gpqa_diamond","gsm8k","amc23","train_gsm8k"]

# <--- 修改：验证函数接受 `current_step` 而不是 `epoch` --->
def run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, current_step):
    """
    在验证集上运行评估。
    这个函数 *不* 传递 "ground_truth"，因此不会触发 PPO 训练。
    """
    print(f"\n--- Step {current_step + 1} / {args.num_steps} 训练完毕. 开始运行验证... ---")

    total_correct = 0
    total_processed = 0

    # 验证时使用固定的、较小的批次大小以避免 OOM
    eval_batch_size = args.batch_size

    eval_iterator = tqdm(range(0, len(eval_samples), eval_batch_size), desc=f"验证 Step {current_step+1}")

    for batch_start in eval_iterator:
        batch_end = min(batch_start + eval_batch_size, len(eval_samples))
        batch_samples = eval_samples[batch_start:batch_end]

        if not batch_samples:
            continue

        prompts_list = []
        sampling_params_list = []
        batch_ground_truth = []

        for sample in batch_samples:
            prompt_text = sample["prompt"][0]["value"]
            batch_ground_truth.append(sample["final_answer"])

            # 1. 准备 Prompt (String)
            prompt = MATH_QUERY_TEMPLATE.format(Question=prompt_text)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

            # 2. 准备验证用的 SamplingParams
            sampling_params_dict = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_generated_tokens,
                "think_end_str": args.think_end_str,
                "n": 1,
                "soft_hard_action": None,
            }

            prompts_list.append(chat_prompt)
            sampling_params_list.append(sampling_params_dict)

        # 4. 执行批处理 (使用 llm.generate)
        try:
            outputs = llm.generate(
                prompts_list,
                sampling_params=sampling_params_list
            )

            # 5. 在客户端进行评估
            for i, output in enumerate(outputs):
                # <--- 修复：llm.generate 在评估时返回 SglGenOutput 对象 ---
                generated_text = output.text
                ground_truth = batch_ground_truth[i]

                # 使用 matheval 在客户端判断
                judge_result, _ = matheval.evaluator_map["gsm8k"].rule_judge(
                    generated_text,
                    ground_truth,
                    True
                )

                if judge_result:
                    total_correct += 1
                total_processed += 1

            current_accuracy = (total_correct / total_processed) * 100
            eval_iterator.set_description(f"验证 Step {current_step+1} (准确率: {current_accuracy:.2f}%)")

        except Exception as e:
            print(f"验证过程中出错: {e}")
            print(get_exception_traceback())

    # 6. 打印 Epoch 验证报告
    if total_processed > 0:
        final_accuracy = (total_correct / total_processed) * 100
        print(f"\n--- 验证完成 ---")
        print(f"  Step: {current_step + 1}")
        print(f"  验证准确率: {final_accuracy:.2f}% ({total_correct} / {total_processed})")
        print("---------------------------\n")
    else:
        print(f"\n--- 验证失败：没有处理任何样本 ---")


def main():
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # --- 1. 参数解析 (基于您的参考脚本) ---
    parser = argparse.ArgumentParser(description='PPO Controller Training Script')

    # sglang 引擎参数
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--model_id_scope', type=str, default=None, help='ModelScope ID')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tp_size)')
    parser.add_argument('--mem_fraction_static', type=float, default=0.8, help='Max memory per GPU')
    parser.add_argument('--max_running_requests', type=int, default=128, help='Max running requests')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_level', type=str, default="info")

    # (PPO): 必须启用 Soft Thinking 来激活 PPO 逻辑
    parser.add_argument("--enable_soft_thinking", action="store_true", default=True)
    parser.add_argument("--max_topk", type=int, default=10, help="K value for Soft Thinking (K=10 for L_t)")

    # 训练参数
    parser.add_argument('--train_dataset', type=str, default="train_gsm8k", help='Name of training dataset')
    parser.add_argument('--dataset_path', type=str, default="./datasets/train_gsm8k.json", help='Path to training JSON file (主训练集 M)')
    parser.add_argument('--eval_dataset_path', type=str, default="/root/shared-nvme/gj/Hybrid-Thinking/datasets/gsm8k.json", help='Path to validation JSON file')

    # <--- 新增：Step-based 训练参数 ---
    parser.add_argument('--num_steps', type=int, default=1000, help='外循环总步数 (总共训练的批次数)')
    parser.add_argument('--wrong_question_set_path', type=str, default=None, help='Path to the wrong question set JSON file (错题本 N)')
    parser.add_argument('--wrong_question_prob', type=float, default=0.0, help='错题本选中概率 (K)')
    parser.add_argument('--eval_interval', type=int, default=200, help='每 N 步运行一次验证')
    # <--- 新增结束 ---

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default="ppo_checkpoints", help='Directory to save PPO agent weights')
    parser.add_argument('--save_interval', type=int, default=50, help='Save checkpoint every N batches')

    # LLM 评判器参数 (用于奖励计算)
    parser.add_argument('--api_base', type=str, default=None, help='API base for LLM judge')
    parser.add_argument('--api_key', type=str, default=None, help='API key for LLM judge')
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14", help='Judge LLM model name')

    # 生成参数 (用于 Rollout)
    parser.add_argument('--max_generated_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--think_end_str', type=str, default="</think>")
    parser.add_argument('--disable_overlap_schedule', action='store_true',
                        help='(PPO Fix) Disable overlap schedule to prevent bugs')
    args = parser.parse_args()

    # --- 2. PPO 依赖设置 ---

    args.enable_soft_thinking = True
    if args.max_topk < 10:
        print(f"Warning: max_topk ({args.max_topk}) is less than 10. Setting to 10 for PPO L_t features.")
        args.max_topk = 10

    os.makedirs(args.save_dir, exist_ok=True)
    if args.train_dataset in MATH_DATASETS:
        print("Setting up matheval client (for backend reward calculation)...")
        matheval.set_client(args.api_base, None, None, args.api_key, args.judge_model_name)

    # --- 3. 加载数据和 Tokenizer ---

    print(f"Loading main training dataset (M) from: {args.dataset_path}")
    try:
        with open(args.dataset_path) as f:
            samples = json.load(f)
    except Exception as e:
        print(f"Failed to load training dataset: {e}")
        sys.exit(1)
    if not samples:
        print(f"Error: Main training dataset (M) is empty. Path: {args.dataset_path}")
        sys.exit(1)
    print(f"Loaded {len(samples)} training samples (M).")

    # <--- 新增：加载错题本 N --->
    wrong_question_set = []
    if args.wrong_question_set_path and args.wrong_question_prob > 0:
        print(f"Loading wrong question set (N) from: {args.wrong_question_set_path}")
        try:
            with open(args.wrong_question_set_path) as f:
                wrong_question_set = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load wrong question set: {e}. Will proceed without it.")
            wrong_question_set = []

        if wrong_question_set:
            print(f"Loaded {len(wrong_question_set)} wrong questions (N).")
            wrong_question_prob_k = args.wrong_question_prob
            print(f"Wrong question sampling probability (K) set to: {wrong_question_prob_k}")
        else:
            print("Warning: Wrong question set (N) is empty. Will only sample from main set (M).")
            wrong_question_prob_k = 0
    else:
        print("Wrong question set path not provided or probability is 0. Will only sample from main set (M).")
        wrong_question_prob_k = 0
    # <--- 新增结束 --->

    # 加载验证集 (与之前相同)
    print(f"Loading validation dataset from: {args.eval_dataset_path}")
    try:
        with open(args.eval_dataset_path) as f:
            eval_samples = json.load(f)
    except Exception as e:
        eval_samples = []

    if eval_samples:
        print(f"Loaded {len(eval_samples)} validation samples.")
    else:
        print("Warning: Validation dataset not found or empty. Skipping validation.")

    print(f"Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    MATH_QUERY_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}".strip()

    # --- 4. 初始化 SGLang 引擎 ---
    print("Initializing sglang.Engine (this will load the PPO Agent)...")
    llm = sgl.Engine(
        model_path=args.model_name,
        tp_size=args.num_gpus,
        log_level=args.log_level,
        trust_remote_code=True,
        random_seed=args.random_seed,
        max_running_requests=args.max_running_requests,
        mem_fraction_static=args.mem_fraction_static,
        disable_overlap_schedule=args.disable_overlap_schedule,
        enable_soft_thinking=args.enable_soft_thinking,
        max_topk=args.max_topk,
        ppo_save_dir=args.save_dir,
        ppo_save_interval=args.save_interval
    )
    print("sglang.Engine initialized.")


    # --- 5. PPO 训练循环 (已替换为 Step-based 混合采样逻辑) ---
    print(f"--- 开始 Step-Based 训练 (共 {args.num_steps} 步) ---")

    # 1. 初始化状态索引
    j_main_set_idx = 0 # 主训练集 M (samples) 的指针

    # 2. 外循环 (按 step 迭代)
    main_iterator = tqdm(range(args.num_steps), desc="Training Steps")

    for i_step in main_iterator:
        # 3. 内循环 (构建单个批次)
        batch_list = []

        while len(batch_list) < args.batch_size:
            # 4. 数据采样逻辑
            rand_num = random.random()

            if rand_num < wrong_question_prob_k and wrong_question_set:
                # 情况 A (抽错题): 从 N 中随机抽取
                sample = random.choice(wrong_question_set)
            else:
                # 情况 B (抽主训练集): 从 M 中按顺序获取
                sample = samples[j_main_set_idx]

                # 更新 j 的指向 (自动循环)
                j_main_set_idx = (j_main_set_idx + 1) % len(samples)

            batch_list.append(sample)

        # 5. 触发训练 (这是 *原有的* 批次处理逻辑)
        prompts_list = []
        sampling_params_list = []

        for idx, sample in enumerate(batch_list):
            prompt_text = sample["prompt"][0]["value"]
            ground_truth_answer = sample["final_answer"]

            # 准备 Prompt
            prompt = MATH_QUERY_TEMPLATE.format(Question=prompt_text)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

            # 准备 SamplingParams (带 ground_truth 以触发训练)
            sampling_params_dict = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_generated_tokens,
                "think_end_str": args.think_end_str,
                "ground_truth": ground_truth_answer, # <-- 触发训练
                "n": 1,
                "soft_hard_action": None,
            }

            prompts_list.append(chat_prompt)
            sampling_params_list.append(sampling_params_dict)

        # 6. 执行批处理
        try:
            start_time = time.time()
            # 训练时, llm.generate 返回 dict 列表
            outputs = llm.generate(
                prompts_list,
                sampling_params=sampling_params_list
            )
            end_time = time.time()

            main_iterator.set_description(f"Step {i_step+1}/{args.num_steps} (Batch Time: {end_time - start_time:.2f}s)")

            # (可选) 调试打印
            if (i_step + 1) % 20 == 0:
                try:
                    print(f"\n--- [Debug] Batch {i_step + 1} Finished ---")
                    out_text = outputs[0]["text"] # 训练时返回 dict
                    sample = batch_list[0]
                    print(f"  Q: {sample['prompt'][0]['value'][:50]}...")
                    print(f"  A: {out_text.strip().replace(chr(10), ' ')[-100:]}")
                    print(f" GT: {sample['final_answer']}")
                    print("--------------------")
                except Exception as e:
                    print(f"Debug 日志打印出错: {e} (这不影响主训练)")

        except Exception as e:
            print(f"Error during sglang.generate (training): {e}")
            print(get_exception_traceback())

        # 7. <--- 新增：定期验证 ---
        if (i_step + 1) % args.eval_interval == 0 and eval_samples:
            # 在一个 no_grad 上下文中运行验证
            with torch.no_grad():
                run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, i_step)

    # --- 训练循环结束 ---

    print("--- PPO 训练全部完成 ---")

    # <--- 新增：在训练全部结束后，再运行一次最终验证 ---
    if eval_samples:
        print("--- 运行最终验证 ---")
        with torch.no_grad():
            run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, args.num_steps - 1)

    try:
        pass
    finally:
        llm.shutdown()
        print("sglang.Engine shutdown.")


# (download_model_if_needed 函数保持不变)
def download_model_if_needed(local_model_path, modelscope_id):
    """Checks if the model exists locally, otherwise downloads from ModelScope."""
    config_path = os.path.join(local_model_path, "config.json")

    if os.path.exists(config_path):
        print(f"Model found locally at: {local_model_path}")
        return

    print(f"Model not found locally at {local_model_path}.")

    if not modelscope_id:
        print(f"Error: Model not found locally and --model_id_scope was not provided. Please download the model manually to {local_model_path} or provide the ModelScope ID.")
        sys.exit(1)

    print(f"Attempting to download model '{modelscope_id}' from ModelScope...")
    try:
        os.makedirs(local_model_path, exist_ok=True)

        print("Step 1: Downloading model to cache...")
        cache_path = snapshot_download(model_id=modelscope_id,
                                       ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf", "consolidated.safesensors"])

        print(f"Step 2: Copying files from {cache_path} to {local_model_path}...")
        shutil.copytree(cache_path, local_model_path, dirs_exist_ok=True)

        print(f"Model successfully downloaded and copied to: {local_model_path}")

    except Exception as e:
        print(f"Error during model download or copy: {e}")
        print(f"Please check the model ID '{modelscope_id}' and your network connection.")
        if os.path.exists(local_model_path):
            print(f"Cleaning up potentially incomplete directory: {local_model_path}")
            shutil.rmtree(local_model_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
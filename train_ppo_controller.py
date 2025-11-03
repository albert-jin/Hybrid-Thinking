"""
train_ppo_controller.py

主训练脚本，用于"遥控"嵌入在 sglang 调度器内部的 PPO Agent。
它负责：
1. 加载 sglang 引擎 (引擎会自动加载 PPO Agent)。
2. 加载训练数据集 (例如 train_gsm8k.json)。
3. 批量调用 llm.run_batch()，并通过 sampling_params "喂"入
   ground_truth，以触发后端的 PPO 训练逻辑。
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
    parser.add_argument('--dataset_path', type=str, default="./datasets/train_gsm8k.json", help='Path to training JSON file')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
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

    # (PPO): 必须启用 soft thinking
    args.enable_soft_thinking = True

    # (PPO): 确保 K 与 Agent 的 L_t 维度匹配
    if args.max_topk < 10:
        print(f"Warning: max_topk ({args.max_topk}) is less than 10. Setting to 10 for PPO L_t features.")
        args.max_topk = 10

    # (PPO): 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # (PPO): 设置 matheval 客户端 (被后端使用)
    if args.train_dataset in MATH_DATASETS:
        print("Setting up matheval client (for backend reward calculation)...")
        matheval.set_client(args.api_base, None, None, args.api_key, args.judge_model_name)

    # --- 3. 加载数据和 Tokenizer ---

    print(f"Loading training dataset from: {args.dataset_path}")
    try:
        with open(args.dataset_path) as f:
            samples = json.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    print(f"Loaded {len(samples)} training samples.")

    print(f"Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    MATH_QUERY_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}".strip()

    # --- 4. 初始化 SGLang 引擎 ---
    # 这一步会启动 sglang 后端
    # 我们修改后的 scheduler.py __init__ 将在此被调用
    # 它会自动加载 PPO Agent
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
        # --- PPO 关键参数 ---
        # (这些参数需要被添加到 server_args.py 和 scheduler.py __init__ 中)
        enable_soft_thinking=args.enable_soft_thinking,
        max_topk=args.max_topk,
        ppo_save_dir=args.save_dir,
        ppo_save_interval=args.save_interval
        # --- ---
    )
    print("sglang.Engine initialized.")

    # --- 5. PPO 训练循环 ---
    # --- 5. PPO 训练循环 (已修复) ---
    train_step = 0
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1} / {args.num_epochs} ---")

        # 打乱数据集
        random.shuffle(samples)

        batch_iterator = tqdm(range(0, len(samples), args.batch_size), desc=f"Epoch {epoch+1}")

        for batch_start in batch_iterator:
            batch_end = min(batch_start + args.batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]

            if not batch_samples:
                continue

            # --- === PPO 修复: 使用 llm.generate() API === ---

            prompts_list = []
            sampling_params_list = []

            for idx, sample in enumerate(batch_samples):
                prompt_text = sample["prompt"][0]["value"]
                ground_truth_answer = sample["final_answer"]

                # 1. 准备 Prompt (String)
                prompt = MATH_QUERY_TEMPLATE.format(Question=prompt_text)
                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False
                )

                # 2. 准备 SamplingParams (以 *字典* 形式)
                #    sglang 后端 (io_struct.py) 期望的是 dict，而不是 SamplingParams 对象。
                sampling_params_dict = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "max_new_tokens": args.max_generated_tokens,
                    "think_end_str": args.think_end_str,

                    # --- PPO 训练载荷 ---
                    "ground_truth": ground_truth_answer,

                    # --- PPO 阶段三 依赖 ---
                    # 确保 'n' 存在, 因为 io_struct.py 正在访问 .get("n", 1)
                    "n": 1,

                    # (PPO): 我们必须把所有在 sglang 后端
                    # (sampling_params.py) 中定义的 PPO 参数也加在这里
                    # (尽管它们在后端会被覆盖，但传递 None 以确保安全)
                    "soft_hard_action": None,
                }

                prompts_list.append(chat_prompt)
                sampling_params_list.append(sampling_params_dict) # <-- 传递字典

            # 4. 执行批处理 (使用 llm.generate)
            #    后端 (mixin.py) 会自动处理 (H_t, L_t) 提取、
            #    Agent 决策、奖励计算和训练。
            try:
                start_time = time.time()
                # (PPO): llm.generate() 将返回 SglGenOutput 列表
                #        我们不需要 "answer" 键，因为 .text 就包含答案
                outputs = llm.generate(
                    prompts_list,
                    sampling_params=sampling_params_list
                )
                end_time = time.time()

                batch_iterator.set_description(f"Epoch {epoch+1} (Batch Time: {end_time - start_time:.2f}s)")
                train_step += 1

                # (可选) 打印一些输出用于调试
                if train_step % 20 == 0: # 每 20 步打印一次
                    print(f"\n--- [Debug] Batch {train_step} Finished ---")
                    # 'outputs' 是一个列表, 包含 SglGenOutput 对象
                    out_text = outputs[0]["text"]  # <--- 修复点
                    sample = batch_samples[0]
                    print(f"  Q: {sample['prompt'][0]['value'][:50]}...")
                    print(f"  A: {out_text.strip().replace(chr(10), ' ')[-100:]}")
                    print(f" GT: {sample['final_answer']}")
                    print("--------------------")

            except Exception as e:
                print(f"Error during sglang.generate: {e}")
                print(get_exception_traceback())

    print("--- PPO 训练完成 ---")

    try:
        # 训练循环结束，后端会自动按 interval 保存
        pass
    finally:
        llm.shutdown()
        print("sglang.Engine shutdown.")


# (PPO): 复制 download_model_if_needed 函数
def download_model_if_needed(local_model_path, modelscope_id):
    """Checks if the model exists locally, otherwise downloads from ModelScope."""
    config_path = os.path.join(local_model_path, "config.json")

    if os.path.exists(config_path):
        print(f"Model found locally at: {local_model_path}")
        return # 模型已存在，直接返回

    print(f"Model not found locally at {local_model_path}.")

    if not modelscope_id:
        print(f"Error: Model not found locally and --model_id_scope was not provided. Please download the model manually to {local_model_path} or provide the ModelScope ID.")
        sys.exit(1) # 无法下载，退出脚本

    print(f"Attempting to download model '{modelscope_id}' from ModelScope...")
    try:
        os.makedirs(local_model_path, exist_ok=True) # 确保目标目录存在

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
        sys.exit(1) # 下载失败，退出脚本


if __name__ == "__main__":
    main()
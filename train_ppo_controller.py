#!/usr/bin/env python
"""
train_ppo_controller.py (混合采样 Step-based 训练版)
(已修改: 1. 验证逻辑 2. 详细训练日志 3. 动态错题本)
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
import random
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client, AIMEEvaluator
from modelscope.hub.snapshot_download import snapshot_download
import asyncio
import matheval
import torch
import uvloop
from sglang.utils import get_exception_traceback

MATH_DATASETS = ["math500", "aime2024", "aime2025", "gpqa_diamond", "gsm8k", "amc23", "train_gsm8k"]


def run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, current_step):
    """
    在验证集上运行评估。
    """
    print(f"\n--- Step {current_step + 1} / {args.num_steps} 训练完毕. 开始运行验证... ---")

    total_correct = 0
    total_processed = 0

    eval_batch_size = args.batch_size

    eval_iterator = tqdm(range(0, len(eval_samples), eval_batch_size), desc=f"验证 Step {current_step + 1}")

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

            prompt = MATH_QUERY_TEMPLATE.format(Question=prompt_text)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

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

        try:
            outputs = llm.generate(
                prompts_list,
                sampling_params=sampling_params_list
            )

            for i, output in enumerate(outputs):
                generated_text = output["text"]
                ground_truth = batch_ground_truth[i]

                rule_judge_result, extracted_answer = matheval.evaluator_map["gsm8k"].rule_judge(
                    generated_text,
                    ground_truth,
                    True
                )

                llm_judge_result = None
                if not rule_judge_result and args.use_llm_judge:
                    try:
                        llm_judge_result = matheval.evaluator_map["gsm8k"].llm_judge(
                            generated_text,
                            ground_truth,
                            extracted_answer,
                            True
                        )
                    except Exception as e:
                        print(f"LLM Judge (验证) 失败: {e}")
                        llm_judge_result = False

                finally_judge_result = rule_judge_result or llm_judge_result

                if finally_judge_result:
                    total_correct += 1
                total_processed += 1

            current_accuracy = (total_correct / total_processed) * 100
            eval_iterator.set_description(f"验证 Step {current_step + 1} (准确率: {current_accuracy:.2f}%)")

        except Exception as e:
            print(f"验证过程中出错: {e}")
            print(get_exception_traceback())

    if total_processed > 0:
        final_accuracy = (total_correct / total_processed) * 100
        print(f"\n--- 验证完成 ---")
        print(f"  Step: {current_step + 1}")
        print(
            f"  验证准确率: {final_accuracy:.2f}% ({total_correct} / {total_processed}) (LLM Judge: {'启用' if args.use_llm_judge else '禁用'})")
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

    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description='PPO Controller Training Script')

    # (sglang 引擎参数)
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--model_id_scope', type=str, default=None, help='ModelScope ID')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tp_size)')
    parser.add_argument('--mem_fraction_static', type=float, default=0.8, help='Max memory per GPU')
    parser.add_argument('--max_running_requests', type=int, default=128, help='Max running requests')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_level', type=str, default="info")

    # (PPO 参数)
    parser.add_argument("--enable_soft_thinking", action="store_true", default=True)
    parser.add_argument("--max_topk", type=int, default=10, help="K value for Soft Thinking (K=10 for L_t)")

    # (训练参数)
    parser.add_argument('--train_dataset', type=str, default="train_gsm8k", help='Name of training dataset')
    parser.add_argument('--dataset_path', type=str, default="./datasets/train_gsm8k.json",
                        help='Path to training JSON file (主训练集 M)')
    parser.add_argument('--eval_dataset_path', type=str,
                        default="/root/shared-nvme/gj/Hybrid-Thinking/datasets/gsm8k.json",
                        help='Path to validation JSON file')

    parser.add_argument('--num_steps', type=int, default=1000, help='外循环总步数 (总共训练的批次数)')

    # <--- 修改：错题本路径现在是可选的，用于 *初始化* 错题本 ---
    parser.add_argument('--wrong_question_set_path', type=str, default=None,
                        help='(可选) 用于 *初始化* 错题本 N 的 JSON 文件路径')
    # <--- 修改结束 ---

    parser.add_argument('--wrong_question_prob', type=float, default=0.0, help='错题本选中概率 (K)')
    parser.add_argument('--eval_interval', type=int, default=200, help='每 N 步运行一次验证')
    parser.add_argument('--run_validation', action='store_true',
                        help='(新)是否在训练过程中运行验证集评估 (默认为 False)')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default="ppo_checkpoints", help='Directory to save PPO agent weights')
    parser.add_argument('--save_interval', type=int, default=50, help='Save checkpoint every N batches')

    parser.add_argument('--log_train_results', action='store_true', help='是否保存详细的训练批次日志 (jsonl 格式)')
    parser.add_argument('--train_log_interval', type=int, default=50, help='每 N 步保存一次训练日志并报告训练准确率')

    # (LLM 评判器参数)
    parser.add_argument('--api_base', type=str, default=None, help='API base for LLM judge')
    parser.add_argument('--api_key', type=str, default=None, help='API key for LLM judge')
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14", help='Judge LLM model name')
    parser.add_argument('--use_llm_judge', action='store_true',
                        help='(训练后端)如果规则评估失败，是否使用 LLM Judge 来计算奖励')

    # (生成参数)
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
        args.max_topk = 10
    os.makedirs(args.save_dir, exist_ok=True)
    if args.train_dataset in MATH_DATASETS:
        print("Setting up matheval client (for backend reward calculation)...")
        matheval.set_client(args.api_base, None, None, args.api_key, args.judge_model_name)

    # --- 3. 加载数据和 Tokenizer ---
    print(f"Loading main training dataset (M) from: {args.dataset_path}")
    try:
        with open(args.dataset_path) as f:
            all_samples = json.load(f)
        samples = []
        for i, sample in enumerate(all_samples):
            # <--- 关键：确保 M 中的每个样本都有一个唯一的 "original_idx"
            if "original_idx" not in sample:
                sample["original_idx"] = f"M_{i}"  # 如果没有，创建一个
            samples.append(sample)
    except Exception as e:
        print(f"Failed to load training dataset: {e}")
        sys.exit(1)
    if not samples:
        print(f"Error: Main training dataset (M) is empty. Path: {args.dataset_path}")
        sys.exit(1)
    print(f"Loaded {len(samples)} training samples (M).")

    # <--- 修改：错题本 N 现在是动态的 ---
    wrong_question_set = []  # 1. 错题本 N 一开始是空的
    wrong_question_set_ids = set()  # 2. 用于快速查找的 ID 集合

    if args.wrong_question_set_path:
        print(f"Loading *initial* wrong question set (N) from: {args.wrong_question_set_path}")
        try:
            with open(args.wrong_question_set_path) as f:
                initial_wrong_questions = json.load(f)

            # 预填充错题本
            for sample in initial_wrong_questions:
                # 确保样本有 ID
                if "original_idx" not in sample:
                    # 如果初始错题本没有 ID，我们无法管理它
                    print(f"Warning: Skipping sample in wrong set (no 'original_idx'): {str(sample)[:50]}...")
                    continue

                sample_id = sample["original_idx"]
                if sample_id not in wrong_question_set_ids:
                    wrong_question_set.append(sample)
                    wrong_question_set_ids.add(sample_id)

        except Exception as e:
            print(f"Warning: Failed to load initial wrong question set: {e}. Starting with an empty set.")
            wrong_question_set = []
            wrong_question_set_ids = set()

    if wrong_question_set:
        print(f"Loaded {len(wrong_question_set)} initial wrong questions (N).")
    else:
        print("Starting with an empty wrong question set (N).")

    wrong_question_prob_k = args.wrong_question_prob
    if wrong_question_prob_k > 0:
        print(f"Wrong question sampling probability (K) set to: {wrong_question_prob_k}")
    # <--- 修改结束 ---

    # (加载验证集)
    eval_samples = []
    if args.run_validation:
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
    else:
        print("Validation is disabled (--run_validation=False). Skipping validation dataset loading.")

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
        ppo_save_interval=args.save_interval,
        use_llm_judge=args.use_llm_judge
    )
    print("sglang.Engine initialized.")

    # --- 5. PPO 训练循环 (Step-based) ---
    print(f"--- 开始 Step-Based 训练 (共 {args.num_steps} 步) ---")

    j_main_set_idx = 0
    train_results_buffer = []
    train_stats_correct = 0
    train_stats_total = 0
    train_log_file_path = os.path.join(args.save_dir, "train_results_log.jsonl")

    main_iterator = tqdm(range(args.num_steps), desc="Training Steps")

    for i_step in main_iterator:
        # 3. 内循环 (构建单个批次)
        batch_list = []
        batch_sources = []  # <--- 新增：跟踪样本来源

        while len(batch_list) < args.batch_size:
            # 4. 数据采样逻辑
            rand_num = random.random()

            # <--- 修改：检查错题本是否 *非空* ---
            if rand_num < wrong_question_prob_k and wrong_question_set:
                # 情况 A (抽错题): 从 N 中随机抽取
                sample = random.choice(wrong_question_set)
                batch_sources.append("N")  # 标记来源为错题本
            else:
                # 情况 B (抽主训练集): 从 M 中按顺序获取
                sample = samples[j_main_set_idx]
                j_main_set_idx = (j_main_set_idx + 1) % len(samples)
                batch_sources.append("M")  # 标记来源为主训练集

            batch_list.append(sample)

        # 5. 触发训练
        prompts_list = []
        sampling_params_list = []

        for idx, sample in enumerate(batch_list):
            prompt_text = sample["prompt"][0]["value"]
            ground_truth_answer = sample["final_answer"]

            prompt = MATH_QUERY_TEMPLATE.format(Question=prompt_text)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

            sampling_params_dict = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_generated_tokens,
                "think_end_str": args.think_end_str,
                "ground_truth": ground_truth_answer,
                "n": 1,
                "soft_hard_action": None,
                # "use_llm_judge": args.use_llm_judge
            }
            prompts_list.append(chat_prompt)
            sampling_params_list.append(sampling_params_dict)

        # 6. 执行批处理
        try:
            start_time = time.time()
            outputs = llm.generate(
                prompts_list,
                sampling_params=sampling_params_list,
                return_logprob=True,
                top_logprobs_num=args.max_topk
            )
            end_time = time.time()
            main_iterator.set_description(
                f"Step {i_step + 1}/{args.num_steps} (Batch Time: {end_time - start_time:.2f}s, 错题本: {len(wrong_question_set)})")

        except Exception as e:
            print(f"Error during sglang.generate (training): {e}")
            print(get_exception_traceback())
            continue

            # 7. <--- 修改：记录日志 + 动态更新错题本 ---
        try:
            for i, output in enumerate(outputs):
                sample = batch_list[i]
                source = batch_sources[i]  # 获取来源
                sample_id = sample.get("original_idx", f"unknown_id_{random.randint(1000, 9999)}")
                generated_text = output["text"]
                ground_truth = sample["final_answer"]

                # --- vvv 从这里开始替换 vvv ---

                # 1. 从后端获取已计算的评判结果
                judge_info = output["meta_info"].get("judge_info")

                if judge_info is None:
                    # 这不是一个训练请求 (e.g., ground_truth=None)
                    # 或者后端没有返回评判结果，跳过
                    print(f"Warning: No judge_info found in meta_info for {sample_id}. Skipping notebook/log update.")
                    continue

                # 2. 解包结果 (现在我们有了所有信息)
                rule_judge_result = judge_info.get("rule_judge_result")
                llm_judge_result = judge_info.get("llm_judge_result")
                finally_judge_result = judge_info.get("finally_judge_result")

                if finally_judge_result is None:
                    print(f"Error: judge_info dict is incomplete for {sample_id}. Skipping.")
                    continue

                if finally_judge_result:
                    train_stats_correct += 1
                train_stats_total += 1
                pass_val = 1.0 if finally_judge_result else 0.0

                # --- ^^^ 替换结束 ^^^ ---

                # --- 评估结束 --- (这行保留)

                # --- !!! 动态错题本逻辑 !!! --- (这行保留)
                if finally_judge_result == True:
                    # 答对了
                    if source == "N" and sample_id in wrong_question_set_ids:
                        # 答对了 *错题本* 中的题，将其移除
                        wrong_question_set_ids.remove(sample_id)
                        wrong_question_set = [q for q in wrong_question_set if q.get("original_idx") != sample_id]
                else:
                    # 答错了
                    if source == "M" and sample_id not in wrong_question_set_ids:
                        # 答错了 *主训练集* 的题，将其加入
                        wrong_question_set.append(sample)
                        wrong_question_set_ids.add(sample_id)
                # --- 动态逻辑结束 ---

                # --- 构建日志字典 (仅在开启日志时) ---
                if args.log_train_results:
                    result_dict = {
                        "hyperparams": str(args),
                        "prompt": sample["prompt"][0]["value"],
                        "completion": [generated_text],
                        "ground_truth": ground_truth,
                        "generated_tokens": [output["meta_info"]["completion_tokens"]],
                        "avg_generated_tokens": output["meta_info"]["completion_tokens"],
                        "idx": sample_id,
                        "n": 1,
                        "finish_generation": [output["meta_info"]["finish_reason"]],

                        # (这里的 'judge_info' 变量现在来自 meta_info，可以被正确记录)
                        "judge_info": [{"rule_judge_result": rule_judge_result, "llm_judge_result": llm_judge_result,
                                        "finally_judge_result": finally_judge_result}],
                        "passat1": pass_val,
                        "passat1_list": [pass_val],
                        # "output_topk_probs_list": output["meta_info"].get("output_topk_probs_list"),
                        # "output_topk_indices_list": output["meta_info"].get("output_topk_indices_list"),
                        # "input_token_logprobs": output["meta_info"].get("input_token_logprobs"),
                        # "output_token_logprobs": output["meta_info"].get("output_token_logprobs"),
                        # "input_top_logprobs": output["meta_info"].get("input_top_logprobs"),
                        # "output_top_logprobs": output["meta_info"].get("output_top_logprobs"),
                        # "input_token_ids_logprobs": output["meta_info"].get("input_token_ids_logprobs"),
                        # "output_token_ids_logprobs": output["meta_info"].get("output_token_ids_logprobs"),
                    }
                    train_results_buffer.append(result_dict)

        except Exception as e:
            print(f"Error during training logging/dynamic update: {e}")
            print(get_exception_traceback())
        # <--- 修改结束 ---

        # 8. (定期保存日志和打印统计)
        if args.log_train_results and (i_step + 1) % args.train_log_interval == 0:
            print(f"\n--- 保存训练日志 (Step {i_step + 1}) ---")

            if train_stats_total > 0:
                accuracy = (train_stats_correct / train_stats_total) * 100
                print(f"  [训练集准确率] (过去 {train_stats_total} 个样本): {accuracy:.2f}%")

            try:
                with open(train_log_file_path, "a", encoding="utf-8") as f:
                    for record in train_results_buffer:
                        f.write(json.dumps(record) + "\n")
                print(f"  已追加 {len(train_results_buffer)} 条记录到 {train_log_file_path}")
            except Exception as e:
                print(f"  保存训练日志失败: {e}")

            train_results_buffer = []
            train_stats_correct = 0
            train_stats_total = 0
            print("--------------------------------------\n")

        # 9. (定期验证)
        if args.run_validation and (i_step + 1) % args.eval_interval == 0 and eval_samples:
            with torch.no_grad():
                run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, i_step)

    # --- 训练循环结束 ---

    print("--- PPO 训练全部完成 ---")

    # (最终验证)
    if args.run_validation and eval_samples:
        print("--- 运行最终验证 ---")
        with torch.no_grad():
            run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, args.num_steps - 1)

    try:
        # <--- 新增：在训练结束时，保存最终的错题本 ---
        final_wrong_set_path = os.path.join(args.save_dir, "final_wrong_question_set.json")
        try:
            with open(final_wrong_set_path, "w", encoding="utf-8") as f:
                # 只保存 ID 在 set 中的错题
                final_wrong_set = [q for q in wrong_question_set if q.get("original_idx") in wrong_question_set_ids]
                json.dump(final_wrong_set, f, indent=4)
            print(f"--- 最终错题本 ({len(final_wrong_set)} 条) 已保存到: {final_wrong_set_path} ---")
        except Exception as e:
            print(f"--- 保存最终错题本失败: {e} ---")
        # <--- 新增结束 ---

    finally:
        llm.shutdown()
        print("sglang.Engine shutdown.")


# (download_model_if_needed 函数保持不变)
def download_model_if_needed(local_model_path, modelscope_id):
    config_path = os.path.join(local_model_path, "config.json")
    if os.path.exists(config_path):
        print(f"Model found locally at: {local_model_path}")
        return
    print(f"Model not found locally at {local_model_path}.")
    if not modelscope_id:
        print(
            f"Error: Model not found locally and --model_id_scope was not provided. Please download the model manually to {local_model_path} or provide the ModelScope ID.")
        sys.exit(1)
    print(f"Attempting to download model '{modelscope_id}' from ModelScope...")
    try:
        os.makedirs(local_model_path, exist_ok=True)
        print("Step 1: Downloading model to cache...")
        cache_path = snapshot_download(model_id=modelscope_id,
                                       ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf",
                                                        "consolidated.safesensors"])
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
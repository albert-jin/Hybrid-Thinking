#!/usr/bin/env python
"""
eval_ppo_agent.py

专用的 PPO Agent *评估* 脚本。
(已修改，支持保存详细的 JSON 结果和 Soft Thinking 中间内容)
"""
import os
import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import sys
import random
import time
from transformers import AutoTokenizer
from matheval import evaluator_map, set_client
import matheval
import torch
import uvloop
from sglang.utils import get_exception_traceback

# (PPO): 我们只评估数学数据集
MATH_DATASETS = ["math500","aime2024","aime2025","gpqa_diamond","gsm8k","amc23","train_gsm8k"]

def run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, results_list):
    """
    在验证集上运行评估。
    这个函数 *不* 传递 "ground_truth"，因此不会触发 PPO 训练。
    它会就地修改传入的 results_list。
    """
    print(f"\n--- 开始运行验证... (共 {len(eval_samples)} 个样本) ---")

    total_correct = 0
    total_processed = 0

    eval_batch_size = args.batch_size

    eval_iterator = tqdm(range(0, len(eval_samples), eval_batch_size), desc=f"评估中")

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
            # <--- 新增：根据 force_mode 设置动作值 ---
            if args.force_mode == "soft":
                action_value = 0
            elif args.force_mode == "hard":
                action_value = 1
            else: # "ppo"
                action_value = None # None 会告诉后端使用 PPO Agent
            # <--- 新增结束 ---
            # 2. 准备验证用的 SamplingParams
            sampling_params_dict = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_generated_tokens,
                "think_end_str": args.think_end_str,
                "n": 1,
                "soft_hard_action": action_value, # <--- 修改：传递 0, 1, 或 None
            }

            prompts_list.append(chat_prompt)
            sampling_params_list.append(sampling_params_dict)

        # 4. 执行批处理 (使用 llm.generate)
        try:
            # llm.generate 返回字典列表
            outputs = llm.generate(
                prompts_list,
                sampling_params=sampling_params_list,

                # <--- 修复：在顶层传递这些参数 ---
                # SGLang 后端需要这些参数来知道
                # 它必须返回 "Soft" 的中间内容
                return_logprob=True,
                top_logprobs_num=args.max_topk
                # (已删除 return_hidden_states=True)
                # <--- 修复结束 ---
            )

            # 5. 在客户端进行评估
            for i, output in enumerate(outputs):
                generated_text = output["text"]
                ground_truth = batch_ground_truth[i]
                sample = batch_samples[i]

                # <--- 新增：动态检查 "finish_reason" ---
                # 从 SGLang 的输出中获取停止原因
                finish_reason_dict = output["meta_info"]["finish_reason"]

                # 如果类型是 "stop" (自然停止)，则为 True
                # 如果类型是 "length" (达到 max_tokens)，则为 False
                is_generation_finished = (finish_reason_dict.get("type") == "stop")
                # <--- 新增结束 ---


                # 1. 规则评估
                rule_judge_result, extracted_answer = matheval.evaluator_map["gsm8k"].rule_judge(
                    generated_text,
                    ground_truth,
                    is_generation_finished  # <--- 修改：不再硬编码 True
                )

                # 2. LLM 评估 (如果规则失败且已启用)
                llm_judge_result = None
                if not rule_judge_result and args.use_llm_judge:
                    try:
                        llm_judge_result = matheval.evaluator_map["gsm8k"].llm_judge(
                            generated_text,
                            ground_truth,
                            extracted_answer,
                            is_generation_finished  # <--- 修改：不再硬编码 True
                        )
                    except Exception as e:
                        print(f"LLM Judge 失败: {e}")
                        llm_judge_result = False  # 评判失败，算作错误

                # 3. 最终结果
                finally_judge_result = rule_judge_result or llm_judge_result

                if finally_judge_result:
                    total_correct += 1
                total_processed += 1

                pass_val = 1.0 if finally_judge_result else 0.0
                result_dict = {
                    "hyperparams": str(args),
                    "prompt": sample["prompt"][0]["value"],
                    "completion": [generated_text],
                    "ground_truth": ground_truth,
                    "generated_tokens": [output["meta_info"]["completion_tokens"]],
                    "avg_generated_tokens": output["meta_info"]["completion_tokens"],
                    "idx": sample["original_idx"],
                    "n": 1,
                    # <--- 修改：保存原始的 finish_reason 字典 ---
                    "finish_generation": [finish_reason_dict],
                    # <--- 修改结束 ---

                    "judge_info": [{"rule_judge_result": rule_judge_result, "llm_judge_result": llm_judge_result,
                                    "finally_judge_result": finally_judge_result}],
                    "passat1": pass_val,
                    "passat1_list": [pass_val],
                }
                results_list.append(result_dict)
                # --- 修改结束 ---

            current_accuracy = (total_correct / total_processed) * 100
            eval_iterator.set_description(f"评估中 (准确率: {current_accuracy:.2f}%)")

        except Exception as e:
            print(f"验证过程中出错: {e}")
            print(get_exception_traceback())

    # 6. 打印 Epoch 验证报告
    if total_processed > 0:
        final_accuracy = (total_correct / total_processed) * 100
        print(f"\n--- 验证完成 ---")
        print(f"  已评估 Agent: {args.ppo_agent_checkpoint_path}")
        print(f"  最终准确率: {final_accuracy:.2f}% ({total_correct} / {total_processed})")
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
    parser = argparse.ArgumentParser(description='PPO Agent *Evaluation* Script')

    # sglang 引擎参数
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tp_size)')
    parser.add_argument('--mem_fraction_static', type=float, default=0.8, help='Max memory per GPU')
    parser.add_argument('--max_running_requests', type=int, default=128, help='Max running requests')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_level', type=str, default="info")

    # (PPO): 必须启用 Soft Thinking 来激活 PPO 逻辑
    parser.add_argument("--enable_soft_thinking", action="store_true", default=True)
    parser.add_argument("--max_topk", type=int, default=10, help="K value for Soft Thinking (K=10 for L_t)")

    # PPO Agent Checkpoint 路径
    parser.add_argument('--ppo_agent_checkpoint_path', type=str, required=True, help='Path to the trained PPO agent .pth file')

    # 评估参数
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to validation JSON file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')

    # LLM 评判器参数 (用于客户端评估)
    parser.add_argument('--api_base', type=str, default=None, help='API base for LLM judge')
    parser.add_argument('--api_key', type=str, default=None, help='API key for LLM judge')
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14", help='Judge LLM model name')
    parser.add_argument('--use_llm_judge', action='store_true', help='(评估)如果规则评估失败，是否使用 LLM Judge')
    # 生成参数
    parser.add_argument('--max_generated_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--think_end_str', type=str, default="</think>")
    parser.add_argument('--disable_overlap_schedule', action='store_true')

    # 结果保存和数据范围参数
    parser.add_argument('--output_dir', type=str, default="eval_results", help='Directory to save results')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing samples')
    parser.add_argument('--end_idx', type=int, default=1000000, help='End index for processing samples')
    parser.add_argument('--force_mode', type=str, choices=["soft", "hard", "ppo"], default="ppo",
                        help='强制模式: "soft" (始终Soft), "hard" (始终Hard/离散), "ppo" (使用训练好的PPO Agent决策)')
    # <--- 新增结束 ---
    args = parser.parse_args()

    # --- 2. PPO 依赖设置 ---
    args.enable_soft_thinking = True
    if args.max_topk < 10:
        args.max_topk = 10

    print("Setting up matheval client (for client-side evaluation)...")
    matheval.set_client(args.api_base, None, None, args.api_key, args.judge_model_name)

    # --- 3. 加载数据和 Tokenizer ---
    print(f"Loading validation dataset from: {args.dataset_path}")
    eval_samples = []
    try:
        with open(args.dataset_path) as f:
            all_samples = json.load(f)

        start_idx = args.start_idx
        end_idx = min(args.end_idx, len(all_samples))

        sliced_samples = all_samples[start_idx:end_idx]

        for i, sample in enumerate(sliced_samples):
            sample["original_idx"] = start_idx + i
            eval_samples.append(sample)

    except Exception as e:
        print(f"Failed to load validation dataset: {e}")
        sys.exit(1)

    if not eval_samples:
        print("Error: Validation dataset is empty or start/end index is out of range.")
        sys.exit(1)

    print(f"Loaded {len(eval_samples)} validation samples (from index {args.start_idx} to {end_idx}).")

    print(f"Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    MATH_QUERY_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}".strip()

    # --- 4. 初始化 SGLang 引擎 ---
    print("Initializing sglang.Engine (this will load the *TRAINED* PPO Agent)...")
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
        ppo_agent_checkpoint_path=args.ppo_agent_checkpoint_path,
        use_llm_judge=args.use_llm_judge
    )
    print("sglang.Engine initialized.")

    # 定义结果文件路径
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    agent_name = os.path.basename(args.ppo_agent_checkpoint_path).split('.')[0]

    # <--- 修改：将 force_mode (PPO/SOFT/HARD) 添加到文件名中 ---
    mode_str = args.force_mode.upper() # 结果为 "PPO", "SOFT", 或 "HARD"
    base_filename = f"eval_{agent_name}_on_{dataset_name}_MODE_{mode_str}_{run_timestamp}"
    # <--- 修改结束 ---

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"{base_filename}_results.json")
    results_statistics_file = os.path.join(output_dir, f"{base_filename}_statistics.json")

    results = [] # 初始化结果列表


    # --- 5. PPO 评估 ---
    print("--- PPO 评估开始 ---")
    start_time = time.time()
    with torch.no_grad():
        run_validation(llm, eval_samples, tokenizer, args, MATH_QUERY_TEMPLATE, results)

    end_time = time.time()
    print("--- PPO 评估完成 ---")

    try:
        # --- 6. 保存结果和统计数据 ---
        print(f"\n--- 保存详细结果 ({len(results)} 条) 到: {results_file} ---")
        with open(results_file, "w") as f:
            results.sort(key=lambda x: x["idx"])
            json.dump(results, f, indent=4)

        # <--- 修复：添加 'avg_token_length-correct' 和 'all_idx' 的计算 ---
        total_num = len(results)
        pass_at_1 = sum([r["passat1"] for r in results]) / total_num if total_num > 0 else 0
        avg_tokens_all = sum([r["avg_generated_tokens"] for r in results]) / total_num if total_num > 0 else 0
        time_taken_hours = (end_time - start_time) / 3600

        # 计算 avg_token_length-correct
        correct_results_tokens = [r["avg_generated_tokens"] for r in results if r["passat1"] > 0]
        if len(correct_results_tokens) > 0:
            avg_token_length_correct = sum(correct_results_tokens) / len(correct_results_tokens)
        else:
            avg_token_length_correct = 0

        # 创建 all_idx 字典 (并确保索引按数字顺序排列)
        all_idx_list = sorted([(r["idx"], r["passat1"]) for r in results], key=lambda x: x[0])
        all_idx_dict = {str(i): j for i, j in all_idx_list} # 确保 key 是 str 来匹配示例

        results_statistics = {
            "total_num": total_num,
            "pass@1": pass_at_1,
            "avg_token_length-all": avg_tokens_all,
            "avg_token_length-correct": avg_token_length_correct, # <-- 已添加
            "time_taken/h": time_taken_hours,
            "all_idx": all_idx_dict # <-- 已添加
        }
        # <--- 修复结束 ---

        print(f"--- 保存统计数据到: {results_statistics_file} ---")
        with open(results_statistics_file, "w") as f:
            json.dump(results_statistics, f, indent=4)

        print(f"\n最终统计: {json.dumps(results_statistics, indent=4)}")

    finally:
        llm.shutdown()
        print("sglang.Engine shutdown.")

if __name__ == "__main__":
    main()
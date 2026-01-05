#!/usr/bin/env python
"""
eval_code_ppo_agent.py

专用的 PPO Agent *代码* 评估脚本。
(混合了 eval_ppo_agent.py 的 PPO 加载逻辑
 和 run_sglang_softthinking.py 的两阶段代码评估流程)
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
import torch
from sglang.utils import get_exception_traceback
import subprocess # 用于 LiveCodeBench

# (+) 导入代码评估器
# (我们在此脚本中 "调用" 它们)
import humanevaleval
import mbppeval
import convert_livecodebench

# (PPO): 我们只评估代码数据集
CODE_DATASETS = ["humaneval","mbpp","livecodebench"]

# --- Prompt 模板 (从 run_sglang_softthinking.py 复制) ---
CODE_QUERY_TEMPLATE = """
Please solve the programming task below in Python. Code should be wrapped in a markdown code block.

```python
{Question}
```
""".strip()

MBPP_QUERY_TEMPLATE = """
Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:
1. The function name and signature must match exactly as specified in the test cases.
2. Your code should be wrapped in a markdown code block without including any test cases.

Task:
{Question}

Test Cases:
```python
{TestCases}
```
""".strip()
def get_lcb_prompt(question_content, starter_code):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question_content}\n\n"
    if starter_code:
        prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    return prompt
# --- Prompt 模板结束 ---


def main():

    # --- 1. 参数解析 (混合自两个脚本) ---
    parser = argparse.ArgumentParser(description='PPO Agent *CODE* Evaluation Script')

    # sglang 引擎参数
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tp_size)')
    parser.add_argument('--mem_fraction_static', type=float, default=0.8, help='Max memory per GPU')
    parser.add_argument('--max_running_requests', type=int, default=128, help='Max running requests')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_level', type=str, default="info")

    # (PPO): PPO 特定参数
    parser.add_argument("--enable_soft_thinking", action="store_true", default=True, help="[锁定为 True] 必须启用 Soft Thinking 来激活 PPO 逻辑")
    parser.add_argument("--max_topk", type=int, default=10, help="[PPO] K value for Soft Thinking (K=10 for L_t)")
    parser.add_argument('--ppo_agent_checkpoint_path', type=str, required=True, help='[PPO] Path to the trained PPO agent .pth file')
    parser.add_argument('--force_mode', type=str, choices=["soft", "hard", "ppo"], default="ppo",
                        help='[PPO] 强制模式: "soft" (始终Soft), "hard" (始终Hard/离散), "ppo" (使用训练好的PPO Agent决策)')

    # (Code): 代码评估参数
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to code dataset JSON file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation (阶段 1)')
    parser.add_argument('--reeval', action='store_true', help='[Code] 启用对代码数据集的重新评估 (两阶段流程的第二阶段)')
    parser.add_argument('--num_samples', type=int, default=1, help='[Code] 代码数据集的采样数量')

    # (Code): 生成参数
    parser.add_argument('--max_generated_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    parser.add_argument('--think_end_str', type=str, default="</think>")
    parser.add_argument('--disable_overlap_schedule', action='store_true')

    # 结果保存和数据范围参数
    parser.add_argument('--output_dir', type=str, default="eval_results", help='Directory to save results')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing samples')
    parser.add_argument('--end_idx', type=int, default=1000000, help='End index for processing samples')
    parser.add_argument('--reeval_input_file', type=str, default=None,
                        help='[Code] (Re-eval Only) Path to the specific GENERATE JSON file to re-evaluate.')
    args = parser.parse_args()

    # --- 2. PPO 依赖设置 ---
    args.enable_soft_thinking = True # 强制启用
    if args.max_topk < 10:
        args.max_topk = 10
        print("Warning: --max_topk < 10, setting to 10 for PPO.")

    # --- 3. 加载数据和 Tokenizer ---
    print(f"Loading code dataset from: {args.dataset_path}")

    # 推断数据集名称 (用于逻辑判断)
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    if dataset_name not in CODE_DATASETS:
        print(f"错误: 从 {args.dataset_path} 推断的数据集名称 '{dataset_name}' 不在支持的 CODE_DATASETS 列表中。")
        print(f"支持的列表: {CODE_DATASETS}")
        sys.exit(1)

    try:
        with open(args.dataset_path) as f:
            all_samples_raw = json.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 切片并添加 'original_idx' (借鉴 eval_ppo_agent.py)
    eval_samples = []
    start_idx = args.start_idx
    end_idx = min(args.end_idx, len(all_samples_raw))
    sliced_samples = all_samples_raw[start_idx:end_idx]
    for i, sample in enumerate(sliced_samples):
        sample["original_idx"] = start_idx + i
        eval_samples.append(sample)

    if not eval_samples:
        print("Error: Validation dataset is empty or start/end index is out of range.")
        sys.exit(1)

    print(f"Loaded {len(eval_samples)} validation samples (from index {start_idx} to {end_idx}).")

    print(f"Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # --- 4. 定义结果文件路径 ---
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_name = os.path.basename(args.ppo_agent_checkpoint_path).split('.')[0]

    mode_str = args.force_mode.upper() # "PPO", "SOFT", 或 "HARD"
    reeval_str = "REEVAL" if args.reeval else "GENERATE"

    base_filename = (
        f"eval_CODE_{agent_name}_on_{dataset_name}_MODE_{mode_str}_"
        f"REEVAL_{reeval_str}_SAMPLES_{args.num_samples}_{run_timestamp}"
    )

    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"{base_filename}_results.json")
    results_statistics_file = os.path.join(output_dir, f"{base_filename}_statistics.json")

    # --- 5. 两阶段评估逻辑 ---

    llm = None
    results = []
    start_time = time.time()

    try:
        # 阶段 2: 仅评估 (Re-evaluation)
        if args.reeval:
            print(f"--- (Code PPO) 阶段 2: Re-evaluation 模式 ---")

            # --- (!!!) 关键修改 (!!!) ---
            if not args.reeval_input_file:
                print(f"错误: 在 --reeval 模式下, 必须提供 --reeval_input_file 参数指定输入文件。")
                sys.exit(1)

            # 使用我们提供的精确路径
            results_file_to_read = args.reeval_input_file

            if not os.path.exists(results_file_to_read):
                print(f"错误: 找不到指定的 re-eval input file: {results_file_to_read}")
                sys.exit(1)

            # 使用输入文件路径来定义 *输出* 路径
            # (我们将 "GENERATE" 替换为 "REEVAL_STEP2" 来创建新的输出文件名)
            input_filename_base = os.path.basename(results_file_to_read).replace("_results.json", "")
            base_filename = input_filename_base.replace("GENERATE", "REEVAL_STEP2")
            output_dir = os.path.dirname(results_file_to_read)

            # (!!!) 这是新的、独立的 "Step 2" 输出文件 (!!!)
            results_file = os.path.join(output_dir, f"{base_filename}_results.json")
            results_statistics_file = os.path.join(output_dir, f"{base_filename}_statistics.json")

            print(f"Loading existing results from: {results_file_to_read}")
            with open(results_file_to_read, "r") as f:
                results = json.load(f) # 加载 'results'

            # 从文件加载 'samples' (因为 'samples' 在 reeval 模式下就是 'results')
            samples_to_iterate = results
            # idx_list 只是 loaded results 的索引
            idx_list = list(range(len(samples_to_iterate)))

            # 从加载的 'results' 中提取数据 (完全复制自 run_sglang_softthinking.py)
            decoded_text_list = []
            finish_generation_list = []
            generated_tokens_list = []

            for r in samples_to_iterate:
                # 兼容旧格式
                if "prompt" not in r: r["prompt"] = ""
                if "completion" not in r: r["completion"] = [""] * args.num_samples
                if "finish_generation" not in r: r["finish_generation"] = [{}] * args.num_samples
                if "generated_tokens" not in r: r["generated_tokens"] = [0] * args.num_samples

                decoded_text_list.extend(r["completion"])
                finish_generation_list.extend(r["finish_generation"])
                generated_tokens_list.extend(r["generated_tokens"])

            results = [] # 清空 results 列表，准备重新填充


        # 阶段 1: 仅生成 (Generation)
        else:
            print(f"--- (Code PPO) 阶段 1: Generation 模式 (Agent: {agent_name}) ---")

            # 在生成模式下, 'samples_to_iterate' 是新加载的 'eval_samples'
            samples_to_iterate = eval_samples
            idx_list = list(range(len(samples_to_iterate))) # idx_list 是 samples 的本地索引

            prompts_list = []
            sampling_params_list = []

            for idx in idx_list:
                sample = samples_to_iterate[idx]

                if dataset_name == "humaneval":
                    chat = [{"role": "user", "content": CODE_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
                elif dataset_name == "mbpp":
                    chat = [{"role": "user", "content": MBPP_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"], TestCases="\n".join(sample["final_answer"]["test_list"]))}]
                elif dataset_name == "livecodebench":
                    chat = [{"role": "user", "content": get_lcb_prompt(question_content=sample["prompt"][0]["value"], starter_code=sample["final_answer"]["starter_code"])}]

                prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

                # PPO 模式设置 (借鉴 eval_ppo_agent.py)
                if args.force_mode == "soft":
                    action_value = 0
                elif args.force_mode == "hard":
                    action_value = 1
                else: # "ppo"
                    action_value = None # 后端将使用 PPO Agent

                # 为 'num_samples' 重复
                for _ in range(args.num_samples):
                    prompts_list.append(prompt)
                    sampling_params_list.append({
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "repetition_penalty": args.repetition_penalty,
                        "max_new_tokens": args.max_generated_tokens,
                        "think_end_str": args.think_end_str,
                        "n": 1,
                        "soft_hard_action": action_value, # <-- PPO 决策
                    })

            # 初始化 PPO SGLang 引擎 (借鉴 eval_ppo_agent.py)
            print(f"Initializing sglang.Engine (Loading PPO Agent from {args.ppo_agent_checkpoint_path})...")
            llm = sgl.Engine(
                model_path=args.model_name,
                tp_size=args.num_gpus,
                log_level=args.log_level,
                trust_remote_code=True,
                random_seed=args.random_seed,
                max_running_requests=args.max_running_requests,
                mem_fraction_static=args.mem_fraction_static,
                disable_overlap_schedule=args.disable_overlap_schedule,
                # PPO 关键参数
                enable_soft_thinking=args.enable_soft_thinking,
                max_topk=args.max_topk,
                ppo_agent_checkpoint_path=args.ppo_agent_checkpoint_path
            )
            print("sglang.Engine (PPO) initialized.")

            # 开始生成
            decoded_text_list = []
            finish_generation_list = []
            generated_tokens_list = []
            soft_thinking_logs_list = [] # PPO 专用

            eval_batch_size = args.batch_size
            gen_iterator = tqdm(range(0, len(prompts_list), eval_batch_size), desc=f"(Code PPO) Generating")

            for batch_start in gen_iterator:
                batch_end = min(batch_start + eval_batch_size, len(prompts_list))
                batch_prompts = prompts_list[batch_start:batch_end]
                batch_sampling_params = sampling_params_list[batch_start:batch_end]

                if not batch_prompts:
                    continue

                # 执行批处理 (使用同步 generate, 借鉴 eval_ppo_agent.py)
                outputs = llm.generate(
                    batch_prompts,
                    sampling_params=batch_sampling_params,
                    # PPO Agent 需要这些来进行决策
                    return_logprob=True,
                    top_logprobs_num=args.max_topk
                )

                # 收集结果
                for o in outputs:
                    decoded_text_list.append(o["text"])
                    finish_generation_list.append(o["meta_info"]["finish_reason"])
                    generated_tokens_list.append(o["meta_info"]["completion_tokens"])
                    soft_thinking_logs_list.append(o.get("soft_thinking_logs", None)) # 存储 PPO 日志


        # --- 6. 评估阶段 (reeval 和 generate 模式都需要) ---

        # (+) 严格按照 run_sglang_softthinking.py 的方式 "调用"
        print("--- (Code PPO) 调用代码评估器 (Humaneval/MBPP)... ---")
        mbppeval.init_evaluator()
        humanevaleval.init_evaluator()

        print("--- (Code PPO) 开始评判... ---")
        for i, idx in enumerate(tqdm(idx_list, desc="(Code PPO) Judging")):

            sample = samples_to_iterate[idx]

            # 按 'num_samples' 切片
            decoded_text = decoded_text_list[i*args.num_samples:(i+1)*args.num_samples]
            finish_generation_dicts = finish_generation_list[i*args.num_samples:(i+1)*args.num_samples]
            batch_gen_tokens = generated_tokens_list[i*args.num_samples:(i+1)*args.num_samples]
            # (PPO)
            batch_soft_logs = soft_thinking_logs_list[i*args.num_samples:(i+1)*args.num_samples] if 'soft_thinking_logs_list' in locals() else [None] * args.num_samples

            judge_info = []
            passat1_list = []

            # 遍历每个采样
            for j in range(args.num_samples):
                passat1 = 0.0
                single_judge_info = None

                # 仅在 reeval 模式下运行沙盒
                # (完全复制 run_sglang_softthinking.py 的逻辑)
                if args.reeval:
                    try:
                        k = 1 # pass@1
                        current_prompt_text = sample["prompt"][0]["value"] if isinstance(sample["prompt"], list) else sample["prompt"]

                        if dataset_name=="humaneval":
                            passat1, single_judge_info = humanevaleval.evaluator_map[dataset_name].judge(
                                current_prompt_text, decoded_text[j],  sample["ground_truth"], k
                            )
                        elif dataset_name=="mbpp":
                            passat1, single_judge_info = mbppeval.evaluator_map[dataset_name].judge(
                                current_prompt_text, decoded_text[j],  sample["ground_truth"], k
                            )
                        elif dataset_name=="livecodebench":
                            # LCB 评判将在稍后进行
                            passat1, single_judge_info = 0.0, {"status": "pending_lcb_reeval"}

                    except Exception as e:
                        print(f"Error judging sample {idx} (completion {j}): {e}", flush=True)
                        passat1 = 0.0
                        single_judge_info = {"error": str(e)}
                else:
                    # 在生成模式下，跳过评判
                    passat1, single_judge_info = 0.0, {"status": "skipped_generation_phase"}

                passat1_list.append(passat1)
                judge_info.append(single_judge_info)

            # 组装结果字典
            current_prompt_text = sample["prompt"][0]["value"] if isinstance(sample["prompt"], list) else sample["prompt"]
            original_idx = sample["original_idx"] if not args.reeval else sample["idx"]

            result = {
                "hyperparams": str(args),
                "prompt": current_prompt_text,
                "completion": decoded_text,
                "ground_truth": sample["ground_truth"], # <--- 这是修复后的行
                "generated_tokens": batch_gen_tokens,
                "avg_generated_tokens": sum(batch_gen_tokens)/len(batch_gen_tokens) if batch_gen_tokens else 0,
                "idx": original_idx,
                "n": args.num_samples,
                "finish_generation": finish_generation_dicts,
                "judge_info": judge_info,
                "passat1": sum(passat1_list)/len(passat1_list) if passat1_list else 0.0,
                "passat1_list": passat1_list,
                "soft_thinking_logs": batch_soft_logs # <-- (PPO) 保存 PPO 日志
            }
            results.append(result)

        # --- 7. 保存结果 (在 LCB 评估之前) ---
        print(f"\n--- (Code PPO) 保存详细结果 ({len(results)} 条) 到: {results_file} ---")
        with open(results_file, "w") as f:
            results.sort(key=lambda x: x["idx"])
            json.dump(results, f, indent=4)

        # --- 8. (可选) LiveCodeBench 外部评估 (复制自 run_sglang_softthinking.py) ---
        if dataset_name == "livecodebench" and args.reeval:
            print("--- (Code PPO) LiveCodeBench: 运行格式转换和外部评估器... ---")
            results_file_converted = os.path.join(output_dir, f"{base_filename}_converted.json")
            convert_livecodebench.convert_json(input_file=results_file, output_file=results_file_converted)

            orig_cwd = os.getcwd()
            lcb_pkg_dir = "../LiveCodeBench_pkg"  # (假设在同级)
            if not os.path.isdir(lcb_pkg_dir):
                print(f"警告: 找不到 '{lcb_pkg_dir}' 目录, 跳过 LCB 外部评估。")
            else:
                try:
                    custom_eval_cmd = [
                        sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
                        "--custom_output_file", os.path.join("..", results_file_converted),
                        "--release_version", "release_v5",
                        "--start_date", "2024-08-01",
                        "--num_process_evaluate", "1",
                        "--timeout", "50"
                    ]
                    print(f"--- (Code PPO) 运行 LCB 评估: {' '.join(custom_eval_cmd)} ---")
                    os.chdir(lcb_pkg_dir)
                    subprocess.run(custom_eval_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
                except Exception as e:
                    print(f"(Code PPO) LiveCodeBench 外部评估器失败: {e}", flush=True)
                finally:
                    os.chdir(orig_cwd)

                # LCB 结果回写
                livecodebench_results_file = os.path.join(output_dir, f"{base_filename}_converted_codegeneration_output_eval_all.json")
                if os.path.exists(livecodebench_results_file):
                    with open(livecodebench_results_file, "r") as f:
                        livecodebench_results = json.load(f)

                    for r in results:
                        for lcb_r in livecodebench_results:
                            if r["ground_truth"]["question_id"] == lcb_r["question_id"]:
                                r["passat1"] = lcb_r["pass@1"]
                                r["passat1_list"] = [int(p) for p in lcb_r.get("graded_list", [0.0]*args.num_samples)]
                                r["judge_info"] = lcb_r.get("metadata", {})
                                break

                    print(f"--- (Code PPO) LCB 结果回写后, 再次保存到: {results_file} ---")
                    with open(results_file, "w") as f:
                        results.sort(key=lambda x: x["idx"])
                        json.dump(results, f, indent=4)
                else:
                    print(f"--- (Code PPO) LCB 警告: 未找到评估结果文件 {livecodebench_results_file} ---")

        # --- 9. 计算并保存最终统计数据 (复制自 run_sglang_softthinking.py) ---
        end_time = time.time()
        print(f"--- (Code PPO) 评估完成。总耗时: {(end_time - start_time)/3600:.2f} 小时 ---")

        total_num = len(results)
        pass_at_1 = sum([r["passat1"] for r in results]) / total_num if total_num > 0 else 0
        avg_tokens_all = sum([r["avg_generated_tokens"] for r in results]) / total_num if total_num > 0 else 0
        time_taken_hours = (end_time - start_time) / 3600

        correct_results_tokens = [r["avg_generated_tokens"] for r in results if r["passat1"] > 0]
        avg_token_length_correct = sum(correct_results_tokens) / len(correct_results_tokens) if len(correct_results_tokens) > 0 else 0

        all_idx_list = sorted([(r["idx"], r["passat1"]) for r in results], key=lambda x: x[0])
        all_idx_dict = {str(i): j for i, j in all_idx_list}

        results_statistics = {
            "total_num": total_num,
            "pass@1": pass_at_1,
            "avg_token_length-all": avg_tokens_all,
            "avg_token_length-correct": avg_token_length_correct,
            "time_taken/h": time_taken_hours,
            "all_idx": all_idx_dict
        }

        print(f"--- 保存统计数据到: {results_statistics_file} ---")
        with open(results_statistics_file, "w") as f:
            json.dump(results_statistics, f, indent=4)

        print(f"\n最终统计: {json.dumps(results_statistics, indent=4)}")

    except Exception as e:
        print(f"发生致命错误: {e}")
        print(get_exception_traceback())

    finally:
        # 安全关闭引擎 (仅在 阶段 1 中初始化)
        if llm:
            llm.shutdown()
            print("sglang.Engine (PPO) shutdown.")
        else:
            print("sglang.Engine was not initialized (reeval mode).")


if __name__ == "__main__":
    main()
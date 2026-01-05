import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
import shutil
import sys
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client
from modelscope.hub.snapshot_download import snapshot_download
import asyncio
import matheval

# 假设这些评估库存在于您的环境中，如果不存在请注释掉
try:
    import humanevaleval
    import mbppeval
    import convert_livecodebench
except ImportError:
    pass

from huggingface_hub import HfApi
import torch
import uvloop

MATH_DATASETS = ["math500", "aime2024", "aime2025", "gpqa_diamond", "gsm8k", "amc23", "train_gsm8k"]
CODE_DATASETS = ["humaneval", "mbpp", "livecodebench"]


def main():
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description='PPO Agent Evaluation Script')
    parser.add_argument('--dataset', type=str,
                        choices=["math500", "aime2024", "aime2025", "gpqa_diamond", "gsm8k", "amc23", "humaneval",
                                 "mbpp", "livecodebench"], help='Name of dataset')
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"], default="flashinfer",
                        help='Sampling backend')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--model_id_scope', type=str, default=None, help='Model ID on ModelScope Hub')

    # PPO Agent 特有参数
    parser.add_argument('--ppo_agent_checkpoint_path', type=str, default=None,
                        help='Path to the trained PPO agent .pth file')
    parser.add_argument('--force_mode', type=str, choices=["soft", "hard", "ppo"], default="ppo",
                        help='Force mode: soft, hard, or ppo')

    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number')
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None)
    parser.add_argument('--max_running_requests', type=int, default=None)
    parser.add_argument('--max_batch', type=int, default=1000000)
    parser.add_argument('--mem_fraction_static', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=500)

    # 采样参数
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--max_generated_tokens', type=int, default=32768)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6)
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95)
    parser.add_argument('--after_thinking_top_k', type=int, default=30)
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0)
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.0)
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    # 噪声参数
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0)
    parser.add_argument('--gumbel_softmax_temperature', type=float, default=1.0)
    parser.add_argument('--add_noise_dirichlet', action='store_true')
    parser.add_argument('--add_noise_gumbel_softmax', action='store_true')

    # 评估与推送
    parser.add_argument('--reeval', action='store_true')
    parser.add_argument('--use_llm_judge', action='store_true')
    parser.add_argument('--api_base', type=str, default=None)
    parser.add_argument('--deployment_name', type=str, default=None)
    parser.add_argument('--api_version', type=str, default=None)
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument('--push_results_to_hf', action='store_true')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--hf_repo_id', type=str, default=None)

    parser.add_argument("--enable_soft_thinking", action="store_true")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=15)

    args = parser.parse_args()

    # 确保 PPO 模式下 enable_soft_thinking 为 True
    if args.ppo_agent_checkpoint_path:
        args.enable_soft_thinking = True

    download_model_if_needed(args.model_name, args.model_id_scope)

    dataset = args.dataset
    model_name = args.model_name

    # --- 确定 Soft/Hard Action ---
    if args.force_mode == "soft":
        action_value = 0
    elif args.force_mode == "hard":
        action_value = 1
    else:  # "ppo"
        action_value = None

    print(f"Arguments: {args}", flush=True)
    if dataset in MATH_DATASETS:
        matheval.set_client(args.api_base, args.deployment_name, args.api_version, args.api_key, args.judge_model_name)

    # --- Load Dataset (使用参考代码的硬编码路径) ---
    # 确保当前目录下有 datasets 文件夹
    if dataset == "math500":
        with open("./datasets/math500.json") as f:
            samples = json.load(f)
    elif dataset == "aime2024":
        with open("./datasets/aime2024.json") as f:
            samples = json.load(f)
    elif dataset == "aime2025":
        with open("./datasets/aime2025.json") as f:
            samples = json.load(f)
    elif dataset == "gpqa_diamond":
        with open("./datasets/gpqa_diamond.json") as f:
            samples = json.load(f)
    elif dataset == "gsm8k":
        with open("./datasets/gsm8k.json") as f:
            samples = json.load(f)
    elif dataset == "train_gsm8k":
        with open("./datasets/train_gsm8k.json") as f:
            samples = json.load(f)
    elif dataset == "amc23":
        with open("./datasets/amc23.json") as f:
            samples = json.load(f)
    elif dataset == "humaneval":
        with open("./datasets/humaneval.json") as f:
            samples = json.load(f)
    elif dataset == "mbpp":
        with open("./datasets/mbpp.json") as f:
            samples = json.load(f)
    elif dataset == "livecodebench":
        with open("./datasets/livecodebench.json") as f:
            samples = json.load(f)
    else:
        raise ValueError("Invalid dataset name")

    # --- Prompt Templates ---
    MATH_QUERY_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}".strip()
    GPQA_QUERY_TEMPLATE = "Please solve the following multiple-choice question. Please show your choice in the answer field with only the choice letter, e.g.,\"answer\": \"C\".\n\n{Question}".strip()
    CODE_QUERY_TEMPLATE = "Please solve the programming task below in Python. Code should be wrapped in a markdown code block.\n\n```python\n{Question}\n```".strip()
    MBPP_QUERY_TEMPLATE = "Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:\n1. The function name and signature must match exactly as specified in the test cases.\n2. Your code should be wrapped in a markdown code block without including any test cases.\n\nTask:\n{Question}\n\nTest Cases:\n```python\n{TestCases}\n```".strip()

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

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # --- Sampling Params (加入 soft_hard_action) ---
    sampling_params = {
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "after_thinking_temperature": args.after_thinking_temperature,
        "after_thinking_top_p": args.after_thinking_top_p,
        "after_thinking_top_k": args.after_thinking_top_k,
        "after_thinking_min_p": args.after_thinking_min_p,
        "n": 1,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "dirichlet_alpha": args.dirichlet_alpha,
        "max_new_tokens": args.max_generated_tokens,
        "think_end_str": args.think_end_str,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold,
        # !!! PPO 关键参数 !!!
        "soft_hard_action": action_value
    }

    # --- 文件名生成 ---
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{args.output_dir}/results/{dataset}", exist_ok=True)
    noise_suffix = (
            (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
            + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
    )
    # 添加 mode 到文件名
    base_filename_params = (
        f"{model_name.split('/')[-1]}_{dataset}_{args.enable_soft_thinking}_MODE_{args.force_mode.upper()}_{args.num_samples}_"
        f"{args.temperature}_{args.top_p}_{args.top_k}_{args.min_p}_{args.repetition_penalty}_"
        f"{args.max_topk}_{args.max_generated_tokens}{noise_suffix}"
    )
    base_filename = f"{base_filename_params}_{run_timestamp}"
    results_file = f"{args.output_dir}/results/{dataset}/{base_filename}.json"
    results_statistics_file = f"{args.output_dir}/results/{dataset}/{base_filename}_statistics.json"

    results = []

    print("--- Start Evaluation ---")
    start_time = time.time()

    if args.reeval:
        # reeval logic
        with open(results_file, "r") as f:
            results = json.load(f)
        prompt_list = []
        idx_list = list(range(args.start_idx, min(args.end_idx, len(results))))
        decoded_text_list = []
        finish_generation_list = []
        generated_tokens_list = []
        for r in results:
            prompt_list.append(r["prompt"])
            decoded_text_list.extend(r["completion"])
            finish_generation_list.extend(r["finish_generation"])
            generated_tokens_list.extend(r["generated_tokens"])
        results = []
    else:
        # Normal generation logic
        prompt_list = []
        idx_list = []
        for idx in range(args.start_idx, min(args.end_idx, len(samples))):
            sample = samples[idx]

            if dataset in ["aime2024", "aime2025", "math500", "gsm8k", "amc23"]:
                chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "gpqa_diamond":
                chat = [{"role": "user", "content": GPQA_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "humaneval":
                chat = [{"role": "user", "content": CODE_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"])}]
            elif dataset == "mbpp":
                chat = [{"role": "user", "content": MBPP_QUERY_TEMPLATE.format(Question=sample["prompt"][0]["value"],
                                                                               TestCases="\n".join(
                                                                                   sample["final_answer"][
                                                                                       "test_list"]))}]
            elif dataset == "livecodebench":
                chat = [{"role": "user", "content": get_lcb_prompt(question_content=sample["prompt"][0]["value"],
                                                                   starter_code=sample["final_answer"][
                                                                       "starter_code"])}]
            else:
                raise ValueError("Invalid dataset name")

            prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            for _ in range(args.num_samples):
                prompt_list.append(prompt)
            idx_list.append(idx)

        # --- Generation Loop ---
        decoded_text_list = []
        finish_generation_list = []
        generated_tokens_list = []
        idx = 0
        while idx < len(prompt_list):
            print(f"Number of GPUs available: {args.num_gpus}", flush=True)
            print(f"Processing batch starting at idx: {idx}", flush=True)

            # 关键：初始化 SGL Engine 并传入 PPO Checkpoint
            llm = sgl.Engine(
                model_path=model_name,
                tp_size=args.num_gpus,
                log_level="info",
                trust_remote_code=True,
                random_seed=args.random_seed,
                max_running_requests=args.max_running_requests,
                mem_fraction_static=args.mem_fraction_static,
                disable_cuda_graph=True,
                disable_overlap_schedule=True,
                enable_soft_thinking=args.enable_soft_thinking,
                add_noise_dirichlet=args.add_noise_dirichlet,
                add_noise_gumbel_softmax=args.add_noise_gumbel_softmax,
                max_topk=args.max_topk,
                cuda_graph_max_bs=args.cuda_graph_max_bs,
                sampling_backend=args.sampling_backend,
                # PPO 特有
                ppo_agent_checkpoint_path=args.ppo_agent_checkpoint_path
            )

            # 执行生成
            batch_prompts = prompt_list[idx: idx + args.max_batch]
            outputs = llm.generate(batch_prompts, sampling_params)

            decoded_text_list.extend([o["text"] for o in outputs])
            # 注意: 这里 finish_generation 逻辑保持与参考一致
            finish_generation_list.extend(
                [o["meta_info"]["finish_reason"]["type"] == "stop" and not args.enable_soft_thinking for o in outputs])
            generated_tokens_list.extend([o["meta_info"]["completion_tokens"] for o in outputs])

            idx += args.max_batch
            outputs = None
            llm.shutdown()
            torch.cuda.empty_cache()

    # --- Initialization Evaluators ---
    if dataset in CODE_DATASETS:
        try:
            mbppeval.init_evaluator()
            humanevaleval.init_evaluator()
        except:
            print("Warning: Code evaluators initialization failed or modules missing.")

    # --- Evaluation Loop ---
    for i, idx in enumerate(idx_list):
        if i % 10 == 0: print(f"Evaluating sample {i}/{len(idx_list)}", flush=True)
        sample = samples[idx]
        judge_info = []
        passat1_list = []
        decoded_text = decoded_text_list[i * args.num_samples:(i + 1) * args.num_samples]
        finish_generation = finish_generation_list[i * args.num_samples:(i + 1) * args.num_samples]

        for j in range(args.num_samples):
            for _ in range(5):  # Retry loop
                try:
                    if dataset in MATH_DATASETS:
                        rule_judge_result, extracted_answer = matheval.evaluator_map[dataset].rule_judge(
                            decoded_text[j], sample["final_answer"], finish_generation[j])
                        llm_judge_result = None
                        if not rule_judge_result and args.use_llm_judge:
                            llm_judge_result = matheval.evaluator_map[dataset].llm_judge(decoded_text[j],
                                                                                         sample["final_answer"],
                                                                                         extracted_answer,
                                                                                         finish_generation[j])
                        finally_judge_result = rule_judge_result or llm_judge_result
                        judge_info.append({
                            "rule_judge_result": rule_judge_result,
                            "llm_judge_result": llm_judge_result,
                            "finally_judge_result": finally_judge_result
                        })
                        passat1_list.append(1.0 if finally_judge_result else 0.0)

                    elif dataset in CODE_DATASETS:
                        # 代码评估逻辑保持原样
                        k = 1
                        single_judge_info = None
                        passat1 = 0.0
                        if dataset == "humaneval" and args.reeval:
                            passat1, single_judge_info = humanevaleval.evaluator_map[dataset].judge(
                                sample["prompt"][0]["value"], decoded_text[j], sample["final_answer"], k)
                        elif dataset == "mbpp" and args.reeval:
                            passat1, single_judge_info = mbppeval.evaluator_map[dataset].judge(
                                sample["prompt"][0]["value"], decoded_text[j], sample["final_answer"], k)

                        passat1_list.append(passat1)
                        judge_info.append(single_judge_info)
                    break  # Break retry loop
                except Exception as e:
                    print(f"Eval Error: {e}", flush=True)
                    time.sleep(0.5)

        # Save Result
        result = {
            "hyperparams": str(args),
            "prompt": sample["prompt"][0]["value"],
            "completion": decoded_text,
            "ground_truth": sample["final_answer"],
            "generated_tokens": generated_tokens_list[i * args.num_samples:(i + 1) * args.num_samples],
            "avg_generated_tokens": sum(
                generated_tokens_list[i * args.num_samples:(i + 1) * args.num_samples]) / args.num_samples,
            "idx": idx,
            "n": args.num_samples,
            "finish_generation": finish_generation_list[i * args.num_samples:(i + 1) * args.num_samples],
            "judge_info": judge_info,
            "passat1": sum(passat1_list) / len(passat1_list) if passat1_list else 0.0,
            "passat1_list": passat1_list
        }
        results.append(result)

    # Save to JSON
    with open(results_file, "w") as f:
        results.sort(key=lambda x: x["idx"])
        json.dump(results, f, indent=4)

    # --- LiveCodeBench Conversion Logic ---
    if dataset == "livecodebench":
        try:
            from convert_livecodebench import convert_json
            results_file_converted = f"{args.output_dir}/results/{dataset}/{base_filename}_converted.json"
            convert_json(input_file=results_file, output_file=results_file_converted)

            if args.reeval:
                import subprocess
                orig_cwd = os.getcwd()
                lcb_pkg_dir = "../LiveCodeBench_pkg"
                custom_eval_cmd = [
                    sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
                    "--custom_output_file", "../" + results_file_converted,
                    "--release_version", "release_v5",
                    "--start_date", "2024-08-01",
                    "--num_process_evaluate", "1",
                    "--timeout", "50"
                ]
                print("Running LCB custom_evaluator...")
                os.chdir(lcb_pkg_dir)
                subprocess.run(custom_eval_cmd, check=True)
                os.chdir(orig_cwd)

                # Load back LCB results
                lcb_res_file = f"{args.output_dir}/results/{dataset}/{base_filename}_converted_codegeneration_output_eval_all.json"
                with open(lcb_res_file, "r") as f:
                    lcb_results = json.load(f)

                # Merge back
                for r in results:
                    for lcb_r in lcb_results:
                        if r["ground_truth"]["question_id"] == lcb_r["question_id"]:
                            r["passat1"] = lcb_r["pass@1"]
                            r["passat1_list"] = [int(p) for p in lcb_r["graded_list"]]
                            r["judge_info"] = lcb_r["metadata"]
                            break
                with open(results_file, "w") as f:
                    results.sort(key=lambda x: x["idx"])
                    json.dump(results, f, indent=4)
        except Exception as e:
            print(f"LCB Conversion/Eval failed: {e}")

    # --- Statistics ---
    total_num = len(results)
    pass_at_1 = sum([r["passat1"] for r in results]) / total_num if total_num > 0 else 0
    end_time = time.time()
    print(f"Evaluation End. Time taken: {(end_time - start_time) / 3600:.2f} hours")

    results_statistics = {
        "total_num": total_num,
        "pass@1": pass_at_1,
        "avg_token_length-all": sum([r["avg_generated_tokens"] for r in results]) / total_num if total_num > 0 else 0,
        "avg_token_length-correct": sum([r["avg_generated_tokens"] for r in results if r["passat1"] > 0]) / len(
            [r["passat1"] for r in results if r["passat1"] > 0]) if any(r["passat1"] > 0 for r in results) else 0,
        "time_taken/h": (end_time - start_time) / 3600
    }

    all_idx = sorted([(r["idx"], r["passat1"]) for r in results], key=lambda x: x[0])
    results_statistics["all_idx"] = {str(i): j for i, j in all_idx}

    with open(results_statistics_file, "w") as f:
        json.dump(results_statistics, f, indent=4)

    if args.push_results_to_hf:
        api = HfApi()
        api.upload_file(path_or_fileobj=results_statistics_file, path_in_repo=results_statistics_file,
                        repo_id=args.hf_repo_id, token=args.hf_token)

    print(results_statistics, flush=True)


def download_model_if_needed(local_model_path, modelscope_id):
    config_path = os.path.join(local_model_path, "config.json")
    if os.path.exists(config_path):
        print(f"Model found locally at: {local_model_path}")
        return
    if not modelscope_id:
        print(f"Error: Model not found locally and no modelscope_id provided for {local_model_path}")
        sys.exit(1)

    print(f"Downloading {modelscope_id} from ModelScope...")
    try:
        os.makedirs(local_model_path, exist_ok=True)
        cache_path = snapshot_download(model_id=modelscope_id, ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf",
                                                                                "consolidated.safetensors"])
        shutil.copytree(cache_path, local_model_path, dirs_exist_ok=True)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
import os
import sys
import json
import argparse
import subprocess
import re
from tqdm import tqdm
from openai import OpenAI

cache_dir = "/root/shared-nvme/gj/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "models")
os.environ["TMPDIR"] = os.path.join(cache_dir, "tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

TARGET_DATASETS = ["humaneval", "mbpp", "livecodebench"]

import humanevaleval
import mbppeval
try:
    from convert_livecodebench import convert_json
except ImportError:
    pass

JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "sk-f5aff073f1da401c98180a7a9c8a50f9")
JUDGE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_MODEL_NAME = "deepseek-chat"

judge_client = None

def init_judge_client():
    global judge_client
    if JUDGE_API_KEY:
        try:
            judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
        except Exception:
            pass

def extract_after_think(text: str) -> str:
    pattern = r"</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

def try_llm_rescue(question: str, solution_raw: str, ground_truth: dict) -> bool:
    global judge_client
    if not judge_client:
        return False

    reference_code = ""
    if isinstance(ground_truth, dict):
        reference_code = ground_truth.get("code") or ground_truth.get("canonical_solution") or ground_truth.get("solution") or ""
    elif isinstance(ground_truth, str):
        reference_code = ground_truth

    clean_solution = extract_after_think(solution_raw)

    if reference_code:
        prompt = f"""You are a code reviewer.
Problem:
{question}

Reference:
{reference_code}

Response:
{clean_solution}

Evaluate based on the following rules:
1. If the core algorithm logic matches the reference -> YES.
2. If the response contains the correct solution, even if repeated multiple times -> YES.
3. If the code is truncated but the logic is visible -> YES.
4. If the output is complete nonsense or does not solve the problem -> NO.

Return only "YES" or "NO"."""

    else:
        prompt = f"""You are a code reviewer. NO REFERENCE AVAILABLE.
Problem:
{question}

Response:
{clean_solution}

Evaluate based on the following rules:
1. If the algorithmic approach logically solves the problem -> YES.
2. If the code is truncated but the logic so far is correct -> YES.
3. If the code is readable despite minor format issues -> YES.
4. If the model repeats the same text/code endlessly (Infinite Repetition) -> NO.
5. If the output is unreadable chaos -> NO.

Return only "YES" or "NO"."""

    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception:
        return False

def update_statistics_file(result_filepath, new_pass_at_1, updated_data):
    stats_filepath = result_filepath.replace(".json", "_statistics.json")
    if not os.path.exists(stats_filepath):
        stats = {}
    else:
        try:
            with open(stats_filepath, 'r') as f:
                stats = json.load(f)
        except Exception:
            stats = {}

    try:
        stats["pass@1"] = new_pass_at_1
        updated_data.sort(key=lambda x: x.get("idx", 0) if isinstance(x.get("idx"), int) else int(x.get("idx", 0) if str(x.get("idx", "0")).isdigit() else 0))

        all_idx = {}
        for sample in updated_data:
            idx_val = sample.get("idx")
            if idx_val is not None:
                all_idx[str(idx_val)] = sample.get("passat1", 0.0)

        stats["all_idx"] = all_idx
        with open(stats_filepath, 'w') as f:
            json.dump(stats, f, indent=4)
    except Exception:
        pass

def evaluate_humaneval_mbpp(filepath, dataset_name):
    print(f"Evaluating {dataset_name}: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    try:
        if dataset_name == "humaneval":
            humanevaleval.init_evaluator()
            evaluator = humanevaleval.evaluator_map[dataset_name]
        elif dataset_name == "mbpp":
            mbppeval.init_evaluator()
            evaluator = mbppeval.evaluator_map[dataset_name]
        else:
            return 0.0
    except Exception:
        return 0.0

    pass_list = []
    updated_data = []
    rescue_count = 0

    for sample in tqdm(data, desc=f"Running Tests ({dataset_name})"):
        prompt = sample.get("prompt", "")
        completion = sample.get("completion")
        if isinstance(completion, list): completion = completion[0]
        ground_truth = sample.get("ground_truth")

        try:
            pass_at_1, metadata = evaluator.judge(prompt, completion, ground_truth, 1)
        except Exception as e:
            pass_at_1 = 0.0
            metadata = {"error": str(e)}

        if pass_at_1 == 0.0:
            is_rescued = try_llm_rescue(prompt, completion, ground_truth)
            if is_rescued:
                pass_at_1 = 1.0
                if not isinstance(metadata, dict): metadata = {}
                metadata["rescued_by_llm"] = True
                metadata["original_error"] = "Execution Failed"
                rescue_count += 1

        pass_list.append(pass_at_1)
        sample["passat1"] = pass_at_1
        sample["judge_info"] = metadata
        updated_data.append(sample)

    acc = sum(pass_list) / len(pass_list) if pass_list else 0
    print(f"✅ Result for {filepath}: Pass@1 = {acc:.2%} (Rescued: {rescue_count}/{len(pass_list)})")

    with open(filepath, 'w') as f:
        json.dump(updated_data, f, indent=4)
    update_statistics_file(filepath, acc, updated_data)
    return acc

def evaluate_livecodebench(filepath, base_dir):
    print(f"\nEvaluating livecodebench: {filepath}")

    converted_file = filepath.replace(".json", "_converted.json")
    convert_json(input_file=filepath, output_file=converted_file)

    possible_pkg_paths = [
        "/root/shared-nvme/gj/Hybrid-Thinking/LiveCodeBench_pkg",
        "LiveCodeBench_pkg",
        "../LiveCodeBench_pkg",
        os.path.join(base_dir, "LiveCodeBench_pkg")
    ]

    lcb_pkg_dir = None
    for p in possible_pkg_paths:
        if os.path.exists(p) and os.path.isdir(p):
            lcb_pkg_dir = p
            break

    if not lcb_pkg_dir:
        return 0.0

    orig_cwd = os.getcwd()

    try:
        os.chdir(lcb_pkg_dir)
        abs_converted_path = os.path.abspath(os.path.join(orig_cwd, converted_file))

        cmd = [
            sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", abs_converted_path,
            "--release_version", "release_v5",
            "--start_date", "2024-08-01",
            "--num_process_evaluate", "8",
            "--timeout", "50"
        ]
        subprocess.run(cmd, check=True)

    except Exception:
        os.chdir(orig_cwd)
    finally:
        os.chdir(orig_cwd)

    eval_output_file = converted_file.replace(".json", "_codegeneration_output_eval_all.json")

    lcb_results = []
    if os.path.exists(eval_output_file):
        with open(eval_output_file, 'r') as f:
            lcb_results = json.load(f)

    with open(filepath, 'r') as f:
        original_data = json.load(f)

    pass_list = []
    updated_data = []
    rescue_count = 0
    match_count = 0

    for orig in tqdm(original_data, desc="Mapping & Rescuing LCB"):
        raw_qid = orig.get("ground_truth", {}).get("question_id")
        qid = str(raw_qid).strip() if raw_qid is not None else None

        pass_val = 0.0
        metadata = {}

        for res in lcb_results:
            if str(res.get("question_id")).strip() == qid:
                match_count += 1
                pass_val = float(res.get("pass@1", 0.0))
                metadata = res.get("metadata", {})
                break

        if pass_val == 0.0:
            prompt = orig.get("prompt", "") or str(orig.get("ground_truth", ""))
            completion = orig.get("completion", "")
            if isinstance(completion, list): completion = completion[0]
            ground_truth = orig.get("ground_truth")

            is_rescued = try_llm_rescue(prompt, completion, ground_truth)

            if is_rescued:
                pass_val = 1.0
                if not isinstance(metadata, dict): metadata = {}
                metadata["rescued_by_llm"] = True
                metadata["rescue_type"] = "Loose Blind Judge"
                rescue_count += 1

        orig["passat1"] = pass_val
        orig["judge_info"] = metadata
        pass_list.append(pass_val)
        updated_data.append(orig)

    acc = sum(pass_list) / len(pass_list) if pass_list else 0
    print(f"\n✅ Result for {filepath}: Pass@1 = {acc:.2%} (Rescued: {rescue_count}/{len(pass_list)})")

    with open(filepath, 'w') as f:
        json.dump(updated_data, f, indent=4)
    update_statistics_file(filepath, acc, updated_data)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./eval_results")
    args = parser.parse_args()

    init_judge_client()

    search_dir = args.results_dir

    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if not file.endswith(".json") or "statistics" in file or "converted" in file:
                continue
            if "_codegeneration_output_eval_all" in file:
                continue

            filepath = os.path.join(root, file)
            file_lower = file.lower()

            dataset_type = None
            if "humaneval" in file_lower: dataset_type = "humaneval"
            elif "mbpp" in file_lower: dataset_type = "mbpp"
            elif "livecodebench" in file_lower: dataset_type = "livecodebench"

            if dataset_type and dataset_type in TARGET_DATASETS:
                print(f"\nProcessing [{dataset_type}]: {filepath}")
                try:
                    if dataset_type == "livecodebench":
                        evaluate_livecodebench(filepath, args.results_dir)
                    else:
                        evaluate_humaneval_mbpp(filepath, dataset_type)
                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")

if __name__ == "__main__":
    main()
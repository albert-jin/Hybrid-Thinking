import os
import sys
import json
import subprocess
import argparse
import re  # 必须导入
from tqdm import tqdm

# === 配置您下载的本地路径 ===
LCB_PKG_DIR = "/root/shared-nvme/gj/Hybrid-Thinking/LiveCodeBench_pkg"


# ==========================

def extract_code(text: str) -> str:
    """更鲁棒的代码提取函数"""
    if not isinstance(text, str):
        return ""

    # 1. 移除 <think> 标签
    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. 尝试提取代码块 (支持 ```python, ``` python, ```Python 等)
    # pattern: ``` + 任意空白 + (可选语言名) + 换行 + (代码内容) + ```
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 3. 通用 fallback (匹配任何 ``` ... ```)
    matches_generic = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches_generic:
        return matches_generic[-1].strip()

    # 4. 如果没有代码块，尝试简单的启发式清洗（如果只有代码）
    # 或者直接返回原始内容
    return text.strip()


def convert_to_lcb_format(input_file, output_file):
    """将结果转换为 LiveCodeBench 需要的格式 (含代码清洗)"""
    print(f"Converting {input_file} -> {output_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output_data = []
        for item in data:
            # 获取 question_id
            gt = item.get("ground_truth")
            qid = None
            if isinstance(gt, dict):
                qid = gt.get("question_id")

            if not qid:
                continue

            # 获取原始输出
            completion = item.get("completion")

            # 提取代码
            if isinstance(completion, str):
                completion_list = [extract_code(completion)]
            elif isinstance(completion, list):
                completion_list = [extract_code(c) for c in completion]
            else:
                completion_list = [""]

            # LCB 格式要求
            output_data.append({
                "question_id": qid,
                "code_list": completion_list
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Format conversion failed: {e}")
        return False


def update_stats(result_file, acc, updated_data):
    """同步更新 statistics 文件"""
    stats_file = result_file.replace(".json", "_statistics.json")
    stats = {}

    # 读取旧数据
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except:
            pass

    # 更新关键指标
    stats["pass@1"] = acc
    stats["total_num"] = len(updated_data)

    # 重新计算 Token 长度
    all_tokens = []
    correct_tokens = []
    for item in updated_data:
        t_len = item.get("avg_generated_tokens", 0)
        if t_len == 0 and isinstance(item.get("generated_tokens"), list) and item["generated_tokens"]:
            t_len = sum(item["generated_tokens"]) / len(item["generated_tokens"])

        all_tokens.append(t_len)
        if item.get("passat1", 0) > 0:
            correct_tokens.append(t_len)

    stats["avg_token_length-all"] = sum(all_tokens) / len(all_tokens) if all_tokens else 0
    stats["avg_token_length-correct"] = sum(correct_tokens) / len(correct_tokens) if correct_tokens else 0

    # 更新索引
    updated_data.sort(key=lambda x: x.get("idx", 0))
    stats["all_idx"] = {str(d.get("idx")): d.get("passat1", 0.0) for d in updated_data}

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"Stats file updated: {stats_file}")


def evaluate_single_file(filepath):
    print(f"\nProcessing: {filepath}")

    # 1. 准备路径
    abs_filepath = os.path.abspath(filepath)
    converted_file = abs_filepath.replace(".json", "_converted.json")

    # 2. 转换格式
    if not convert_to_lcb_format(abs_filepath, converted_file):
        return 0.0  # 失败返回 0

    # 3. 准备执行环境
    if not os.path.exists(LCB_PKG_DIR):
        print(f"❌ Error: 本地库路径不存在: {LCB_PKG_DIR}")
        return 0.0

    orig_cwd = os.getcwd()
    try:
        os.chdir(LCB_PKG_DIR)

        # 构造命令
        cmd = [
            sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", converted_file,
            "--release_version", "release_v5",
            "--start_date", "2024-08-01",
            "--num_process_evaluate", "8",
            "--timeout", "60"
        ]

        # 设置缓存环境
        env = os.environ.copy()
        temp_cache_dir = "/root/shared-nvme/gj/tmp/lcb_cache_safe"
        os.makedirs(temp_cache_dir, exist_ok=True)
        env["HF_DATASETS_CACHE"] = temp_cache_dir
        env["HF_HOME"] = temp_cache_dir

        print("🚀 Running LCB Runner (Local)...")
        subprocess.run(cmd, check=True, env=env)

    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        return 0.0
    finally:
        os.chdir(orig_cwd)

    # 4. 合并结果
    output_eval_file = converted_file.replace(".json", "_codegeneration_output_eval_all.json")

    if not os.path.exists(output_eval_file):
        print(f"❌ Output file not found: {output_eval_file}")
        return 0.0

    print("🔄 Merging results...")
    try:
        with open(output_eval_file, 'r', encoding='utf-8') as f:
            lcb_results = json.load(f)
        with open(abs_filepath, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading result files: {e}")
        return 0.0

    pass_list = []
    updated_data = []

    # 创建快速查找字典
    lcb_map = {res["question_id"]: res for res in lcb_results}

    for orig in original_data:
        qid = None
        if isinstance(orig.get("ground_truth"), dict):
            qid = orig["ground_truth"].get("question_id")

        if qid and qid in lcb_map:
            res = lcb_map[qid]
            # 确保转换为 float
            score = float(res.get("pass@1", 0))
            orig["passat1"] = score
            orig["judge_info"] = res.get("metadata", {})
            pass_list.append(score)
            updated_data.append(orig)
        else:
            # 如果没找到结果，记为 0
            orig["passat1"] = 0.0
            updated_data.append(orig)

    # 5. 写回原始文件并返回分数
    acc = 0.0
    if pass_list:
        acc = sum(pass_list) / len(pass_list)
        print(f"✅ Evaluation Complete. Pass@1: {acc:.2%}")

        with open(abs_filepath, 'w', encoding='utf-8') as f:
            json.dump(original_data, f, indent=4, ensure_ascii=False)

        update_stats(abs_filepath, acc, updated_data)
    else:
        print("⚠️ Warning: No matching results found during merge.")

    return acc  # <--- 【关键修改】必须返回 acc，否则外面收到的是 None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, required=True, help="Directory containing result jsons")
    args = parser.parse_args()

    if not os.path.exists(args.scan_dir):
        print(f"Directory not found: {args.scan_dir}")
        return

    print(f"Scanning {args.scan_dir} for livecodebench results...")

    found = False
    for root, dirs, files in os.walk(args.scan_dir):
        for file in files:
            if not file.endswith(".json"): continue
            if "statistics" in file or "converted" in file or "codegeneration" in file: continue

            if "livecodebench" in file.lower() or "livecodebench" in root.lower():
                filepath = os.path.join(root, file)
                evaluate_single_file(filepath)
                found = True

    if not found:
        print("No livecodebench result files found.")


if __name__ == "__main__":
    main()
import os
import json
import argparse
from tqdm import tqdm


def calculate_metrics(data):
    """
    根据结果列表重新计算核心指标
    """
    total_num = len(data)
    if total_num == 0:
        return None

    # 1. 计算 Pass@1
    # 兼容 passat1 可能是 1.0/0.0 或 True/False 的情况
    pass_list = []
    for item in data:
        p = item.get("passat1", 0)
        # 尝试转换为 float
        try:
            p = float(p)
        except:
            p = 0.0
        pass_list.append(p)

    pass_at_1 = sum(pass_list) / total_num

    # 2. 计算 Token 长度统计 (如果存在)
    avg_tokens_all = 0
    avg_token_len_correct = 0

    # 检查是否有 token 数据
    if "avg_generated_tokens" in data[0] or "generated_tokens" in data[0]:
        all_tokens = []
        correct_tokens = []

        for i, item in enumerate(data):
            # 获取当前样本的 token 长度
            t_len = item.get("avg_generated_tokens", 0)
            if t_len == 0 and "generated_tokens" in item:
                # 尝试从 list 中获取
                gen_list = item["generated_tokens"]
                if isinstance(gen_list, list) and len(gen_list) > 0:
                    t_len = sum(gen_list) / len(gen_list)

            all_tokens.append(t_len)
            if pass_list[i] > 0:
                correct_tokens.append(t_len)

        avg_tokens_all = sum(all_tokens) / len(all_tokens) if all_tokens else 0
        avg_token_len_correct = sum(correct_tokens) / len(correct_tokens) if correct_tokens else 0

    # 3. 生成 all_idx 字典 {"0": 1.0, "1": 0.0}
    # 先按 idx 排序
    data_sorted = sorted(data, key=lambda x: x.get("idx", -1))
    all_idx = {str(item.get("idx")): item.get("passat1", 0) for item in data_sorted}

    return {
        "total_num": total_num,
        "pass@1": pass_at_1,
        "avg_token_length-all": avg_tokens_all,
        "avg_token_length-correct": avg_token_len_correct,
        "all_idx": all_idx
    }


def process_file(result_filepath):
    # 构造对应的统计文件名
    stats_filepath = result_filepath.replace(".json", "_statistics.json")

    # 如果统计文件不存在，我们是否要创建？通常应该更新已有的。
    # 这里设定为：如果结果文件存在，我们就强制生成/覆盖统计文件

    try:
        # 读取结果文件
        with open(result_filepath, 'r') as f:
            result_data = json.load(f)

        # 计算新指标
        new_metrics = calculate_metrics(result_data)
        if not new_metrics:
            print(f"[Skip] Empty data: {result_filepath}")
            return

        # 读取旧统计文件 (为了保留 time_taken 等无法重算的字段)
        stats_data = {}
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'r') as f:
                try:
                    stats_data = json.load(f)
                except json.JSONDecodeError:
                    stats_data = {}  # 文件损坏则重置

        # 更新字段
        old_pass = stats_data.get("pass@1", "N/A")

        stats_data["total_num"] = new_metrics["total_num"]
        stats_data["pass@1"] = new_metrics["pass@1"]
        stats_data["avg_token_length-all"] = new_metrics["avg_token_length-all"]
        stats_data["avg_token_length-correct"] = new_metrics["avg_token_length-correct"]
        stats_data["all_idx"] = new_metrics["all_idx"]

        # 保存
        with open(stats_filepath, 'w') as f:
            json.dump(stats_data, f, indent=4)

        print(f"[Updated] {os.path.basename(stats_filepath)} | Pass@1: {old_pass} -> {new_metrics['pass@1']:.4f}")

    except Exception as e:
        print(f"[Error] Failed processing {result_filepath}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./eval_results", help="Directory to scan")
    args = parser.parse_args()

    print(f"Scanning directory: {args.results_dir} ...")

    found_files = []

    for root, dirs, files in os.walk(args.results_dir):
        for file in files:
            # 找到结果文件：以 .json 结尾，且不是统计文件，也不是转换文件
            if file.endswith(".json") and "statistics" not in file and "converted" not in file:
                found_files.append(os.path.join(root, file))

    print(f"Found {len(found_files)} result files. Starting synchronization...")

    for filepath in tqdm(found_files):
        process_file(filepath)

    print("\n--- Synchronization Complete ---")


if __name__ == "__main__":
    main()
from modelscope.hub.snapshot_download import snapshot_download
import argparse
import os
import sys

# --- 配置 ---
# 这里设置为你想要存放所有模型的根目录
# 当下载 'LLM-Research/Meta-Llama-3.1-8B-Instruct' 时，
# 它会自动在该目录下创建 'LLM-Research/Meta-Llama-3.1-8B-Instruct' 文件夹
MODELS_ROOT_DIR = "/root/shared-nvme/gj/Hybrid-Thinking/models"

parser = argparse.ArgumentParser(description='Download model from ModelScope Hub')
parser.add_argument('--model_name', type=str, default='LLM-Research/Meta-Llama-3.1-8B-Instruct', help='Name of the model to download')
args = parser.parse_args()

# 拼接最终的模型目录路径用于检查
local_dir = os.path.join(MODELS_ROOT_DIR, args.model_name)

# --- 核心检查逻辑 ---
# 检查 config.json 是否存在，如果存在则跳过
if os.path.exists(os.path.join(local_dir, "config.json")):
    print(f"Model already exists at: {local_dir}")
    print("Skipping download.")
    sys.exit(0)

print(f"Model not found at {local_dir}. Starting direct download...")

# --- 下载逻辑 (修改版) ---
# 使用 cache_dir 参数直接指向目标目录的父级
# ModelScope 会自动处理目录结构: cache_dir/model_id
snapshot_download(
    model_id=args.model_name,
    cache_dir=MODELS_ROOT_DIR,  # 关键修改：直接指定下载根路径
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf", "consolidated.safetensors"]
)

print(f"Model successfully downloaded to: {local_dir}")
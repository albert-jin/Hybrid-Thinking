from modelscope.hub.snapshot_download import snapshot_download
import argparse
import os
import shutil
import sys  # 导入 sys 库用于正常退出脚本
# 模型从model scope下载
parser = argparse.ArgumentParser(description='Download model from ModelScope Hub')
parser.add_argument('--model_name', type=str, default='qwen/Qwen-72B-Chat', help='Name of the model to download')
args = parser.parse_args()

local_dir = "./models/" + args.model_name

# --- 新增的核心检查逻辑 ---
# 检查一个代表性的文件（如 config.json）是否存在于目标目录中
# 如果存在，就认为模型已经下载好了
if os.path.exists(os.path.join(local_dir, "config.json")):
    print(f"Model already exists at: {local_dir}")
    print("Skipping download.")
    sys.exit(0)  # 打印消息后正常退出脚本，返回码为 0
# --- 检查逻辑结束 ---

# 如果模型不存在，则继续执行下面的下载和复制流程
print(f"Model not found at {local_dir}. Starting download...")
os.makedirs(local_dir, exist_ok=True)

# 第 1 步: 下载模型到 ModelScope 的默认缓存目录
print("Step 1: Downloading model to cache...")
cache_path = snapshot_download(model_id=args.model_name,
                               ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf", "consolidated.safetensors"])

# 第 2 步: 将缓存目录中的文件递归复制到我们的目标目录
print(f"Step 2: Copying files from {cache_path} to {local_dir}...")
shutil.copytree(cache_path, local_dir, dirs_exist_ok=True)

print(f"Model successfully downloaded and copied to: {local_dir}")
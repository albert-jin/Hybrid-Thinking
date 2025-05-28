from calendar import month_name
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser(description='Download model from Hugging Face Hub')
parser.add_argument('--model_name', type=str, default='Qwen/QwQ-32B', help='Name of the model to download')
args = parser.parse_args()

local_dir = "./models/"+args.model_name
snapshot_download(repo_id=args.model_name,
                  ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf", "consolidated.safetensors"] ,
                  local_dir=local_dir)

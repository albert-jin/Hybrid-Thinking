# train_rl_strategy.py

import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
import shutil
import sys
import random # 新增
import math # 新增
import logging # 新增

from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
import asyncio
import dataclasses
import matheval #
from matheval import evaluator_map, set_client # 保留评估器用于获取奖励
# from modelscope.hub.snapshot_download import snapshot_download # 暂时保留，根据需要决定是否移除
import torch

# --- 导入 RL 相关模块 ---
from strategy_selector.strategy_model import DQN
from strategy_selector.replay_buffer import Trajectory # 只需要 Trajectory 定义

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 数据集常量 (保持不变) ---
MATH_DATASETS = ["gsm8k"] # 训练专注于 GSM8K
# CODE_DATASETS = ["humaneval","mbpp","livecodebench"] # 暂时不用于训练

# --- 模型下载函数 (暂时保留，可能需要调整) ---
def download_model_if_needed(local_model_path, modelscope_id):
    # ... (原有下载逻辑，可以暂时保留) ...
    config_path = os.path.join(local_model_path, "config.json")
    if os.path.exists(config_path):
        logger.info(f"Model found locally at: {local_model_path}")
        return local_model_path # 返回路径
    logger.warning(f"Model not found locally at {local_model_path}.")
    # TODO: 添加下载逻辑 (modelscope_download or huggingface_hub)
    raise NotImplementedError("Model download logic needs implementation if not found locally.")
    # return downloaded_path

# --- 主函数 ---
def main():
    # === 1. 参数解析 ===
    parser = argparse.ArgumentParser(description='Train RL Strategy Selector for Soft Thinking.')

    # --- LLM and SGLang Args (基本保留 run_sglang_softthinking.py 的参数) ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to the base LLM model directory')
    # parser.add_argument('--model_id_scope', type=str, default=None, help='Model ID on ModelScope for download') # 可选
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to the tokenizer, if different from model_path')
    parser.add_argument('--dataset_path', type=str, default="./datasets/gsm8k_train.json", help='Path to the training dataset (e.g., train_gsm8k.JSON)') # 明确训练集
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for SGLang Tensor Parallelism (tp_size)') # 训练时可能先用单卡调试
    parser.add_argument('--mem_fraction_static', type=float, default=0.8, help='GPU memory fraction for SGLang')
    # ... (可以保留 SGLang 的其他高级参数如 max_running_requests, sampling_backend 等，但先用默认值)
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"], default="flashinfer", help='Sampling backend')

    # --- RL Training Args (新增) ---
    parser.add_argument('--total_train_steps', type=int, default=100000, help='Total number of training steps (problems solved)')
    parser.add_argument('--reduced_dim', type=int, default=512, help='Dimension after reducing LLM embeddings for QNetwork input')
    parser.add_argument('--lstm_hidden_dim', type=int, default=512, help='LSTM hidden dimension in QNetwork')
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of LSTM layers in QNetwork')
    parser.add_argument('--replay_buffer_capacity', type=int, default=10000, help='Capacity of the main replay buffer')
    parser.add_argument('--error_buffer_capacity', type=int, default=2000, help='Capacity of the error replay buffer ("wrong answer book")')
    parser.add_argument('--error_buffer_sampling_prob', type=float, default=0.3, help='Probability of sampling from the error buffer')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the DQN optimizer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update factor for target network')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DQN training')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=50000, help='Exponential decay rate for epsilon (steps)')
    parser.add_argument('--state_max_len', type=int, default=35, help='Max sequence length for RL state')
    parser.add_argument('--include_prompt_in_state', action='store_true', help='Include prompt embeddings in the RL state')
    parser.add_argument('--train_interval_steps', type=int, default=4, help='Train DQN every N problems solved')
    # parser.add_argument('--target_update_interval', type=int, default=100, help='Update target network every N training steps (batches)') # target 更新频率 (DQN类中通过tau软更新，这个参数可以去掉)
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints_rl", help='Directory to save RL model checkpoints')
    parser.add_argument('--save_interval_steps', type=int, default=5000, help='Save RL checkpoint every N problems solved')
    parser.add_argument('--log_interval_steps', type=int, default=100, help='Log training progress every N problems solved')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to load a pre-trained RL checkpoint')
    parser.add_argument('--reward_correct', type=float, default=1.0, help='Reward for correct answer')
    parser.add_argument('--reward_wrong', type=float, default=-1.0, help='Reward for wrong answer')

    # --- Soft Thinking Params (传递给 SGLang) ---
    parser.add_argument("--enable_soft_thinking_global", action="store_true", help="Globally enable soft thinking capability in SGLang (RL decides per step)")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=15, help="Max top-k for soft thinking concept token")
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.0)
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256)

    # --- Generation Params (传递给 SGLang) ---
    parser.add_argument('--max_generated_tokens', type=int, default=2048, help='Max new tokens to generate for each problem') # 为训练设置一个合理的长度
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature (for discrete steps)')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling (for discrete steps)')
    # ... (其他采样参数, 如果需要的话)

    # --- Evaluation Args (用于获取 Reward) ---
    parser.add_argument('--use_llm_judge', action='store_true', help='Use LLM judge for evaluation (reward signal)')
    parser.add_argument('--judge_model_name', type=str, default="deepseek-chat", help='Judge LLM model name (e.g., deepseek-chat)')
    parser.add_argument('--api_base', type=str, default="https://api.deepseek.com", help='API base URL for the judge model')
    parser.add_argument('--api_key_env', type=str, default="DEEPSEEK_API_KEY", help='Environment variable name for the judge API key')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default="gsm8k", help='Name of the dataset (e.g., gsm8k) for evaluator mapping')
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # === 2. 初始化 ===
    # --- 设置随机种子 ---
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # --- 加载数据集 ---
    try:
        with open(args.dataset_path, "r") as f:
            train_data = json.load(f)
            logger.info(f"Loaded dataset from {args.dataset_path}, total samples: {len(train_data)}")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {args.dataset_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {args.dataset_path}")
        sys.exit(1)

    # --- 初始化 Tokenizer ---
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded from {tokenizer_path}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        sys.exit(1)

    # --- 临时获取 LLM 嵌入维度 (在初始化 SGLang 之前) ---
    # 这是一个临时的、开销较大的方法，但可以确保在创建 DQN 之前拿到维度
    # TODO: 优化为直接从 SGLang Engine 配置中获取
    try:
        logger.info("Temporarily loading model config to get embedding dim...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        embedding_dim = config.hidden_size
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"Invalid embedding_dim obtained: {embedding_dim}")
        logger.info(f"Detected LLM embedding dimension: {embedding_dim}")
        del config # 释放内存
    except Exception as e:
        logger.error(f"Failed to get embedding dimension from model config at {args.model_path}: {e}")
        sys.exit(1)

    # --- 初始化 DQN 模型 ---
    dqn_model = DQN(
        embedding_dim=embedding_dim,
        reduced_dim=args.reduced_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_lstm_layers=args.lstm_num_layers,
        replay_buffer_capacity=args.replay_buffer_capacity,
        error_buffer_capacity=args.error_buffer_capacity,
        error_buffer_sampling_prob=args.error_buffer_sampling_prob,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        state_max_len=args.state_max_len,
        include_prompt_in_state=args.include_prompt_in_state,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- 加载检查点 (如果指定) ---
    if args.load_checkpoint:
        dqn_model.load_checkpoint(args.load_checkpoint)

    # --- 初始化 SGLang 引擎 ---
    logger.info("Initializing SGLang Engine and passing DQN model...")
    sgl_engine = sgl.Engine(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        tp_size=args.num_gpus,
        mem_fraction_static=args.mem_fraction_static,
        log_level="info",
        trust_remote_code=True,
        random_seed=args.random_seed,
        sampling_backend=args.sampling_backend,
        # 传递 Soft Thinking 相关参数给 SGLang
        enable_soft_thinking=args.enable_soft_thinking_global,
        max_topk=args.max_topk,
        think_end_str=args.think_end_str,
        early_stopping_entropy_threshold=args.early_stopping_entropy_threshold,
        early_stopping_length_threshold=args.early_stopping_length_threshold,

        # ==========================================================
        # == BEGIN: 关键修改 - 传递 RL 模型实例 =====================
        # ==========================================================
        # rl_model=dqn_model
        # ==========================================================
        # == END: 关键修改 - 传递 RL 模型实例 =======================
        # ==========================================================
    )
    logger.info("SGLang Engine Initialized.")
    # --- 新增：命令 Scheduler 子进程初始化其本地的 RL 模型 ---
    logger.info("Sending RPC call to initialize RL model in subprocess...")
    sgl_engine.init_rl_model(
        # 我们必须传递所有 DQN.__init__ 需要的参数
        embedding_dim=embedding_dim,
        reduced_dim=args.reduced_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_lstm_layers=args.lstm_num_layers,
        action_dim=2, # (硬编码为 2)
        # 注意：子进程中的模型不需要 ReplayBuffer 或 Optimizer 参数
        # 它只需要 QNetwork 的结构参数
        state_max_len=args.state_max_len,
        include_prompt_in_state=args.include_prompt_in_state,
        device="cuda" # 子进程会在它自己的 GPU 上创建模型
    )
    logger.info("RPC init_rl_model complete.")
    # --- 新增结束 ---
    # --- 初始化评估器 (用于奖励) ---
    api_key = os.getenv(args.api_key_env)
    if args.use_llm_judge and not api_key:
        logger.warning(f"LLM Judge enabled but environment variable {args.api_key_env} not set. LLM Judge may fail.")

    # 假设 args.dataset 总是 "gsm8k" (根据 MATH_DATASETS 定义)
    try:
        matheval.set_client(api_base=args.api_base, api_key=api_key, model_name=args.judge_model_name)
        evaluator = matheval.evaluator_map.get(args.dataset) # args.dataset 应该是 "gsm8k"
        if evaluator is None:
            raise ValueError(f"No evaluator found for dataset: {args.dataset}")
        logger.info(f"Evaluator for {args.dataset} initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        sgl_engine.shutdown()
        sys.exit(1)

    # --- 创建检查点保存目录 ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # === 3. 主训练循环 ===
    logger.info("Starting training loop...")
    start_time = time.time()
    total_steps = args.total_train_steps
    data_index = 0 # 用于循环数据集
    losses = []
    accuracies = [] # 记录每个 log interval 的准确率
    correct_count_interval = 0
    total_count_interval = 0

    pbar = tqdm(range(dqn_model.steps_done, total_steps), desc="Training Steps")

    for current_step in pbar:
        # --- a. 选择问题 ---
        # (暂时禁用错题本采样，保持按顺序循环)
        if data_index >= len(train_data):
            data_index = 0 # 循环数据集
            logger.info("Dataset looped.")

        current_sample = train_data[data_index]
        data_index += 1

        try:
            current_prompt_text = current_sample["prompt"][0]["value"] # 假设格式
            ground_truth_answer = current_sample["final_answer"] # 假设格式
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Skipping sample {data_index-1} due to unexpected format.")
            continue

        # --- b. 准备 Prompt ---
        MATH_QUERY_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}"
        chat = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=current_prompt_text)}]
        prompt_for_sglang = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        # --- c. 与 SGLang 交互生成轨迹 ---
        # 【【【关键交互点 - 移除占位符，使用真实调用】】】
        try:
            # 定义基础采样参数 (不包括 RL 决策的参数)
            sampling_params = SamplingParams(
                max_new_tokens=args.max_generated_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                # ... 其他需要的非 RL 采样参数 ...
            )

            # 实际调用 SGLang 引擎
            # sgl_engine.generate 返回一个列表（如果 batch_size=1）或迭代器
            # 我们使用非流式调用，它应该返回一个列表
            output_list = sgl_engine.generate(
                prompt=prompt_for_sglang,
                sampling_params=dataclasses.asdict(sampling_params),
                stream=False
            )

            # 提取第一个（也是唯一一个）结果
            output = output_list[0]

            # --- d. 解包数据 ---
            generated_text = output["text"]
            meta_info = output["meta_info"]

            # 从 meta_info 提取历史记录
            # (需要确保 SGLang 修改正确，且 meta_info 中包含这些键)
            embedding_history = meta_info.get("embedding_history")
            action_history = meta_info.get("action_history")

            if embedding_history is None or action_history is None:
                logger.warning(f"Step {current_step}: Missing embedding_history or action_history in meta_info. Skipping step.")
                continue
            # --- 新增：获取 Prompt 长度并拆分嵌入 ---
            # SGLang 默认会在 meta_info 中返回 prompt_token_len
            prompt_token_len = meta_info.get("prompt_token_len")

            if prompt_token_len is None or prompt_token_len < 0:
                logger.warning(f"Step {current_step}: Invalid prompt_token_len ({prompt_token_len}) in meta_info. Skipping step.")
                continue

            # 验证历史记录的长度是否匹配
            # embedding_history 包含 prompt + reasoning
            # action_history 只包含 reasoning
            if len(embedding_history) != prompt_token_len + len(action_history):
                logger.warning(f"Step {current_step}: Mismatch lengths. Embeddings: {len(embedding_history)}, Prompt: {prompt_token_len}, Actions: {len(action_history)}. Skipping.")
                continue

            # 正确拆分
            prompt_embeds = embedding_history[:prompt_token_len]
            reasoning_embeds = embedding_history[prompt_token_len:]
            # --- 新增结束 ---
        except Exception as e:
            logger.error(f"Error during SGLang generation for step {current_step}: {e}", exc_info=True)
            continue # 跳过这个错误的步骤

        # --- e. 评估答案 & 计算奖励 ---
        final_result = False
        extracted_answer = "N/A" # 用于日志
        try:
            # 假设 evaluator.rule_judge 返回 (bool, str)
            final_result, extracted_answer = evaluator.rule_judge(generated_text, ground_truth_answer)
            if not final_result and args.use_llm_judge:
                # TODO: 考虑是否需要异步执行 LLM Judge
                final_result = evaluator.llm_judge(generated_text, ground_truth_answer, extracted_answer)
        except Exception as e:
            logger.error(f"Error during evaluation for step {current_step}: {e}", exc_info=True)
            # 评估失败，按错误处理
            final_result = False

        # 根据评估结果确定最终奖励
        final_reward = args.reward_correct if final_result else args.reward_wrong

        correct_count_interval += 1 if final_result else 0
        total_count_interval += 1

        # --- f. 存储经验 ---
        # TODO: 实现 prompt_embeddings 的获取和存储 (目前为空列表)
        trajectory = Trajectory(
            prompt_embeddings=prompt_embeds,           # 使用正确拆分的 prompt 嵌入
            reasoning_embeddings=reasoning_embeds,     # 使用正确拆分的 reasoning 嵌入
            actions=action_history,                    # 从 meta_info 获取
            reward=final_reward,
            final_result=final_result,
            start_reasoning_index=prompt_token_len     # 使用正确的索引
        )

        dqn_model.store_trajectory(trajectory)

        # --- g. 训练 DQN ---
        if current_step % args.train_interval_steps == 0 and current_step > 0:
            loss = dqn_model.train()
            if loss is not None:
                losses.append(loss)
                # logger.debug(f"Step {current_step}: Training Loss: {loss:.4f}")

        # --- h. 日志记录 ---
        if current_step % args.log_interval_steps == 0 and current_step > 0:
            avg_loss = sum(losses) / len(losses) if losses else 0
            accuracy = correct_count_interval / total_count_interval if total_count_interval > 0 else 0
            accuracies.append(accuracy)

            # 计算 epsilon (需要访问 dqn_model.steps_done)
            current_epsilon = dqn_model.epsilon_end + (dqn_model.epsilon_start - dqn_model.epsilon_end) * \
                              math.exp(-1. * dqn_model.steps_done / dqn_model.epsilon_decay)

            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{accuracy:.3f}",
                "Epsilon": f"{current_epsilon:.4f}",
                "Buffer": f"{len(dqn_model.memory)}",
            })
            logger.info(f"Step {current_step}/{total_steps} | Loss: {avg_loss:.4f} | Accuracy (last {args.log_interval_steps} steps): {accuracy:.3f} | Epsilon: {current_epsilon:.4f} | Buffer Size: {len(dqn_model.memory)}")

            # 重置 interval 计数器
            losses = []
            correct_count_interval = 0
            total_count_interval = 0
            start_time = time.time() # 重置计时器 (如果需要计算 steps/sec)


        # --- i. 保存检查点 ---
        if current_step % args.save_interval_steps == 0 and current_step > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"dqn_step_{current_step}.pt")
            dqn_model.save_checkpoint(checkpoint_path)

    # === 4. 训练结束 ===
    logger.info("Training finished.")
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "dqn_final.pt")
    dqn_model.save_checkpoint(final_checkpoint_path)

    # === 5. 清理 ===
    logger.info("Shutting down SGLang Engine...")
    sgl_engine.shutdown()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    # 确保 asyncio 循环已设置 (SGLang Engine 需要)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    main()
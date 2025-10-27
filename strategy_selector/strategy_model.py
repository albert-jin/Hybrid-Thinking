# strategy_selector/strategy_model.py

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from collections import namedtuple
from typing import Optional, Tuple, List

#  Replay Buffer
from .replay_buffer import ReplayBuffer, Trajectory, Transition

# Critic
class QNetwork(nn.Module):
    """
    DQN 网络 (Critic)，使用 降维层 + LSTM 处理嵌入序列。
    输出两个动作的 Q 值：离散生成 vs Soft Thinking 连续生成。
    设计为通用模型，需要根据基础 LLM 初始化正确的 embedding_dim。
    """
    def __init__(self,
                 embedding_dim: int,
                 reduced_dim: int = 512, # 降维后的维度
                 lstm_hidden_dim: int = 512,
                 num_lstm_layers: int = 1,
                 action_dim: int = 2):
        super(QNetwork, self).__init__()
        if embedding_dim <= 0 or reduced_dim <= 0 or lstm_hidden_dim <= 0:
            raise ValueError("embedding_dim, reduced_dim, 和 lstm_hidden_dim 都必须是正整数。")

        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.action_dim = action_dim

        self.fc_reduce_dim = nn.Linear(embedding_dim, reduced_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=reduced_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.fc_q_values = nn.Linear(lstm_hidden_dim, action_dim)

        print(f"[*] QNetwork Initialized:")
        print(f"    - Embedding Dim (Input): {embedding_dim}")
        print(f"    - Reduced Dim (To LSTM): {reduced_dim}")
        print(f"    - LSTM Hidden Dim:       {lstm_hidden_dim}")
        print(f"    - LSTM Layers:           {num_lstm_layers}")
        print(f"    - Action Dim (Output):   {action_dim}")

    def forward(self, state_sequence: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        batch_size, seq_len, _ = state_sequence.shape
        reduced_sequence = self.relu(self.fc_reduce_dim(state_sequence))
        lstm_out, last_hidden = self.lstm(reduced_sequence, hidden_state)
        last_time_step_out = lstm_out[:, -1, :]    # LSTM 最后一步的输出, 形状 [batch_size, lstm_hidden_dim]
        q_values = self.fc_q_values(last_time_step_out)  # 线性层将 [..., lstm_hidden_dim] 映射到 [..., action_dim=2]
        return q_values    #[Q1,Q2] 对应0和1的奖励

# --- DQN  类定义 ---
class DQN(nn.Module):
    """
    DQN 类，负责决策、学习和经验管理。
    """
    def __init__(self,
                 embedding_dim: int,
                 reduced_dim: int = 512, # QNetwork 的降维维度
                 lstm_hidden_dim: int = 512,
                 num_lstm_layers: int = 1,
                 action_dim: int = 2,
                 replay_buffer_capacity: int = 1000,
                 error_buffer_capacity: int = 500,
                 error_buffer_sampling_prob: float = 0.2,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 64,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 10000,
                 state_max_len: int = 35, # 新增：状态序列的最大长度
                 include_prompt_in_state: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super(DQN, self).__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.state_max_len = state_max_len # 存储状态最大长度
        self.device = torch.device(device)
        self.include_prompt_in_state = include_prompt_in_state ## <--- 存储超参数
        self.policy_net = QNetwork(embedding_dim, reduced_dim, lstm_hidden_dim, num_lstm_layers, action_dim).to(self.device)
        self.target_net = QNetwork(embedding_dim, reduced_dim, lstm_hidden_dim, num_lstm_layers, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # AdamW
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayBuffer(replay_buffer_capacity, error_buffer_capacity, error_buffer_sampling_prob)
        self.steps_done = 0
        # 打印参数部分
        print(f"[*] DQN Model Initialized on device: {self.device}")
        print(f"    - LLM Embedding Dim:          {embedding_dim}")
        print(f"    - State Max Length:         {state_max_len}")
        print(f"    - Include Prompt in State:  {self.include_prompt_in_state}")
        print(f"    - Learning Rate:            {learning_rate}")
        print(f"    - Gamma (Discount Factor):    {gamma}")
        print(f"    - Tau (Target Update):      {tau}")
        print(f"    - Batch Size:               {batch_size}")
        print(f"    - Epsilon Start / End / Decay:{epsilon_start} / {epsilon_end} / {epsilon_decay}")
        print(f"    - Replay Buffer Capacity:   {replay_buffer_capacity} (Main) / {error_buffer_capacity} (Error)")
        print(f"    - Error Buffer Sampling %:  {error_buffer_sampling_prob * 100:.1f}%")
        print(f"    - Policy Network & Target Network created (using QNetwork).")
        print(f"    - Optimizer: AdamW")

    def select_action(self, state_sequence: List[torch.Tensor], is_training: bool = True) -> int:
        """
        根据当前状态 (嵌入列表) 和 Epsilon-greedy 策略选择动作。

        Args:
            state_sequence (List[torch.Tensor]): 当前状态的嵌入序列 (Tensor 列表)。
                                                 长度 <= state_max_len。
            is_training (bool): 是否处于训练模式。

        Returns:
            int: 选择的动作 (0 或 1)。
        """
        sample = random.random()
        # epsilon 探索率
        # self.epsilon_start (起始 Epsilon):  推荐默认值: 0.9 或 1.0
        # self.epsilon_end (结束 Epsilon) 推荐默认值: 0.05 或 0.1
        # self.epsilon_decay (Epsilon 衰减速率/步数)  就是我们设置的最大步数 10000
        # self.steps_done (已完成步数) 记录了 select_action 方法在训练模式下被调用的次数。
        # 根据步数进行指数衰减
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)

        if is_training:
            self.steps_done += 1  # 更新步数，持续衰减
        # 分支概率为 epsilon
        if is_training and sample < epsilon:
            return random.randrange(self.action_dim) # 随机返回 0 或 1
        # 分支概率为 1 - epsilon
        else:
            with torch.no_grad():
                # --- 准备输入 ---
                # 填充状态序列
                padded_state = self._pad_sequence(state_sequence, self.state_max_len)
                if padded_state is None: # 如果序列为空
                    # 可以返回随机动作或默认动作，这里返回随机
                    print("[Warning] select_action received empty state sequence, returning random action.")
                    return random.randrange(self.action_dim) # 随机返回 0 或 1

                # 增加 batch 维度（适应lstm输入） 并移动到设备
                state_tensor = padded_state.unsqueeze(0).to(self.device) # 形状 (1, max_len, embed_dim)

                # --- 获取 Q 值并选择动作 ---
                # policy_net Qnet 根据LSTM层 forword输出q_values tensor
                q_values = self.policy_net(state_tensor) # 形状 (1, action_dim)
                action = q_values.max(1)[1].item() # q_values.max(1)返回 (values, indices)形状的tensor q_values.max(1)[1]则是[0]或[1]
                return action

    def store_trajectory(self, trajectory: Trajectory):
        """将完整的轨迹存入 Replay Buffer"""
        # 可以考虑在这里做一些预处理，比如将 Tensor 移到 CPU 存储以节省 GPU 内存
        # trajectory_cpu = Trajectory(
        #     prompt_embeddings=[e.cpu() for e in trajectory.prompt_embeddings],
        #     reasoning_embeddings=[e.cpu() for e in trajectory.reasoning_embeddings],
        #     actions=trajectory.actions,
        #     reward=trajectory.reward,
        #     final_result=trajectory.final_result,
        #     start_reasoning_index=trajectory.start_reasoning_index
        # )
        # self.memory.add(trajectory_cpu)
        self.memory.add(trajectory) # 暂时直接存储

    def train(self) -> Optional[float]:
        """
        执行一次 DQN 训练步骤。返回损失值或 None (如果未训练)。
        """
        if len(self.memory) < self.batch_size:
            return None

        # --- 采样并处理 Transitions（暂定） ---
        transitions = self._sample_and_process_transitions()
        if not transitions:
            return None

        # 将一批 Transition 解构成独立的批次
        # batch.state/next_state 已经是填充好的 Tensor
        batch = Transition(*zip(*transitions))

        # 过滤掉 next_state 为 None 的样本 (对应终止状态)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                      device=self.device, dtype=torch.bool)

        # 准备 next_states 张量 (只包含非终止状态的)
        non_final_next_states = None
        if non_final_mask.any():
            next_state_list = [s for s in batch.next_state if s is not None]
            if next_state_list: # 再次检查以防万一
                non_final_next_states = torch.stack(next_state_list).to(self.device)
                # 验证形状
                if non_final_next_states.shape[1] != self.state_max_len or non_final_next_states.shape[2] != self.embedding_dim:
                    print(f"[Error] Unexpected shape for non_final_next_states: {non_final_next_states.shape}")
                    # 可能需要进一步调试 _pad_sequence 或 _sample_and_process_transitions
                    return None # 暂时跳过此 batch
            else:
                non_final_mask.fill_(False) # 如果 next_state_list 为空，强制 mask 为 False


        # 准备 state, action, reward 张量
        # 确保 batch.state 中的 None (如果 _pad_sequence 返回 None) 被处理
        valid_states = [s for s in batch.state if s is not None]
        if len(valid_states) != self.batch_size:
            # 这通常不应该发生，除非 _pad_sequence 对初始状态返回了 None
            print(f"[Error] Number of valid states ({len(valid_states)}) does not match batch size ({self.batch_size}). Skipping batch.")
            return None
        state_batch = torch.stack(valid_states).to(self.device)
        # 验证形状
        if state_batch.shape[1] != self.state_max_len or state_batch.shape[2] != self.embedding_dim:
            print(f"[Error] Unexpected shape for state_batch: {state_batch.shape}")
            return None

        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

        # --- 计算 Q 值 ---
        # 获取当前状态下，实际采取动作的 Q 值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # --- 计算目标 Q 值 ---
        next_state_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)
        if non_final_next_states is not None: # 确保张量存在
            with torch.no_grad():
                # 使用 target_net 预测下一状态的最大 Q 值
                next_q_values = self.target_net(non_final_next_states) # (num_non_final, action_dim)
                next_state_values[non_final_mask] = next_q_values.max(1)[0] # .max 返回 (values, indices)

        # 计算 TD Target: R_t + gamma * max_a Q_target(S_{t+1}, a)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # --- 计算损失 ---
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # --- 优化模型 ---
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # 使用 clip_grad_norm_ 更常用
        self.optimizer.step()

        # --- 软更新目标网络 ---
        self._soft_update_target_network()

        return loss.item() # 返回损失值

    def _soft_update_target_network(self):
        """执行目标网络的软更新"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    # 这里的实现还是不完整  可能需要调整
    # 目前实现逻辑：
    # 从 memory 中采样一批 Trajectory 对象，转换成DQN训练所需的 Transition 元组列表 （暂定）(state, action, reward, next_state, done)
    # TODO    这里需要考虑，如果这里加入了include_prompt_in_state为true 即加入了提示之后
    #  就会出现 提示一直是一样的，然后每一次的采样的初始上下文总是存在
    #  [提示embeddings，+初始推理tokens的embedding的情况]，会对训练产生什么影响？？？？
    def _sample_and_process_transitions(self) -> Optional[List[Tuple]]:
        """
        (内部辅助函数) 从 Replay Buffer 采样轨迹，并将其处理成 DQN 需要的 Transition 元组列表。
        根据 include_prompt_in_state 参数决定是否包含 prompt 嵌入。
        """
        sampled_trajectories = self.memory.sample(self.batch_size)
        if not sampled_trajectories:
            return None

        transitions = []

        for traj in sampled_trajectories:
            # 动作列表
            actions = traj.actions
            reward = traj.reward
            # num_steps 已经推理了多少步 = 已经记录了多少个动作 所以可以直接取
            num_steps = len(actions)

            if num_steps == 0: continue

            # --- 根据超参数决定 full_embeddings ---
            if self.include_prompt_in_state:
                # 包含 prompt: 拼接 prompt 和 reasoning 嵌入
                # 假设 traj.prompt_embeddings 和 traj.reasoning_embeddings 都是 List[Tensor]
                # 需要确保它们内部的 Tensor 可以在 CPU/GPU 间正确移动
                # TODO: 验证 prompt_embeddings 是否存在且非空
                prompt_embeds = traj.prompt_embeddings if traj.prompt_embeddings else []
                reasoning_embeds = traj.reasoning_embeddings if traj.reasoning_embeddings else []
                full_embeddings = prompt_embeds + reasoning_embeds # 简单列表拼接
                start_offset = len(prompt_embeds) # 推理动作对应的嵌入从 prompt 之后开始
            else:
                # 不包含 prompt: 只使用 reasoning 嵌入
                full_embeddings = traj.reasoning_embeddings if traj.reasoning_embeddings else []
                start_offset = 0 # 推理动作对应的嵌入从列表开头开始

            # 如果 full_embeddings 为空，则跳过此轨迹
            if not full_embeddings: continue

            # 遍历推理链的每一步 (对应 actions 列表) 来创建 Transition
            for t in range(num_steps):
                # 状态 s_t: 包含到时间步 t 为止的嵌入 (相对于 full_embeddings)
                # 动作 a_t 发生在状态 s_t 之后，该状态由之前的嵌入决定
                # 嵌入应该结束的位置
                state_end_idx_in_full = start_offset + t
                # 嵌入应该开始的位置或0（不够35的时候）
                state_start_idx_in_full = max(0, state_end_idx_in_full - self.state_max_len)
                # 截取[当前最近的35个tokens对应的embedding] ps：如果狗的话
                current_state_embeds_list = full_embeddings[state_start_idx_in_full : state_end_idx_in_full]

                # 动作 a_t
                current_action = actions[t]

                # 下一状态 s_{t+1}: 包含到时间步 t+1 为止的嵌入 (相对于 full_embeddings)
                # 同样的方法构造next的数据
                next_state_end_idx_in_full = start_offset + t + 1
                next_state_start_idx_in_full = max(0, next_state_end_idx_in_full - self.state_max_len)
                next_state_embeds_list = full_embeddings[next_state_start_idx_in_full : next_state_end_idx_in_full]

                # 奖励 r_t 和是否结束 done
                # 最后一步才是1
                step_reward = reward if (t == num_steps - 1) else 0.0
                done = (t == num_steps - 1)

                # 填充/截断序列 在_pad_sequence中实现，为了保持35的长度
                current_state_padded = self._pad_sequence(current_state_embeds_list, self.state_max_len)
                # 最后一步就不处理了
                next_state_padded = self._pad_sequence(next_state_embeds_list, self.state_max_len) if not done else None

                # 添加 Transition
                if current_state_padded is not None:
                    transitions.append((current_state_padded, current_action, step_reward, next_state_padded, done))
        # 返回最后的组装数据 （bachsize * （推理链长度））
        return transitions if transitions else None
    # 这里是为了处理不满足 最大序列（我们设置的默认是35）的情况   短 向前补0 长 保留最近35  保证保留最近的信息     保证滑动窗口（max_len）的长度
    def _pad_sequence(self, sequence: List[torch.Tensor], max_len: int) -> Optional[torch.Tensor]:
        """
        (内部辅助函数) 将嵌入序列(Tensor列表)填充或截断到固定长度 max_len。
        在序列前面填充零向量。确保返回 Tensor 在正确的设备上。
        如果输入序列为空，返回 None。
        """
        current_len = len(sequence)

        if current_len == 0:
            return None

        # 确保所有张量都在 CPU 上（如果它们来自 Replay Buffer）
        # 或者确保它们与 agent 的 device 一致
        # 这里假设输入的 sequence 已经在正确的设备上或可以移动
        try:
            # 尝试直接 stack，如果失败说明可能在不同设备或类型不一致
            test_stack = torch.stack(sequence)
            first_embedding = test_stack[0]
        except Exception as e:
            print(f"[Error] Failed to stack embeddings in _pad_sequence: {e}")
            print(f"    - Number of embeddings: {current_len}")
            # 打印每个 embedding 的形状、设备和类型
            # for i, emb in enumerate(sequence):
            #     if isinstance(emb, torch.Tensor):
            #         print(f"    - Embedding {i}: shape={emb.shape}, device={emb.device}, dtype={emb.dtype}")
            #     else:
            #         print(f"    - Embedding {i}: Not a tensor, type={type(emb)}")
            return None # 或者返回全零 Tensor，避免训练中断

        embed_dim = first_embedding.shape[-1]
        dtype = first_embedding.dtype
        # 使用 agent 配置的 device
        device = self.device

        if current_len >= max_len:
            # 截取最后的 max_len 个
            # 使用 test_stack 避免再次 stack
            padded_sequence = test_stack[-max_len:].to(device=device, dtype=dtype)
        else:
            # 计算填充数量
            padding_len = max_len - current_len
            # 创建零向量填充，确保在正确的设备和类型
            padding = torch.zeros((padding_len, embed_dim), dtype=dtype, device=device)
            # 拼接，使用 test_stack
            sequence_tensor = test_stack.to(device=device, dtype=dtype)
            padded_sequence = torch.cat((padding, sequence_tensor), dim=0)

        # 最终形状应该是 (max_len, embedding_dim)
        return padded_sequence


    def save_checkpoint(self, filepath: str):
        """保存模型检查点"""
        # 确保模型在 CPU 上保存，以增加兼容性
        self.policy_net.cpu()
        self.target_net.cpu()
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, filepath)
        # 将模型移回原来的设备
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        print(f"[*] Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        try:
            # 加载到当前 agent 配置的 device
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']
            self.target_net.eval()
            # 确保模型在正确的设备上
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            print(f"[*] Checkpoint loaded from {filepath}")
        except FileNotFoundError:
            print(f"[!] Checkpoint file not found at {filepath}. Starting from scratch.")
        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}. Starting from scratch.")

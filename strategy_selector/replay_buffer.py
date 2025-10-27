# strategy_selector/replay_buffer.py

import random
from collections import deque, namedtuple
from typing import List, Tuple
import torch

# 定义存储在缓冲区中的单个经验单元（轨迹片段或完整轨迹）的结构
# 我们可以根据需要调整这个结构，目前包含：
# state: 当前状态 (嵌入序列)
# action: 采取的动作 (0 或 1)
# reward: 获得的奖励
# next_state: 下一个状态 (嵌入序列)
# done: 是否是轨迹的结束 (True/False)
# 实际应用中可能需要存储完整的轨迹信息，包括 prompt 嵌入、推理链嵌入、动作序列等
# 这里为了 DQN 的标准实现，先定义为单个 transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# 我们可以为完整的轨迹定义一个更复杂的结构
Trajectory = namedtuple('Trajectory', [
    'prompt_embeddings',      # List[torch.Tensor] 或 torch.Tensor: 问题 Prompt 部分的原始高维嵌入序列
    'reasoning_embeddings',   # List[torch.Tensor]: LLM 推理链（思考过程）部分的原始高维嵌入序列
    'actions',                # List[int]: 在推理链的每一步，RL Agent 选择的动作 (0 或 1) 序列
    'reward',                 # float: 整个问题解答完成后获得的最终奖励
    'final_result',           # bool: 最终答案是否正确 (True/False)
    'start_reasoning_index'   # int: 推理链嵌入在完整嵌入序列中的起始索引 (用于区分 prompt 和 reasoning)
    # 可以考虑增加其他字段，例如：
    # 'question_text': str, # 原始问题文本，方便调试和错题本回顾
    # 'generated_text': str # LLM 生成的完整文本
])


class ReplayBuffer:
    """
    经验回放缓冲区，包含一个主缓冲区和一个用于存储错误经验的“错题本”。
    """
    def __init__(self, buffer1_max_len: int = 1000, buffer2_max_len: int = 500, error_buffer_sampling_prob: float = 0.2):
        """
        初始化 ReplayBuffer。

        Args:
            buffer1_max_len (int): 主缓冲区 (buffer1) 的最大长度。
            buffer2_max_len (int): 错误经验缓冲区 (buffer2, "错题本") 的最大长度。
            error_buffer_sampling_prob (float): 从错误缓冲区 (buffer2) 采样的概率 (例如 0.2 代表 20%)。
        """
        self.buffer1 = deque(maxlen=buffer1_max_len) # 存储正确或一般经验
        self.buffer2 = deque(maxlen=buffer2_max_len) # 存储错误经验 (“错题本”)
        self.error_buffer_sampling_prob = error_buffer_sampling_prob

        print(f"[*] ReplayBuffer Initialized:")
        print(f"    - Main Buffer Max Length:   {buffer1_max_len}")
        print(f"    - Error Buffer Max Length:  {buffer2_max_len}")
        print(f"    - Error Buffer Sampling %:  {error_buffer_sampling_prob * 100:.1f}%")

    def add(self, trajectory: Trajectory):
        """
        向缓冲区添加一条完整的轨迹经验。

        Args:
            trajectory (Trajectory): 要存储的完整轨迹数据。
                                     需要包含 final_result (True/False) 来判断存入哪个缓冲区。
        """
        if trajectory.final_result: # 如果最终结果正确
            self.buffer1.append(trajectory)
        else: # 如果最终结果错误
            self.buffer2.append(trajectory)


    def sample(self, batch_size: int) -> List[Trajectory]:
        """
        从缓冲区中采样一批经验。

        Args:
            batch_size (int): 要采样的批量大小。

        Returns:
            List[Trajectory]: 采样得到的经验列表。如果缓冲区大小不足，返回空列表。
        """
        total_size = len(self.buffer1) + len(self.buffer2)

        # 只有当缓冲区总大小大于等于 batch_size 时才进行采样
        if total_size < batch_size:
            return []

        sampled_trajectories = []
        num_sampled_from_buffer2 = 0

        # 优先确保 buffer2 (错题本) 中有样本时会被采样到
        if len(self.buffer2) > 0:
            # 根据概率决定从 buffer2 采样的数量，但至少采样一个（如果buffer2非空）
            num_to_sample_from_buffer2 = max(1, int(batch_size * self.error_buffer_sampling_prob))
            # 不能超过 buffer2 的实际大小
            num_to_sample_from_buffer2 = min(num_to_sample_from_buffer2, len(self.buffer2))

            sampled_buffer2 = random.sample(self.buffer2, num_to_sample_from_buffer2)
            sampled_trajectories.extend(sampled_buffer2)
            num_sampled_from_buffer2 = num_to_sample_from_buffer2

        # 从 buffer1 采样剩余的数量
        num_to_sample_from_buffer1 = batch_size - num_sampled_from_buffer2
        if num_to_sample_from_buffer1 > 0 and len(self.buffer1) > 0:
             # 不能超过 buffer1 的实际大小
            num_to_sample_from_buffer1 = min(num_to_sample_from_buffer1, len(self.buffer1))
            sampled_buffer1 = random.sample(self.buffer1, num_to_sample_from_buffer1)
            sampled_trajectories.extend(sampled_buffer1)

        # 如果因为某个buffer为空导致采样数量不足batch_size，需要补充
        current_sampled_count = len(sampled_trajectories)
        if current_sampled_count < batch_size:
            remaining_needed = batch_size - current_sampled_count
            # 尝试从另一个非空的 buffer 补充
            if len(self.buffer1) > num_to_sample_from_buffer1: # 如果 buffer1 还有剩余
                 additional_needed = min(remaining_needed, len(self.buffer1) - num_to_sample_from_buffer1)
                 additional_samples = random.sample([t for t in self.buffer1 if t not in sampled_trajectories], additional_needed)
                 sampled_trajectories.extend(additional_samples)
            elif len(self.buffer2) > num_sampled_from_buffer2: # 如果 buffer2 还有剩余
                 additional_needed = min(remaining_needed, len(self.buffer2) - num_sampled_from_buffer2)
                 additional_samples = random.sample([t for t in self.buffer2 if t not in sampled_trajectories], additional_needed)
                 sampled_trajectories.extend(additional_samples)


        random.shuffle(sampled_trajectories) # 打乱顺序
        return sampled_trajectories

    def __len__(self) -> int:
        """
        返回缓冲区中存储的总经验数量。
        """
        return len(self.buffer1) + len(self.buffer2)

# 示例用法:
if __name__ == '__main__':
    # 假设 Trajectory 结构如上定义
    buffer = ReplayBuffer(buffer1_max_len=10, buffer2_max_len=5, error_buffer_sampling_prob=0.3)

    # 添加一些虚拟轨迹数据
    for i in range(15):
        is_correct = random.choice([True, False, True]) # 假设 True 概率更高
        traj = Trajectory(
            prompt_embeddings=f"prompt_{i}", # 用字符串代替嵌入
            reasoning_embeddings=f"reasoning_{i}",
            actions=[random.randint(0, 1) for _ in range(5)], # 虚拟动作序列
            reward=1 if is_correct else -1, # 虚拟奖励
            final_result=is_correct
        )
        buffer.add(traj)
        print(f"Added trajectory {i}. Buffer1 size: {len(buffer.buffer1)}, Buffer2 size: {len(buffer.buffer2)}, Total: {len(buffer)}")

    # 采样
    batch_size = 4
    if len(buffer) >= batch_size:
        sample_batch = buffer.sample(batch_size)
        print(f"\nSampled batch of size {len(sample_batch)}:")
        for traj in sample_batch:
            print(f"  - Final Result: {traj.final_result}, Reward: {traj.reward}")

        # 验证采样比例是否大致符合预期
        errors_in_sample = sum(1 for t in sample_batch if not t.final_result)
        print(f"  - Errors in sample: {errors_in_sample} (Expected around {max(1, int(batch_size * buffer.error_buffer_sampling_prob))} if buffer2 has enough samples)")
    else:
        print(f"\nBuffer size ({len(buffer)}) is less than batch size ({batch_size}). No sampling.")
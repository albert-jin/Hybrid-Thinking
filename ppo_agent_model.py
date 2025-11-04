"""
ppo_agent_model.py

定义 PPO 控制器 (Agent) 的模型架构。
这是一个 Actor-Critic (演员-评论家) 模型，用于决策 "Soft" vs "Hard"
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import math # <--- 新增：导入 math 库以使用 log (ln)

# --- 1. 共享主干 (Shared Backbone) ---
# 负责将高维的 LLM 状态 (H_t) 和 Logits 特征 (L_t)
# 压缩为低维的"状态摘要 (State Summary)"

class SharedBackbone(nn.Module):
    def __init__(self, hidden_state_dim: int, logits_feature_dim: int, summary_dim: int):
        """
        初始化共享主干。

        参数:
        hidden_state_dim (int): LLM 隐藏状态的维度 (例如 4096)，动态传入。
        logits_feature_dim (int): Logits 特征的维度 (例如 Top-K=10)。
        summary_dim (int): 我们希望压缩成的"状态摘要"维度 (例如 128)。
        """
        super().__init__()

        # 核心层：将高维的 LLM 隐藏状态 "投影" (Projection)
        # 到一个更小、更易于处理的维度
        self.projection_layer = nn.Linear(hidden_state_dim, summary_dim)

        # 假设 logits 特征维度已经很小，我们不再压缩
        # 最终的"状态摘要"维度将是 summary_dim + logits_feature_dim

    def forward(self, H_t: torch.Tensor, L_t: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
        H_t (torch.Tensor): LLM 隐藏状态张量, shape [Batch, hidden_state_dim]
        L_t (torch.Tensor): Logits 特征张量, shape [Batch, logits_feature_dim]

        返回:
        torch.Tensor: 状态摘要张量, shape [Batch, summary_dim + logits_feature_dim]
        """

        # 1. 压缩 LLM 隐藏状态
        #    [B, 4096] -> [B, 128]
        H_proj = self.projection_layer(H_t)
        H_proj = torch.relu(H_proj) # 添加非线性激活

        # 2. 拼接
        #    [B, 128] + [B, 10] -> [B, 138]
        state_summary = torch.cat([H_proj, L_t.to(H_proj.dtype)], dim=-1)

        return state_summary

# --- 2. Actor和 Critic

class ActorCriticAgent(nn.Module):
    """
    PPO Actor-Critic Agent (控制器)

    该模型接收 LLM 的 (H_t, L_t) 作为状态，并并行输出：
    1. Actor: 动作 {Soft, Hard} 的概率
    2. Critic: 当前状态的 V 值 (预期回报)
    """

    def __init__(self,
                 hidden_state_dim: int,     # 必须! 从 config.hidden_size 动态传入
                 logits_feature_dim: int = 10,    # 假设我们用 Top-10 概率
                 projection_dim: int = 128,     # "投影" 后的维度
                 mlp_hidden_dim: int = 64):     # 决策MLP的隐藏维度
        """
        初始化 Actor-Critic Agent。

        参数:
        hidden_state_dim (int):     LLM 的隐藏状态维度 (config.hidden_size)。
        logits_feature_dim (int):   Logits 特征的维度 (例如 K=10)。
        projection_dim (int):       投影层输出的维度。
        mlp_hidden_dim (int):       Actor 和 Critic 头内部MLP的隐藏维度。
        """
        super().__init__()
        self.logits_feature_dim = logits_feature_dim
        # 动作空间是固定的：{0: "Soft", 1: "Hard"}
        self.action_dim = 2

        # 最终的"状态摘要"维度
        summary_input_dim = projection_dim + logits_feature_dim

        # --- 共享主干 (Backbone) ---
        self.backbone = SharedBackbone(
            hidden_state_dim,
            logits_feature_dim,
            projection_dim
        )

        # --- 演员头 (Actor Head) ---
        # 负责"决策"：输出 Soft/Hard 的概率
        self.actor_head = nn.Sequential(
            nn.Linear(summary_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.action_dim) # 输出 2 个 Logits
        )

        # --- 评论家头 (Critic Head) ---
        # 负责"估价"：输出 V 值
        self.critic_head = nn.Sequential(
            nn.Linear(summary_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1) # 输出 1 个 V-value
        )

        # <--- 新增：初始化偏置（Prior）为 90% Soft ---
        # 目标: Softmax([logit_soft, logit_hard]) = [0.9, 0.1]
        # 这需要 logit_soft - logit_hard = ln(0.9 / 0.1) = ln(9)
        initial_soft_bias = math.log(9.0) # approx 2.197

        # 1. 初始化 Actor Head 的最后一层 (索引 2)
        final_actor_layer = self.actor_head[2]
        # 将权重初始化为 0，使初始决策仅受偏置影响
        torch.nn.init.constant_(final_actor_layer.weight, 0.0)
        # 设置偏置：[logit_soft, logit_hard]
        torch.nn.init.constant_(final_actor_layer.bias, 0.0)
        # 我们假设动作 0 是 "Soft"
        final_actor_layer.bias.data[0] = initial_soft_bias

        # 2. (推荐) 初始化 Critic Head 的最后一层 (索引 2)
        # 使 Agent 初始时对 V-value 的预测为 0 (中立)
        final_critic_layer = self.critic_head[2]
        torch.nn.init.constant_(final_critic_layer.weight, 0.0)
        torch.nn.init.constant_(final_critic_layer.bias, 0.0)
        # <--- 新增结束 ---

    def get_action_and_value(self, H_t: torch.Tensor, L_t: torch.Tensor,
                             action: torch.Tensor = None):
        """
        PPO 训练循环的核心：
        根据状态 s_t = (H_t, L_t)
        1. (Actor) 采样一个动作
        2. (Critic) 估算 V 值
        3. (Actor) 计算所选动作的 log_prob (用于梯度更新)

        参数:
        H_t (torch.Tensor): LLM 隐藏状态张量, shape [Batch, hidden_state_dim]
        L_t (torch.Tensor): Logits 特征张量, shape [Batch, logits_feature_dim]
        action (torch.Tensor, optional): 如果提供了动作，则计算该动作的 log_prob；
                                         如果为 None (Rollout阶段)，则采样一个新动作。

        返回:
        tuple: (
            action (torch.Tensor): 采样/提供的动作, shape [Batch]
            log_prob (torch.Tensor): 所选动作的 log-probability, shape [Batch]
            entropy (torch.Tensor): 策略分布的熵 (用于PPO), shape [Batch]
            v_value (torch.Tensor): 状态的 V 值, shape [Batch]
        )
        """

        # 1. 跑共享主干，获取"状态摘要"
        # s_t = [B, 138]
        state_summary = self.backbone(H_t, L_t)

        # 2. 跑"演员头"，获取动作 Logits
        # action_logits = [B, 2] (例如 [0.2, -0.5])
        action_logits = self.actor_head(state_summary)

        # 3. 跑"评论家头"，获取 V 值
        # v_value = [B, 1] (例如 [0.7])
        v_value = self.critic_head(state_summary)

        # 4. 从 Logits 创建一个"概率分布"
        probs = Categorical(logits=action_logits)

        if action is None:
            # 在"Rollout" (数据收集) 阶段：我们需要采样一个新动作
            action = probs.sample()

        # 5. 计算所选动作的 log_prob 和分布的熵
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, v_value.squeeze(-1)

    def get_value(self, H_t: torch.Tensor, L_t: torch.Tensor) -> torch.Tensor:
        """
        PPO 学习阶段的辅助函数：
        只需要"评论家"对某个状态进行"估价"。

        参数:
        H_t (torch.Tensor): LLM 隐藏状态张量, shape [Batch, hidden_state_dim]
        L_t (torch.Tensor): Logits 特征张量, shape [Batch, logits_feature_dim]

        返回:
        torch.Tensor: 状态的 V 值, shape [Batch]
        """
        state_summary = self.backbone(H_t, L_t)
        v_value = self.critic_head(state_summary)
        return v_value.squeeze(-1)
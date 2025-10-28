import logging
from typing import List,Optional

import torch
import torch.distributed as dist
from torch import nn
from sglang.srt.distributed import get_tensor_model_parallel_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda
import torch.nn.functional as F
# ===============================================
# == BEGIN: 新增 RL DQN 模型导入 ===============
# ===============================================
try:
    # 尝试使用绝对路径导入 (假设执行脚本时 Python 解释器能找到 strategy_selector)
    from strategy_selector.strategy_model import DQN
    print("[INFO] Successfully imported DQN from strategy_selector.strategy_model") # 添加打印以确认导入成功
except ImportError as e:
    # 如果绝对导入失败（例如直接运行此文件或路径问题），打印警告并使用占位符
    print(f"[WARNING] Failed to import DQN via absolute path: {e}")
    print("[WARNING] Using placeholder DQN class. Ensure PYTHONPATH is set correctly or run from project root.")
    class DQN: # 定义一个占位符类，避免后续代码报错
        def __init__(self, *args, **kwargs):
            print("[WARNING] Initializing placeholder DQN!")
        def select_action(self, state, is_training=True):
            print("[WARNING] Placeholder DQN returning default action 0")
            return 0 # 默认返回离散动作
    dqn_imported_ok = False
else:
    dqn_imported_ok = True


# --- 全局 DQN 模型实例 ---
# 假设 dqn_model 实例将由 SGLang 引擎或更高层代码在初始化时设置
# 这个全局变量允许我们在 sampler.forward 中访问 DQN 模型
global dqn_model
dqn_model = None
# ===============================================
# == END: 新增 RL DQN 模型导入 ==================
# ===============================================
if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")


# --- RL Modification Start ---
# ==========================================================
# == BEGIN: 移除占位符函数 ================================
# ==========================================================
# def get_current_states_for_batch(sampling_info: SamplingBatchInfo) -> List[List[torch.Tensor]]:
#     """ (此函数已被移除，状态将从 sampling_info.rl_states 中直接获取) """
#     logger.warning("get_current_states_for_batch not implemented! Returning dummy states.")
#     batch_size = len(sampling_info.req_pool)
#     dummy_state = [torch.randn(768)]
#     return [dummy_state for _ in range(batch_size)]
# ==========================================================
# == END: 移除占位符函数 ==================================
# ==========================================================
# --- RL Modification End ---
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tensor_model_parallel_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        # ==========
        # begin of soft thinking
        # ==========
        enable_soft_thinking: bool = False,
        add_noise_dirichlet: bool = False,
        add_noise_gumbel_softmax: bool = False,
        # ==========
        # end of soft thinking
        # ==========
    )-> Tuple[torch.Tensor, Optional[List[int]]]: # <-- 修改返回值类型:
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        """
        运行采样器，计算 logprobs，并根据 RL 决策（如果启用）选择生成模式。

        Returns:
            Tuple[torch.Tensor, Optional[List[int]]]:
                - batch_next_token_ids: 批次中每个请求的下一个 token ID。
                - current_batch_actions: 批次中每个请求对应的 RL 动作列表 (0 或 1)，
                                         如果未执行 RL 决策则为 None。
        """
        logits = logits_output.next_token_logits
        # ==========
        # begin of soft thinking
        # ==========
        probs_clone = None
        # ==========
        # end of soft thinking
        # ==========
        # --- RL Modification Start ---
        # 初始化动作用于存储 RL 决策结果
        current_batch_actions: Optional[List[int]] = None
        # --- RL Modification End ---
        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            # ==========
            # begin of soft thinking
            # ==========
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            batch_next_token_ids = torch.argmax(logits, -1)

            # --- RL Modification Start ---
            # 贪心模式下，我们假设 RL 默认选择离散动作 (0)
            # 或者可以根据需要添加 RL 决策逻辑
            # 为了保持返回值结构一致，创建一个全 0 的动作列表
            current_batch_actions = [0] * logits.shape[0]
            # --- RL Modification End ---
            if enable_soft_thinking:
                # logits_output.topk_probs, logits_output.topk_indices
                # logits.div_(sampling_info.temperatures)
                # logits[:] = torch.softmax(logits, dim=-1)
                # probs = logits
                # del logits
                # # determine how many top-k to keep (at least 1)
                # # 只用argmax，不用topk以提升速度
                # max_k = max(1, sampling_info.max_topk if sampling_info.max_topk is not None else 1)
                # logits_output.topk_probs = torch.zeros(probs.shape[0], max_k, dtype=probs.dtype, device=probs.device)
                # logits_output.topk_indices = torch.zeros(probs.shape[0], max_k, dtype=torch.long, device=probs.device)

                # # 取对应位置的概率，其余为0
                # logits_output.topk_probs[:, 0] = torch.gather(probs, 1, batch_next_token_ids.unsqueeze(1)).squeeze(1)
                logits_output.topk_probs[:, 0] = 1
                logits_output.topk_indices[:, 0] = batch_next_token_ids
            # ==========
            # end of soft thinking
            # ==========
            else:
                pass
        else:
            # --- 非贪心采样分支 ---
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits # 注意：这里修改了 logits 本身
            # del logits # 现在 probs 就是修改后的 logits

            # --- RL Modification Start ---
            # ========== RL 决策步骤 ==========
            global dqn_model, dqn_imported_ok # 访问全局 DQN 实例
            current_batch_actions = [] # 重置/初始化

            # 1. 检查 DQN 模型是否已导入并初始化
            if dqn_model is not None and dqn_imported_ok:

                # 2. 从 sampling_info 中获取状态  # <--- 修改
                if hasattr(sampling_info, "rl_states") and sampling_info.rl_states is not None:
                    current_batch_states = sampling_info.rl_states # <--- 修改

                    # 确保状态数量与批次大小一致
                    if len(current_batch_states) == probs.shape[0]:
                        try:
                            # 3. 调用 DQN 模型进行决策 (逐个请求)
                            for state_list in current_batch_states:
                                # 假设 dqn_model.select_action 接收 List[Tensor]
                                action = dqn_model.select_action(state_list, is_training=True) # 假设训练模式
                                current_batch_actions.append(action)
                        except Exception as e:
                            logger.error(f"Error during RL action selection: {e}", exc_info=True)
                            current_batch_actions = [0] * probs.shape[0] # 决策异常，默认离散
                    else:
                        logger.error(f"RL state count ({len(current_batch_states)}) mismatch batch size ({probs.shape[0]}). Defaulting to discrete.")
                        current_batch_actions = [0] * probs.shape[0] # 数量不匹配，默认离散
                else:
                    logger.warning("sampling_info.rl_states not found or is None. Defaulting to discrete action (0).")
                    current_batch_actions = [0] * probs.shape[0] # 缺少状态信息，默认离散
            else:
                logger.warning("DQN model not imported or initialized. Defaulting to discrete action (0).")
                current_batch_actions = [0] * probs.shape[0] # DQN 未就绪，默认离散

            # 3. 创建动作掩码 (不变)
            rl_soft_mask = torch.tensor(current_batch_actions, device=probs.device, dtype=torch.bool)
            rl_discrete_mask = ~rl_soft_mask

            # 4. 记录动作 (在 Scheduler 中完成)

            # ========== 根据 RL 动作执行后续操作 (不变) ==========
            entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
            logits_output.entropy = entropy

            batch_size = probs.shape[0]
            batch_next_token_ids = torch.full((batch_size,), -1, dtype=torch.int32, device=probs.device)
            max_k = sampling_info.max_topk if sampling_info.max_topk is not None else 1
            if not hasattr(logits_output, 'topk_probs') or logits_output.topk_probs is None or logits_output.topk_probs.shape[1] < max_k:
                logits_output.topk_probs = torch.zeros(batch_size, max_k, dtype=probs.dtype, device=probs.device)
                logits_output.topk_indices = torch.full((batch_size, max_k), -1, dtype=torch.long, device=probs.device)

            # --- 处理 RL 选择 Soft Thinking (动作 1) 的请求 ---
            if rl_soft_mask.any() and enable_soft_thinking:
                probs_soft = probs[rl_soft_mask]

                # 适配 sampling_info 索引
                top_ps_soft = sampling_info.top_ps[rl_soft_mask]
                top_ks_soft = sampling_info.top_ks[rl_soft_mask]
                min_ps_soft = sampling_info.min_ps[rl_soft_mask]

                need_min_p_soft = sampling_info.need_min_p_sampling
                if hasattr(sampling_info, "need_after_thinking_min_p_sampling"):
                    need_min_p_soft |= sampling_info.need_after_thinking_min_p_sampling

                # --- Soft Thinking 数据准备逻辑 ---
                probs_soft = top_k_renorm_prob(probs_soft, top_ks_soft)
                probs_soft = top_p_renorm_prob(probs_soft, top_ps_soft)

                if need_min_p_soft:
                    max_prob_soft = probs_soft.max(dim=-1, keepdim=True).values
                    min_p_thresholds_soft = max_prob_soft * min_ps_soft.view(-1, 1)
                    min_p_mask_soft = probs_soft < min_p_thresholds_soft
                    probs_soft.masked_fill_(min_p_mask_soft, 0.0)
                    probs_sum_soft = probs_soft.sum(dim=-1, keepdim=True)
                    probs_soft = torch.where(probs_sum_soft > 1e-8, probs_soft / probs_sum_soft, probs_soft)

                topk_probs_soft, topk_indices_soft = torch.topk(probs_soft, k=max_k, dim=-1)
                topk_sum_soft = topk_probs_soft.sum(dim=-1, keepdim=True)
                topk_probs_soft = torch.where(topk_sum_soft > 1e-8, topk_probs_soft / topk_sum_soft, topk_probs_soft)

                logits_output.topk_probs[rl_soft_mask] = topk_probs_soft
                logits_output.topk_indices[rl_soft_mask] = topk_indices_soft
                batch_next_token_ids[rl_soft_mask] = topk_indices_soft[:, 0].to(torch.int32)

            # --- 处理 RL 选择 离散生成 (动作 0) 的请求 ---
            discrete_effective_mask = rl_discrete_mask | (rl_soft_mask & ~enable_soft_thinking)
            if discrete_effective_mask.any():
                probs_discrete = probs[discrete_effective_mask]

                # 适配 sampling_info 索引
                top_ks_discrete = sampling_info.top_ks[discrete_effective_mask]
                top_ps_discrete = sampling_info.top_ps[discrete_effective_mask]
                min_ps_discrete = sampling_info.min_ps[discrete_effective_mask]
                need_min_p_discrete = sampling_info.need_min_p_sampling

                if global_server_args_dict["sampling_backend"] == "flashinfer":
                    if need_min_p_discrete:
                        probs_discrete = top_k_renorm_prob(probs_discrete, top_ks_discrete)
                        probs_discrete = top_p_renorm_prob(probs_discrete, top_ps_discrete)
                        sampled_token_ids_discrete = min_p_sampling_from_probs(
                            probs_discrete, min_ps_discrete
                        )
                    else:
                        check_nan = self.use_nan_detection and crash_on_warnings()
                        sampled_token_ids_discrete = top_k_top_p_sampling_from_probs(
                            probs_discrete,
                            top_ks_discrete,
                            top_ps_discrete,
                            filter_apply_order="joint",
                            check_nan=check_nan,
                        )
                    batch_next_token_ids[discrete_effective_mask] = sampled_token_ids_discrete.to(torch.int32)
                    logits_output.topk_probs[discrete_effective_mask, 0] = 1.0
                    logits_output.topk_indices[discrete_effective_mask, 0] = sampled_token_ids_discrete
                    if max_k > 1:
                        logits_output.topk_probs[discrete_effective_mask, 1:] = 0.0
                        logits_output.topk_indices[discrete_effective_mask, 1:] = -1

                else:
                    raise NotImplementedError("Only flashinfer backend adapted for RL masking.")

            # --- 验证 (不变) ---
            if (batch_next_token_ids == -1).any():
                raise RuntimeError("RL decision/sampling logic failed to cover all requests.")
            # --- RL Modification End ---
        # else:
        #     # Post process logits
        #     logits.div_(sampling_info.temperatures)
        #     logits[:] = torch.softmax(logits, dim=-1)
        #     probs = logits
        #     del logits
        #
        #     if global_server_args_dict["sampling_backend"] == "flashinfer":
        #         if return_logprob:
        #             # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
        #             # https://github.com/flashinfer-ai/flashinfer/issues/708
        #             # so we use the torch implementation.
        #
        #             # clamp to avoid -inf
        #             logprobs = torch.log(
        #                 top_p_normalize_probs_torch(probs, sampling_info.top_ps)
        #             ).clamp(min=torch.finfo(probs.dtype).min)
        #
        #         # ==========
        #         # begin of soft thinking
        #         # ==========
        #         if enable_soft_thinking:
        #             # calculate the entropy
        #             entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
        #             soft_mask = sampling_info.soft_thinking_modes # Shape (B,)
        #             top_ps = torch.where(soft_mask, sampling_info.top_ps, sampling_info.after_thinking_top_ps)
        #             top_ks = torch.where(soft_mask, sampling_info.top_ks, sampling_info.after_thinking_top_ks)
        #             min_ps = torch.where(soft_mask, sampling_info.min_ps, sampling_info.after_thinking_min_ps)
        #             dirichlet_alphas = sampling_info.dirichlet_alphas
        #
        #             # top k top p renorm
        #             probs = top_k_renorm_prob(probs, top_ks)
        #             probs = top_p_renorm_prob(probs, top_ps)
        #
        #             # minp renorm
        #             if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow
        #                 max_prob = probs.max(dim=-1, keepdim=True).values
        #                 min_p_thresholds = max_prob * min_ps.view(-1, 1)
        #                 min_p_mask = probs < min_p_thresholds
        #                 probs.masked_fill_(min_p_mask, 0.0)
        #                 probs = probs / probs.sum(dim=-1, keepdim=True)
        #
        #             # dirichlet noise (not used in paper)
        #             if add_noise_dirichlet: # slow
        #                 conc = probs[soft_mask] * dirichlet_alphas[soft_mask].view(-1, 1)
        #                 gamma_dist = torch.distributions.Gamma(conc, torch.ones_like(conc))
        #                 gamma_samples = gamma_dist.sample()
        #                 probs_new = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
        #                 probs[soft_mask] = probs_new
        #
        #             # max top k
        #             topk_probs, topk_indices = torch.topk(probs, k=sampling_info.max_topk, dim=-1) # slow
        #             topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True))
        #
        #             # gumbel softmax noise (not used in paper)
        #             if add_noise_gumbel_softmax: # slow
        #                 topk_logits = torch.log(topk_probs)
        #                 gumbels = (
        #                     -torch.empty_like(topk_logits)
        #                     .exponential_()
        #                     .log()
        #                 )  # ~Gumbel(0,1)
        #                 gumbels = (topk_logits + gumbels) / sampling_info.gumbel_softmax_temperatures  # ~Gumbel(logits,tau)
        #                 topk_probs = gumbels.softmax(-1)
        #                 # topk_probs = F.gumbel_softmax(topk_logits, tau=sampling_info.dirichlet_alphas) # use dirichlet alpha as temperature
        #                 # sort the topk token according to new probability
        #                 sorted_weights, sorted_idx = torch.sort(topk_probs, dim=-1, descending=True)
        #                 topk_probs = sorted_weights
        #                 topk_indices = torch.gather(topk_indices, dim=1, index=sorted_idx)
        #
        #             # after thinking sampling
        #             non_soft_mask = ~soft_mask
        #             if any(non_soft_mask):
        #                 sampled_token_ids = torch.multinomial(probs, num_samples=1)
        #
        #                 # For rows where soft_thinking_modes is False
        #                 topk_probs[non_soft_mask] = 0.0
        #                 topk_indices[non_soft_mask] = 0
        #
        #                 # Assign the first element of each row to sampled_token_ids and set it to 1.0 in topk_probs
        #                 topk_probs[non_soft_mask, 0] = 1.0
        #                 topk_indices[non_soft_mask, 0] = sampled_token_ids[non_soft_mask].view(-1)
        #
        #             logits_output.topk_probs = topk_probs
        #             logits_output.topk_indices = topk_indices
        #             logits_output.entropy = entropy
        #             batch_next_token_ids = topk_indices[:, 0].to(torch.int32)
        #         # ==========
        #         # end of soft thinking
        #         # ==========
        #         else:
        #             if sampling_info.need_min_p_sampling:
        #                 probs = top_k_renorm_prob(probs, sampling_info.top_ks)
        #                 probs = top_p_renorm_prob(probs, sampling_info.top_ps)
        #                 batch_next_token_ids = min_p_sampling_from_probs(
        #                     probs, sampling_info.min_ps
        #                 )
        #             else:
        #                 # Check Nan will throw exception, only check when crash_on_warnings is True
        #                 check_nan = self.use_nan_detection and crash_on_warnings()
        #                 batch_next_token_ids = top_k_top_p_sampling_from_probs(
        #                     probs,
        #                     sampling_info.top_ks,
        #                     sampling_info.top_ps,
        #                     filter_apply_order="joint",
        #                     check_nan=check_nan,
        #                 )
        #
        #     elif global_server_args_dict["sampling_backend"] == "pytorch":
        #         raise NotImplementedError("Pytorch sampling backend is not implemented")
        #         # A slower fallback implementation with torch native operations.
        #         batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
        #             probs,
        #             sampling_info.top_ks,
        #             sampling_info.top_ps,
        #             sampling_info.min_ps,
        #             sampling_info.need_min_p_sampling,
        #         )
        #
        #         if return_logprob:
        #             # clamp to avoid -inf
        #             logprobs = torch.log(
        #                 top_p_normalize_probs_torch(probs, sampling_info.top_ps)
        #             ).clamp(min=torch.finfo(probs.dtype).min)
        #
        #     else:
        #         raise ValueError(
        #             f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
        #         )

        # Attach logprobs to logits_output (in-place modification)
        # todo 注意: logprob 计算现在需要更小心，因为它应该基于 RL 决策之前的 probs
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )
        # 修改返回值，包含动作列表
        return batch_next_token_ids, current_batch_actions
        # --- RL Modification End ---
        # return batch_next_token_ids

    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    assert len(top_logprobs_nums) == logprobs.shape[0], (
        len(top_logprobs_nums),
        logprobs.shape[0],
    )
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx

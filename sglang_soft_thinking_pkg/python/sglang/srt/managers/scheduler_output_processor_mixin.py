from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.managers.schedule_batch import BaseFinishReason, Req, ScheduleBatch
import torch
import logging

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )
import os
logger = logging.getLogger(__name__)
# --- === PPO 阶段四 导入 (开始) === ---
import torch.nn.functional as F  # <-- (PPO): 新增
import torch.optim as optim
from transformers import AutoConfig
# 假设 ppo_agent_model.py 在 Hybrid-Thinking 根目录
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ppo_agent_model import ActorCriticAgent
import matheval
# --- === PPO 阶段四 导入 (结束) === ---

class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        skip_stream_req = None

        if self.is_generation:
            # --- === PPO 修复: 使用属性访问, 而不是解包 === ---
            model_outputs = result.model_outputs
            next_token_ids = result.next_token_ids
            extend_input_len_per_req = result.extend_input_len_per_req
            extend_logprob_start_len_per_req = result.extend_logprob_start_len_per_req
            bid = result.bid
            # --- === PPO 修复结束 === ---
            # <--- 修复：立即解包，确保 'else' 分支有值 ---
            logits_output = model_outputs[0]
            last_hidden_state = model_outputs[1] # H_t
            # <--- 修复结束 ---
            if self.enable_overlap:
                model_outputs, next_token_ids = ( # (原: logits_output)
                    self.tp_worker.resolve_last_batch_result(
                        launch_done,
                    )
                )
                    # <--- 修复：在 overlap 模式下，model_outputs 被重写，必须再次解包 ---
                logits_output = model_outputs[0]
                last_hidden_state = model_outputs[1] # H_t
                # <--- 修复结束 ---
            else:
                # Move next_token_ids and logprobs to cpu
                next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    if logits_output.next_token_logprobs is not None:
                        logits_output.next_token_logprobs = (
                            logits_output.next_token_logprobs.tolist()
                        )
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = tuple(
                            logits_output.input_token_logprobs.tolist()
                        )
            logits_output = model_outputs[0]
            last_hidden_state = model_outputs[1] # H_t
            # --- === PPO 阶段四 决策 (开始) === ---
        actions_list = []
        if self.enable_soft_thinking:
            # <--- 修改：检查来自客户端的强制模式 ---
            # 假设批次中的所有请求都具有相同的设置
            # 0 = 强制 Soft, 1 = 强制 Hard, None = PPO 动态决策
            forced_action = batch.reqs[0].sampling_params.soft_hard_action

            if forced_action is None:
                # --- 模式 1: PPO 动态决策 (Agent 决定) ---
                # (这是我们之前的 PPO 逻辑)
                H_t = last_hidden_state # [B, H]
                K_dim = self.ppo_agent.logits_feature_dim # (例如 10)
                L_t_probs = logits_output.topk_probs # [B, K_actual]

                # 确保 L_t 维度匹配
                if L_t_probs.shape[1] < K_dim:
                    pad = (0, K_dim - L_t_probs.shape[1])
                    L_t = torch.nn.functional.pad(L_t_probs, pad, "constant", 0)
                else:
                    L_t = L_t_probs[:, :K_dim]

                actions, log_probs, _, v_values = self.ppo_agent.get_action_and_value(H_t, L_t)
                actions_list = actions.tolist() # [0, 1, 1, 0, ...]

                # --- 轨迹存储 (仅在 ground_truth 存在时 - 即训练时) ---
                # (在评估模式下，ground_truth 为 None，此块将自动跳过)
                if batch.reqs[0].sampling_params.ground_truth is not None:
                    for i, req in enumerate(batch.reqs):
                        if req.rid not in self.ppo_trajectory_storage:
                            self.ppo_trajectory_storage[req.rid] = []

                        experience = {
                            "H_t": H_t[i].cpu().detach(),
                            "L_t": L_t[i].cpu().detach(),
                            "action": actions[i].cpu().detach(),
                            "v_value": v_values[i].cpu().detach(),
                            "log_prob": log_probs[i].cpu().detach(),
                            "reward": 0.0 # 奖励将在 stream_output 中设置
                        }
                        self.ppo_trajectory_storage[req.rid].append(experience)

            else:
                # --- 模式 2: 强制静态模式 (客户端决定) ---
                # (跳过 PPO Agent，直接使用客户端的命令)
                # logger.info(f"PPO: 强制使用静态模式: {'Soft' if forced_action == 0 else 'Hard'}")
                actions_list = [forced_action] * batch.batch_size()
            # <--- 修改结束 ---
            # --- === PPO 阶段四 决策 (结束) === ---
            #
            # # --- === PPO 阶段四 决策 (开始) === ---
            # actions_list = []
            # if self.enable_soft_thinking:
            #     # 1. 准备状态 s_t = (H_t, L_t)
            #     H_t = last_hidden_state # [B, H]
            #
            #     # 2. 准备 L_t (Logits 特征)
            #     K_dim = self.ppo_agent.logits_feature_dim # (例如 10)
            #     L_t_probs = logits_output.topk_probs # [B, K_actual]
            #
            #     # 确保 L_t 维度匹配
            #     if L_t_probs.shape[1] < K_dim:
            #         # 如果 K_actual < 10, 填充 0
            #         pad = (0, K_dim - L_t_probs.shape[1])
            #         L_t = torch.nn.functional.pad(L_t_probs, pad, "constant", 0)
            #     else:
            #         # 如果 K_actual >= 10, 截断
            #         L_t = L_t_probs[:, :K_dim]
            #
            #     # 3. PPO Agent 决策 (Rollout)
            #     #    (我们处于 no_grad 上下文中, 这是正确的)
            #     actions, log_probs, _, v_values = self.ppo_agent.get_action_and_value(H_t, L_t)
            #
            #     # 4. 存储经验 (s, a, v, log_p)
            #     for i, req in enumerate(batch.reqs):
            #         if req.rid not in self.ppo_trajectory_storage:
            #             self.ppo_trajectory_storage[req.rid] = []
            #
            #         experience = {
            #             "H_t": H_t[i].cpu().detach(),
            #             "L_t": L_t[i].cpu().detach(),
            #             "action": actions[i].cpu().detach(),
            #             "v_value": v_values[i].cpu().detach(),
            #             "log_prob": log_probs[i].cpu().detach(),
            #             "reward": 0.0 # 奖励将在 stream_output 中设置
            #         }
            #         self.ppo_trajectory_storage[req.rid].append(experience)
            #
            #     actions_list = actions.tolist() # [0, 1, 1, 0, ...]
            # --- === PPO 阶段四 决策 (结束) === ---
            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                    continue

                if req.is_chunked <= 0:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    if req.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        req.grammar.accept_token(next_token_id)
                        req.grammar.finished = req.finished()
                    # ==========
                    # begin of soft thinking
                    # ==========
                    if self.enable_soft_thinking:
                        # --- === PPO 阶段四 执行 (开始) === ---
                        # 1. (PPO): 将 Agent 的决策注入 sampling_params
                        req.sampling_params.soft_hard_action = actions_list[i]

                        # 2. (PPO): 调用"傀儡"执行器
                        req.update_top_k_info(logits_output, i, last_hidden_state[i])
                        # --- === PPO 阶段四 执行 (结束) === ---
                    # ==========
                    # end of soft thinking
                    # ==========
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if req.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = (
                                extend_input_len - extend_logprob_start_len
                            )
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                            logprob_pt += num_input_logprobs

            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()

        else:  # embedding or reward model
            embeddings, bid = result.embeddings, result.bid
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        # --- === PPO 修复: 使用属性访问, 而不是解包 === ---
        model_outputs = result.model_outputs
        next_token_ids = result.next_token_ids
        bid = result.bid
        # --- === PPO 修复结束 === ---

        self.num_generated_tokens += len(batch.reqs)

        if self.enable_overlap:
            model_outputs, next_token_ids = self.tp_worker.resolve_last_batch_result( # (原: logits_output)
                launch_done
            )
            next_token_logprobs = model_outputs[0].next_token_logprobs # (原: logits_output...)
        elif batch.spec_algorithm.is_none():
            # spec decoding handles output logprobs inside verify process.
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                # PPO 修正: model_outputs[0] (logits_output) 才是 LogitsProcessorOutput 对象
                next_token_logprobs = model_outputs[0].next_token_logprobs.tolist()

        # --- === PPO 修改点 3: 分离 H_t === ---
        # model_outputs 是 (logits, H_t) 元组
        logits_output = model_outputs[0]
        last_hidden_state = model_outputs[1] # H_t
        # --- === PPO 修改结束 === ---
        # --- === PPO 阶段四 决策 (开始) === ---
        actions_list = []
        if self.enable_soft_thinking:
            # <--- 修改：检查来自客户端的强制模式 ---
            # 0 = 强制 Soft, 1 = 强制 Hard, None = PPO 动态决策
            forced_action = batch.reqs[0].sampling_params.soft_hard_action

            if forced_action is None:
                # --- 模式 1: PPO 动态决策 (Agent 决定) ---
                H_t = last_hidden_state # [B, H]
                K_dim = self.ppo_agent.logits_feature_dim
                L_t_probs = logits_output.topk_probs # [B, K_actual]

                if L_t_probs.shape[1] < K_dim:
                    pad = (0, K_dim - L_t_probs.shape[1])
                    L_t = torch.nn.functional.pad(L_t_probs, pad, "constant", 0)
                else:
                    L_t = L_t_probs[:, :K_dim]

                actions, log_probs, _, v_values = self.ppo_agent.get_action_and_value(H_t, L_t)
                actions_list = actions.tolist() # [0, 1, 1, 0, ...]

                # --- 轨迹存储 (仅在 ground_truth 存在时 - 即训练时) ---
                if batch.reqs[0].sampling_params.ground_truth is not None:
                    for i, req in enumerate(batch.reqs):
                        if req.rid not in self.ppo_trajectory_storage:
                            self.ppo_trajectory_storage[req.rid] = []

                        experience = {
                            "H_t": H_t[i].cpu().detach(),
                            "L_t": L_t[i].cpu().detach(),
                            "action": actions[i].cpu().detach(),
                            "v_value": v_values[i].cpu().detach(),
                            "log_prob": log_probs[i].cpu().detach(),
                            "reward": 0.0 # 奖励将在 stream_output 中设置
                        }
                        self.ppo_trajectory_storage[req.rid].append(experience)

            else:
                # --- 模式 2: 强制静态模式 (客户端决定) ---
                # logger.info(f"PPO: 强制使用静态模式: {'Soft' if forced_action == 0 else 'Hard'}")
                actions_list = [forced_action] * batch.batch_size()
            # <--- 修改结束 ---
        # --- === PPO 阶段四 决策 (结束) === ---

        # # --- === PPO 阶段四 决策 (开始) === ---
        # actions_list = []
        # if self.enable_soft_thinking:
        #     # 1. 准备状态 s_t = (H_t, L_t)
        #     H_t = last_hidden_state # [B, H]
        #
        #     # 2. 准备 L_t (Logits 特征)
        #     K_dim = self.ppo_agent.logits_feature_dim
        #     L_t_probs = logits_output.topk_probs # [B, K_actual]
        #
        #     # 确保 L_t 维度匹配
        #     if L_t_probs.shape[1] < K_dim:
        #         pad = (0, K_dim - L_t_probs.shape[1])
        #         L_t = torch.nn.functional.pad(L_t_probs, pad, "constant", 0)
        #     else:
        #         L_t = L_t_probs[:, :K_dim]
        #
        #     # 3. PPO Agent 决策 (Rollout)
        #     actions, log_probs, _, v_values = self.ppo_agent.get_action_and_value(H_t, L_t)
        #
        #     # (后续代码在您的片段中缺失, 但在 User #37 中存在)
        #     # 4. 存储经验 (s, a, v, log_p)
        #     for i, req in enumerate(batch.reqs):
        #         if req.rid not in self.ppo_trajectory_storage:
        #             self.ppo_trajectory_storage[req.rid] = []
        #
        #         experience = {
        #             "H_t": H_t[i].cpu().detach(),
        #             "L_t": L_t[i].cpu().detach(),
        #             "action": actions[i].cpu().detach(),
        #             "v_value": v_values[i].cpu().detach(),
        #             "log_prob": log_probs[i].cpu().detach(),
        #             "reward": 0.0 # 奖励将在 stream_output 中设置
        #         }
        #         self.ppo_trajectory_storage[req.rid].append(experience)
        #
        #     actions_list = actions.tolist() # [0, 1, 1, 0, ...]
        # # --- === PPO 阶段四 决策 (结束) === ---

        # (您提供的片段到此为止, 下面是 User #37 中该函数的剩余部分,
        #  其中包含了 PPO 的 "执行" 逻辑)

        self.token_to_kv_pool_allocator.free_group_begin()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one extra delayed token
                if self.page_size == 1:
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[i : i + 1])
                else:
                    # Only free when the extra token is in a new page
                    if (
                        len(req.origin_input_ids) + len(req.output_ids) - 1
                    ) % self.page_size == 0:
                        self.token_to_kv_pool_allocator.free(
                            batch.out_cache_loc[i : i + 1]
                        )
                continue

            if batch.spec_algorithm.is_none():
                # speculative worker will solve the output_ids in speculative decoding
                req.output_ids.append(next_token_id)

            req.check_finished()

            if req.finished():
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None and batch.spec_algorithm.is_none():
                req.grammar.accept_token(next_token_id)
                req.grammar.finished = req.finished()


            # ==========
            # begin of soft thinking
            # ==========
            if self.enable_soft_thinking:
                # --- === PPO 阶段四 执行 (开始) === ---
                # 1. (PPO): 将 Agent 的决策注入 sampling_params
                req.sampling_params.soft_hard_action = actions_list[i]

                # 2. (PPO): 调用"傀儡"执行器
                req.update_top_k_info(logits_output, i, last_hidden_state[i])
                # --- === PPO 阶段四 执行 (结束) === ---
            # ==========
            # end of soft thinking
            # ==========

        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()
        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.attn_tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats()

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Compute input_token_logprobs_val
            # Always pad the first one with None.
            req.input_token_logprobs_val = [None]
            req.input_token_logprobs_val.extend(input_token_logprobs)
            # The last input logprob is for sampling, so just pop it out.
            req.input_token_logprobs_val.pop()

            # Compute input_token_logprobs_idx
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]
            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_logprobs_idx = [
                x if x < self.model_config.vocab_size - 1 else 0
                for x in input_token_logprobs_idx
            ]
            req.input_token_logprobs_idx = input_token_logprobs_idx

            if req.top_logprobs_num > 0:
                req.input_top_logprobs_val = [None]
                req.input_top_logprobs_idx = [None]
                assert len(req.temp_input_token_ids_logprobs_val) == len(
                    req.temp_input_token_ids_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_top_logprobs_val,
                    req.temp_input_top_logprobs_idx,
                    strict=True,
                ):
                    req.input_top_logprobs_val.extend(val)
                    req.input_top_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_top_logprobs_val.pop()
                req.input_top_logprobs_idx.pop()
                req.temp_input_top_logprobs_idx = None
                req.temp_input_top_logprobs_val = None

            if req.token_ids_logprob is not None:
                req.input_token_ids_logprobs_val = [None]
                req.input_token_ids_logprobs_idx = [None]

                for val, idx in zip(
                    req.temp_input_token_ids_logprobs_val,
                    req.temp_input_token_ids_logprobs_idx,
                    strict=True,
                ):
                    req.input_token_ids_logprobs_val.extend(val)
                    req.input_token_ids_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_token_ids_logprobs_val.pop()
                req.input_token_ids_logprobs_idx.pop()
                req.temp_input_token_ids_logprobs_idx = None
                req.temp_input_token_ids_logprobs_val = None

            if req.return_logprob:
                relevant_tokens_len = len(req.origin_input_ids) - req.logprob_start_len
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs_val.append(output.next_token_logprobs[i])
        req.output_token_logprobs_idx.append(next_token_ids[i])

        self.add_input_logprob_return_values(
            i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
        )

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.output_token_ids_logprobs_val.append(
                output.next_token_token_ids_logprobs_val[i]
            )
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        output_hidden_states = None

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        # ==========
        # begin of soft thinking
        # ==========
        # Always initialize soft thinking output lists so they exist regardless of flag
        output_topk_probs_list = []
        output_topk_indices_list = []
        # ==========
        # end of soft thinking
        # ==========
        # --- === PPO 修改点 1: 初始化 H_t 列表 === ---
        output_last_hidden_state_list = []
        # --- === PPO 修改结束 === ---
        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_abort:
                continue

            if (
                req.finished()
                # If stream, follow the given stream_interval
                or (req.stream and len(req.output_ids) % self.stream_interval == 0)
                # If not stream, we still want to output some tokens to get the benefit of incremental decoding.
                # TODO(lianmin): this is wrong for speculative decoding because len(req.output_ids) does not
                # always increase one-by-one.
                or (
                    not req.stream
                    and len(req.output_ids) % 16384 == 0
                    and not self.model_config.is_multimodal_gen
                )
            ):
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()
                decode_ids_list.append(decode_ids)
                read_offsets.append(read_offset)
                if self.skip_tokenizer_init:
                    output_ids.append(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)

                if return_logprob:
                    input_token_logprobs_val.append(req.input_token_logprobs_val)
                    input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                    output_token_logprobs_val.append(req.output_token_logprobs_val)
                    output_token_logprobs_idx.append(req.output_token_logprobs_idx)
                    input_top_logprobs_val.append(req.input_top_logprobs_val)
                    input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                    output_top_logprobs_val.append(req.output_top_logprobs_val)
                    output_top_logprobs_idx.append(req.output_top_logprobs_idx)
                    input_token_ids_logprobs_val.append(
                        req.input_token_ids_logprobs_val
                    )
                    input_token_ids_logprobs_idx.append(
                        req.input_token_ids_logprobs_idx
                    )
                    output_token_ids_logprobs_val.append(
                        req.output_token_ids_logprobs_val
                    )
                    output_token_ids_logprobs_idx.append(
                        req.output_token_ids_logprobs_idx
                    )

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                # ==========
                # begin of soft thinking
                # ==========
                if self.enable_soft_thinking:
                    output_topk_probs_list.append(req.get_output_topk_prob_list())
                    output_topk_indices_list.append(req.get_output_topk_idx_list())
                    # --- === PPO 修改点 2: 收集 H_t 列表 === ---
                    output_last_hidden_state_list.append(
                        req.get_output_last_hidden_state_list()
                    )
                    # --- === PPO 阶段四 触发 "Learn" (开始) === ---

                    # 检查1: 是否处于 PPO 模式
                    # 检查2: 请求是否已完成
                    # 检查3: "正确答案"是否已通过 sampling_params 传入
                    if (req.rid in self.ppo_trajectory_storage and
                        req.sampling_params.ground_truth is not None):

                        try:
                            final_text = req.get_full_output_text()
                            ground_truth = req.sampling_params.ground_truth

                            # 1. 规则评估
                            rule_judge_result, extracted_answer = matheval.evaluator_map["gsm8k"].rule_judge(
                                final_text,
                                ground_truth,
                                True
                            )

                            # 2. LLM 评估 (如果规则失败且已启用)
                            #    我们从 sampling_params 中获取 use_llm_judge 标志
                            use_llm_judge_flag = self.use_llm_judge
                            llm_judge_result = None

                            if not rule_judge_result and use_llm_judge_flag:
                                try:
                                    llm_judge_result = matheval.evaluator_map["gsm8k"].llm_judge(
                                        final_text,
                                        ground_truth,
                                        extracted_answer,
                                        True
                                    )
                                except Exception as e_judge:
                                    logger.error(f"PPO: LLM Judge 失败 for rid: {req.rid}. Error: {e_judge}")
                                    llm_judge_result = False  # 评判失败，算作错误

                            # 3. 计算最终奖励 R_T
                            finally_judge_result = rule_judge_result or llm_judge_result
                            final_reward = 1.0 if finally_judge_result else 0.0

                            # 4. 触发训练
                            self._train_ppo_agent(req.rid, final_reward)

                        except Exception as e:
                            logger.error(f"PPO: Failed to get reward for {req.rid}. Error: {e}")
                            if req.rid in self.ppo_trajectory_storage:
                                del self.ppo_trajectory_storage[req.rid]

                    elif req.rid in self.ppo_trajectory_storage:
                        # 请求已完成，但没有 ground_truth (可能是一个评估请求)
                        # 我们必须清理轨迹，否则会内存泄漏
                        del self.ppo_trajectory_storage[req.rid]

                    # --- === PPO 阶段四 触发 "Learn" (结束) === ---
                # ==========
                # end of soft thinking
                # ==========



        # Send to detokenizer
        if rids:
            if self.model_config.is_multimodal_gen:
                return
            self.send_to_detokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids,
                    finished_reasons,
                    decoded_texts,
                    decode_ids_list,
                    read_offsets,
                    output_ids,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                    no_stop_trim,
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    spec_verify_ct,
                    input_token_logprobs_val,
                    input_token_logprobs_idx,
                    output_token_logprobs_val,
                    output_token_logprobs_idx,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    output_top_logprobs_val,
                    output_top_logprobs_idx,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx,
                    output_hidden_states,
                    # ==========
                    # begin of soft thinking
                    # ==========
                    output_topk_probs_list,
                    output_topk_indices_list,
                    # --- === PPO 修改点 3: 插入 H_t 列表 (修复错位) === ---
                    output_last_hidden_state_list,
                    # --- === PPO 修改结束 === ---
                    # ==========
                    # end of soft thinking
                    # ==========
                )
            )


    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)
        self.send_to_detokenizer.send_pyobj(
            BatchEmbeddingOut(
                rids, finished_reasons, embeddings, prompt_tokens, cached_tokens
            )
        )

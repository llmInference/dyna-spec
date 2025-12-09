import logging
from copy import copy
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple
import threading

import torch
import torch.nn.functional as F

# for queries: begin

import sys
import os
import numpy as np
import bisect
import requests
import json

project_root = os.path.join(os.path.dirname(__file__), '../../../..')
if project_root not in sys.path:
    sys.path.append(project_root)

from queries.query_hnsw import get_hnsw_similar_words
from queries.query_graph import get_co_occurrence

lookup = np.load(project_root + "/queries/generated/lookup.npy") # token_id -> token

with open(project_root + "/queries/assets/custom_static_vocab.txt") as f:
    static_vocab = [int(line.strip()) for line in f]

# for queries: end

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info_v2 import (
    EagleDraftInputV2Mixin,
    EagleVerifyInputV2Mixin,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    align_evict_mask_to_page_size,
    assign_req_to_token_pool_func,
    create_accept_length_filter,
    create_extend_after_decode_spec_info,
    filter_finished_cache_loc_kernel,
    generate_simulated_accept_index,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_npu, next_power_of_2

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

logger = logging.getLogger(__name__)


@dataclass
class EagleVerifyInput(SpecInput, EagleVerifyInputV2Mixin):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        if not _is_npu:
            device = "cuda"
        else:
            device = "npu"
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device=device),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device=device),
            positions=torch.empty((0,), dtype=torch.int64, device=device),
            retrive_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):

        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
            for req in batch.reqs:
                req.kv_allocated_len += 1
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def _print_verification_details(
        self,
        batch: ScheduleBatch,
        candidates: torch.Tensor,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
        predict: torch.Tensor,
        target_probs: Optional[torch.Tensor] = None,
        prev_output_ids_list: Optional[List[List[int]]] = None,
    ):
        """Print detailed verification information for each request."""
        bs = candidates.shape[0]
        accept_index_cpu = accept_index.cpu().tolist()
        candidates_cpu = candidates.cpu().tolist()
        predict_cpu = predict.cpu().tolist()
        accept_length_cpu = accept_length.cpu().tolist()
        
        # Get target probabilities if available
        target_probs_cpu = None
        if target_probs is not None:
            target_probs_cpu = target_probs.cpu()

        for i, req in enumerate(batch.reqs):
            if i >= bs:
                break
                
            # Get tokenizer if available
            tokenizer = getattr(req, 'tokenizer', None)
            if tokenizer is None:
                tokenizer = getattr(batch, 'tokenizer', None)
            
            # Get previous outputs (before this verification round)
            if prev_output_ids_list is not None and i < len(prev_output_ids_list):
                prev_output_ids = prev_output_ids_list[i]
            else:
                # Fallback: try to get from req.output_ids
                num_new_tokens = accept_length_cpu[i] + 1  # +1 for bonus token
                prev_output_ids = req.output_ids[:-num_new_tokens] if len(req.output_ids) > num_new_tokens else []
            context_text = ""
            if tokenizer is not None and len(prev_output_ids) > 0:
                try:
                    # Get last 20 tokens for better context
                    context_tokens = prev_output_ids[-20:] if len(prev_output_ids) >= 20 else prev_output_ids
                    context_text = tokenizer.decode(context_tokens)
                except Exception as e:
                    context_text = f"Token IDs: {prev_output_ids[-10:] if prev_output_ids else []}"
            else:
                context_text = f"Token IDs: {prev_output_ids[-10:] if prev_output_ids else []}"
            
            # Calculate rejection layer
            accept_row = accept_index_cpu[i]
            rejection_layer = -1
            
            # Find the first rejected position (first -1)
            for j in range(len(accept_row)):
                if accept_row[j] == -1:
                    rejection_layer = j
                    break
            
            if rejection_layer == -1:
                rejection_layer = self.draft_token_num
            
            # Get draft candidates for this request
            # candidates shape is (bs, draft_token_num), containing all draft tokens
            # For display, we'll show the first few candidates
            # Note: The actual tree structure is more complex, but for logging purposes
            # we'll show the first topk candidates
            draft_candidates_list = []
            for pos in range(min(self.topk, len(candidates_cpu[i]))):
                if pos < len(candidates_cpu[i]):
                    draft_candidates_list.append(candidates_cpu[i][pos])
            
            # Get bonus token (the first accepted token, which is the bonus)
            bonus_token_id = None
            bonus_token_prob = 0.0
            if len(accept_row) > 0 and accept_row[0] != -1:
                bonus_idx = accept_row[0]
                if bonus_idx < len(predict_cpu):
                    bonus_token_id = predict_cpu[bonus_idx]
                    # Get probability for bonus token
                    # Bonus token is the target model's prediction at the root position
                    if target_probs_cpu is not None:
                        # target_probs shape: (bs, draft_token_num, vocab_size)
                        # Bonus token corresponds to target prediction at position 0 (root)
                        if i < target_probs_cpu.shape[0] and bonus_token_id < target_probs_cpu.shape[2]:
                            bonus_token_prob = target_probs_cpu[i, 0, bonus_token_id].item()
            
            # Print verification details
            print("\n" + "="*60)
            if rejection_layer < self.draft_token_num:
                print(f"Draft Failed at Layer {rejection_layer}.")
                # Invoke handler to add words asynchronously
                def run_handle_failure():
                    try:
                        self._handle_failure(
                            batch=batch,
                            req_index=i,
                            rejection_layer=rejection_layer,
                            candidates=candidates,
                            target_probs=target_probs,
                            accept_index=accept_index,
                            predict=predict,
                        )
                    except Exception as e:
                        logger.exception(f"Error in async failure handling: {e}")
                thread = threading.Thread(target=run_handle_failure, daemon=True)
                thread.start()
            else:
                print(f"Draft Accepted All {self.draft_token_num} Tokens.")
            
            print(f"\nContext: \"{context_text}\"")
            print("-" * 60)
            
            # Print draft candidates
            print(f"❌ Draft Candidates (Top-{self.topk} guess):")
            for k, candidate_id in enumerate(draft_candidates_list[:self.topk]):
                candidate_text = ""
                if tokenizer is not None and candidate_id != -1:
                    try:
                        candidate_text = tokenizer.decode([candidate_id])
                    except Exception as e:
                        candidate_text = f"Token {candidate_id}"
                else:
                    candidate_text = f"Token {candidate_id}"
                
                # Draft probability is not directly available, use placeholder
                draft_prob = 0.0
                print(f"   {k+1}. \"{candidate_text}\" (prob: {draft_prob:.2f})")
            
            print("-" * 60)
            
            # Print target decision (bonus token)
            if bonus_token_id is not None:
                bonus_text = ""
                if tokenizer is not None:
                    try:
                        bonus_text = tokenizer.decode([bonus_token_id])
                    except Exception as e:
                        bonus_text = f"Token {bonus_token_id}"
                else:
                    bonus_text = f"Token {bonus_token_id}"
                
                print(f"✅ Target Decision (Bonus Token):")
                print(f"   -> \"{bonus_text}\" (prob: {bonus_token_prob:.2f})")
            else:
                print("✅ Target Decision (Bonus Token):")
                print("   -> No bonus token")
            
            print("-" * 60)
            
            # Print analysis
            if rejection_layer < self.draft_token_num and len(draft_candidates_list) > 0:
                draft_pred = draft_candidates_list[0] if len(draft_candidates_list) > 0 else -1
                draft_pred_text = ""
                if tokenizer is not None and draft_pred != -1:
                    try:
                        draft_pred_text = tokenizer.decode([draft_pred])
                    except Exception as e:
                        draft_pred_text = f"Token {draft_pred}"
                else:
                    draft_pred_text = f"Token {draft_pred}"
                
                target_wanted_text = ""
                if bonus_token_id is not None and tokenizer is not None:
                    try:
                        target_wanted_text = tokenizer.decode([bonus_token_id])
                    except Exception as e:
                        target_wanted_text = f"Token {bonus_token_id}"
                else:
                    target_wanted_text = f"Token {bonus_token_id}" if bonus_token_id is not None else "None"
                
                print(f"Analyze: Draft predicted '{draft_pred_text}', but Target wanted '{target_wanted_text}'.")
            
            print("="*60 + "\n")

    def _handle_failure(
        self,
        batch: ScheduleBatch,
        req_index: int,
        rejection_layer: int,
        candidates: torch.Tensor,
        target_probs: Optional[torch.Tensor] = None,
        accept_index: Optional[torch.Tensor] = None,
        predict: Optional[torch.Tensor] = None,
    ):
        print("✨ Handle Failure Started")
        """Handle failure analysis for a specific request.
        
        Args:
            batch: The schedule batch
            req_index: Index of the request in the batch
            rejection_layer: The layer where draft failed (0-indexed)
            candidates: Draft tokens (bs, draft_token_num)
            target_probs: Target model probabilities (bs, draft_token_num, vocab_size)
            accept_index: Accepted indices (bs, spec_steps+1)
            predict: Target model predictions (flat tensor)
        """


        # 0. judge!
        accepted_id = -1
        if predict != None:
            draft_pos = rejection_layer - 1
            if draft_pos < self.draft_token_num:
                idx = self.retrive_index[req_index, draft_pos]
                if idx >= 0 and idx < len(predict):
                    accepted_id = predict[idx].item()
        
        # 不知道accepted_id获取对了没

        if accepted_id != -1:
            idx = bisect.bisect_left(static_vocab, accepted_id)
            if (idx < len(static_vocab) and static_vocab[idx] == accepted_id):
                print(f"✨ Accepted token '{lookup[accepted_id]}' is already in static vocab. Skip.")
                # 如果正确答案已经在精简词汇表里了，说明是小模型自己能力差没猜到，就不加词了
                return

        
        # 1. Get top_k
        print("✨ Getting top_k & last_hidden_state...")
        top_k = []
        if target_probs is not None and rejection_layer < self.draft_token_num:
            # target_probs shape: (bs, draft_token_num, vocab_size)
            probs_at_failure = target_probs[req_index, rejection_layer]
            
            # Get top-k probabilities and token ids
            topk = min(self.topk, probs_at_failure.shape[-1])
            top_probs, top_token_ids = torch.topk(probs_at_failure, k=topk)
            
            # print(f"\nTop-{topk} tokens from Target Model at failure layer {rejection_layer}:")
            for k in range(topk):
                token_id = top_token_ids[k].item()
                top_k.append(token_id)


        # 2. Get last hidden state
        if hasattr(batch, 'spec_info') and hasattr(batch.spec_info, 'hidden_states'):
            hidden_states = batch.spec_info.hidden_states

            if rejection_layer < self.draft_token_num:
                flat_index = req_index * self.draft_token_num + rejection_layer
                if flat_index < hidden_states.shape[0]:
                    last_hidden_state = hidden_states[flat_index].float().cpu().numpy()

                else:
                    print(f"\nHidden State not available at index {flat_index}")
            else:
                print(f"\nHidden State: rejection_layer {rejection_layer} >= draft_token_num {self.draft_token_num}")
        else:
            print(f"\nHidden State not available in batch.spec_info")


        # 3. Query and get C_final
        hnsw_result = get_hnsw_similar_words(last_hidden_state, 10)
        C_initial = list(set(top_k + hnsw_result))

        C_co_occurrence = []
        for token in C_initial:
            C_co_occurrence.extend(get_co_occurrence(token))
        
        print(f"✨ HNSW Result: {len(hnsw_result)} tokens; C_co_occurrence: {len(C_co_occurrence)} tokens")
        
        C_final = list(set(C_initial + C_co_occurrence))


        # 4. Add to dyna vocab
        url = "http://127.0.0.1:30000/v1/vocab/add"
        headers = {"Content-Type": "application/json"}
        C_final_words = [lookup[idx] for idx in C_final]
        data = {"words": C_final_words}

        print("✨ Adding words to dyna vocab: [" + ", ".join(C_final_words) + "]")
        
        # 有一些词可能会被重复添加，不过为了性能考虑，就不做判断了
        response = requests.post(url, headers=headers, json=data)
        

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        # Skip verification in prefill-only phase
        # Speculative decoding should only happen in decode phase, not in prefill phase
        is_prefill_only = getattr(batch, 'is_prefill_only', False)
        if is_prefill_only and batch.forward_mode.is_extend():
            # This is a prefill-only batch, should not perform speculative decoding
            logger.warning(
                "Attempted to verify in prefill-only phase. "
                "Speculative decoding should only occur in decode phase. "
                "Skipping verification."
            )
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrive_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrive_index are the indices of the requests that are kept.
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            # This is a relaxed version of penalties for speculative decoding.
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=batch.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # Sample tokens. Force greedy sampling on AMD
        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        target_probs_for_log = None
        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE or _is_npu:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
            # Compute probabilities for logging in greedy case
            target_logits_reshaped = logits_output.next_token_logits.reshape(bs, self.draft_token_num, -1)
            target_probs_for_log = F.softmax(target_logits_reshaped, dim=-1)

        else:
            # apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            if not torch.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            target_probs_for_log = target_probs

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )

            # coins for rejection sampling
            coins = torch.rand_like(
                candidates, dtype=torch.float32, device=batch.device
            )
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0.0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                bs=bs,
                spec_steps=self.spec_steps,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # Save output_ids before modification for logging
        prev_output_ids_list = [list(req.output_ids) for req in batch.reqs]

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                            )
                            raise e
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += (
                sum(1 for idx in accept_index_row if idx != -1) - 1
            )

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Save accept_index before it gets modified
        accept_index_for_log = accept_index.clone()

        # Print detailed verification information for each request
        # Only print in decode phase, not in prefill phase
        # Check if this is a decode phase (not prefill-only)
        # Skip printing if this is a prefill-only batch
        is_prefill_only = getattr(batch, 'is_prefill_only', False)
        is_decode_phase = (
            not batch.forward_mode.is_idle()
            and not is_prefill_only
            and (
                batch.forward_mode.is_decode()
                or batch.forward_mode == ForwardMode.TARGET_VERIFY
                or (hasattr(batch, 'forward_mode') and not batch.forward_mode.is_extend())
            )
        )
        
        if is_decode_phase:
            self._print_verification_details(
                batch=batch,
                candidates=candidates,
                accept_index=accept_index_for_log,
                accept_length=accept_length,
                predict=predict,
                target_probs=target_probs_for_log,
                prev_output_ids_list=prev_output_ids_list,
            )

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        accept_length_cpu = accept_length.cpu()
        # FIXME: this `tolist()` fixes the numerical calculation consistency
        # try to unify the tensor representation and list representation
        accept_length_list = accept_length_cpu.tolist()

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            for i, req in enumerate(batch.reqs):
                req.kv_committed_len += accept_length_list[i] + 1
                req.kv_allocated_len = req.kv_committed_len
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
                for i, req in enumerate(batch.reqs):
                    req.kv_committed_len += accept_length_list[i] + 1
                    req.kv_allocated_len = req.kv_committed_len
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = accept_length[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        # Construct EagleVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(accept_length + 1)
            batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            draft_input = EagleDraftInput(
                hidden_states=batch.spec_info.hidden_states[accept_index],
                verified_id=verified_id,
                accept_length=accept_length,
                accept_length_cpu=accept_length_list,
                seq_lens_for_draft_extend=batch.seq_lens,
                seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
                req_pool_indices_for_draft_extend=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=draft_input.accept_length_cpu,
                accepted_indices=accept_index,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    bs,
                )
                batch.seq_lens.add_(accept_length + 1)
                batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_accept_length_cpu = [
                    accept_length_list[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    batch.seq_lens_cpu.add_(accept_length_cpu + 1)
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                draft_input = EagleDraftInput(
                    hidden_states=batch.spec_info.hidden_states[
                        unfinished_accept_index
                    ],
                    verified_id=predict[unfinished_accept_index],
                    accept_length_cpu=draft_input_accept_length_cpu,
                    accept_length=accept_length[unfinished_index_device],
                    seq_lens_for_draft_extend=batch.seq_lens[unfinished_index_device],
                    seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu[unfinished_index],
                    req_pool_indices_for_draft_extend=batch.req_pool_indices[
                        unfinished_index_device
                    ],
                )
            else:
                draft_input = EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_list,
                accepted_indices=accept_index,
            )


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    # Constant: alloc length per decode step
    ALLOC_LEN_PER_DECODE: ClassVar[int] = None

    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    # Shape info for padding
    num_tokens_per_batch: int = -1
    num_tokens_for_logprob_per_batch: int = -1

    # Inputs for draft extend
    # shape: (b,)
    seq_lens_for_draft_extend: torch.Tensor = None
    seq_lens_for_draft_extend_cpu: torch.Tensor = None
    req_pool_indices_for_draft_extend: torch.Tensor = None

    # Inputs for V2 overlap worker
    future_indices: Optional[FutureIndices] = None
    allocate_lens: Optional[torch.Tensor] = None
    new_seq_lens: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_batch, self.num_tokens_for_logprob_per_batch

    def prepare_for_extend(self, batch: ScheduleBatch):

        if batch.forward_mode.is_idle():
            return

        # Prefill only generate 1 token.
        assert len(self.verified_id) == len(batch.seq_lens)

        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype,
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, hidden_size), device=device, dtype=dtype),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
            accept_length=torch.empty((0,), device=device, dtype=torch.int32),
            accept_length_cpu=[],
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):

        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.verified_id
        batch.extend_lens = [x + 1 for x in batch.spec_info.accept_length_cpu]
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend
        batch.seq_lens_cpu = batch.spec_info.seq_lens_for_draft_extend_cpu
        batch.req_pool_indices = batch.spec_info.req_pool_indices_for_draft_extend
        batch.return_logprob = False
        batch.return_hidden_states = False

        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.accept_length.add_(1)
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)

        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.accept_length,
            self.positions,
            self.verified_id,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = self.accept_length.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            self.allocate_lens = self.allocate_lens[new_indices]
            return

        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            if len(new_indices) != len(self.topk_p):
                logger.warning(
                    f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
                )
            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            self.hidden_states = self.hidden_states[: len(new_indices)]
            self.verified_id = self.verified_id[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            self.hidden_states = self.hidden_states[new_indices]
            self.verified_id = self.verified_id[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            self.allocate_lens = torch.cat(
                [self.allocate_lens, spec_info.allocate_lens]
            )
            return

        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: torch.Tensor

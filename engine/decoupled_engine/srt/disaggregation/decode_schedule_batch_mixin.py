# Adapted from SGLang python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, List

import torch

from decoupled_engine.srt.disaggregation.prebuilt_h2d import (
    list_to_device,
    prebuilt_nonblocking_h2d,
)
from decoupled_engine.srt.managers.overlap_utils import RelayPayload
from decoupled_engine.srt.managers.scheduler_components.phase_timers import phase
from decoupled_engine.srt.mem_cache.common import maybe_cache_unfinished_req
from decoupled_engine.srt.model_executor.forward_batch_info import ForwardMode
from decoupled_engine.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.overlap_utils import FutureMap
    from decoupled_engine.srt.managers.schedule_batch import ScheduleBatch


class ScheduleBatchDisaggregationDecodeMixin:

    def prepare_for_prebuilt(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        Adapted from .prepare_for_extend().
        """

        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        # PREBUILT never enters a model forward. Keep the legacy scalar metadata,
        # but do not flatten and copy every transferred prompt to the GPU only to
        # discard it before the first decode step.
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_range.length for req in reqs)
        extend_num_tokens = total_size
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)
            pre_len = len(req.prefix_indices)

            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                pre_len : pre_len + req.extend_range.length
            ]
            assert (
                offset + req.extend_range.length <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_range.length={req.extend_range.length}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_range.length] = chunk
            offset += req.extend_range.length

            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_range.length
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_range.length={req.extend_range.length}"

            if not req.retracted_stain:
                # Clamp to avoid double-counting: already_computed is seeded from
                # the prefill-reported cached_tokens in _commit_transfer_to_req, so
                # a decode-side prefix shorter than the prefill report must not
                # subtract from cached_tokens.
                delta = max(0, pre_len - req.already_computed)
                req.cached_tokens += delta
                req.cached_tokens_device += delta
                req.already_computed = seq_len
            req.is_retracted = False
            if getattr(req, "pd_rebootstrap_in_progress", False):
                req.pd_rebootstrap_in_progress = False
            pre_lens.append(pre_len)

        # Set fields
        # The first decode input and speculative extras are seeded by
        # process_prebuilt through FutureMap, so merge/forward-entry rebuilds
        # input_ids from the relay.
        self.input_ids = None
        if prebuilt_nonblocking_h2d():
            # SGLANG_PREBUILT_NONBLOCKING_H2D: same tensors, no host wait for
            # the forward in flight (see disaggregation/prebuilt_h2d.py).
            self.req_pool_indices = list_to_device(
                req_pool_indices, torch.int64, self.device
            )
            self.seq_lens = list_to_device(seq_lens, torch.int64, self.device)
            self.orig_seq_lens = list_to_device(seq_lens, torch.int32, self.device)
        else:
            self.req_pool_indices = torch.tensor(
                req_pool_indices, dtype=torch.int64, device=self.device
            )
            self.seq_lens = torch.tensor(
                seq_lens, dtype=torch.int64, device=self.device
            )
            self.orig_seq_lens = torch.tensor(
                seq_lens, dtype=torch.int32, device=self.device
            )
        self.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.logprob.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_range.length for r in reqs]
        self.extend_logprob_start_lens = None
        self.extend_input_logprob_token_ids = None
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt(
        self: ScheduleBatch,
        future_map: FutureMap,
    ):
        """Assign the buffered last input id to schedule batch"""
        # Admission path phases (SGLANG_SCHED_PHASE_TIMERS): prebuilt_reqs,
        # prebuilt_h2d_last_tokens (a copy from pageable host memory on the
        # schedule stream, which waited for the forward stream), and
        # prebuilt_spec_input (the EAGLE draft input: see eagle_disaggregation).
        with phase("prebuilt_reqs"):
            last_tokens: List[int] = []
            for req in self.reqs:
                last_tokens.append(req.output_ids[-1])
                # PREBUILT does not materialize a local SWA branching window.
                if req.swa_branching_seqlen is not None:
                    req.swa_branching_seqlen = None
                maybe_cache_unfinished_req(req, self.tree_cache)
                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        # if it is not None, then the grammar is from a retracted request, and we should not
                        # accept the token as it's already accepted
                        if req.grammar.current_token is None:
                            req.grammar.accept_token(req.output_ids[-1])
                    except ValueError as e:
                        from decoupled_engine.srt.managers.schedule_batch import FINISH_ABORT

                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        # Use to_finish (not finished_reason) so that process_batch_result_prebuilt
                        # handles the release via update_finish_state -> release_kv_cache in one place.
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                        req.to_finish = FINISH_ABORT(
                            error_message, HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                    req.grammar.finished = req.finished()
        with phase("prebuilt_h2d_last_tokens"):
            if prebuilt_nonblocking_h2d():
                last_tokens_tensor = list_to_device(
                    last_tokens, torch.int64, self.device
                )
            else:
                last_tokens_tensor = torch.tensor(
                    last_tokens, dtype=torch.int64, device=self.device
                )

        with phase("prebuilt_spec_input"):
            spec_info = self.spec_algorithm.build_disagg_draft_input(
                self,
                last_tokens_tensor,
                future_map,
            )
        if spec_info is not None:
            self.spec_info = spec_info
        else:
            # Non-spec: stash last token into the relay so the first DECODE's
            # resolve_forward_inputs gathers it like any other decode iter.
            future_map.stash(
                self.req_pool_indices, RelayPayload(bonus_tokens=last_tokens_tensor)
            )
            self.input_ids = None

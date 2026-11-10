# Adapted from SGLang python/sglang/srt/speculative/eagle_disaggregation.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from decoupled_engine.srt.disaggregation.prebuilt_h2d import (
    list_to_device,
    prebuilt_nonblocking_h2d,
    stack_to_device,
)
from decoupled_engine.srt.layers.attention.dsa.utils import (
    should_remap_pd_dsa_seed_to_local_slots,
)
from decoupled_engine.srt.managers.overlap_utils import RelayPayload
from decoupled_engine.srt.managers.scheduler_components.phase_timers import phase
from decoupled_engine.srt.model_executor.forward_batch_info import CaptureHiddenMode
from decoupled_engine.srt.runtime_context import get_spec
from decoupled_engine.srt.speculative.eagle_info import EagleDraftInput

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.overlap_utils import FutureMap
    from decoupled_engine.srt.managers.schedule_batch import ScheduleBatch


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> EagleDraftInput:
    # Adaptive spec moves the step count after publish, and this runs once per
    # prebuilt batch.
    spec = get_spec()
    num_states = spec.speculative_eagle_topk
    if spec.enable_multi_layer_eagle:
        num_states *= spec.speculative_num_steps

    # Admission path phases (SGLANG_SCHED_PHASE_TIMERS): host data to the GPU
    # on the schedule stream right after it waited for the forward stream. A
    # copy from pageable memory or an .item() there blocks the CPU until the
    # forward in flight is done. SGLANG_PREBUILT_NONBLOCKING_H2D: the same
    # tensors from pinned host memory with non-blocking copies
    # (disaggregation/prebuilt_h2d.py). Each helper returns None for inputs it
    # does not cover (then the original blocking copy below runs).
    nonblocking = prebuilt_nonblocking_h2d()
    with phase("prebuilt_h2d_topk"):
        topk_p = topk_index = None
        if nonblocking:
            topk_p = _topk_to_device(
                [req.output_topk_p for req in batch.reqs],
                num_states,
                torch.float32,
                batch.device,
            )
            topk_index = _topk_to_device(
                [req.output_topk_index for req in batch.reqs],
                num_states,
                torch.int64,
                batch.device,
            )
        if topk_p is None:
            topk_p = torch.stack(
                [
                    torch.as_tensor(
                        req.output_topk_p[:num_states],
                        device=batch.device,
                        dtype=torch.float32,
                    )
                    for req in batch.reqs
                ],
                dim=0,
            )
        if topk_index is None:
            topk_index = torch.stack(
                [
                    torch.as_tensor(
                        req.output_topk_index[:num_states],
                        device=batch.device,
                        dtype=torch.int64,
                    )
                    for req in batch.reqs
                ],
                dim=0,
            )

    with phase("prebuilt_h2d_hidden"):
        hidden_list = [req.hidden_states_tensor for req in batch.reqs]
        hidden_states = (
            stack_to_device(hidden_list, batch.device) if nonblocking else None
        )
        if hidden_states is None:
            hidden_states = torch.stack(hidden_list, dim=0).to(batch.device)

    dsa_topk_indices = None
    dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
    if dsa_indices_list and all(t is not None for t in dsa_indices_list):
        with phase("prebuilt_h2d_dsa"):
            dsa_topk_indices = (
                stack_to_device(dsa_indices_list, batch.device) if nonblocking else None
            )
            if dsa_topk_indices is None:
                dsa_topk_indices = torch.stack(dsa_indices_list, dim=0).to(batch.device)
        if should_remap_pd_dsa_seed_to_local_slots():
            # PD sends request-relative positions; fused TopK consumes
            # decode-local physical slots. Remap once before the draft loop/graph.
            req_to_token = batch.req_to_token_pool.req_to_token
            table_width = req_to_token.shape[1]
            valid_positions = dsa_topk_indices >= 0
            gather_positions = dsa_topk_indices.clamp(min=0, max=table_width - 1).to(
                torch.int64
            )
            local_slots = req_to_token[
                batch.req_pool_indices[:, None], gather_positions
            ]
            invalid_rows = torch.any(
                (dsa_topk_indices < -1)
                | (dsa_topk_indices >= batch.seq_lens[:, None])
                | (dsa_topk_indices >= table_width)
                # Slot 0 is the reserved padding sink; real KV allocations
                # start at 1, and untouched req-to-token entries remain 0.
                | (valid_positions & (local_slots <= 0)),
                dim=1,
            )
            local_slots.masked_fill_(~valid_positions, -1)
            local_slots.masked_fill_(invalid_rows[:, None], -1)
            dsa_topk_indices = local_slots
        with phase("prebuilt_item"):
            # The one real sync left on this path (data-dependent: the remap
            # above reads req_to_token on the GPU). It runs after the radix
            # insert in process_prebuilt, so with the flag on the forward in
            # flight usually finished during the insert.
            no_valid_row = torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item()
        if no_valid_row:
            dsa_topk_indices = None

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
        dsa_topk_indices=dsa_topk_indices,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    if batch.enable_overlap:
        spec_info.future_dsa_topk_indices_available = dsa_topk_indices is not None
        spec_info.future_indices = batch.req_pool_indices
        # Seed the relay buf with the known seq_lens; publish's chained record
        # keeps the in-flight forward's fence intact (see FutureMap.publish).
        with phase("prebuilt_future_map"):
            future_map.publish(spec_info.future_indices, batch.seq_lens)
            future_map.stash(
                spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
            )

    return spec_info


def _topk_to_device(rows, num_states: int, dtype: torch.dtype, device):
    """The stacked ``torch.as_tensor(row[:num_states], dtype=dtype)`` rows on
    ``device`` without a host wait; None if the rows are neither all host
    tensors nor all lists/tuples (the caller keeps its original path)."""
    cut = [row[:num_states] for row in rows]
    if all(isinstance(row, (list, tuple)) for row in cut):
        if len({len(row) for row in cut}) != 1:
            return None
        return list_to_device(cut, dtype, device)
    return stack_to_device(cut, device, dtype=dtype)

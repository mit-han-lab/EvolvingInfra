from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from decoupled_engine.srt.managers.overlap_utils import RelayPayload
from decoupled_engine.srt.speculative.dspark_components.dspark_draft import make_next_draft_input

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.overlap_utils import FutureMap
    from decoupled_engine.srt.managers.schedule_batch import ScheduleBatch
    from decoupled_engine.srt.speculative.spec_info import SpecInput


def build_dspark_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> SpecInput:
    spec_info = make_next_draft_input(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )
    if batch.enable_overlap:
        spec_info.future_dsa_topk_indices_available = False
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )
    return spec_info

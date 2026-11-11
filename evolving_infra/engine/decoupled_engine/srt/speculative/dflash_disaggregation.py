from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from decoupled_engine.srt.managers.overlap_utils import RelayPayload
from decoupled_engine.srt.speculative.draft_worker_common import make_draft_input_v2

if TYPE_CHECKING:
    from decoupled_engine.srt.managers.overlap_utils import FutureMap
    from decoupled_engine.srt.managers.schedule_batch import ScheduleBatch
    from decoupled_engine.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def build_dflash_family_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    spec_info = make_draft_input_v2(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )
    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices,
            RelayPayload(bonus_tokens=last_tokens_tensor),
        )
    return spec_info

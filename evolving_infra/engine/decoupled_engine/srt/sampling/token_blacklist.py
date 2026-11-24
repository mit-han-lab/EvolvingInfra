# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: Apache-2.0
"""An engine-level decode-token blacklist (no torch import).

DeepSeek's reference engine (DeepInfer) masks the token ids in
DEEPINFER_DECODE_TOKEN_BLACKLIST to -inf on the target logits before sampling and
before speculative verification, including the token sampled at prefill; a replay
that must run every output to its recorded length masks the EOS this way.
SGLANG_V41_DECODE_TOKEN_BLACKLIST="1" applies the same mask inside the target's
LogitsProcessor, so sampling_info carries no adjustment: a per-request logit_bias
or min_new_tokens would turn off DSpark's blockwise and fused verify paths
(dspark_verify.verify_logits_adjustments_are_noop).
"""

from typing import List, Sequence, Tuple


def parse_token_blacklist(spec: str) -> Tuple[int, ...]:
    """Comma-separated token ids ("1", "1, 128803"); empty disables the mask."""
    ids = []
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        token_id = int(part)
        if token_id < 0:
            raise ValueError(f"negative token id in the token blacklist: {spec!r}")
        if token_id not in ids:
            ids.append(token_id)
    return tuple(ids)


def local_blacklist_columns(
    token_ids: Sequence[int], col_offset: int, width: int
) -> List[int]:
    """Columns of a [*, width] logits tensor (vocab shard starting at col_offset)."""
    return [t - col_offset for t in token_ids if col_offset <= t < col_offset + width]


def apply_token_blacklist(logits, token_ids: Sequence[int], col_offset: int = 0):
    """Set the blacklisted columns of next-token logits to -inf, in place.

    One fill per token on a column view: no index tensor and no host-to-device
    copy, so it is safe inside CUDA graph capture."""
    if logits is None or not token_ids:
        return logits
    for column in local_blacklist_columns(token_ids, col_offset, logits.shape[-1]):
        logits[..., column].fill_(float("-inf"))
    return logits

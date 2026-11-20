# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: Apache-2.0
"""DP rank of a request on a symmetric P/D worker with DP attention.

Each DP rank of such a worker has its own radix tree and host KV tier, and the
router addresses whole workers. The worker's DP controller places requests:

- A decode leg (``kv_import`` without ``kv_export``) goes to
  ``planes.import_dp_rank(transfer_id)``. The prefill leg's owner computes the
  same rank and sends its publications there, so the two legs meet without a
  message between the workers.
- Every other request (prefill legs, local requests) is placed by the router's
  cache-first prefill selection (``select_p_cache_first`` in
  ``router/core/src/scheduler/placement.rs``), applied to the worker's DP ranks
  with the router's default thresholds:
  1. if the ranks' requests are imbalanced (the most loaded rank has more
     than ``BALANCE_ABS_REQS`` more requests and more than ``BALANCE_REL``
     times the least loaded rank's), the least-loaded rank;
  2. else, if the ranks holding the longest prefix of the prompt hold more
     than ``CACHE_THRESHOLD`` of it, the least loaded of them;
  3. else the least-loaded rank.

  Least loaded means fewest requests, then fewest queued uncached tokens,
  then the lowest rank: on an idle worker, sequential requests that no rank
  holds a prefix of all go to rank 0.
  A rank's load comes from its last load snapshot (waiting and running
  requests, ``num_waiting_uncached_tokens``), plus the requests placed since.
  What a rank holds is what this controller sent it: a chained hash per prompt
  page records the ranks that received that page. A conversation's next turn
  therefore returns to the rank that holds its previous turn, also when that
  rank's queue is longer, while a prefix that every rank holds (a shared
  system prompt) prefers no rank. Queued tokens do not outweigh a hit: on a
  saturated worker the ranks' queues differ by more than a whole prompt, and a
  turn sent away from its rank computes its whole prompt again.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Sequence

from decoupled_engine.srt.disaggregation.symmetric.planes import import_dp_rank

# The router's cache-first defaults (placement.rs: default_balance_abs_reqs,
# default_balance_rel, default_cache_threshold).
BALANCE_ABS_REQS = 32
BALANCE_REL = 1.1
CACHE_THRESHOLD = 0.5
# Pages whose ranks are remembered; the least recently placed are dropped
# first. A worker needs the pages of the prompts of the conversations it
# serves: 512 conversations of 134k tokens are 268k pages of 256 tokens.
MAX_PAGES = 1 << 20


def _transfer_id(mark: Any) -> Optional[int]:
    if isinstance(mark, Mapping):
        value = mark.get("transfer_id")
    else:
        value = getattr(mark, "transfer_id", None)
    return int(value) if value is not None else None


class SymmetricDpPlacement:
    """See the module docstring. Not thread-safe: the DP controller's loop owns it."""

    def __init__(self, dp_size: int, page_size: int, max_pages: int = MAX_PAGES) -> None:
        self.dp_size = int(dp_size)
        self.page_size = max(1, int(page_size))
        self.max_pages = int(max_pages)
        # chained page hash -> bit mask of the ranks that received that page
        self._pages: OrderedDict[int, int] = OrderedDict()
        self._reqs = [0] * self.dp_size
        self._queued = [0] * self.dp_size
        self._stamp = [0.0] * self.dp_size

    def update_loads(self, snapshots: Sequence[Any]) -> None:
        """Take each rank's requests and queued uncached tokens from a fresh load snapshot."""
        for snap in snapshots:
            rank = int(getattr(snap, "dp_rank", -1))
            if not 0 <= rank < self.dp_size or snap.timestamp == self._stamp[rank]:
                continue
            self._stamp[rank] = snap.timestamp
            self._reqs[rank] = int(snap.num_waiting_reqs) + int(snap.num_running_reqs)
            self._queued[rank] = int(snap.num_waiting_uncached_tokens)

    def _page_hashes(self, input_ids: Sequence[int]) -> List[int]:
        out, h, ps = [], 0, self.page_size
        for start in range(0, len(input_ids) - ps + 1, ps):
            h = hash((h, tuple(input_ids[start : start + ps])))
            out.append(h)
        return out

    def held_pages(self, hashes: Sequence[int]) -> List[int]:
        """Per rank: how many leading pages of the prompt it received."""
        held = [0] * self.dp_size
        alive = (1 << self.dp_size) - 1
        for i, h in enumerate(hashes):
            alive &= self._pages.get(h, 0)
            if not alive:
                break
            for rank in range(self.dp_size):
                if alive >> rank & 1:
                    held[rank] = i + 1
        return held

    def _rank(self, held: Sequence[int], prompt_len: int) -> int:
        ranks = range(self.dp_size)

        def least_loaded(candidates) -> int:
            return min(candidates, key=lambda r: (self._reqs[r], self._queued[r], r))

        low, high = min(self._reqs), max(self._reqs)
        if high - low > BALANCE_ABS_REQS and high > low * BALANCE_REL:
            return least_loaded(ranks)
        best = max(held)
        if best > 0 and best * self.page_size > CACHE_THRESHOLD * prompt_len:
            return least_loaded([r for r in ranks if held[r] == best])
        return least_loaded(ranks)

    def place(self, input_ids: Optional[Sequence[int]], kv_export: Any = None, kv_import: Any = None) -> int:
        if kv_import is not None and kv_export is None:
            tid = _transfer_id(kv_import)
            if tid is not None:
                return import_dp_rank(tid, self.dp_size)
        ids = input_ids or ()
        hashes = self._page_hashes(ids)
        held = self.held_pages(hashes)
        rank = self._rank(held, len(ids))
        self._reqs[rank] += 1
        self._queued[rank] += max(0, len(ids) - held[rank] * self.page_size)
        bit = 1 << rank
        for h in hashes:
            self._pages[h] = self._pages.pop(h, 0) | bit
        while len(self._pages) > self.max_pages:
            self._pages.popitem(last=False)
        return rank

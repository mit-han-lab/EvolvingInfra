# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cross-rank agreement for symmetric P/D state transitions.

Every TP rank runs its own OwnerPlane / ReaderPlane IO threads, so READY,
FAILED, DONE and timeouts land on the ranks at slightly different wall-clock
times. If one rank commits an import (or frees an export) in tick N and its
peer only in tick N+1, the two schedulers build different batches and the next
collective (broadcast_pyobj / NCCL all-reduce) never completes (an observed
hang had READY land 4 ms apart across TP0/TP1).

`RankConsensus.agree` MIN-reduces a small integer state vector over the
attention-TP gloo group, the same trick PD decode uses in
`disaggregation.utils._all_reduce_polls`. Callers encode states so that MIN
gives the conservative answer:

    -1 = failed / aborted / timed out on this rank   (any rank -> everyone fails)
     0 = still pending                               (any rank -> everyone waits)
     1 = ready / released on this rank               (all ranks -> everyone commits)

The vector must be built from a set the ranks agree on (imports/exports are
added from broadcast requests) in a rank-independent order (sorted by
transfer_id); a length mismatch would make the all-reduce hang, so callers
never include rank-local bookkeeping in the vector.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from decoupled_engine.srt.managers.scheduler_components import collective_guard

FAILED = -1
PENDING = 0
READY = 1


def _torch_min_all_reduce(states: Sequence[int], group) -> List[int]:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(states, dtype=torch.int32, device="cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    return tensor.tolist()


class RankConsensus:
    """MIN-reduce integer state vectors across the TP ranks of one scheduler.

    `size <= 1` (or `group is None`) is the single-rank fast path: `agree`
    returns its input unchanged and performs no collective.
    """

    def __init__(
        self,
        group=None,
        size: int = 1,
        *,
        reduce_fn: Optional[Callable[[Sequence[int], object], List[int]]] = None,
    ) -> None:
        self.group = group
        self.size = size
        self._reduce = reduce_fn or _torch_min_all_reduce
        self.stats_rounds = 0

    @property
    def enabled(self) -> bool:
        return self.size > 1 and self.group is not None

    def agree(self, states: Sequence[int]) -> List[int]:
        # Before the single-rank shortcut, so that TP1 stand-ins in CPU tests
        # also catch a vote inside the symmetric launch thread's result work.
        collective_guard.check("symmetric vote")
        if not self.enabled or not states:
            return list(states)
        self.stats_rounds += 1
        return self._reduce(states, self.group)


def local_consensus() -> RankConsensus:
    """Single-rank consensus (tests, TP1)."""
    return RankConsensus(None, 1)

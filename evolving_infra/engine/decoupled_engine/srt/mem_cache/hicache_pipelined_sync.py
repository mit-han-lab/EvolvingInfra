"""SGLANG_HICACHE_PIPELINED_SYNC: the per-step HiCache agreement, one step late.

Every scheduler step, ``UnifiedRadixCache.check_hicache_events`` MIN-reduces a
small CPU vector across the TP ranks (ready write / load acks, storage control
queue sizes, the GPU-export work flag, the TP digests, the idle-sync in-flight
count) and, under DSv4 window residency, ``SwaBackfillManager.poll`` reduces
the backfill job rows. Both are blocking Gloo collectives on the critical path
of the step. A Gloo all-reduce is a ring of 2(N-1) sequential messages, so its
cost is that many network latencies: about 0.8 ms on bare metal at TP8, about
13 ms each under gVisor, where they turn a 8.4 ms bs1 decode step into 18 ms.

With the switch on, ``check_hicache_events`` instead

1. waits for the vector started at the previous step and applies it (the
   checks first, then write acks, load acks, storage drains, backfill rows),
2. runs the rest of the step's HiCache housekeeping as before, and
3. snapshots this step's local values and starts their MIN reduction with
   ``async_op=True`` on a dedicated Gloo group,

so the collective overlaps the rest of the step (scheduling, the forward
launch, result processing, the request broadcast) instead of blocking it.
The group is created once at ``init_hicache`` on every rank. Collectives of a
Gloo group run in issue order, so sharing the TP CPU group (which, in plain TP,
is also the request-broadcast group) would put the next step's broadcast
behind the in-flight reduce.

Everything a snapshot decides is applied at the same later step on every rank
(the step-sync decisions are rank-uniform), so the lag is uniform. Per
consumer:

* write acks (write-through, window SWA spills): settled one step later. The
  device pages stay locked (write-through) or detached (spill) one step longer;
  a finishing request re-attaches its unacked spills (``reattach_swa_spills``)
  and a retract decision drains them synchronously (``drain_swa_spills``).
* load acks: only the locks and the load-back pin are released one step later;
  the forward waits on the load stream's per-layer events, not on this.
* storage control queues (L3 prefetch hits and acks, backups, host releases):
  drained one step later, so an L3 prefetch completes one step later.
* the TP digests (write-back reclaim victims, structural, backfill job list):
  every rank checks the same snapshot, so a divergence is still caught, one
  step later. The vector has a fixed size, so diverged ranks raise at the
  apply instead of reducing misaligned rows or hanging.
* the idle-sync in-flight count and the GPU-export work flag: one step older.
* backfill rows: a job advances by the row of the snapshot only if it is still
  in the phase and round the snapshot saw (a submit or cancel between the two
  can launch a lookup or finish a job; both run at lockstep points, so every
  rank skips the same rows). Jobs beyond the fixed slot count wait a step.

Acks are matched by identity, so a synchronous ``writing_check`` /
``loading_check`` that settled some of them in between (spill drain before a
retract, the next-batch host prefetch) is not settled twice. Anything that
drains or replaces the storage queues out of band bumps
``_pipelined_sync_generation`` and the snapshot is dropped; a reset or the
synchronous storage drain before a clear waits for the in-flight collective
and drops it first.

Plain TP only: the PP and DP-attention paths keep the per-step reduction, as
does a cache whose reduction group is not the whole world.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

ALGOS = ("allreduce", "alltoall")

# Same timeout as the TP CPU groups (parallel_state.GroupCoordinator).
_GROUP_TIMEOUT = datetime.timedelta(seconds=120 * 60)


@dataclass
class PipelinedSyncLayout:
    """Offsets of the fields of one step-sync vector. Every presence flag is
    configuration, identical on every rank, so every rank reduces a vector of
    the same size."""

    num_storage: int = 0  # 4 + number of extra release queues, or 0
    has_export: bool = False
    has_peer: bool = False
    track_idle: bool = False
    backfill_fields: int = 0  # header + slots * row width, or 0

    WRITE = 0
    LOAD = 1
    STORAGE = 2

    @property
    def export(self) -> int:
        return self.STORAGE + self.num_storage

    @property
    def reclaim_digest(self) -> int:
        return self.peer + int(self.has_peer)

    @property
    def peer(self) -> int:
        """Optional peer control work bit, reduced with existing cache counts."""
        return self.export + int(self.has_export)

    @property
    def structural_digest(self) -> int:
        return self.reclaim_digest + 2

    @property
    def idle(self) -> int:
        return self.structural_digest + 2

    @property
    def backfill(self) -> int:
        return self.idle + int(self.track_idle)

    @property
    def size(self) -> int:
        return self.backfill + self.backfill_fields


@dataclass
class PendingStepSync:
    """One started step-sync reduction and the local snapshot it decides."""

    work: Any  # .wait() -> the reduction is in ``vec`` (None: nothing to wait)
    vec: torch.Tensor
    layout: PipelinedSyncLayout
    generation: int
    # This rank's ready ack prefixes; the reduced count settles their head.
    write_acks: list = field(default_factory=list)
    load_acks: list = field(default_factory=list)
    # The storage control queues counted, in vector order, and the extra
    # release pools' names (the queues after the first four).
    storage_queues: tuple = ()
    extra_pool_names: tuple = ()
    # SwaBackfillManager.lockstep_rows snapshot, or None without backfill.
    backfill: Optional[list] = None

    def wait(self) -> list[int]:
        if self.work is not None:
            self.work.wait()
            self.work = None
        return [int(v) for v in self.vec.tolist()]


class AllToAllMinWork:
    """MIN reduction as one all-to-all: every rank sends its whole vector to
    every rank (one message latency instead of the ring's 2(N-1)) and takes
    the element-wise MIN of the N vectors it received, identical on every
    rank."""

    def __init__(self, vec: torch.Tensor, group, world_size: int):
        self._vec = vec
        self._world = world_size
        # Both buffers stay referenced until wait(): the collective runs on a
        # Gloo worker thread.
        self._send = vec.repeat(world_size)
        self._recv = torch.empty_like(self._send)
        self._work = torch.distributed.all_to_all_single(
            self._recv, self._send, group=group, async_op=True
        )

    def wait(self) -> None:
        self._work.wait()
        self._vec.copy_(self._recv.view(self._world, -1).min(dim=0).values)


def start_min_reduce(vec: torch.Tensor, group, world_size: int, algo: str):
    """Start the MIN reduction of ``vec`` (in place) on ``group``; returns an
    object whose ``wait()`` completes it."""
    if algo == "alltoall":
        return AllToAllMinWork(vec, group, world_size)
    return torch.distributed.all_reduce(
        vec, op=torch.distributed.ReduceOp.MIN, group=group, async_op=True
    )


def reduction_group_or_reason(groups, tp_group, tp_world_size: int):
    """The one process group UnifiedRadixCache._all_reduce_attn_groups reduces
    over, or (None, why) when the pipelined sync cannot replace it with a
    dedicated group created on every rank of the world."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return None, "torch.distributed is not initialized"
    active = [g for g in groups if g is not None and dist.get_world_size(group=g) > 1]
    if not active and tp_world_size > 1:
        active = [tp_group]
    if not active:
        return None, "no TP peers"
    if len(active) > 1:
        return None, "attention CP and TP both span several ranks"
    group = active[0]
    ranks = dist.get_process_group_ranks(group)
    if len(ranks) != dist.get_world_size():
        # DP attention (or another layout with several such groups): every
        # rank would have to create every group, in the same order.
        return None, f"reduction group {ranks} is not the whole world"
    try:
        backend = dist.get_backend(group)
    except Exception:  # noqa: BLE001 - unknown backends keep the old path
        backend = None
    if backend != "gloo":
        return None, f"reduction group backend {backend!r} is not gloo"
    return group, None


def create_pipelined_sync_group(groups, tp_group, tp_world_size: int):
    """Create the dedicated Gloo group (a collective over the whole world, so
    every rank calls this at the same point). Returns (group, world_size) or
    (None, why)."""
    group, why = reduction_group_or_reason(groups, tp_group, tp_world_size)
    if group is None:
        return None, why
    ranks = torch.distributed.get_process_group_ranks(group)
    new = torch.distributed.new_group(
        ranks=ranks, backend="gloo", timeout=_GROUP_TIMEOUT
    )
    return new, len(ranks)

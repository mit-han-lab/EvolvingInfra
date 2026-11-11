"""Per-op observability of PD decode HiCache restores (load-then-admit).

A decode worker restores a request's host-tier prefix before the request joins
the running batch (``DecodeHiCacheTransferMixin``): no decode forward consumes
the load, and the request is admitted only after every TP rank has the rows.
Whether a restore op also delays the decode steps of the requests already
running depends on the host-dedup fan-out:

- ``nccl`` (DSA and DeepSeek-V4 dedup), ``nccl_symm``, ``ipc``, and ``shared``
  without its loader thread make the caller's stream (the scheduler stream)
  wait for the whole op, and every later forward waits for that stream: the
  next decode step of every running request waits for the restore.
- ``ce_peer`` (C2g) and ``shared`` with its loader thread return without a
  stream wait; ``none`` (no dedup) loads each rank's own host pool and does not
  wait either.

TP rank 0 logs one line when a restore op is issued and one when this rank
observes it done (``SGLANG_DECODE_HICACHE_RESTORE_LOG``, default on; it only
runs on a decode worker with decode radix + HiCache)::

    decode_hicache_restore start op=7 slot=1 reqs=2 tokens=32768 pages=512 mb=2031.6 fanout=nccl stalls_fwd=1
    decode_hicache_restore done op=7 slot=1 reqs=2 tokens=32768 pages=512 fanout=nccl copy_ms=41.2 wall_ms=48.7 stream_wait_ms=44.9

``tokens``/``pages``/``mb`` are the op's host->device rows and bytes (all
pools: target + MTP draft KV and indexer). ``copy_ms`` is the op's GPU time
between its start and finish events (after the load fence). ``wall_ms`` is
CPU time from issue until this rank's scheduler saw the op finished (tick
granular). ``stream_wait_ms`` is the GPU time between two events recorded on
the caller's stream around the issue: with a stalling fan-out it is the time
that stream (and so the next decode forward) waited for the op, ~0 otherwise.
A value is ``na`` when the device has no timing events.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "decode_hicache_restore"
STALL_WARNING = (
    "PD decode HiCache: decode load-back blocks the forward stream; expect ITL "
    "spikes (host-dedup fan-out=%s: each host->device restore op makes the "
    "scheduler stream, and so the next decode forward of every running "
    "request, wait for the whole op). SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer "
    "avoids the wait where it is available."
)


def restore_fanout(cache_controller: Any) -> Tuple[str, bool]:
    """(fan-out name, whether a restore op makes the caller's stream wait)."""
    dedup = getattr(cache_controller, "host_dedup", None)
    if dedup is None:
        dedup = getattr(
            getattr(cache_controller, "mem_pool_host", None), "host_dedup", None
        )
    if dedup is None:
        return "none", False
    has_loader = getattr(dedup, "loader", None) is not None
    if getattr(dedup, "ce_peer", False):
        return "ce_peer", not has_loader
    if getattr(dedup, "shared", False):
        return "shared", not has_loader
    if getattr(dedup, "ipc_fanout", None) is not None:
        return "ipc", True
    if getattr(dedup, "_symm_handle", None) is not None:
        return "nccl_symm", True
    return "nccl", True


def warn_if_stalling_restore_fanout(tree_cache: Any) -> Optional[str]:
    """One WARNING at decode startup when restores stall the forward stream.

    Returns the fan-out name when it warned, else None."""
    cache_controller = getattr(tree_cache, "cache_controller", None)
    if cache_controller is None:
        return None
    fanout, stalls = restore_fanout(cache_controller)
    if not stalls:
        return None
    logger.warning(STALL_WARNING, fanout)
    return fanout


def _fmt_ms(value: Optional[float]) -> str:
    return "na" if value is None else f"{value:.1f}"


@dataclass
class _RestoreOp:
    op: int
    slot: int
    reqs: int
    tokens: int
    pages: int
    fanout: str
    ack: Any
    t_issue: float
    pre: Any = None
    post: Any = None


class DecodeRestoreOpLog:
    """TP0's start/done lines for decode HiCache restore ops.

    Never blocks: completion is polled with ``query()`` on the op's own finish
    event, so a reused producer slot cannot be mistaken for this op.
    """

    def __init__(self, *, enabled: bool, page_size: int, device_module: Any = None):
        self.enabled = enabled
        self.page_size = max(1, int(page_size or 1))
        self.device_module = device_module
        self.pending: List[_RestoreOp] = []
        self.next_op = 0
        self.fanout: Optional[str] = None
        self.stalls_fwd = False

    def _timing_event(self):
        dm = self.device_module
        if dm is None:
            return None
        try:
            event = dm.Event(enable_timing=True)
            event.record()
            return event
        except Exception:  # noqa: BLE001 - observability must not fail serving
            return None

    def before_issue(self) -> Any:
        """Event on the caller's stream just before ``ready_to_load_host_cache``."""
        if not self.enabled:
            return None
        return self._timing_event()

    def after_issue(
        self, cache_controller: Any, slot: int, num_reqs: int, pre: Any
    ) -> None:
        if not self.enabled or slot is None or slot < 0:
            return
        post = self._timing_event() if pre is not None else None
        if self.fanout is None:
            self.fanout, self.stalls_fwd = restore_fanout(cache_controller)
        acks = getattr(cache_controller, "ack_load_queue", None) or ()
        ack = acks[-1] if acks else None
        tokens = int(getattr(ack, "num_tokens", 0) or 0)
        num_bytes = int(getattr(ack, "num_bytes", 0) or 0)
        rec = _RestoreOp(
            op=self.next_op,
            slot=int(slot),
            reqs=int(num_reqs),
            tokens=tokens,
            pages=tokens // self.page_size,
            fanout=self.fanout,
            ack=ack,
            t_issue=time.monotonic(),
            pre=pre,
            post=post,
        )
        self.next_op += 1
        self.pending.append(rec)
        logger.info(
            "%s start op=%d slot=%d reqs=%d tokens=%d pages=%d mb=%.1f "
            "fanout=%s stalls_fwd=%d",
            LOG_PREFIX,
            rec.op,
            rec.slot,
            rec.reqs,
            rec.tokens,
            rec.pages,
            num_bytes / 1e6,
            rec.fanout,
            int(self.stalls_fwd),
        )

    @staticmethod
    def _done(event: Any) -> bool:
        return event is None or bool(event.query())

    @staticmethod
    def _elapsed_ms(start: Any, end: Any) -> Optional[float]:
        if start is None or end is None:
            return None
        try:
            return float(start.elapsed_time(end))
        except Exception:  # noqa: BLE001 - no timing on this device/event
            return None

    def poll(self) -> None:
        """Log every pending op that finished on this rank (non-blocking)."""
        if not self.pending:
            return
        still: List[_RestoreOp] = []
        for rec in self.pending:
            try:
                finish = getattr(rec.ack, "finish_event", None)
                if not (self._done(finish) and self._done(rec.post)):
                    still.append(rec)
                    continue
                copy_ms = (
                    self._elapsed_ms(rec.ack.start_event, finish)
                    if getattr(rec.ack, "timing_enabled", False)
                    else None
                )
                wait_ms = self._elapsed_ms(rec.pre, rec.post)
            except Exception as e:  # noqa: BLE001 - the load path reports errors
                logger.warning(
                    "%s op=%d: dropping its record (%s)", LOG_PREFIX, rec.op, e
                )
                continue
            logger.info(
                "%s done op=%d slot=%d reqs=%d tokens=%d pages=%d fanout=%s "
                "copy_ms=%s wall_ms=%.1f stream_wait_ms=%s",
                LOG_PREFIX,
                rec.op,
                rec.slot,
                rec.reqs,
                rec.tokens,
                rec.pages,
                rec.fanout,
                _fmt_ms(copy_ms),
                (time.monotonic() - rec.t_issue) * 1e3,
                _fmt_ms(wait_ms),
            )
        self.pending = still

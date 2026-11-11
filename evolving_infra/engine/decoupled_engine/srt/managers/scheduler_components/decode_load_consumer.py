"""SGLANG_HICACHE_DECODE_CLEAR_CONSUMER: a decode pass drops the HiCache load
consumers that its batch object still carries from an earlier pass.

The defect (it exists without this flag). Only a new prefill batch gets a load
consumer:

- ``hicache_consumer_index``: the ring slot of the host-to-device loads that
  its admissions queued (``Scheduler._get_new_batch_prefill_raw``);
- ``hicache_extra_waits``: the in-flight bootstrap-prefetch ops that its
  requests adopted (``BootstrapPrefetcher._attach_consume_waits``).

When the running batch is empty after that pass, ``get_next_batch_to_run``
makes the extend batch object itself the running batch
(``running_batch = last_batch``), and nothing resets the two fields. Until the
running batch empties again, every decode pass on that object launches as a
consumer:

- ``TpModelWorker.forward_batch_generation`` calls ``set_consumer(stale
  slot)``. ``LayerDoneCounter.set_consumer`` runs ``wait_gate`` on whatever
  load reuses that ring slot now (whole-op slot: a CPU wait for the loader's
  submission and a stream wait on the load's last event). An eager (non-graph)
  forward also waits per layer in ``wait_until``. The extra waits are waited
  again at launch.
- The gate-pending reorder (``gate_pending_for_batch``) can fire on such a
  decode pass, and then that rank alone processes the previous result before
  the launch.
- With SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH, the import walk runs before such a
  launch (its lockstep superset counts any consumer), so the old wait stays.

Why no decode pass needs a load wait.

- Loads go to the batch that admits their requests. The extend forward that
  consumes them waits for them on the forward stream: ``set_consumer`` at
  launch, per layer in ``wait_until``, and ``wait_consumed_loads``. Its last
  layer waits for the last part, which the loader records after the whole op.
- Every later forward runs on the same forward stream, so it is ordered after
  the loads.
- A request joins a decode batch only after its extend forward, or as a
  PREBUILT import. An import that restores host KV is held in ``pop_ready``
  until the load's ack, and its PREBUILT batch is built with no consumer.
- Decode formation queues no load. A CPU test scans the package for any new
  site that gives a batch a consumer.

The fix. ``Scheduler.update_running_batch`` resets both fields right before
``prepare_for_decode``. The reset reads only the batch, which is the same on
every TP rank, so every rank clears alike. Consumers attached after formation
in the same pass (none for a decode batch today) are kept.

Default off. With the flag off, the path is unchanged for every existing
baseline. The flag is a no-op on standard P/D workers: a D worker never forms
an extend batch with a consumer, and a P worker runs no decode pass.
SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH requires this flag (startup refusal).

Logs. Every TP rank logs one startup line, then a line at the 1st, 2nd, 4th,
8th, ... clear:
    DECODE_CLEAR_CONSUMER rank=<r> on
    DECODE_CLEAR_CONSUMER rank=<r> cleared=N index=<i> extra_waits=<n>
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

STATS: Counter = Counter()
LOG_TAG = "DECODE_CLEAR_CONSUMER"


def log_enabled(tp_rank: int) -> None:
    """Startup (dispatch_event_loop), when Scheduler.decode_clears_load_consumer."""
    logger.info("%s rank=%d on", LOG_TAG, tp_rank)


def clear_load_consumer(batch: Any, tp_rank: int = 0) -> bool:
    """At decode formation (``update_running_batch``, before
    ``prepare_for_decode``): reset the load consumer that ``batch`` carries
    from an earlier pass. Returns True if there was one."""
    idx = getattr(batch, "hicache_consumer_index", -1)
    waits = getattr(batch, "hicache_extra_waits", None)
    stale_idx = idx is not None and idx >= 0
    if not stale_idx and not waits:
        return False
    batch.hicache_consumer_index = -1
    if waits:
        # A new list: the bootstrap prefetcher may still hold the old one.
        batch.hicache_extra_waits = []
    STATS["cleared"] += 1
    n = STATS["cleared"]
    if n & (n - 1) == 0:
        logger.info(
            "%s rank=%d cleared=%d index=%s extra_waits=%d",
            LOG_TAG,
            tp_rank,
            n,
            idx,
            len(waits or ()),
        )
    return True

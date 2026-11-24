# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Symmetric P/D: a P-leg that finds every export aux slot of its rank taken waits for one.

A rank's aux slots are held by its exports until their readers are done, so a rank whose exports
wait for a busy decode worker runs out of them. A P-leg rejected then fails while its D-leg keeps
waiting for KV that never comes, which stalls a closed-loop session for the rest of a replay.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="needs the engine (decoupled-engine)")
pytest.importorskip("decoupled_engine", reason="needs the engine (decoupled-engine)")

from decoupled_engine.srt.disaggregation.pull.api import ExportMark  # noqa: E402
from decoupled_engine.srt.disaggregation.symmetric.export_table import ExportTable  # noqa: E402
from decoupled_engine.srt.disaggregation.symmetric.fake_planes import FakeOwnerPlane  # noqa: E402
from decoupled_engine.srt.disaggregation.symmetric.scheduler_mixin import SchedulerSymmetricMixin  # noqa: E402
from decoupled_engine.srt.disaggregation.utils import ReqToMetadataIdxAllocator  # noqa: E402
from decoupled_engine.srt.managers.io_struct import AbortReq  # noqa: E402


class _Req:
    """The request fields the symmetric admission path reads."""

    def __init__(self, n):
        self.rid = f"r{n}"
        self.transfer_id = 1000 + n
        self.export_mark = ExportMark(transfer_id=self.transfer_id, reader_worker="d", reader_addr="tcp://d:1")
        self.import_mark = None
        self.metadata_buffer_index = -1
        self.inflight_middle_chunks = 0
        self.swa_branching_seqlen = None
        self.time_stats = SimpleNamespace(set_wait_queue_entry_time=lambda: None)

    def finished(self):
        return False


class _NoImports:
    defer_admission = False

    def pop_ready(self, max_ready):
        return [], []

    def admit_pending(self):
        pass

    def abort_matching(self, rid, abort_all):
        return []

    def __len__(self):
        return 0


class _Sched(SchedulerSymmetricMixin):
    """A P rank: a real ExportTable over a fake owner plane with `aux_slots` aux slots."""

    def __init__(self, aux_slots):
        self.owner = FakeOwnerPlane()
        self.export_table = ExportTable(
            self.owner,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            metadata_buffers=None,
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(aux_slots),
            page_size=1,
        )
        self.import_queue = _NoImports()
        self.tree_cache = None
        self.chunked_req = None
        self.req_to_token_pool = SimpleNamespace(size=64)
        self.max_running_requests = 64
        self.waiting_queue = []
        self.rejected, self.freed, self.sent = [], [], []
        self.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=lambda out, req: self.sent.append(out))
        )

    def symmetric_reject(self, req, error_msg):
        self.rejected.append((req.rid, error_msg))

    def _symmetric_free_exported(self, req, reason):
        self.freed.append(req.rid)

    def _abort_on_queued_limit(self, req):
        return False

    def _symmetric_materialize_images(self, req):
        return True

    def _prefetch_kvcache(self, req):
        pass

    def arrive(self, *reqs):
        for req in reqs:
            self.symmetric_add_request(req)

    def tick(self):
        self.symmetric_pre_schedule(SimpleNamespace(batch_size=lambda: 0, reqs=[]))

    def queued(self):
        return [r.rid for r in self.waiting_queue]


def test_a_p_leg_finding_every_aux_slot_taken_waits_and_starts_after_a_release():
    sched = _Sched(aux_slots=2)
    r0, r1, r2 = _Req(0), _Req(1), _Req(2)
    sched.arrive(r0, r1, r2)
    assert sched.rejected == []
    assert sched.queued() == ["r0", "r1"]
    assert sched.export_table.aux_waiting == 1 and r2.transfer_id not in sched.export_table
    sched.tick()  # no release: r2 keeps waiting
    assert sched.queued() == ["r0", "r1"] and sched.export_table.aux_waiting == 1
    sched.owner.release(r0.transfer_id)  # r0's reader is done: its slot frees in the next pass
    sched.tick()
    assert sched.freed == ["r0"]
    assert sched.queued() == ["r0", "r1", "r2"] and r2.transfer_id in sched.export_table
    assert sched.export_table.aux_waiting == 0 and sched.export_table.stats_aux_waited == 1


def test_waiting_p_legs_take_free_slots_in_arrival_order():
    sched = _Sched(aux_slots=1)
    r0, r1, r2, r3 = _Req(0), _Req(1), _Req(2), _Req(3)
    sched.arrive(r0, r1, r2)
    sched.export_table.abort_export(r0, "aborted by client")  # frees r0's slot between two passes
    assert len(sched.export_table) == 0 and not sched.symmetric_is_idle()
    sched.arrive(r3)  # a free slot, but r1 and r2 wait ahead of r3
    assert sched.queued() == ["r0"] and sched.export_table.aux_waiting == 3
    sched.tick()
    assert sched.queued() == ["r0", "r1"] and sched.export_table.aux_waiting == 2
    assert sched.rejected == []


def test_an_abort_removes_a_waiting_p_leg(monkeypatch):
    import decoupled_engine.srt.managers.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "_make_abort_req", lambda req: ("abort", req.rid))
    sched = _Sched(aux_slots=1)
    r0, r1 = _Req(0), _Req(1)
    sched.arrive(r0, r1)
    sched.symmetric_abort_queued(AbortReq(rid="r1"))
    assert sched.sent == [("abort", "r1")] and sched.export_table.aux_waiting == 0
    sched.owner.release(r0.transfer_id)
    sched.tick()
    assert sched.queued() == ["r0"] and r1.transfer_id not in sched.export_table


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

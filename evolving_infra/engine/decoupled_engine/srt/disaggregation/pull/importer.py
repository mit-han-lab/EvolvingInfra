"""ReaderPlane: the D-leg side of the pull plane (api.ReaderPlane).

This worker is the Reader importing KV from the Owner.

Per import: consume CHUNK_READY, map the owner's page indices to remote
addresses with the owner's Registration (exchanged once per owner channel) and
the reader's own indices from ``dst_pages``, issue one blocking batch_read per
chunk on a transfer thread, verify generations on FINAL, read the owner's aux +
state slots (FinalMeta) into the reader's slots the scheduler pre-allocated on
the ImportMark (``local_aux_index`` / ``local_state_indices``), send DONE, set
READY. Homogeneous TP only: rank i reads rank i.

Threading: the scheduler thread calls begin_import / poll / final_meta /
failure_reason / abort_import. One daemon IO thread owns the ZMQ sockets and
runs the protocol state machine; a small thread pool runs the blocking READs.
``dst_pages`` is invoked on the IO thread and must be a read-only lookup of
pages the scheduler allocated before begin_import.

Late D (disaggregation/late_d.py): the scheduler of
reader rank 0 attaches a late inbox (``set_late_inbox``). The IO thread then
turns an owner's ADMIT into an ``admit`` control, a stage-2 RELEASE into a
``release`` control, and an owner ABORT for a transfer without an import (a
never-admitted late entry) into a ``fail`` control; the ABORT is still
parked as before. Without an inbox (late D off, or ranks other than 0) ADMIT
and RELEASE are dropped with a warning and nothing else changes.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import zmq

from decoupled_engine.srt.disaggregation.pull import protocol as wire
from decoupled_engine.srt.disaggregation.pull.api import (
    ChunkReady,
    FinalMeta,
    ImportMark,
    ReadEngine,
    ReadNotQuiescentError,
    TransferState,
)
from decoupled_engine.srt.disaggregation.pull.protocol import AttemptGate, Registration
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.scheduler_components.phase_timers import phase

logger = logging.getLogger(__name__)

# IO-thread poll/flush interval; see envs.SGLANG_SYMMETRIC_PD_PLANE_POLL_MS.
_POLL_INTERVAL_MS = max(1, int(envs.SGLANG_SYMMETRIC_PD_PLANE_POLL_MS.get()))
_REGISTER_RETRY_S = 1.0
_SEND_HWM = 10_000
# Chunks a single owner may publish ahead of its REGISTER before we give up.
_MAX_PENDING_PER_OWNER = 4096
# Publications that arrive before the scheduler's begin_import() for their
# transfer_id (P-leg finished first, e.g. on a radix hit:
# publication before begin_import is buffered). Parked per transfer and
# replayed on begin_import; bounded and expired so a leg that never arrives
# cannot pin memory.
_MAX_EARLY_TRANSFERS = 4096
_MAX_EARLY_MSGS_PER_TRANSFER = 4096
# Remember exact completed attempts independently of the shorter diagnostic
# history. No timer releases resources: eviction only ends late-message
# suppression, and an unknown transfer still cannot issue a READ.
_MAX_TERMINAL_ATTEMPTS = 4096

REASON_TIMEOUT = "timeout"
REASON_OWNER_EVICTED = "owner_evicted"
REASON_READ_FAILED = "read_failed"
REASON_LAYOUT = "layout_mismatch"
REASON_INCOMPLETE = "incomplete"

_ACTIVE = (TransferState.WAITING, TransferState.TRANSFERRING)


class _Import:
    __slots__ = (
        "mark",
        "req_id",
        "dst_pages",
        "state",
        "reason",
        "deadline_mono",
        "announced",
        "read_done",
        "last_seq",
        "max_token_end",
        "reusable_prefix_tokens",
        "final",
        "final_gens",
        "finalizing",
        "inflight",
        "pending_terminal",
        "finished",
        "lock",
        "timeline",
    )

    def __init__(
        self,
        *,
        mark: ImportMark,
        req_id: str,
        dst_pages: Callable[[ChunkReady], Dict[str, Sequence[int]]],
        deadline_mono: float,
    ) -> None:
        self.mark = mark
        self.req_id = req_id
        self.dst_pages = dst_pages
        self.state = TransferState.WAITING
        self.reason: Optional[str] = None
        self.deadline_mono = deadline_mono
        # {pool: {owner page: generation}} as announced by CHUNK_READY.
        self.announced: Dict[str, Dict[int, int]] = {}
        self.read_done: Set[int] = set()
        self.last_seq: Optional[int] = None
        self.max_token_end = 0
        self.reusable_prefix_tokens = 0
        self.final: Optional[FinalMeta] = None
        self.final_gens: Optional[Dict[str, Dict[int, int]]] = None
        self.finalizing = False
        self.inflight = 0
        self.pending_terminal = None
        self.finished = threading.Event()
        self.lock = threading.Lock()
        # Wall-clock telemetry (time.time()) for M0 checks, see snapshot().
        self.timeline: Dict[str, Any] = {
            "req_id": req_id,
            "owner_addr": mark.owner_addr,
            "begin_at": time.time(),
            "first_chunk_recv_at": None,
            "first_read_start_at": None,
            # [chunk_seq, token_start, token_end, is_last, nbytes, read_start, read_end, rc]
            "reads": [],
            "final_recv_at": None,
            "final_read_at": None,
            "ready_at": None,
            "state": None,
            "reason": None,
        }


_SNAPSHOT_KEEP = 256


class ReaderPlane:
    """api.ReaderPlane over ZMQ + api.ReadEngine.

    The reader's pre-allocated aux slot and per-state-type slots arrive on
    ``ImportMark.local_aux_index`` / ``ImportMark.local_state_indices`` (filled
    by the scheduler before begin_import); FINAL's owner slots are READ into
    them. ``final_meta`` returns the OWNER's FinalMeta as received on the wire.
    """

    def __init__(
        self,
        *,
        registration: Registration,
        read_engine: ReadEngine,
        bind_addr: str,
        advertised_addr: Optional[str] = None,
        zmq_ctx: Optional[zmq.Context] = None,
        waiting_timeout_s: Optional[float] = None,
        num_transfer_threads: int = 4,
        worker_name: str = "",
        send_begin: Optional[bool] = None,
    ) -> None:
        self._registration = registration
        self._engine = read_engine
        # Reader-first exports (SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER): tell the
        # owner when the scheduler admitted the import (wire BEGIN).
        self._send_begin = (
            bool(envs.SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER.get())
            if send_begin is None
            else bool(send_begin)
        )
        self.stats_begin_sent = 0
        # Late D, reader rank 0 (set_late_inbox): the scheduler's control
        # inbox (late_d.LateDInbox). None: ADMIT is dropped, no FAIL control.
        self._late_inbox: Optional[Any] = None
        self.stats_admit_received = 0
        self.stats_admit_dropped = 0
        self.stats_fail_forwarded = 0
        # Owner channels whose REGISTER named another late-D mode: logged
        # once; the late-D reader ends their entries (late_d_mismatched_owners).
        self._late_d_mismatch_logged: Set[str] = set()
        self._late_d_mismatched: Set[str] = set()
        self._bind_addr = bind_addr
        self.advertised_addr = (
            advertised_addr if advertised_addr is not None else bind_addr
        )
        self._ctx = zmq_ctx if zmq_ctx is not None else zmq.Context.instance()
        self._timeout_s = (
            float(envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get())
            if waiting_timeout_s is None
            else float(waiting_timeout_s)
        )
        self._worker_name = worker_name

        self._mutex = threading.Lock()
        self._imports: Dict[int, _Import] = {}
        self._gate = AttemptGate()
        # owner_addr -> Registration once REGISTER arrived; None while requested.
        self._owners: Dict[str, Optional[Registration]] = {}
        self._register_sent_mono: Dict[str, float] = {}
        # Publications that overtook REGISTER on a fresh owner channel, replayed
        # in arrival order once the blob lands (IO thread only).
        self._pending_by_owner: Dict[str, List[wire.Message]] = {}
        # transfer_id -> (first-seen monotonic, messages) for publications that
        # overtook begin_import (guarded by _mutex; replayed on the IO thread).
        self._early: Dict[int, Tuple[float, List[wire.Message]]] = {}
        self._terminal_attempts: OrderedDict[Tuple[int, int], None] = OrderedDict()
        self.stats_terminal_dropped = 0
        self._replay: queue.Queue[wire.Message] = queue.Queue()
        self.stats_early_parked = 0
        self.stats_early_replayed = 0
        self.stats_early_expired = 0
        self._outbound: queue.Queue[Tuple[str, bytes]] = queue.Queue()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, num_transfer_threads),
            thread_name_prefix="pull-reader-xfer",
        )
        self.stats_send_dropped = 0
        self.stats_tier_served: Dict[str, int] = {}
        self._history: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.stats_ready = 0
        self.stats_failed = 0
        self.stats_bytes_read = 0

        self._stop = threading.Event()
        self._started = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._io_loop, daemon=True, name="pull-reader-io"
        )
        self._thread.start()
        if not self._started.wait(timeout=30):
            self._stop.set()
            raise RuntimeError(f"ReaderPlane channel startup timed out on {bind_addr}")
        if self._startup_error is not None:
            raise RuntimeError(
                f"ReaderPlane failed to bind {bind_addr}"
            ) from self._startup_error

    # ------------------------------------------------------------------
    # api.ReaderPlane (scheduler thread)
    # ------------------------------------------------------------------

    def begin_import(
        self,
        mark: ImportMark,
        req_id: str,
        dst_pages: Callable[[ChunkReady], Dict[str, Sequence[int]]],
    ) -> None:
        if mark.local_aux_index is None:
            raise ValueError(
                f"begin_import({req_id}, transfer {mark.transfer_id}): "
                "ImportMark.local_aux_index not allocated by the scheduler"
            )
        imp = _Import(
            mark=mark,
            req_id=req_id,
            dst_pages=dst_pages,
            deadline_mono=time.monotonic() + self._timeout_s,
        )
        with self._mutex:
            self._bind_mark_attempt(mark)
            self._imports[mark.transfer_id] = imp
            if mark.owner_addr not in self._owners:
                self._owners[mark.owner_addr] = None
            early = self._early.pop(mark.transfer_id, None)
            if early is not None:
                self._bind_early_attempt(mark.transfer_id, early[1])
            attempt = self._gate.attempt_of(mark.transfer_id)
        if self._send_begin:
            # The destination pages are allocated: the owner may queue the
            # P-leg now (reader-first exports).
            if envs.SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION.get() and attempt:
                imp.reusable_prefix_tokens = wire.begin_reuse_prefix(
                    {"reusable_prefix_tokens": mark.local_reuse_prefix_tokens})
            begin = (wire.make_begin(transfer_id=mark.transfer_id, attempt_id=attempt or 0,
                                     reusable_prefix_tokens=imp.reusable_prefix_tokens)
                     if imp.reusable_prefix_tokens else
                     wire.make_begin(transfer_id=mark.transfer_id, attempt_id=attempt or 0))
            self._send(mark.owner_addr, begin)
            self.stats_begin_sent += 1
        # The IO thread sends REGISTER_REQ for owners whose blob is still None.
        if early is not None:
            # Publications that beat us here: hand them to the IO thread in
            # arrival order (dispatch must not run on the scheduler thread).
            for msg in early[1]:
                self._replay.put(msg)
            self.stats_early_replayed += len(early[1])

    def poll(self, transfer_id: int) -> TransferState:
        with self._mutex:
            imp = self._imports.get(transfer_id)
        if imp is None:
            return TransferState.FAILED
        with imp.lock:
            if imp.state not in _ACTIVE and not imp.finished.is_set():
                return TransferState.TRANSFERRING
            return imp.state

    def final_meta(self, transfer_id: int) -> Optional[FinalMeta]:
        """The OWNER's FinalMeta (wire FINAL) once READY; None before. The
        reader's own slots are ``mark.local_aux_index`` / ``local_state_indices``."""
        with self._mutex:
            imp = self._imports.get(transfer_id)
        if imp is None:
            return None
        with imp.lock:
            if imp.state != TransferState.READY:
                return None
            return imp.final

    def failure_reason(self, transfer_id: int) -> Optional[str]:
        with self._mutex:
            imp = self._imports.get(transfer_id)
        if imp is None:
            return "unknown_transfer"
        with imp.lock:
            return imp.reason

    def abort_import(self, transfer_id: int, reason: str) -> None:
        with self._mutex:
            imp = self._imports.get(transfer_id)
        if imp is None:
            return
        self._request_terminal(imp, TransferState.ABORTED, reason)
        # The scheduler frees destination pages when this call returns. This
        # exceptional path must drain READs, including queued aux operations.
        # A stuck transport fails closed; it must never permit page reuse.
        if not imp.finished.wait(timeout=self._timeout_s):
            raise RuntimeError(f"Import {transfer_id} failed to quiesce after abort")

    def abort_pending(self, mark: ImportMark, reason: str) -> None:
        """Cancel before preallocation; no destination pages or READs exist."""
        self._end_pending(mark, reason, notify_owner=True)

    def release_pending(self, mark: ImportMark, reason: str) -> None:
        """Late D: end a never-admitted import without telling the owner
        (stage-2 RELEASE; an owner ABORT already ended it). As
        abort_pending otherwise: the terminal record drops late
        publications, and parked ones go at once on every rank."""
        self._end_pending(mark, reason, notify_owner=False)

    def _end_pending(
        self, mark: ImportMark, reason: str, *, notify_owner: bool
    ) -> None:
        with self._mutex:
            if mark.transfer_id in self._imports:
                raise RuntimeError("abort_pending called for an active import")
            self._bind_mark_attempt(mark)
            early = self._early.pop(mark.transfer_id, None)
            if early is not None:
                self._bind_early_attempt(mark.transfer_id, early[1])
        imp = _Import(
            mark=mark,
            req_id=f"pending:{mark.transfer_id}",
            dst_pages=lambda chunk: {},
            deadline_mono=time.monotonic(),
        )
        # Install a terminal record before late publications can be admitted.
        self._request_terminal(
            imp, TransferState.ABORTED, reason, notify_owner=notify_owner
        )
        with self._mutex:
            self._early.pop(mark.transfer_id, None)
            self._imports[mark.transfer_id] = imp
        self.forget(mark.transfer_id)

    def set_late_inbox(self, inbox: Optional[Any]) -> None:
        """Late D, reader rank 0: the inbox (late_d.LateDInbox) that receives
        the ``admit`` and ``fail`` controls of the IO thread."""
        with self._mutex:
            self._late_inbox = inbox

    def request_register(self, owner_addr: str) -> None:
        """Late D: register with ``owner_addr`` now (the IO thread sends
        REGISTER_REQ), before any begin_import: a DEFERRED entry may never
        reach it, and its owner's late-D mode must be known."""
        with self._mutex:
            if owner_addr not in self._owners:
                self._owners[owner_addr] = None

    def late_d_mismatched_owners(self) -> Set[str]:
        """Owner channels whose REGISTER named another late-D mode."""
        with self._mutex:
            return set(self._late_d_mismatched)

    def forget(self, transfer_id: int) -> None:
        """Drop a concluded import (scheduler calls after consuming its result)."""
        with self._mutex:
            imp = self._imports.get(transfer_id)
            if imp is not None and not imp.finished.is_set():
                raise RuntimeError(f"Cannot forget active import {transfer_id}")
            attempt = self._gate.attempt_of(transfer_id)
            if imp is not None and attempt is not None:
                self._terminal_attempts[(transfer_id, attempt)] = None
                while len(self._terminal_attempts) > _MAX_TERMINAL_ATTEMPTS:
                    self._terminal_attempts.popitem(last=False)
            self._imports.pop(transfer_id, None)
            self._gate.unbind(transfer_id)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # state machine (IO thread + transfer threads)
    # ------------------------------------------------------------------

    def _conclude(
        self, imp: _Import, *, state: TransferState, reason: Optional[str]
    ) -> bool:
        """Terminal transition; False if already concluded."""
        with imp.lock:
            if imp.state not in _ACTIVE or imp.inflight:
                return False
            if state == TransferState.READY and imp.pending_terminal is not None:
                return False
            imp.state = state
            imp.reason = reason
            now = time.time()
            imp.timeline["state"] = state.name
            imp.timeline["reason"] = reason
            if state == TransferState.READY:
                imp.timeline["ready_at"] = now
            timeline = dict(imp.timeline)
        with self._mutex:
            self._history[imp.mark.transfer_id] = timeline
            while len(self._history) > _SNAPSHOT_KEEP:
                self._history.popitem(last=False)
            if state == TransferState.READY:
                self.stats_ready += 1
            else:
                self.stats_failed += 1
        if state != TransferState.READY:
            logger.info(
                "pull reader: transfer %d (%s) %s: %s",
                imp.mark.transfer_id,
                imp.req_id,
                state.name,
                reason,
            )
        else:
            logger.info(
                "pull reader: transfer %d (%s) READY reads=%d first_read=%s final_recv=%s",
                imp.mark.transfer_id,
                imp.req_id,
                len(timeline["reads"]),
                timeline["first_read_start_at"],
                timeline["final_recv_at"],
            )
        return True

    def _request_terminal(
        self,
        imp: _Import,
        state: TransferState,
        reason: str,
        *,
        notify_owner: bool = True,
    ) -> None:
        with imp.lock:
            if imp.state not in _ACTIVE:
                return
            if imp.pending_terminal is None:
                imp.pending_terminal = (state, reason, notify_owner)
        self._finish_pending(imp)

    def _finish_pending(self, imp: _Import) -> bool:
        with imp.lock:
            pending = imp.pending_terminal
            if pending is None:
                return False
            if imp.inflight:
                return True
        state, reason, notify_owner = pending
        if self._conclude(imp, state=state, reason=reason):
            if notify_owner:
                with self._mutex:
                    attempt = self._gate.attempt_of(imp.mark.transfer_id)
                self._send(
                    imp.mark.owner_addr,
                    wire.make_abort(
                        transfer_id=imp.mark.transfer_id,
                        attempt_id=attempt or 0,
                        reason=reason,
                    ),
                )
            imp.finished.set()
        return True

    def _read_finished(self, imp: _Import) -> None:
        with imp.lock:
            imp.inflight -= 1
            assert imp.inflight >= 0
        self._finish_pending(imp)

    def _quarantine(self, imp: _Import, reason: str) -> None:
        # Retain the in-flight reservation and both sides' pages. No terminal
        # state or ABORT acknowledgement may escape without a transport fence.
        # Recovery requires coordinated worker teardown, not a longer timer.
        with imp.lock:
            imp.timeline["quarantined_at"] = time.time()
            imp.timeline["reason"] = reason
        self._request_terminal(imp, TransferState.FAILED, reason)
        logger.error(
            "pull reader: transfer %d quarantined: %s", imp.mark.transfer_id, reason
        )

    def snapshot(self) -> Dict[str, Any]:
        """Introspection for /server_info (symmetric_pd.reader)."""
        with self._mutex:
            live = {}
            for tid, imp in self._imports.items():
                with imp.lock:
                    live[str(tid)] = dict(imp.timeline, state=imp.state.name)
            history = {str(tid): dict(t) for tid, t in self._history.items()}
            owners = {addr: (reg is not None) for addr, reg in self._owners.items()}
            out = {
                "bind_addr": self._bind_addr,
                "advertised_addr": self.advertised_addr,
                "session_id": self._registration.session_id,
                "tp_rank": self._registration.tp_rank,
                "pools": sorted(self._registration.pools),
                "owners_registered": owners,
                "imports": live,
                "import_count": len(live),
                "ready_total": self.stats_ready,
                "failed_total": self.stats_failed,
                "bytes_read_total": self.stats_bytes_read,
                "tier_served": dict(self.stats_tier_served),
                "send_dropped": self.stats_send_dropped,
                "early_parked": self.stats_early_parked,
                "early_replayed": self.stats_early_replayed,
                "early_expired": self.stats_early_expired,
                "terminal_attempts": len(self._terminal_attempts),
                "terminal_dropped": self.stats_terminal_dropped,
                "early_waiting": len(self._early),
                "dropped": dict(self._gate.dropped),
                "begin_sent": self.stats_begin_sent,
                "admit_received": self.stats_admit_received,
                "admit_dropped": self.stats_admit_dropped,
                "fail_forwarded": self.stats_fail_forwarded,
                "history": history,
            }
        return out

    def _fail(self, imp: _Import, reason: str, *, notify_owner: bool = True) -> None:
        self._request_terminal(
            imp, TransferState.FAILED, reason, notify_owner=notify_owner
        )

    def _on_chunk(
        self, imp: _Import, msg: wire.Message, owner_reg: Registration
    ) -> None:
        chunk = msg.chunk()
        if (envs.SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION.get()
                and chunk.chunk_seq == 0 and chunk.token_start > imp.reusable_prefix_tokens):
            self._fail(imp, f"{REASON_INCOMPLETE}:unowned_omitted_prefix={chunk.token_start}")
            return
        with imp.lock:
            if (
                imp.state not in _ACTIVE
                or imp.pending_terminal is not None
                or chunk.chunk_seq in imp.read_done
            ):
                return
            imp.state = TransferState.TRANSFERRING
            if imp.timeline["first_chunk_recv_at"] is None:
                imp.timeline["first_chunk_recv_at"] = time.time()
            for pp in chunk.pools:
                imp.announced.setdefault(pp.pool, {}).update(
                    zip(
                        (int(p) for p in pp.page_indices),
                        (int(g) for g in pp.generations),
                    )
                )
            if chunk.is_last:
                imp.last_seq = chunk.chunk_seq
            imp.max_token_end = max(imp.max_token_end, chunk.token_end)
            imp.inflight += 1
        try:
            # Phases of this IO thread are named io.* (phase_timers.py).
            with phase("io.dst_pages"):
                local_pages = imp.dst_pages(chunk)
        except Exception as e:
            self._fail(imp, f"dst_pages:{e}")
            self._read_finished(imp)
            return
        self.stats_tier_served[chunk.tier] = (
            self.stats_tier_served.get(chunk.tier, 0) + 1
        )
        self._pool.submit(self._read_chunk, imp, chunk, local_pages, owner_reg)

    def _read_chunk(
        self,
        imp: _Import,
        chunk: ChunkReady,
        local_pages: Dict[str, Sequence[int]],
        owner_reg: Registration,
    ) -> None:
        t0 = time.time()
        with imp.lock:
            cancelled = imp.pending_terminal is not None
        if cancelled:
            self._read_finished(imp)
            return
        with imp.lock:
            if imp.timeline["first_read_start_at"] is None:
                imp.timeline["first_read_start_at"] = t0
        nbytes = 0
        try:
            # Phases of the transfer threads are named xfer.* (phase_timers.py).
            with phase("xfer.chunk_plan"):
                plan = _chunk_plan(
                    chunk=chunk,
                    local_pages=local_pages,
                    local=self._registration,
                    remote=owner_reg,
                )
                nbytes = int(sum(plan[2]))
            # This READ writes the reader's device pages outside every CUDA
            # stream; nothing here orders it after the reader's forwards. The
            # scheduler must hand begin_import pages that no forward in flight
            # touches (M4 load quarantine: clean pages only, alloc_import).
            # xfer.read_call: the binding's list conversion, submit and wait.
            with phase("xfer.read_call"):
                rc = (
                    self._engine.batch_read(owner_reg.session_id, *plan)
                    if plan[2]
                    else 0
                )
        except ReadNotQuiescentError as e:
            self._quarantine(imp, str(e))
            return
        except wire.ProtocolError as e:
            rc = None
            reason = f"{REASON_LAYOUT}:{e}"
        except Exception as e:
            rc = None
            reason = f"{REASON_READ_FAILED}:{e}"
        t1 = time.time()
        with imp.lock:
            if rc == 0:
                imp.read_done.add(chunk.chunk_seq)
            imp.timeline["reads"].append(
                [
                    chunk.chunk_seq,
                    chunk.token_start,
                    chunk.token_end,
                    chunk.is_last,
                    nbytes,
                    t0,
                    t1,
                    rc,
                ]
            )
        if rc == 0:
            with self._mutex:
                self.stats_bytes_read += nbytes
        logger.info(
            "pull reader: transfer %d (%s) READ seq=%d tokens=[%d,%d) last=%s "
            "descriptors=%d bytes=%d %.1f ms rc=%s",
            imp.mark.transfer_id,
            imp.req_id,
            chunk.chunk_seq,
            chunk.token_start,
            chunk.token_end,
            chunk.is_last,
            len(plan[2]) if rc is not None else -1,
            nbytes,
            (t1 - t0) * 1e3,
            rc,
        )
        if rc is None:
            self._fail(imp, reason)
            self._read_finished(imp)
            return
        if rc != 0:
            self._fail(imp, f"{REASON_READ_FAILED}:rc={rc}")
            self._read_finished(imp)
            return
        self._read_finished(imp)
        self._maybe_finalize(imp, owner_reg)

    def _on_final(
        self, imp: _Import, msg: wire.Message, owner_reg: Registration
    ) -> None:
        with imp.lock:
            if (
                imp.state not in _ACTIVE
                or imp.pending_terminal is not None
                or imp.final is not None
            ):
                return
            imp.final = msg.final()
            imp.final_gens = msg.final_generations()
            imp.timeline["final_recv_at"] = time.time()
        self._maybe_finalize(imp, owner_reg)

    def _maybe_finalize(self, imp: _Import, owner_reg: Registration) -> None:
        """Once FINAL and every chunk READ are in: verify, read aux + state, DONE."""
        with imp.lock:
            if (
                imp.state not in _ACTIVE
                or imp.pending_terminal is not None
                or imp.final is None
                or imp.finalizing
            ):
                return
            if imp.last_seq is None or imp.inflight > 0:
                return
            if any(seq not in imp.read_done for seq in range(imp.last_seq + 1)):
                return
            if imp.final.total_tokens != imp.max_token_end:
                mismatch = (imp.final.total_tokens, imp.max_token_end)
            else:
                mismatch = None
            imp.finalizing = True
            announced = {k: dict(v) for k, v in imp.announced.items()}
            final_gens = imp.final_gens or {}
            final = imp.final
        if mismatch is not None:
            self._fail(
                imp,
                f"{REASON_INCOMPLETE}:total_tokens={mismatch[0]} read={mismatch[1]}",
            )
            return
        if not _generations_match(announced=announced, final=final_gens):
            # Name the pages so the owner-side cause can be traced (which pool,
            # how far into the prompt, announced vs FINAL generation).
            self._fail(
                imp,
                f"{REASON_OWNER_EVICTED}:{_generation_mismatches(announced=announced, final=final_gens)}",
            )
            return
        with imp.lock:
            if imp.pending_terminal is not None or imp.state not in _ACTIVE:
                return
            imp.inflight += 1
        self._pool.submit(self._read_final_items, imp, final, owner_reg)

    def _read_final_items(
        self, imp: _Import, final: FinalMeta, owner_reg: Registration
    ) -> None:
        with imp.lock:
            cancelled = imp.pending_terminal is not None
        if cancelled:
            self._read_finished(imp)
            return
        try:
            with phase("xfer.final_read"):
                plan = _final_plan(
                    final=final,
                    local_aux_index=int(imp.mark.local_aux_index),
                    local_state_indices=imp.mark.local_state_indices,
                    local=self._registration,
                    remote=owner_reg,
                )
                rc = (
                    self._engine.batch_read(owner_reg.session_id, *plan)
                    if plan[2]
                    else 0
                )
        except ReadNotQuiescentError as e:
            self._quarantine(imp, str(e))
            return
        except wire.ProtocolError as e:
            self._fail(imp, f"{REASON_LAYOUT}:{e}")
            self._read_finished(imp)
            return
        except Exception as e:
            self._fail(imp, f"{REASON_READ_FAILED}:{e}")
            self._read_finished(imp)
            return
        if rc != 0:
            self._fail(imp, f"{REASON_READ_FAILED}:aux rc={rc}")
            self._read_finished(imp)
            return
        with imp.lock:
            imp.timeline["final_read_at"] = time.time()
        with self._mutex:
            self.stats_bytes_read += int(sum(plan[2]))
        self._read_finished(imp)
        if self._conclude(imp, state=TransferState.READY, reason=None):
            with self._mutex:
                attempt = self._gate.attempt_of(imp.mark.transfer_id)
            self._send(
                imp.mark.owner_addr,
                wire.make_done(
                    transfer_id=imp.mark.transfer_id, attempt_id=attempt or 0
                ),
            )
            imp.finished.set()

    # ------------------------------------------------------------------
    # IO thread
    # ------------------------------------------------------------------

    def _send(self, addr: str, msg: wire.Message) -> None:
        self._outbound.put((addr, wire.encode(msg)))

    def _io_loop(self) -> None:
        pull = self._ctx.socket(zmq.PULL)
        pushes: Dict[str, zmq.Socket] = {}
        try:
            pull.setsockopt(zmq.LINGER, 0)
            pull.setsockopt(zmq.IPV6, 1)  # an IPv6-only worker binds tcp://[addr]:port
            pull.bind(self._bind_addr)
            poller = zmq.Poller()
            poller.register(pull, zmq.POLLIN)
        except Exception as exc:
            self._startup_error = exc
            pull.close(linger=0)
            self._started.set()
            return
        self._started.set()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                self._request_registrations(now)
                self._expire_imports(now)
                self._expire_early(now)
                self._drain_replay()
                self._flush_outbound(pushes)
                if not poller.poll(_POLL_INTERVAL_MS):
                    continue
                while True:
                    try:
                        raw = pull.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    self._handle_raw(raw)
        finally:
            pull.close(linger=0)
            for sock in pushes.values():
                sock.close(linger=0)

    def _request_registrations(self, now: float) -> None:
        with self._mutex:
            pending = [a for a, reg in self._owners.items() if reg is None]
        for addr in pending:
            if now - self._register_sent_mono.get(addr, -1e9) < _REGISTER_RETRY_S:
                continue
            self._register_sent_mono[addr] = now
            self._send(
                addr,
                wire.make_register_req(
                    reader_addr=self.advertised_addr,
                    tp_rank=self._registration.tp_rank,
                    reader_worker=self._worker_name,
                ),
            )

    def _expire_imports(self, now: float) -> None:
        with self._mutex:
            imps = list(self._imports.values())
        for imp in imps:
            with imp.lock:
                expired = imp.state in _ACTIVE and imp.deadline_mono <= now
            if expired:
                self._fail(imp, REASON_TIMEOUT)

    def _flush_outbound(self, pushes: Dict[str, zmq.Socket]) -> None:
        while True:
            try:
                addr, raw = self._outbound.get_nowait()
            except queue.Empty:
                return
            sock = pushes.get(addr)
            if sock is None:
                sock = self._ctx.socket(zmq.PUSH)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDHWM, _SEND_HWM)
                sock.setsockopt(zmq.IPV6, 1)
                sock.connect(addr)
                pushes[addr] = sock
            try:
                sock.send(raw, flags=zmq.NOBLOCK)
            except zmq.Again:
                self.stats_send_dropped += 1
                logger.warning("pull reader: send to %s dropped", addr)

    def _handle_raw(self, raw: bytes) -> None:
        try:
            msg = wire.decode(raw)
        except wire.ProtocolError as e:
            logger.warning("pull reader: bad message: %s", e)
            return
        if msg.type == wire.MSG_REGISTER:
            self._on_register(msg)
            return
        if msg.type in (wire.MSG_ADMIT, wire.MSG_RELEASE):
            # Never dispatched: _dispatch rejects them and _admit would park
            # them.
            self._on_admit(msg)
            return
        if msg.type == wire.MSG_ABORT:
            self._forward_owner_abort(msg)
        self._dispatch(msg)

    def _late_control(self, kind: str, msg: wire.Message) -> Any:
        # Imported here: late_d needs the scheduler-side modules, the plane
        # does not (the IO thread runs this only with a late inbox attached).
        from decoupled_engine.srt.disaggregation.late_d import LateDControl

        return LateDControl(
            kind=kind,
            room=msg.transfer_id,
            seq=msg.seq,
            attempt=msg.attempt_id,
            t_recv=time.monotonic(),
            kv_session=wire.admit_kv_session(msg),
        )

    def _on_admit(self, msg: wire.Message) -> None:
        """Late D: owner rank 0 admits this rank-0 reader's import (ADMIT) or,
        in stage 2, releases it (RELEASE). Drop it when the attempt already
        ended here, or without a late inbox (late D off, or a rank other than
        0: a configuration error)."""
        with self._mutex:
            self.stats_admit_received += 1
            inbox = self._late_inbox
            terminal = (msg.transfer_id, msg.attempt_id) in self._terminal_attempts
            if inbox is None or terminal:
                self.stats_admit_dropped += 1
            dropped = self.stats_admit_dropped
        if inbox is None:
            if dropped <= 5 or dropped % 1000 == 0:
                logger.warning(
                    "pull reader: %s for transfer %d dropped: late D is off "
                    "on this reader (mixed --disaggregation-late-d?); %d dropped",
                    msg.type,
                    msg.transfer_id,
                    dropped,
                )
            return
        if terminal:
            logger.debug(
                "pull reader: %s for ended transfer %d attempt %d dropped",
                msg.type,
                msg.transfer_id,
                msg.attempt_id,
            )
            return
        from decoupled_engine.srt.disaggregation.late_d import CONTROL_ADMIT, CONTROL_RELEASE

        kind = CONTROL_RELEASE if msg.type == wire.MSG_RELEASE else CONTROL_ADMIT
        inbox.put(self._late_control(kind, msg))

    def _forward_owner_abort(self, msg: wire.Message) -> None:
        """Late D: an owner ABORT for a transfer without an import ends
        a late entry that never allocated (DEFERRED, ADMITTED or HELD). The
        message is still parked by _dispatch as before."""
        with self._mutex:
            inbox = self._late_inbox
            if inbox is None:
                return
            key = (msg.transfer_id, msg.attempt_id)
            if msg.transfer_id in self._imports or key in self._terminal_attempts:
                return
            self.stats_fail_forwarded += 1
        from decoupled_engine.srt.disaggregation.late_d import CONTROL_FAIL

        inbox.put(self._late_control(CONTROL_FAIL, msg))

    def _dispatch(self, msg: wire.Message) -> None:
        if msg.type not in (wire.MSG_CHUNK_READY, wire.MSG_FINAL, wire.MSG_ABORT):
            logger.warning("pull reader: unexpected %s", msg.type)
            return
        imp, owner_reg = self._admit(msg)
        if imp is None:
            return
        if msg.type == wire.MSG_ABORT:
            self._request_terminal(
                imp, TransferState.ABORTED, f"owner_abort:{msg.reason}"
            )
            return
        if owner_reg is None:
            # The owner may publish before our REGISTER_REQ reached it; park the
            # message until REGISTER lands (the import deadline still applies).
            pending = self._pending_by_owner.setdefault(imp.mark.owner_addr, [])
            if len(pending) >= _MAX_PENDING_PER_OWNER:
                self._fail(imp, f"{REASON_INCOMPLETE}:register_backlog")
                return
            pending.append(msg)
            return
        if msg.type == wire.MSG_CHUNK_READY:
            self._on_chunk(imp, msg, owner_reg)
        else:
            self._on_final(imp, msg, owner_reg)

    def _admit(
        self, msg: wire.Message
    ) -> Tuple[Optional[_Import], Optional[Registration]]:
        """Bind the transfer to its first attempt; drop unknown / stale."""
        with self._mutex:
            # A chunk can already be on the wire when Reader abort finishes
            # and the scheduler forgets its import. Do not misclassify that
            # chunk as a publication preceding begin_import, or let it bind
            # a new import reusing the transfer id to the retired attempt.
            # Exact attempt identity permits legitimate retries to arrive early.
            if (msg.transfer_id, msg.attempt_id) in self._terminal_attempts:
                self.stats_terminal_dropped += 1
                return None, None
            imp = self._imports.get(msg.transfer_id)
            if imp is None:
                if self._park_early(msg):
                    return None, None
                self._gate.dropped[AttemptGate.DROP_UNKNOWN] += 1
                return None, None
            if imp.mark.attempt_id is not None and msg.attempt_id == 0:
                self._gate.dropped[AttemptGate.DROP_ATTEMPT] += 1
                return None, None
            if self._gate.attempt_of(msg.transfer_id) is None:
                self._gate.bind(transfer_id=msg.transfer_id, attempt_id=msg.attempt_id)
            verdict = self._gate.check(msg)
            owner_reg = self._owners.get(imp.mark.owner_addr)
        if verdict != AttemptGate.ACCEPT:
            logger.debug(
                "pull reader: %s for %d dropped (%s)",
                msg.type,
                msg.transfer_id,
                verdict,
            )
            return None, None
        with imp.lock:
            if imp.state not in _ACTIVE:
                return None, None
        return imp, owner_reg

    def _bind_mark_attempt(self, mark: ImportMark) -> None:
        """Bind before any publication or cancellation; caller holds _mutex."""
        if mark.attempt_id is None:
            return
        bound = self._gate.attempt_of(mark.transfer_id)
        if bound is not None and bound != mark.attempt_id:
            raise ValueError(f"Transfer {mark.transfer_id} already has a bound attempt")
        self._gate.bind(transfer_id=mark.transfer_id, attempt_id=mark.attempt_id)

    def _bind_early_attempt(
        self, transfer_id: int, messages: Sequence[wire.Message]
    ) -> None:
        """Remember the first admissible publication's identity before a peer
        rank can cancel us ahead of IO replay. Caller holds _mutex; this does
        not dispatch messages or start READs. Match _admit's first-message
        binding, including its rejection of already retired attempts."""
        if self._gate.attempt_of(transfer_id) is not None:
            return
        for msg in messages:
            if (transfer_id, msg.attempt_id) not in self._terminal_attempts:
                self._gate.bind(transfer_id=transfer_id, attempt_id=msg.attempt_id)
                return

    def _park_early(self, msg: wire.Message) -> bool:
        """Park a publication whose begin_import has not happened yet. Caller
        holds _mutex. Returns False when the message is dropped instead."""
        entry = self._early.get(msg.transfer_id)
        if entry is None:
            if len(self._early) >= _MAX_EARLY_TRANSFERS:
                return False
            entry = (time.monotonic(), [])
            self._early[msg.transfer_id] = entry
        if len(entry[1]) >= _MAX_EARLY_MSGS_PER_TRANSFER:
            return False
        entry[1].append(msg)
        self.stats_early_parked += 1
        return True

    def _drain_replay(self) -> None:
        while True:
            try:
                msg = self._replay.get_nowait()
            except queue.Empty:
                return
            self._dispatch(msg)

    def _expire_early(self, now: float) -> None:
        with self._mutex:
            stale = [
                tid
                for tid, (since, _) in self._early.items()
                if now - since > self._timeout_s
            ]
            for tid in stale:
                self.stats_early_expired += len(self._early.pop(tid)[1])
        if stale:
            logger.info(
                "pull reader: dropped early publications for %d transfer(s) that "
                "never reached begin_import within %.0f s",
                len(stale),
                self._timeout_s,
            )

    def _on_register(self, msg: wire.Message) -> None:
        reg = msg.registration()
        problem = wire.check_compatible(local=self._registration, remote=reg)
        mismatch = wire.late_d_mismatch(local=self._registration, remote=reg)
        with self._mutex:
            if mismatch is None:
                self._late_d_mismatched.discard(msg.owner_addr)
            else:
                self._late_d_mismatched.add(msg.owner_addr)
        if mismatch is not None and msg.owner_addr not in self._late_d_mismatch_logged:
            # Never a reason to refuse the owner; late rooms between the two
            # do not bind ("mixed late-D flags").
            self._late_d_mismatch_logged.add(msg.owner_addr)
            logger.error(
                "pull reader: owner %s (%s) registered with a different late-D "
                "mode: %s (reader first); run the same "
                "--disaggregation-late-d on every symmetric worker",
                msg.owner_addr,
                reg.worker,
                mismatch,
            )
        with self._mutex:
            if problem is None:
                self._owners[msg.owner_addr] = reg
            victims = [
                imp
                for imp in self._imports.values()
                if imp.mark.owner_addr == msg.owner_addr
            ]
        if problem is not None:
            self._pending_by_owner.pop(msg.owner_addr, None)
            for imp in victims:
                self._fail(imp, f"{REASON_LAYOUT}:{problem}")
            return
        for parked in self._pending_by_owner.pop(msg.owner_addr, []):
            self._dispatch(parked)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _chunk_plan(
    *,
    chunk: ChunkReady,
    local_pages: Dict[str, Sequence[int]],
    local: Registration,
    remote: Registration,
) -> Tuple[List[int], List[int], List[int]]:
    lp_all: List[int] = []
    rp_all: List[int] = []
    ln_all: List[int] = []
    for pp in chunk.pools:
        if pp.pool not in local.pools or pp.pool not in remote.pools:
            raise wire.ProtocolError(f"pool {pp.pool} unknown on one side")
        if pp.pool not in local_pages:
            raise wire.ProtocolError(f"dst_pages has no pool {pp.pool}")
        lpages, rpages = local_pages[pp.pool], pp.page_indices
        if len(lpages) == len(rpages) and (
            any(int(p) < 0 for p in lpages) or any(int(p) < 0 for p in rpages)
        ):
            # Reader-side -1 (symmetric/import_queue.py): the page is already
            # resident locally (prefix reuse) or was never allocated (SWA-tail
            # import) -> drop the pair, no READ. Owner-side -1
            # (export_table.owner_swa_valid_floor): the owner has no live page
            # there; if the reader still needs it the import cannot be served
            # by this owner -> ProtocolError -> layout_mismatch -> recompute.
            kept = [(r, l) for r, l in zip(rpages, lpages) if int(l) >= 0]
            missing = sum(1 for r, _ in kept if int(r) < 0)
            if missing:
                raise wire.ProtocolError(
                    f"pool {pp.pool}: owner has no live page for {missing} of "
                    f"{len(kept)} page(s) the reader needs (owner SWA validity floor)"
                )
            rpages = [r for r, _ in kept]
            lpages = [l for _, l in kept]
        lp, rp, ln = wire.build_page_read_plan(
            local=local.pools[pp.pool],
            remote=remote.pools[pp.pool],
            local_pages=lpages,
            remote_pages=rpages,
        )
        lp_all += lp
        rp_all += rp
        ln_all += ln
    return lp_all, rp_all, ln_all


def _final_plan(
    *,
    final: FinalMeta,
    local_aux_index: int,
    local_state_indices: Dict[str, List[int]],
    local: Registration,
    remote: Registration,
) -> Tuple[List[int], List[int], List[int]]:
    lp_all, rp_all, ln_all = wire.build_item_read_plan(
        local=local.aux,
        remote=remote.aux,
        local_index=local_aux_index,
        remote_index=final.aux_index,
    )
    for name, owner_idx in final.state_indices.items():
        if not owner_idx:
            # Nothing to read for this state (e.g. DSv4 C128 rows are empty when
            # total_tokens % 128 == 0); tolerate an owner that still lists it.
            continue
        if name not in remote.states or name not in local.states:
            raise wire.ProtocolError(f"state {name} unknown on one side")
        local_idx = local_state_indices.get(name)
        if local_idx is None or len(local_idx) != len(owner_idx):
            raise wire.ProtocolError(f"state {name}: no matching local slots")
        for li, ri in zip(local_idx, owner_idx):
            lp, rp, ln = wire.build_item_read_plan(
                local=local.states[name],
                remote=remote.states[name],
                local_index=li,
                remote_index=ri,
            )
            lp_all += lp
            rp_all += rp
            ln_all += ln
    return lp_all, rp_all, ln_all


def _generation_mismatches(
    *,
    announced: Dict[str, Dict[int, int]],
    final: Dict[str, Dict[int, int]],
    limit: int = 6,
) -> str:
    """Compact description of the (pool, page, announced->final) triples that
    fail _generations_match, plus per-pool totals: 'full:3/1200 swa:0/300 |
    full p4711 3->4, ...'. Diagnostic only (failure reason string)."""
    parts: List[str] = []
    samples: List[str] = []
    for pool, pages in announced.items():
        now = final.get(pool)
        if now is None:
            continue
        bad = [(p, g, now[p]) for p, g in pages.items() if p in now and now[p] != g]
        parts.append(f"{pool}:{len(bad)}/{len(pages)}")
        for p, g, n in sorted(bad)[: max(0, limit - len(samples))]:
            samples.append(f"{pool} p{p} {g}->{n}")
    return " ".join(parts) + (" | " + ", ".join(samples) if samples else "")


def _generations_match(
    *, announced: Dict[str, Dict[int, int]], final: Dict[str, Dict[int, int]]
) -> bool:
    """Every page FINAL reports must carry the generation CHUNK_READY announced.
    Pools absent from FINAL are unverifiable and pass."""
    for pool, pages in announced.items():
        now = final.get(pool)
        if now is None:
            continue
        for page, gen in pages.items():
            if page in now and now[page] != gen:
                return False
    return True

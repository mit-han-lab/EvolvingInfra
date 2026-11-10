"""OwnerPlane: the P-leg side of the pull plane (api.OwnerPlane).

This worker is the Owner of the exported KV; the D-leg is the Reader.

Threading: the scheduler thread calls begin_export / publish_chunk /
publish_final / poll_released / abort_export and never touches a socket. One
daemon IO thread owns the ZMQ sockets: it binds the per-rank PULL channel,
drains the outbound queue into per-reader PUSH sockets, and answers
REGISTER_REQ / receives DONE and ABORT. Lease pins and unpins (radix tree
lock_ref) run on the scheduler thread only; expiry is flagged by the Lease and
drained in poll_released (GpuLeaseTable in mem_cache/peer_kv_servicer.py).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import zmq

from decoupled_engine.srt.disaggregation.pull import protocol as wire
from decoupled_engine.srt.disaggregation.pull.api import (
    ChunkReady,
    ExportCandidate,
    ExportMark,
    FinalMeta,
)
from decoupled_engine.srt.disaggregation.pull.protocol import AttemptGate, Registration
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.peer_kv_servicer import GpuLeaseTable

logger = logging.getLogger(__name__)

# IO-thread poll/flush interval; see envs.SGLANG_SYMMETRIC_PD_PLANE_POLL_MS.
_POLL_INTERVAL_MS = max(1, int(envs.SGLANG_SYMMETRIC_PD_PLANE_POLL_MS.get()))
# Bounded so a dead reader cannot make the IO thread hoard memory; drops are counted.
_SEND_HWM = 10_000
# BEGINs that overtook begin_export (reader-first exports): bounded, expired
# after the import waiting timeout.
_MAX_EARLY_BEGIN = 4096


class _Export:
    __slots__ = (
        "mark",
        "req_id",
        "attempt_id",
        "cands",
        "committed",
        "winner_gone",
        "begin_seen",
        "reusable_prefix_tokens",
        "chunks",
        "final_sent",
        "aborted",
        "timeline",
    )

    def __init__(
        self, mark: ExportMark, req_id: str, attempt_id: Optional[int]
    ) -> None:
        self.mark = mark
        self.req_id = req_id
        # None: a late-D stage-2 candidate export before its BEGIN binds it.
        self.attempt_id = attempt_id
        # Late D stage 2: attempt -> candidate (reader_addr rank-rewritten);
        # None for a one-reader export.
        self.cands: Optional[Dict[int, ExportCandidate]] = None
        # Rank 0: the attempt the binder committed the export to, and
        # whether that winner sent ABORT before its BEGIN bound it.
        self.committed: Optional[int] = None
        self.winner_gone = False
        self.begin_seen = False
        self.reusable_prefix_tokens = 0
        self.chunks = 0
        self.final_sent = False
        self.aborted = False
        # Wall-clock (time.time()) telemetry for M0 (check:
        # "first READ before last prefill chunk"), exposed via snapshot().
        self.timeline: Dict[str, Any] = {
            "req_id": req_id,
            "reader_addr": mark.reader_addr,
            "begin_at": time.time(),
            "chunks": [],  # [chunk_seq, token_start, token_end, is_last, published_at]
            "final_at": None,
            "released_at": None,
            "reason": None,
        }


_SNAPSHOT_KEEP = 256


class OwnerPlane:
    """api.OwnerPlane over ZMQ + GpuLeaseTable.

    ``bind_addr`` is the ZMQ endpoint the PULL channel binds ("tcp://ip:port" or
    "inproc://..."); ``advertised_addr`` is what the router puts in
    ImportMark.owner_addr (defaults to bind_addr). ``node_refs_of(chunk)``
    returns the radix node refs whose lock protects the chunk's GPU pages when
    the pages are radix-resident (Session KV reuse); for request-held
    pages (the request stays in EXPORTED and never frees) it returns nothing and
    the Lease only tracks (slot, generation) and expiry.
    """

    def __init__(
        self,
        *,
        registration: Registration,
        lease: GpuLeaseTable,
        bind_addr: str,
        advertised_addr: Optional[str] = None,
        zmq_ctx: Optional[zmq.Context] = None,
        node_refs_of: Optional[Callable[[ChunkReady], Sequence[Any]]] = None,
        worker_name: str = "",
        await_reader: Optional[bool] = None,
    ) -> None:
        self._registration = registration
        # Reader-first exports (SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER): readers send
        # BEGIN when they admit the import; poll_begun hands the transfer ids
        # to the scheduler, which queues the held P-leg. Off: BEGIN is ignored.
        self._await_reader = (
            bool(envs.SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER.get())
            if await_reader is None
            else bool(await_reader)
        )
        self._begun: queue.Queue[int] = queue.Queue()
        # transfer_id -> (monotonic arrival, attempt_id, reusable prefix) of BEGINs that came
        # before begin_export (guarded by _mutex).
        self._early_begin: Dict[int, Tuple[float, int, int]] = {}
        self._early_begin_ttl_s = float(
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        )
        self._early_begin_swept = time.monotonic()
        self.stats_begin_received = 0
        self.stats_begin_early = 0
        self.stats_begin_dropped = 0
        self.stats_winner_gone = 0
        self._lease = lease
        self._bind_addr = bind_addr
        self.advertised_addr = (
            advertised_addr if advertised_addr is not None else bind_addr
        )
        self._ctx = zmq_ctx if zmq_ctx is not None else zmq.Context.instance()
        self._node_refs_of = node_refs_of
        self._worker_name = worker_name

        self._mutex = threading.Lock()
        self._exports: Dict[int, _Export] = {}
        self._gate = AttemptGate()
        self._next_attempt = 1
        # (transfer_id, reason) events from the IO thread, drained by poll_released.
        self._inbound: queue.Queue[Tuple[int, str]] = queue.Queue()
        self._outbound: queue.Queue[Tuple[str, bytes]] = queue.Queue()
        self._released_pending: List[int] = []
        self.stats_send_dropped = 0
        # Concluded exports' timelines (bounded), for snapshot().
        self._history: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self.stats_released = 0

        self._stop = threading.Event()
        self._started = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._io_loop, daemon=True, name="pull-owner-io"
        )
        self._thread.start()
        if not self._started.wait(timeout=30):
            self._stop.set()
            raise RuntimeError(f"OwnerPlane channel startup timed out on {bind_addr}")
        if self._startup_error is not None:
            raise RuntimeError(
                f"OwnerPlane failed to bind {bind_addr}"
            ) from self._startup_error

    # ------------------------------------------------------------------
    # api.OwnerPlane (scheduler thread)
    # ------------------------------------------------------------------

    def begin_export(self, mark: ExportMark, req_id: str) -> None:
        with self._mutex:
            if mark.candidates:
                # Late D stage 2 (sketch h): no reader and no gate binding
                # until the committed winner's BEGIN binds them.
                ex = _Export(mark, req_id, None)
                ex.cands = {c.attempt_id: c for c in mark.candidates}
                self._exports[mark.transfer_id] = ex
                early = self._early_begin.pop(mark.transfer_id, None)
                begun = early is not None and self._bind_on_begin(ex, early[1])
            else:
                attempt = mark.attempt_id
                if attempt is None:
                    attempt = self._next_attempt
                    self._next_attempt += 1
                ex = _Export(mark, req_id, attempt)
                self._exports[mark.transfer_id] = ex
                self._gate.bind(transfer_id=mark.transfer_id, attempt_id=attempt)
                early = self._early_begin.pop(mark.transfer_id, None)
                begun = early is not None and self._begin_matches(ex, early[1])
            if begun:
                self._remember_begin_prefix(ex, early[1], early[2])
        if begun:
            # The reader admitted the import before this P-leg arrived.
            self._begun.put(mark.transfer_id)

    def publish_chunk(self, chunk: ChunkReady) -> None:
        """Pin the chunk's pages under the Lease, then send CHUNK_READY. Pages are
        covered by the Lease from the moment of publication."""
        with self._mutex:
            ex = self._exports.get(chunk.transfer_id)
        if ex is None or ex.aborted or ex.attempt_id is None:
            logger.warning(
                "publish_chunk for unknown/aborted/unbound transfer %d",
                chunk.transfer_id,
            )
            return
        node_refs = self._node_refs_of(chunk) if self._node_refs_of is not None else ()
        refusal = self._lease.pin(
            transfer_id=chunk.transfer_id,
            pages={pp.pool: (pp.page_indices, pp.generations) for pp in chunk.pools},
            node_refs=node_refs,
        )
        if refusal is not None:
            self.abort_export(chunk.transfer_id, f"lease_refused:{refusal}")
            return
        ex.chunks += 1
        now = time.time()
        ex.timeline["chunks"].append(
            [chunk.chunk_seq, chunk.token_start, chunk.token_end, chunk.is_last, now]
        )
        self._send(
            ex.mark.reader_addr,
            wire.make_chunk_ready(chunk=chunk, attempt_id=ex.attempt_id),
        )
        logger.info(
            "pull owner: transfer %d (%s) CHUNK_READY seq=%d tokens=[%d,%d) last=%s -> %s",
            chunk.transfer_id,
            ex.req_id,
            chunk.chunk_seq,
            chunk.token_start,
            chunk.token_end,
            chunk.is_last,
            ex.mark.reader_addr,
        )

    def publish_final(self, final: FinalMeta) -> None:
        with self._mutex:
            ex = self._exports.get(final.transfer_id)
        if ex is None or ex.aborted or ex.attempt_id is None:
            logger.warning(
                "publish_final for unknown/aborted/unbound transfer %d",
                final.transfer_id,
            )
            return
        ex.final_sent = True
        ex.timeline["final_at"] = time.time()
        logger.info(
            "pull owner: transfer %d (%s) FINAL total_tokens=%d aux=%d",
            final.transfer_id,
            ex.req_id,
            final.total_tokens,
            final.aux_index,
        )
        self._send(
            ex.mark.reader_addr,
            wire.make_final(
                final=final,
                attempt_id=ex.attempt_id,
                generations=self._lease.current_generations(final.transfer_id),
            ),
        )

    def poll_released(self) -> List[int]:
        """Drain DONE/ABORT notices and Lease expiries; unpin and return the
        transfer_ids whose pages the scheduler may now free."""
        released: List[int] = []
        while True:
            try:
                tid, reason = self._inbound.get_nowait()
            except queue.Empty:
                break
            if self._finish(tid, reason=reason):
                released.append(tid)
        for tid in self._lease.drain_expired():
            if self._finish(tid, reason="lease_expired"):
                released.append(tid)
        with self._mutex:
            released.extend(self._released_pending)
            self._released_pending.clear()
        return released

    def poll_begun(self) -> List[int]:
        """Reader-first exports: transfer_ids whose reader sent BEGIN (import
        admitted) since the last call, on this rank. The scheduler agrees on
        them across ranks before it queues the held P-leg. Always [] when
        SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER is off."""
        begun: List[int] = []
        while True:
            try:
                begun.append(self._begun.get_nowait())
            except queue.Empty:
                break
        now = time.monotonic()
        if now - self._early_begin_swept >= 1.0:
            self._early_begin_swept = now
            with self._mutex:
                stale = [
                    tid
                    for tid, (since, _, _) in self._early_begin.items()
                    if now - since > self._early_begin_ttl_s
                ]
                for tid in stale:
                    del self._early_begin[tid]
                self.stats_begin_dropped += len(stale)
        return begun

    def reusable_prefix_tokens(self, transfer_id: int) -> int:
        """The first accepted BEGIN claim for the currently bound attempt."""
        with self._mutex:
            ex = self._exports.get(transfer_id)
            return ex.reusable_prefix_tokens if ex is not None and not ex.aborted else 0

    @staticmethod
    def _remember_begin_prefix(ex: _Export, attempt_id: int, prefix: int) -> None:
        if ex.begin_seen:
            return
        ex.begin_seen = True
        if (envs.SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION.get()
                and attempt_id > 0 and attempt_id == ex.attempt_id and ex.chunks == 0):
            ex.reusable_prefix_tokens = prefix

    def attempt_of(self, transfer_id: int) -> Optional[int]:
        """The attempt this rank exports ``transfer_id`` under, or None when
        it has no live export (or a stage-2 candidate export not bound yet).
        Late D: owner rank 0 quotes it in ADMIT."""
        with self._mutex:
            ex = self._exports.get(transfer_id)
            return None if ex is None else ex.attempt_id

    def note_committed(self, transfer_id: int, attempt_id: int) -> None:
        """Late D stage 2, owner rank 0: the binder committed the
        candidate export to ``attempt_id``. Only rank 0 commits, so only rank
        0 can accept that winner's ABORT before its BEGIN binds the export."""
        with self._mutex:
            ex = self._exports.get(transfer_id)
            if ex is not None and ex.attempt_id is None:
                ex.committed = int(attempt_id)

    def winner_gave_up(self, transfer_id: int) -> bool:
        """Rank 0: the committed winner sent ABORT before its BEGIN (the
        scheduler votes FAILED for the room; the MIN fails it on every rank)."""
        with self._mutex:
            ex = self._exports.get(transfer_id)
            return ex is not None and ex.winner_gone

    def release_local(self, transfer_id: int) -> bool:
        """Late D stage 2, LOCAL (LC-3): the owner runs the P-leg itself. End
        the export without an ABORT (no reader is bound, nothing is leased);
        the RELEASEs to the candidates went out with the commit."""
        return self._finish(transfer_id, reason="late_d_local")

    def abort_export(self, transfer_id: int, reason: str) -> None:
        with self._mutex:
            ex = self._exports.get(transfer_id)
            if ex is None or ex.aborted:
                return
            ex.aborted = True
            if ex.attempt_id is None:
                # A candidate export before its bind: every candidate gets an
                # ABORT under its own attempt (sketch h, 3.1 item 7).
                targets = [(c.reader_addr, a) for a, c in sorted(ex.cands.items())]
            else:
                targets = [(ex.mark.reader_addr, ex.attempt_id)]
        for addr, attempt in targets:
            self._send(
                addr,
                wire.make_abort(
                    transfer_id=transfer_id, attempt_id=attempt, reason=reason
                ),
            )
        # An ABORT is a revocation request, not proof that remote READs have
        # stopped. Published pages stay pinned until the reader acknowledges
        # after draining its queued and in-flight operations.
        if ex.chunks == 0 and self._finish(transfer_id, reason=f"owner_abort:{reason}"):
            with self._mutex:
                self._released_pending.append(transfer_id)

    @property
    def lease(self) -> GpuLeaseTable:
        """The Lease this plane pins published pages in (scheduler thread)."""
        return self._lease

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _finish(self, transfer_id: int, *, reason: str) -> bool:
        """Release the Lease (scheduler thread) and forget the export."""
        with self._mutex:
            ex = self._exports.pop(transfer_id, None)
            self._gate.unbind(transfer_id)
        self._lease.release(transfer_id=transfer_id)
        if ex is None:
            return False
        ex.timeline["released_at"] = time.time()
        ex.timeline["reason"] = reason
        with self._mutex:
            self._history[transfer_id] = ex.timeline
            while len(self._history) > _SNAPSHOT_KEEP:
                self._history.popitem(last=False)
            self.stats_released += 1
        logger.info(
            "pull owner: transfer %d (%s) released: %s chunks=%d final=%s",
            transfer_id,
            ex.req_id,
            reason,
            ex.chunks,
            ex.final_sent,
        )
        return True

    def snapshot(self) -> Dict[str, Any]:
        """Introspection for /server_info (symmetric_pd.owner): live exports, the
        Lease table size and the bounded history of concluded transfers."""
        with self._mutex:
            active = {str(tid): dict(ex.timeline) for tid, ex in self._exports.items()}
            history = {str(tid): dict(t) for tid, t in self._history.items()}
            released = self.stats_released
            begin = {
                "await_reader": self._await_reader,
                "received": self.stats_begin_received,
                "early": self.stats_begin_early,
                "early_waiting": len(self._early_begin),
                "dropped": self.stats_begin_dropped,
                "winner_gone": self.stats_winner_gone,
            }
        lease_count = self._lease.active_count()
        return {
            "bind_addr": self._bind_addr,
            "advertised_addr": self.advertised_addr,
            "session_id": self._registration.session_id,
            "tp_rank": self._registration.tp_rank,
            "pools": sorted(self._registration.pools),
            "active": active,
            "active_count": len(active),
            "lease_count": lease_count,
            "released_total": released,
            "send_dropped": self.stats_send_dropped,
            "begin": begin,
            "history": history,
        }

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
                logger.warning(
                    "pull owner: send to %s dropped (reader not draining)", addr
                )

    def _handle_raw(self, raw: bytes) -> None:
        try:
            msg = wire.decode(raw)
        except wire.ProtocolError as e:
            logger.warning("pull owner: bad message: %s", e)
            return
        if msg.type == wire.MSG_REGISTER_REQ:
            self._send(
                msg.reader_addr,
                wire.make_register(
                    reg=self._registration, owner_addr=self.advertised_addr
                ),
            )
            return
        if msg.type == wire.MSG_BEGIN:
            self._on_begin(msg)
            return
        if msg.type not in (wire.MSG_DONE, wire.MSG_ABORT):
            logger.warning("pull owner: unexpected %s", msg.type)
            return
        with self._mutex:
            ex = self._exports.get(msg.transfer_id)
            winner_gone = (
                msg.type == wire.MSG_ABORT
                and ex is not None
                and ex.attempt_id is None
                and ex.committed is not None
                and int(msg.attempt_id) == ex.committed
            )
            if winner_gone:
                # Rank 0: the committed winner gave up before BEGIN.
                # Not through _inbound: the other ranks drop this ABORT (their
                # export is unbound), so a local release would wait in the MIN.
                ex.winner_gone = True
                self.stats_winner_gone += 1
        if winner_gone:
            logger.warning(
                "pull owner: transfer %d: the committed reader (attempt %d) "
                "aborted before BEGIN: %s",
                msg.transfer_id,
                msg.attempt_id,
                msg.reason,
            )
            return
        with self._mutex:
            ex = self._exports.get(msg.transfer_id)
            if (
                ex is not None
                and ex.mark.attempt_id is not None
                and msg.attempt_id == 0
            ):
                # Explicitly identified attempts never accept a legacy wildcard
                # cancellation that could belong to a previous retry.
                verdict = AttemptGate.DROP_ATTEMPT
                self._gate.dropped[verdict] += 1
            else:
                verdict = self._gate.check(msg)
        if verdict != AttemptGate.ACCEPT:
            logger.debug(
                "pull owner: %s for transfer %d dropped (%s)",
                msg.type,
                msg.transfer_id,
                verdict,
            )
            return
        reason = (
            "reader_done" if msg.type == wire.MSG_DONE else f"reader_abort:{msg.reason}"
        )
        self._inbound.put((msg.transfer_id, reason))

    def _bind_on_begin(self, ex: _Export, attempt_id: int) -> bool:
        """Late D stage 2, caller holds ``_mutex``: the first BEGIN that
        names a candidate's attempt binds the export to that reader (only the
        committed winner can begin). Attempt 0, an unknown attempt or an
        aborted export never binds (V1 m4). Setting ``mark.attempt_id`` makes
        the owner refuse a wildcard ABORT from then on."""
        if ex.attempt_id is not None:
            return attempt_id == ex.attempt_id
        cand = ex.cands.get(attempt_id) if ex.cands else None
        if cand is None or ex.aborted:
            return False
        ex.attempt_id = attempt_id
        ex.mark.attempt_id = attempt_id
        ex.mark.reader_addr = cand.reader_addr
        ex.mark.reader_worker = cand.reader_worker
        ex.timeline["reader_addr"] = cand.reader_addr
        self._gate.bind(transfer_id=ex.mark.transfer_id, attempt_id=attempt_id)
        return True

    @staticmethod
    def _begin_matches(ex: _Export, attempt_id: int) -> bool:
        """A BEGIN names this export's attempt, or none (attempt 0: a reader
        whose mark carries no attempt and that has seen no publication yet,
        which under reader-first exports is every legacy reader). A stale
        BEGIN only queues the P-leg early: today's behaviour."""
        return attempt_id == AttemptGate.WILDCARD_ATTEMPT or attempt_id == ex.attempt_id

    def _on_begin(self, msg: wire.Message) -> None:
        if not self._await_reader:
            return
        with self._mutex:
            self.stats_begin_received += 1
            ex = self._exports.get(msg.transfer_id)
            if ex is None:
                # The D-leg was admitted before this P-leg reached us: park.
                if len(self._early_begin) >= _MAX_EARLY_BEGIN:
                    self.stats_begin_dropped += 1
                    return
                self.stats_begin_early += 1
                attempt = int(msg.attempt_id)
                prior = self._early_begin.get(msg.transfer_id)
                prefix = wire.begin_reuse_prefix(msg.payload)
                if prior is not None and prior[1] == attempt:
                    # Freeze the first claim even before begin_export. Keep
                    # the existing arrival refresh and different-attempt rule.
                    prefix = prior[2]
                self._early_begin[msg.transfer_id] = (
                    time.monotonic(),
                    attempt,
                    prefix,
                )
                return
            if ex.cands is not None:
                matches = self._bind_on_begin(ex, int(msg.attempt_id))
            else:
                matches = self._begin_matches(ex, int(msg.attempt_id))
            if not matches:
                self.stats_begin_dropped += 1
            else:
                self._remember_begin_prefix(ex, int(msg.attempt_id), wire.begin_reuse_prefix(msg.payload))
        if matches:
            self._begun.put(msg.transfer_id)

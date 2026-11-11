"""In-process loopback OwnerPlane / ReaderPlane for CPU tests and dry runs.

Implements the `pull/api.py` protocols without any transport: a chunk that an
owner publishes is immediately visible to the reader paired on the same
`FakePlaneHub` (by transfer_id). No bytes move; the reader records the
`dst_pages(chunk)` mapping it would have READ into. Used by
`planes.build_planes` as the fallback when `disaggregation.pull.owner` /
`importer` are not importable, and directly by
`test/registered/unit/disaggregation/test_symmetric_scheduler.py`.

Covers chunk publication and the failure paths for CPU tests.
Pure Python on purpose (no torch).

Late D: a reader built with ``addr`` is reachable
by address. ``FakePlaneHub.late_sender()`` stands in for the owner's late-D
sender thread: it routes an encoded ADMIT frame to the reader at its
endpoint, which puts an ``admit`` control into its late inbox
(``set_late_inbox``); a stage-2 RELEASE frame becomes a ``release``
control. An owner ABORT for a transfer the reader has not begun also becomes
a ``fail`` control there.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

import msgspec

from decoupled_engine.srt.disaggregation.pull import protocol as wire
from decoupled_engine.srt.disaggregation.pull.api import (
    ChunkReady,
    ExportMark,
    FinalMeta,
    ImportMark,
    TransferState,
)

logger = logging.getLogger(__name__)

DstPagesFn = Callable[[ChunkReady], Dict[str, Sequence[int]]]


class _ExportState(msgspec.Struct, kw_only=True):
    mark: ExportMark
    req_id: str
    chunks: List[ChunkReady] = []
    final: Optional[FinalMeta] = None
    aborted_reason: Optional[str] = None


class _ImportState(msgspec.Struct, kw_only=True):
    mark: ImportMark
    req_id: str
    dst_pages: DstPagesFn
    state: TransferState = TransferState.WAITING
    chunks: List[ChunkReady] = []
    # chunk_seq -> {pool: reader page indices}; what a real reader READs into.
    dst_map: Dict[int, Dict[str, List[int]]] = {}
    saw_last: bool = False
    final: Optional[FinalMeta] = None
    failure: Optional[str] = None
    done_sent: bool = False


class FakePlaneHub:
    """Routes CHUNK_READY / FINAL / DONE / ABORT between the owners and readers
    registered on it. Publications that arrive before `begin_import` are
    buffered per transfer_id (the router dispatches P-leg and D-leg
    simultaneously, so either side may be first)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: Dict[int, FakeOwnerPlane] = {}
        self._readers: Dict[int, FakeReaderPlane] = {}
        self._pending_chunks: Dict[int, List[ChunkReady]] = {}
        self._pending_final: Dict[int, FinalMeta] = {}
        self._pending_abort: Dict[int, str] = {}
        # Reader BEGINs (import admitted) that came before begin_export.
        self._pending_begin: set = set()
        # Late D: readers by endpoint (FakeReaderPlane(addr=...)), and ADMIT
        # frames sent to an endpoint without a reader.
        self._reader_addrs: Dict[str, FakeReaderPlane] = {}
        self.stats_admit_unrouted = 0

    def pair(
        self, *, auto_release: bool = True
    ) -> tuple[FakeOwnerPlane, FakeReaderPlane]:
        return FakeOwnerPlane(self), FakeReaderPlane(self, auto_release=auto_release)

    # -- owner side -----------------------------------------------------
    def register_owner(self, transfer_id: int, owner: FakeOwnerPlane) -> None:
        with self._lock:
            self._owners[transfer_id] = owner
            begun = transfer_id in self._pending_begin
            self._pending_begin.discard(transfer_id)
        if begun:
            owner._on_reader_begin(transfer_id)

    def route_chunk(self, chunk: ChunkReady) -> None:
        with self._lock:
            reader = self._readers.get(chunk.transfer_id)
            if reader is None:
                self._pending_chunks.setdefault(chunk.transfer_id, []).append(chunk)
                return
        reader._deliver_chunk(chunk)

    def route_final(self, final: FinalMeta) -> None:
        with self._lock:
            reader = self._readers.get(final.transfer_id)
            if reader is None:
                self._pending_final[final.transfer_id] = final
                return
        reader._deliver_final(final)

    def route_owner_abort(
        self,
        transfer_id: int,
        reason: str,
        *,
        reader_addr: Optional[str] = None,
        attempt_id: int = 0,
    ) -> None:
        with self._lock:
            reader = self._readers.get(transfer_id)
            if reader is None:
                self._pending_abort[transfer_id] = reason
                late = self._reader_addrs.get(reader_addr) if reader_addr else None
        if reader is None:
            if late is not None:
                # Late D: the reader may hold a never-begun late entry.
                late._forward_owner_abort(transfer_id, attempt_id)
            return
        reader._deliver_owner_abort(transfer_id, reason)

    # -- late D -----------------------------------------------------------
    def register_reader_addr(self, addr: str, reader: FakeReaderPlane) -> None:
        with self._lock:
            self._reader_addrs[addr] = reader

    def late_sender(self) -> _FakeLateSender:
        """Stands in for late_d.LateDSender: delivers to the reader at the
        endpoint, synchronously."""
        return _FakeLateSender(self)

    def route_late_frame(self, addr: str, frame: bytes) -> None:
        with self._lock:
            reader = self._reader_addrs.get(addr)
            if reader is None:
                self.stats_admit_unrouted += 1
                return
        reader._on_late_msg(wire.decode(frame))

    # -- reader side ----------------------------------------------------
    def register_reader(self, transfer_id: int, reader: FakeReaderPlane) -> None:
        with self._lock:
            self._readers[transfer_id] = reader
            chunks = self._pending_chunks.pop(transfer_id, [])
            final = self._pending_final.pop(transfer_id, None)
            abort = self._pending_abort.pop(transfer_id, None)
        for chunk in chunks:
            reader._deliver_chunk(chunk)
        if final is not None:
            reader._deliver_final(final)
        if abort is not None:
            reader._deliver_owner_abort(transfer_id, abort)

    def notify_done(self, transfer_id: int) -> None:
        """Reader DONE / ABORT -> the owner may release its Lease."""
        with self._lock:
            owner = self._owners.get(transfer_id)
        if owner is not None:
            owner._on_reader_done(transfer_id)

    def notify_begin(self, transfer_id: int) -> None:
        """Reader BEGIN (import admitted) -> the owner may queue its P-leg
        (reader-first exports). Parked until begin_export if it comes first."""
        with self._lock:
            owner = self._owners.get(transfer_id)
            if owner is None:
                self._pending_begin.add(transfer_id)
                return
        owner._on_reader_begin(transfer_id)

    def forget(self, transfer_id: int) -> None:
        with self._lock:
            self._owners.pop(transfer_id, None)
            self._readers.pop(transfer_id, None)
            self._pending_chunks.pop(transfer_id, None)
            self._pending_final.pop(transfer_id, None)
            self._pending_abort.pop(transfer_id, None)
            self._pending_begin.discard(transfer_id)

    def forget_reader(self, transfer_id: int) -> None:
        """Reader side only (late D release_pending): drop the parked
        publications; the owner's registration stays."""
        with self._lock:
            self._readers.pop(transfer_id, None)
            self._pending_chunks.pop(transfer_id, None)
            self._pending_final.pop(transfer_id, None)
            self._pending_abort.pop(transfer_id, None)


class _FakeLateSender:
    """late_d.LateDSender stand-in: one encoded pull-plane frame per send."""

    def __init__(self, hub: FakePlaneHub) -> None:
        self.hub = hub
        self.sent: List[tuple] = []

    def send(self, endpoint: str, is_ipv6: bool, frames) -> None:
        frame = frames if isinstance(frames, (bytes, bytearray)) else frames[0]
        self.sent.append((endpoint, bytes(frame)))
        self.hub.route_late_frame(endpoint, bytes(frame))


class FakeOwnerPlane:
    """OwnerPlane over a FakePlaneHub (pull/api.py OwnerPlane protocol)."""

    def __init__(self, hub: Optional[FakePlaneHub] = None) -> None:
        self.hub = hub if hub is not None else FakePlaneHub()
        self.exports: Dict[int, _ExportState] = {}
        self._released: List[int] = []
        self._begun: List[int] = []
        # Late D stage 2 (rank 0): committed attempts, and the transfers
        # whose committed winner aborted before its BEGIN.
        self._committed: Dict[int, int] = {}
        self._winner_gone: set = set()
        self._lock = threading.RLock()

    def begin_export(self, mark: ExportMark, req_id: str) -> None:
        with self._lock:
            self.exports[mark.transfer_id] = _ExportState(mark=mark, req_id=req_id)
        self.hub.register_owner(mark.transfer_id, self)

    def publish_chunk(self, chunk: ChunkReady) -> None:
        with self._lock:
            state = self.exports.get(chunk.transfer_id)
            if state is None:
                raise KeyError(
                    f"publish_chunk before begin_export: {chunk.transfer_id}"
                )
            state.chunks.append(chunk)
        self.hub.route_chunk(chunk)

    def publish_final(self, final: FinalMeta) -> None:
        with self._lock:
            state = self.exports.get(final.transfer_id)
            if state is None:
                raise KeyError(
                    f"publish_final before begin_export: {final.transfer_id}"
                )
            state.final = final
        self.hub.route_final(final)

    def poll_released(self) -> List[int]:
        with self._lock:
            released, self._released = self._released, []
        for transfer_id in released:
            self.exports.pop(transfer_id, None)
            self.hub.forget(transfer_id)
        return released

    def poll_begun(self) -> List[int]:
        """Reader-first exports: transfer_ids whose reader admitted the import."""
        with self._lock:
            begun, self._begun = self._begun, []
        return begun

    def late_sender(self) -> _FakeLateSender:
        """The late-D ADMIT sender for this hub (the real owner uses
        late_d.LateDSender)."""
        return self.hub.late_sender()

    def attempt_of(self, transfer_id: int) -> Optional[int]:
        """As OwnerPlane.attempt_of; a mark without an attempt exports as 0."""
        with self._lock:
            state = self.exports.get(transfer_id)
        if state is None:
            return None
        return state.mark.attempt_id or 0

    def note_committed(self, transfer_id: int, attempt_id: int) -> None:
        """As OwnerPlane.note_committed (late D stage 2, rank 0)."""
        with self._lock:
            if transfer_id in self.exports:
                self._committed[transfer_id] = int(attempt_id)

    def winner_gave_up(self, transfer_id: int) -> bool:
        with self._lock:
            return transfer_id in self._winner_gone

    def release_local(self, transfer_id: int) -> bool:
        """As OwnerPlane.release_local: end the export without an ABORT."""
        with self._lock:
            state = self.exports.pop(transfer_id, None)
            self._committed.pop(transfer_id, None)
        self.hub.forget(transfer_id)
        return state is not None

    def reader_abort_before_begin(self, transfer_id: int, attempt_id: int) -> None:
        """Test hook: a candidate's ABORT before any BEGIN. Only the
        committed winner's counts; every other one is dropped."""
        with self._lock:
            if self._committed.get(transfer_id) == int(attempt_id):
                self._winner_gone.add(transfer_id)

    def abort_export(self, transfer_id: int, reason: str) -> None:
        with self._lock:
            state = self.exports.get(transfer_id)
            if state is not None:
                state.aborted_reason = reason
        self.hub.route_owner_abort(
            transfer_id,
            reason,
            reader_addr=state.mark.reader_addr if state is not None else None,
            attempt_id=(state.mark.attempt_id or 0) if state is not None else 0,
        )
        if state is not None and not state.chunks:
            self.exports.pop(transfer_id, None)
            self.hub.forget(transfer_id)

    # -- hub / test hooks ----------------------------------------------
    def _on_reader_done(self, transfer_id: int) -> None:
        with self._lock:
            if transfer_id in self.exports and transfer_id not in self._released:
                self._released.append(transfer_id)

    def _on_reader_begin(self, transfer_id: int) -> None:
        with self._lock:
            if transfer_id in self.exports and transfer_id not in self._begun:
                self._begun.append(transfer_id)

    def release(self, transfer_id: int) -> None:
        """Test helper: simulate the reader's DONE for this transfer."""
        self._on_reader_done(transfer_id)

    def published_chunks(self, transfer_id: int) -> List[ChunkReady]:
        state = self.exports.get(transfer_id)
        return list(state.chunks) if state is not None else []


class FakeReaderPlane:
    """ReaderPlane over a FakePlaneHub (pull/api.py ReaderPlane protocol).

    `auto_release=True` mimics a reader whose READs complete instantly: the
    import becomes READY as soon as FINAL and the is_last chunk have both
    arrived, and DONE is sent to the owner at that moment. With
    `auto_release=False` the test drives `send_done()` explicitly so it can
    observe the owner holding pages under the Lease.
    """

    def __init__(
        self,
        hub: Optional[FakePlaneHub] = None,
        *,
        auto_release: bool = True,
        aux_reader: Optional[Callable[[FinalMeta, int], None]] = None,
        addr: Optional[str] = None,
    ) -> None:
        self.hub = hub if hub is not None else FakePlaneHub()
        self.auto_release = auto_release
        # Stands in for the READ of the owner's aux slot (FinalMeta.aux_index)
        # into the reader's local slot (ImportMark.local_aux_index, allocated
        # by the scheduler before begin_import). Skipped when the mark has none.
        self.aux_reader = aux_reader
        self.imports: Dict[int, _ImportState] = {}
        self._lock = threading.RLock()
        # Late D: the endpoint ADMITs reach this reader at, its late inbox
        # (set_late_inbox; None = ADMIT dropped), and the transfers it ended
        # before allocation (an ADMIT for them is dropped).
        self.addr = addr
        self._late_inbox = None
        self._ended: set = set()
        self.stats_admit_received = 0
        self.stats_admit_dropped = 0
        self.stats_fail_forwarded = 0
        if addr is not None:
            self.hub.register_reader_addr(addr, self)

    def begin_import(
        self, mark: ImportMark, req_id: str, dst_pages: DstPagesFn
    ) -> None:
        with self._lock:
            self.imports[mark.transfer_id] = _ImportState(
                mark=mark, req_id=req_id, dst_pages=dst_pages
            )
        self.hub.register_reader(mark.transfer_id, self)
        # Reader-first exports: the import is admitted (wire BEGIN).
        self.hub.notify_begin(mark.transfer_id)

    def poll(self, transfer_id: int) -> TransferState:
        state = self.imports.get(transfer_id)
        if state is None:
            return TransferState.FAILED
        return state.state

    def final_meta(self, transfer_id: int) -> Optional[FinalMeta]:
        """The OWNER's FinalMeta once READY (pull/api.py contract)."""
        state = self.imports.get(transfer_id)
        if state is None or state.state != TransferState.READY:
            return None
        return state.final

    def failure_reason(self, transfer_id: int) -> Optional[str]:
        state = self.imports.get(transfer_id)
        if state is None:
            return "unknown transfer_id"
        return state.failure

    def abort_import(self, transfer_id: int, reason: str) -> None:
        with self._lock:
            state = self.imports.pop(transfer_id, None)
            if state is not None:
                state.state = TransferState.ABORTED
                state.failure = reason
        self.hub.notify_done(transfer_id)
        self.hub.forget(transfer_id)

    def abort_pending(self, mark: ImportMark, reason: str) -> None:
        self._ended.add(mark.transfer_id)
        self.hub.notify_done(mark.transfer_id)
        self.hub.forget(mark.transfer_id)

    def release_pending(self, mark: ImportMark, reason: str) -> None:
        """As ReaderPlane.release_pending: no DONE / ABORT to the owner."""
        self._ended.add(mark.transfer_id)
        self.hub.forget_reader(mark.transfer_id)

    # -- late D ---------------------------------------------------------
    def set_late_inbox(self, inbox) -> None:
        self._late_inbox = inbox

    def _on_late_msg(self, msg: wire.Message) -> None:
        if msg.type not in (wire.MSG_ADMIT, wire.MSG_RELEASE):
            logger.warning("fake reader: unexpected late message %s", msg.type)
            return
        from decoupled_engine.srt.disaggregation.late_d import (
            CONTROL_ADMIT,
            CONTROL_RELEASE,
            LateDControl,
        )

        with self._lock:
            self.stats_admit_received += 1
            if self._late_inbox is None or msg.transfer_id in self._ended:
                self.stats_admit_dropped += 1
                return
            inbox = self._late_inbox
        inbox.put(
            LateDControl(
                kind=CONTROL_RELEASE if msg.type == wire.MSG_RELEASE else CONTROL_ADMIT,
                room=msg.transfer_id,
                seq=msg.seq,
                attempt=msg.attempt_id,
                t_recv=time.monotonic(),
                # As ReaderPlane._late_control: the optional ADMIT tag
                # (SGLANG_KV_SESSION_ADMIT_TAG); None for RELEASE.
                kv_session=wire.admit_kv_session(msg),
            )
        )

    def _forward_owner_abort(self, transfer_id: int, attempt_id: int) -> None:
        from decoupled_engine.srt.disaggregation.late_d import CONTROL_FAIL, LateDControl

        with self._lock:
            inbox = self._late_inbox
            if (
                inbox is None
                or transfer_id in self.imports
                or transfer_id in self._ended
            ):
                return
            self.stats_fail_forwarded += 1
        inbox.put(
            LateDControl(
                kind=CONTROL_FAIL,
                room=transfer_id,
                attempt=attempt_id,
                t_recv=time.monotonic(),
            )
        )

    # -- hub delivery ---------------------------------------------------
    def _deliver_chunk(self, chunk: ChunkReady) -> None:
        with self._lock:
            state = self.imports.get(chunk.transfer_id)
            if state is None or state.state in (
                TransferState.FAILED,
                TransferState.ABORTED,
            ):
                return  # late CHUNK_READY after ABORT is ignored
            # A real reader READs `chunk.pools` into these local pages.
            dst = {pool: list(pages) for pool, pages in state.dst_pages(chunk).items()}
            state.dst_map[chunk.chunk_seq] = dst
            state.chunks.append(chunk)
            if chunk.is_last:
                state.saw_last = True
            if state.state == TransferState.WAITING:
                state.state = TransferState.TRANSFERRING
        self._maybe_ready(chunk.transfer_id)

    def _deliver_final(self, final: FinalMeta) -> None:
        with self._lock:
            state = self.imports.get(final.transfer_id)
            if state is None or state.state in (
                TransferState.FAILED,
                TransferState.ABORTED,
            ):
                return
            state.final = final
        self._maybe_ready(final.transfer_id)

    def _deliver_owner_abort(self, transfer_id: int, reason: str) -> None:
        # Owner died / evicted mid-transfer: reader fails, scheduler recomputes.
        self.fail(transfer_id, f"owner aborted: {reason}")
        self.hub.notify_done(transfer_id)

    def _maybe_ready(self, transfer_id: int) -> None:
        send_done = False
        with self._lock:
            state = self.imports.get(transfer_id)
            if state is None or state.state == TransferState.READY:
                return
            if state.final is None or not state.saw_last:
                return
            if self.aux_reader is not None and state.mark.local_aux_index is not None:
                self.aux_reader(state.final, int(state.mark.local_aux_index))
            state.state = TransferState.READY
            send_done = self.auto_release and not state.done_sent
            if send_done:
                state.done_sent = True
        if send_done:
            self.hub.notify_done(transfer_id)

    # -- test hooks -----------------------------------------------------
    def fail(self, transfer_id: int, reason: str) -> None:
        with self._lock:
            state = self.imports.get(transfer_id)
            if state is None:
                return
            state.state = TransferState.FAILED
            state.failure = reason

    def send_done(self, transfer_id: int) -> None:
        """Explicit DONE when constructed with auto_release=False."""
        with self._lock:
            state = self.imports.get(transfer_id)
            if state is None or state.done_sent:
                return
            state.done_sent = True
        self.hub.notify_done(transfer_id)

    def received_chunks(self, transfer_id: int) -> List[ChunkReady]:
        state = self.imports.get(transfer_id)
        return list(state.chunks) if state is not None else []

    def dst_map(self, transfer_id: int) -> Dict[int, Dict[str, List[int]]]:
        state = self.imports.get(transfer_id)
        return dict(state.dst_map) if state is not None else {}


_DEFAULT_HUB: Optional[FakePlaneHub] = None
_DEFAULT_HUB_LOCK = threading.Lock()


def get_default_hub() -> FakePlaneHub:
    """Process-wide hub so two symmetric schedulers in one process (or the
    owner/reader halves of one scheduler) can loop back to each other."""
    global _DEFAULT_HUB
    with _DEFAULT_HUB_LOCK:
        if _DEFAULT_HUB is None:
            _DEFAULT_HUB = FakePlaneHub()
        return _DEFAULT_HUB


def build_fake_planes(
    hub: Optional[FakePlaneHub] = None,
    *,
    auto_release: bool = True,
    aux_reader: Optional[Callable[[FinalMeta, int], None]] = None,
    reader_addr: Optional[str] = None,
) -> tuple[FakeOwnerPlane, FakeReaderPlane]:
    hub = hub if hub is not None else get_default_hub()
    return FakeOwnerPlane(hub), FakeReaderPlane(
        hub, auto_release=auto_release, aux_reader=aux_reader, addr=reader_addr
    )

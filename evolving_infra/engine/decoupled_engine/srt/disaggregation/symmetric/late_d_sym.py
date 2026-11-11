"""Late D in symmetric mode (stage 1, S-1a, and stage 2): the ports of the
mode-neutral late-D core (disaggregation/late_d.py) for a symmetric worker.

Ports:

* Owner side (``SymOwnerBinderPort``, rank 0): a late P-leg is held after
  ``begin_export`` (no aux slot). The binder (``LateDPrefillBinder``)
  sends ADMIT {seq} to the reader rank 0 over the late-D sender thread when
  the P-leg enters its lead window. The reader's BEGIN, agreed on every rank
  in ``ExportTable.poll_released``, then queues the P-leg at its arrival
  position. Rank 0 owns the admission and bind timers
  (``SymOwnerTimer``); it votes FAILED and the MIN spreads it.
* Reader side (``SymReaderPort``, every rank; inbox on rank 0): a D-leg waits
  DEFERRED in ``ImportQueue.pending``; the ADMIT, turned into an ``admit``
  control on the reader IO thread, rides the request broadcast; the walk
  allocates it (``_prealloc`` + ``begin_import``, which sends BEGIN) or holds
  it. A never-allocated entry ends with an error and no recompute (rule X):
  DEFERRED / HELD TTL (``expire``, ABORT to the owner), an owner ABORT
  (``fail``, no ABORT), or an owner in another late-D mode (``fail``).

Stage 2 (``choose``, C8): a P-leg names its candidate readers in
``kv_export.candidates`` and carries the router's ``late_d`` field. The owner
port's ``commit`` sends ADMIT(a_w) to the winner and RELEASE(a_i) to every
other candidate, or, for LOCAL (WINNER_LOCAL), puts a ``local`` control into
the owner inbox (every rank converts the held P-leg into a Local request)
and RELEASEs every candidate. The winner's BEGIN binds the owner's export
(OwnerPlane, sketch h).

Rank-0 data rule (core docstring): the owner port reads only lockstep or
rank-0 private state and never calls ``poll_released`` / ``poll_begun`` or a
consensus round.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from decoupled_engine.srt.disaggregation.late_d import (
    ADMITTED,
    CONTROL_FAIL,
    CONTROL_LOCAL,
    DEFERRED,
    FAIL_MODE_MISMATCH,
    HELD,
    LATE_D_LOCAL_KEY,
    LATE_D_MODE_MISMATCH_ERR,
    LATE_D_OWNER_ABORT_ERR,
    LATE_D_RELEASED_ERR,
    LATE_D_TTL_ERR,
    WINNER_LOCAL,
    AheadAll,
    AheadByArrival,
    LateDBinderPort,
    LateDChoice,
    LateDControl,
    LateDInbox,
    LateDReaderPort,
    late_d_request_error,
    late_d_timer_verdict,
)
from decoupled_engine.srt.disaggregation.pull import protocol as wire
from decoupled_engine.srt.disaggregation.utils import is_aborted
from decoupled_engine.srt.environ import envs

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.symmetric.export_table import ExportTable
    from decoupled_engine.srt.disaggregation.symmetric.import_queue import (
        ImportEntry,
        ImportQueue,
    )
    from decoupled_engine.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Late states of a reader entry that hold nothing yet (rule X applies).
_UNALLOCATED = (DEFERRED, ADMITTED, HELD)


class SymDest(NamedTuple):
    """A late P-leg's reader: its rank-0 pull-plane endpoint (the binder's
    per-reader cap key), its worker name (logs) and, in stage 2, its
    candidate attempt (ADMIT and RELEASE quote it)."""

    d_key: str
    label: str
    attempt: int = 0


# Stage 2: the owner itself as the winner (no reader, no ADMIT).
LOCAL_DEST = SymDest(d_key=LATE_D_LOCAL_KEY, label="local")


def _is_ipv6(endpoint: str) -> bool:
    return "[" in endpoint


def symmetric_late_d_leg_error(
    field_value: Any, export_mark: Any, import_mark: Any, engine_mode: str,
    *, compute_before_allocation: bool = False,
) -> Optional[str]:
    """Why a symmetric leg's marks and ``late_d`` field (stage 2)
    are refused with HTTP 400, or None. ``field_value`` None: a P-leg must
    not carry candidates. A P-leg's candidates and the field must agree
    (every cand_idx below k; the default is a candidate, or LOCAL with
    local_ok); a D-leg must quote its candidate attempt."""
    if field_value is None:
        if export_mark is not None and export_mark.candidates:
            return "kv_export.candidates needs the late_d field (choose mode)"
        return None
    role = (
        "prefill"
        if export_mark is not None
        else "decode" if import_mark is not None else None
    )
    error = late_d_request_error(field_value, engine_mode, role, symmetric=True)
    if error is not None:
        return error
    if role == "decode":
        if import_mark.attempt_id is None:
            return "a late_d D-leg needs kv_import.attempt_id"
        return None
    if not export_mark.candidates:
        return "a late_d P-leg needs kv_export.candidates"
    if compute_before_allocation and field_value.get("local_ok", False):
        return "symmetric compute-before-allocation choose requires remote D candidates (router LOCAL off)"
    k = field_value["k"]
    idx = {c.cand_idx for c in export_mark.candidates}
    if max(idx) >= k:
        return f"kv_export.candidates cand_idx must be below late_d.k ({k})"
    default = field_value["default_idx"]
    if default != WINNER_LOCAL and default not in idx:
        return f"late_d.default_idx {default} names no candidate"
    return None


class SymOwnerTimer:
    """Owner rank 0: the admission timer (hold -> first ADMIT) and the bind
    timer (first ADMIT -> BEGIN agreed), both on the ExportTable clock."""

    def __init__(self, admission_s: float, bind_s: float):
        self.admission_s = float(admission_s)
        self.bind_s = float(bind_s)

    def verdict(
        self, *, created: float, first_admit: Optional[float], now: float
    ) -> Optional[Tuple[str, float]]:
        return late_d_timer_verdict(
            created, first_admit, now, self.admission_s, self.bind_s
        )


class SymOwnerBinderPort(LateDBinderPort):
    """Owner rank 0 of a symmetric worker: late P-legs wait held in the
    ExportTable; ADMIT goes to the reader's rank-0 pull-plane channel through
    ``sender`` (late_d.LateDSender, own PUSH sockets, never the OwnerPlane
    IO thread)."""

    role = "prefill"
    symmetric = True

    def __init__(
        self,
        table: ExportTable,
        sender: Any,
        *,
        is_leader: bool = True,
        arrival_order: bool = True,
        choose: bool = False,
        owner_inbox: Optional[LateDInbox] = None,
    ):
        self.table = table
        self.sender = sender
        self.is_leader = is_leader
        self.arrival_order = arrival_order
        # Stage 2: rooms with candidates bind by choice; a LOCAL commit puts
        # a ``local`` control into ``owner_inbox`` (drained by rank 0's
        # control source, never held back). ``bound_view``: this owner's
        # bound view (5.1), for local_load; None reports 0.
        self.choose = choose
        self.owner_inbox = owner_inbox
        self.bound_view: Optional[Callable[[], int]] = None
        # SGLANG_KV_SESSION_ADMIT_TAG: this worker's retention registry
        # (KvSessionRetention), resolved at call time; None, or a getter that
        # returns None, without --enable-kv-session-retention.
        self.kv_sessions: Optional[Callable[[], Any]] = None

    def is_bound_runnable(self, req: Req) -> bool:
        if is_aborted(req):
            return False
        if getattr(self.table, "compute_before_allocation", False) is True:
            # Only this mode queues unbound P legs. Keep its state required;
            # legacy count-mode waiting requests were already bound at entry.
            return not req.pending_bootstrap
        return True

    def ahead(self, waiting: List[Req]) -> Any:
        if self.arrival_order:
            return AheadByArrival(waiting, self)
        return AheadAll(waiting, self)

    def is_late(self, req: Req) -> bool:
        entry = self.table.entries.get(req.transfer_id)
        return entry is not None and entry.late

    def is_failed(self, key: int) -> bool:
        entry = self.table.entries.get(key)
        return entry is None or entry.abort_reason is not None

    def destination(self, key: int, req: Req) -> Optional[SymDest]:
        mark = req.export_mark
        if mark is None or mark.reader_addr is None:
            # A candidate export binds by choice only (stage 2).
            return None
        return SymDest(d_key=mark.reader_addr, label=mark.reader_worker)

    def note_admit(self, key: int, dest: SymDest, now: float) -> None:
        # One clock (V2 m3): the table's, never the binder's time.time().
        entry = self.table.entries.get(key)
        if entry is not None and entry.first_admit_at is None:
            entry.first_admit_at = self.table.clock()

    def send_admit(self, dest: SymDest, key: int, seq: int) -> None:
        if dest.d_key == LATE_D_LOCAL_KEY:
            return  # LOCAL: the ``local`` control was the commit
        attempt = getattr(dest, "attempt", 0)
        if not attempt:
            # Stage 1: the export's own attempt.
            attempt_of = getattr(self.table.owner_plane, "attempt_of", None)
            attempt = (attempt_of(key) if callable(attempt_of) else None) or 0
        frame = wire.encode(
            wire.make_admit(
                transfer_id=key,
                attempt_id=attempt,
                seq=seq,
                kv_session=self.admit_kv_session(key),
            )
        )
        self.sender.send(dest.d_key, _is_ipv6(dest.d_key), frame)

    def admit_kv_session(self, key: int) -> Optional[dict]:
        """SGLANG_KV_SESSION_ADMIT_TAG: the P
        leg's kv_session tag, sent in the ADMIT so that only the chosen reader
        registers its D leg.

        Only a live P registration is sent, as the registration's own
        (session_id, version). A P leg whose tag the registry fenced (a
        version older than the session's current one, or an ended session)
        keeps its raw body tag but holds no live registration; sending that
        tag could create a live registration on the reader at a version the
        router never releases. None when the flag is off, retention is off,
        or the P leg has no live registration; the ADMIT is then unchanged.

        A stage-2 commit runs on the router-link thread. This only reads one
        registry entry and its state, which the scheduler thread changes; a
        concurrent change gives the old or the new answer, and both are
        safe."""
        if not envs.SGLANG_KV_SESSION_ADMIT_TAG.get():
            return None
        tracker = self.kv_sessions() if callable(self.kv_sessions) else None
        entry = self.table.entries.get(key)
        if tracker is None or entry is None:
            return None
        reg = getattr(entry.req, "_kv_session_registration", None)
        if not tracker._live(reg):
            return None
        return {"session_id": reg.session_id, "version": reg.version}

    def send_release(self, dest: SymDest, key: int, seq: int) -> None:
        frame = wire.encode(
            wire.make_release(transfer_id=key, attempt_id=dest.attempt, seq=seq)
        )
        self.sender.send(dest.d_key, _is_ipv6(dest.d_key), frame)

    # -- stage 2 ------------------------------------------------------------
    def choice(self, req: Req) -> Optional[LateDChoice]:
        mark = req.export_mark
        field_value = getattr(req, "late_d", None)
        if not self.choose or field_value is None or mark is None:
            return None
        if not mark.candidates:
            return None  # a one-reader P-leg binds as in stage 1
        return LateDChoice.of(field_value)

    def local_load(self, key: int) -> int:
        """L4 local_load: this owner's bound view minus
        the room's own prompt, which runs here in both outcomes."""
        if self.bound_view is None:
            return 0
        entry = self.table.entries.get(key)
        tokens = len(entry.req.origin_input_ids) if entry is not None else 0
        return max(0, int(self.bound_view()) - tokens)

    def candidates(self, key: int) -> Dict[int, SymDest]:
        """Every candidate of the P-leg (known at arrival: no INTENT in
        symmetric mode), with its rank-0 endpoint and attempt."""
        entry = self.table.entries.get(key)
        mark = entry.req.export_mark if entry is not None else None
        if mark is None or not mark.candidates:
            return {}
        return {
            c.cand_idx: SymDest(
                d_key=c.reader_addr, label=c.reader_worker, attempt=c.attempt_id
            )
            for c in mark.candidates
        }

    def commit(
        self, key: int, seq: int, winner_idx: int, now: float
    ) -> Optional[Tuple[SymDest, List[SymDest]]]:
        """The room's only commit: ADMIT(a_w) to the winner and
        RELEASE(a_i) to the other candidates; LOCAL: a ``local`` control into
        the owner inbox and RELEASE to every candidate. None (nothing sent)
        for a room that is gone or failed, or a winner it cannot take."""
        entry = self.table.entries.get(key)
        if entry is None or entry.abort_reason is not None or is_aborted(entry.req):
            return None
        cands = self.candidates(key)
        if winner_idx == WINNER_LOCAL:
            # A started P export cannot be requeued as a fresh local prefill.
            # Request validation refuses this offer; keep the commit defensive.
            if entry.compute_before_allocation:
                return None
            choice = self.choice(entry.req)
            if choice is None or not choice.local_ok or self.owner_inbox is None:
                return None
            winner = LOCAL_DEST
            losers = [cands[i] for i in sorted(cands)]
            self.owner_inbox.put(
                LateDControl(
                    kind=CONTROL_LOCAL, room=key, seq=seq, t_recv=time.monotonic()
                )
            )
        else:
            winner = cands.get(winner_idx)
            if winner is None:
                return None
            losers = [cands[i] for i in sorted(cands) if i != winner_idx]
            note = getattr(self.table.owner_plane, "note_committed", None)
            if callable(note):
                # Rank 0 can now accept the winner's ABORT before BEGIN.
                note(key, winner.attempt)
            self.send_admit(winner, key, seq)
        self.note_admit(key, winner, now)
        for loser in losers:
            self.send_release(loser, key, seq)
        return winner, losers

    def forget(self, key: int) -> None:
        """The ExportTable owns the room's state."""

    def created_at(self, req: Req, now: float) -> float:
        # On the binder's clock: ``now`` minus the time held so far.
        entry = self.table.entries.get(req.transfer_id)
        if entry is None:
            return now
        created = entry.created_at if entry.compute_before_allocation else entry.held_since
        return now if created is None else now - max(0.0, self.table.clock() - created)

    def bound_time(self, req: Req) -> float:
        # The P-leg entered the waiting queue (_queue_for_prefill).
        entry = self.table.entries.get(req.transfer_id)
        if entry is not None and entry.compute_before_allocation:
            return entry.bound_at
        return req.time_stats.wait_queue_entry_time

    def first_admit_time(self, key: int) -> Optional[float]:
        entry = self.table.entries.get(key)
        return None if entry is None else entry.first_admit_at


class SymReaderPort(LateDReaderPort):
    """Every reader rank of a symmetric worker. Rank 0 owns the inbox the
    ReaderPlane IO thread fills (ADMIT, owner ABORT for a never-begun
    import) and the ended-key LRU; ``expire`` and ``fail`` end an entry on
    every rank in the same iteration (ImportQueue.late_end)."""

    role = "decode"
    hold_unknown = True
    symmetric = True
    # Keys of late entries that left the late states (allocated or ended):
    # their resent or stale controls are dropped, not held back.
    ENDED_CAP = 8192

    def __init__(self, queue: ImportQueue, *, is_leader: bool):
        self.queue = queue
        self.is_leader = is_leader
        self.inbox = LateDInbox()
        self._ended: OrderedDict[int, None] = OrderedDict()
        if is_leader:
            set_inbox = getattr(queue.reader_plane, "set_late_inbox", None)
            if callable(set_inbox):
                set_inbox(self.inbox)

    # -- rank 0 -----------------------------------------------------------
    def drain_inbox(self) -> List[LateDControl]:
        return self.inbox.drain() + self._mode_mismatch_fails()

    def _mode_mismatch_fails(self) -> List[LateDControl]:
        """Entries whose owner registered with another late-D mode
        end at once (503 late_d_mode_mismatch), not after the TTL."""
        get = getattr(self.queue.reader_plane, "late_d_mismatched_owners", None)
        owners = get() if callable(get) else None
        if not owners:
            return []
        now = time.monotonic()
        out = []
        for entry in self.queue.pending:
            mark = entry.req.import_mark
            if (
                entry.late_d_state in _UNALLOCATED
                and mark is not None
                and mark.owner_addr in owners
            ):
                out.append(
                    LateDControl(
                        kind=CONTROL_FAIL,
                        room=entry.transfer_id,
                        t_recv=now,
                        detail=FAIL_MODE_MISMATCH,
                    )
                )
        return out

    def key_state(self, key: int) -> str:
        if key in self.queue.active:
            return "active"
        if key in self._ended:
            return "ended"
        return "unknown"

    def note_ended(self, key: int) -> None:
        self._ended[key] = None
        self._ended.move_to_end(key)
        while len(self._ended) > self.ENDED_CAP:
            self._ended.popitem(last=False)

    def batch_import_keys(self, recv_reqs: Any) -> Set[int]:
        """The transfer ids of the D-legs in the raw batch about to
        be broadcast (sub-requests of a batch input included)."""
        keys: Set[int] = set()
        for raw in recv_reqs or ():
            subs = getattr(raw, "batch", None)
            for sub in subs if isinstance(subs, list) else (raw,):
                mark = getattr(sub, "kv_import", None)
                if isinstance(mark, dict) and "transfer_id" in mark:
                    try:
                        keys.add(int(mark["transfer_id"]))
                    except (TypeError, ValueError):
                        pass
        return keys

    # -- every rank ---------------------------------------------------------
    def expire(self, entry: ImportEntry, key: int, reason: str) -> None:
        # DEFERRED / HELD TTL: 503 and ABORT to the owner (rule X).
        self.queue.late_end(
            entry, err_type=LATE_D_TTL_ERR, message=reason, notify_owner=True
        )

    def release(self, entry: ImportEntry, key: int) -> None:
        # Stage 2: another candidate won the transfer, or the owner
        # runs it Local. 409, no ABORT to the owner, no recompute.
        self.queue.late_end(
            entry,
            err_type=LATE_D_RELEASED_ERR,
            message=f"room {key} released",
            notify_owner=False,
            status_code=HTTPStatus.CONFLICT,
        )

    def fail(self, entry: ImportEntry, key: int, detail: str) -> None:
        if detail == FAIL_MODE_MISMATCH:
            # The owner never holds this P-leg for us: tell it (ABORT) so its
            # export does not wait for this reader.
            self.queue.late_end(
                entry,
                err_type=LATE_D_MODE_MISMATCH_ERR,
                message=f"Late D: owner of transfer {key} runs another late-D mode",
                notify_owner=True,
            )
            return
        # The owner ended the transfer; no ABORT back, no recompute.
        self.queue.late_end(
            entry,
            err_type=LATE_D_OWNER_ABORT_ERR,
            message=f"Late D: owner aborted transfer {key} before allocation",
            notify_owner=False,
        )

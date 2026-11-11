"""Defer decode KV allocation or decode worker selection.

``--disaggregation-late-d alloc`` (default ``off``, identical on prefill and
decode; separate prefill/decode uses Mooncake):

* The router still sends the decode (D) leg at arrival. D keeps the request
  DEFERRED: it is in the preallocation queue and its bootstrap handshake is
  done, but it holds no KV, no request slot and no metadata buffer. D attn-TP
  rank 0 sends INTENT(room, D endpoint) to the prefill (P) attn-TP rank 0.
* The P rank-0 binder sends ADMIT(room) at the configured allocation trigger
  (``LateDPrefillBinder``). The count and work modes prepare D before P
  starts. The need mode lets P compute before D is ready, subject to a KV
  budget and a timeout for retaining completed prefill KV. The first_chunk
  trigger sends ADMIT at the first prefill CPU forward-launch boundary;
  this marker is not a GPU kernel start time. P state changes remain
  synchronized across TP ranks.
* D rank 0 injects each ADMIT into the request broadcast, so every D rank
  applies it in the same iteration. D allocates the ADMITted request just in
  time; one that does not fit is HELD and skipped (never a FIFO break), and
  the oldest HELD request becomes a reservation claim after ``claim_iters``
  preallocation passes (``LateDDecode``).
* P admission and bind timers run on the leader. D expiration for a deferred
  request is decided on rank 0 and broadcast like an ADMIT.

``--disaggregation-late-d choose`` adds deferred D selection:

* The router sends the P leg and one D leg to each of K candidate decodes
  at arrival. Each leg carries a ``late_d`` field: the P leg names
  K and the default candidate, each D leg its candidate index. A request
  without the field binds as in alloc mode.
* Each candidate keeps its leg DEFERRED and sends INTENT with its candidate
  index (INTENT v2). When the configured selection trigger fires, the P binder
  offers the candidates whose INTENT it holds and that have headroom to the
  router over the control link (``late_d_link.LateDLink``; CHOICE_PENDING).
  The router's decision, or at the choice timeout the default candidate,
  is committed on the link thread: ADMIT to the winner, RELEASE to every
  other candidate with an INTENT, then a tombstone, so a later INTENT gets
  RELEASE at once. Without a router the binder commits the default itself.
* A released candidate ends its leg with HTTP 409 on every rank in lockstep
  (a ``release`` control in the request broadcast) and never sends ABORT.

The binder and the decode walk are mode-neutral. They reach the transport and
the queues of a disaggregation mode through two ports: ``LateDBinderPort``
(the owner side, which prefills) and ``LateDReaderPort`` (the reader side,
which decodes). The PD (Mooncake) adapter is ``LateDConnState`` with
``PdBinderPort`` and ``PdReaderPort``.

Symmetric mode supports alloc and choose with count mode. Computing before
reader allocation uses alloc + need + first_chunk with a fixed reader,
overlap scheduling and context parallel size 1. It does not support choose
or the lead trigger with need mode.
"""

from __future__ import annotations


import abc
import asyncio
import bisect
import dataclasses
import itertools
import logging
import math
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import zmq

from decoupled_engine.srt.disaggregation.base.conn import KVPoll
from decoupled_engine.srt.disaggregation.late_d_link import (
    REASON_NO_ROUTER,
    WINNER_LOCAL,
    ChoiceItem,
    LateDLink,
)
from decoupled_engine.srt.disaggregation.utils import is_aborted, prepare_abort
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.common.conn import CommonKVManager
    from decoupled_engine.srt.disaggregation.decode import DecodeRequest
    from decoupled_engine.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

LATE_D_OFF = "off"
LATE_D_ALLOC = "alloc"
LATE_D_CHOOSE = "choose"
LATE_D_MODES = (LATE_D_OFF, LATE_D_ALLOC, LATE_D_CHOOSE)

# Allocation preparation (--disaggregation-late-d-lead-mode): count requests,
# estimate prefill work, or compute before binding within a retained-KV budget.
LEAD_COUNT = "count"
LEAD_WORK = "work"
LEAD_NEED = "need"
LEAD_MODES = (LEAD_COUNT, LEAD_WORK, LEAD_NEED)

# First frame of the connector messages. INTENT: D rank 0 -> P rank 0
# bootstrap thread. ADMIT and RELEASE (stage 2): P rank 0 -> D rank 0 decode
# thread.
INTENT_HEADER = b"LATE_D_INTENT"
ADMIT_HEADER = b"LATE_D_ADMIT"
RELEASE_HEADER = b"LATE_D_RELEASE"
# Optional last frame of a decode KV-args registration (capability check).
REGISTRATION_TAG = b"late_d="

# D entry states (DecodeRequest.late_d_state; None = not a late entry).
DEFERRED = "deferred"
ADMITTED = "admitted"
HELD = "held"
ALLOCATED = "allocated"
RELEASED = "released"  # stage 2: another candidate won the room

# P binder room states (P leader bookkeeping only).
CHOICE_PENDING = "choice_pending"  # stage 2: waits for the link's commit
ADMIT_SENT = "admit_sent"
HELD_AT_D = "held_at_d"
BOUND = "bound"

# Broadcast control kinds. ``fail`` and ``local`` are symmetric-mode kinds;
# PD never produces them.
CONTROL_ADMIT = "admit"
CONTROL_EXPIRE = "expire"
CONTROL_RELEASE = "release"
CONTROL_FAIL = "fail"
CONTROL_LOCAL = "local"

# The per-request ``late_d`` field (stage 2, set by the router): version and
# the largest candidate count (the link's eligible mask is a u64).
LATE_D_FIELD_VERSION = 1
LATE_D_MAX_CANDIDATES = 64
# Symmetric stage 2: the binder's per-reader key of a LOCAL commit (the owner
# runs the P-leg itself; WINNER_LOCAL is its candidate index).
LATE_D_LOCAL_KEY = "local"
# FINISH_ABORT err_type of a released candidate (HTTP 409).
LATE_D_RELEASED_ERR = "late_d_released"
# FINISH_ABORT err_type of a DEFERRED entry ended by its TTL (HTTP 503).
LATE_D_TTL_ERR = "late_d_ttl"
# Symmetric mode. Rule X: a late entry that never got KV ends with an
# error, never with a recompute. These errors (HTTP 503 unless noted): a
# never-admitted reader entry whose owner aborted or whose owner runs
# another late-D mode; an owner P-leg ended by its admission or bind timer,
# or because its reader gave up while the P-leg was held.
LATE_D_OWNER_ABORT_ERR = "late_d_owner_abort"
LATE_D_MODE_MISMATCH_ERR = "late_d_mode_mismatch"
LATE_D_ADMISSION_TIMEOUT_ERR = "late_d_admission_timeout"
LATE_D_BIND_TIMEOUT_ERR = "late_d_bind_timeout"
LATE_D_READER_GONE_ERR = "late_d_reader_gone"
# LateDControl.detail of a ``fail`` control (why the entry ends).
FAIL_OWNER_ABORT = "owner_abort"
FAIL_MODE_MISMATCH = "mode_mismatch"
# A late-D outcome's err_type is "late_d_<reason>". /v1/completions sends the
# reason in this header.
LATE_D_ERR_PREFIX = "late_d_"
LATE_D_REASON_HEADER = "X-Late-D-Reason"
# Symmetric stage 2: a choose P-leg's 2xx answer says whether it
# is the stream ("local": the owner runs it) or a one-token stub ("stub").
LATE_D_LEG_HEADER = "X-Late-D-Leg"
LATE_D_OUTCOME_LOCAL = "local"
# Process-local mark that /v1/completions sets on the requests it converts:
# the route answers LateDLegEnded with the leg's HTTP status. In v1 choose
# mode serves only this route; /generate and chat keep in-band errors.
LATE_D_STATUS_ROUTE_ATTR = "_late_d_status_route"


def _resolved(server_args, field):
    """The resolved value of ``field`` (declarations included), not the raw input."""
    from decoupled_engine.srt.arg_groups.overrides import resolution_result

    return resolution_result(server_args, field)


@dataclass(frozen=True)
class LateDConfig:
    lead_min_reqs: int = 2
    lead_max_reqs: int = 8
    lead_max_tokens: int = 4 * 8192
    max_outstanding_per_d: int = 4
    hold_detect_s: float = 0.2
    claim_iters: int = 20
    admission_timeout_s: float = 1800.0
    deferred_ttl_s: float = 1920.0
    # Reader rank 0, ports with hold_unknown only: how long a control may wait
    # for the leg of its key, and how many may wait.
    early_control_ttl_s: float = 60.0
    early_control_cap: int = 4096
    # Stage 2: how long the owner waits for the router's decision.
    choice_timeout_s: float = 0.02
    # Lead window sizing (PD): LEAD_COUNT or LEAD_WORK. Work mode only:
    # lead_margin_iters is added to ceil(T_bind / T_iter); lead_kv_max_tokens
    # caps the prompt tokens bound ahead (D KV held early; 0: no cap).
    lead_mode: str = LEAD_COUNT
    allocation_trigger: str = "lead"
    lead_margin_iters: int = 1
    lead_kv_max_tokens: int = 0
    # Symmetric mode only (for_symmetric; the defaults keep PD unchanged).
    # bind_timeout_s: the owner's bind timer (PD keeps the bootstrap timeout).
    # held_ttl_s: reader rank 0 ends an ADMITTED / HELD entry not allocated
    # within it (0: off). claim_holds_local: a memory-blocked claim holds back
    # requests without an export. claim_escalate_passes: after that many
    # memory-blocked passes the claim also charges new P-leg starts, capped
    # (0: off). arrival_order: order bound work by arrival. walk_digest_every:
    # debug lockstep check. export_aux_reserve: imports never take the last aux
    # slots.
    bind_timeout_s: float = 0.0
    held_ttl_s: float = 0.0
    claim_holds_local: bool = False
    claim_escalate_passes: int = 0
    arrival_order: bool = False
    walk_digest_every: int = 0
    export_aux_reserve: int = 0

    @classmethod
    def from_disagg(cls, disagg: Any, chunked_prefill_size: Optional[int]):
        admission = admission_timeout_s()
        ttl = envs.SGLANG_LATE_D_DEFERRED_TTL.get()
        lead_max_tokens = disagg.disaggregation_late_d_lead_max_tokens
        if lead_max_tokens is None:
            cps = chunked_prefill_size if (chunked_prefill_size or 0) > 0 else 8192
            lead_max_tokens = 4 * cps
        return cls(
            lead_min_reqs=disagg.disaggregation_late_d_lead_min_reqs,
            lead_max_reqs=disagg.disaggregation_late_d_lead_max_reqs,
            lead_max_tokens=lead_max_tokens,
            max_outstanding_per_d=disagg.disaggregation_late_d_max_outstanding_per_d,
            hold_detect_s=disagg.disaggregation_late_d_hold_detect_ms / 1000.0,
            claim_iters=disagg.disaggregation_late_d_claim_iters,
            admission_timeout_s=admission,
            deferred_ttl_s=float(ttl) if ttl is not None else admission + 120.0,
            # A stage-1 config namespace may not carry the stage-2 field.
            choice_timeout_s=getattr(
                disagg, "disaggregation_late_d_choice_timeout_ms", 20
            )
            / 1000.0,
            lead_mode=getattr(disagg, "disaggregation_late_d_lead_mode", LEAD_COUNT),
            allocation_trigger=getattr(disagg, "disaggregation_late_d_allocation_trigger", "lead"),
            lead_margin_iters=getattr(
                disagg, "disaggregation_late_d_lead_margin_iters", 1
            ),
            # Work mode: an explicit token cap bounds the D KV held early.
            lead_kv_max_tokens=disagg.disaggregation_late_d_lead_max_tokens or 0,
        )

    @classmethod
    def for_symmetric(cls, disagg: Any, chunked_prefill_size: Optional[int]):
        """The symmetric-mode config."""
        base = cls.from_disagg(disagg, chunked_prefill_size)
        bind = symmetric_bind_timeout_s()
        held = envs.SGLANG_LATE_D_HELD_TTL.get()
        escalate = envs.SGLANG_LATE_D_CLAIM_ESCALATE_PASSES.get()
        reserve = getattr(disagg, "disaggregation_late_d_export_aux_reserve", None)
        return dataclasses.replace(
            base,
            early_control_ttl_s=float(envs.SGLANG_LATE_D_EARLY_CONTROL_TTL.get()),
            bind_timeout_s=bind,
            held_ttl_s=float(held) if held is not None else bind + 120.0,
            claim_holds_local=bool(envs.SGLANG_LATE_D_CLAIM_HOLDS_LOCAL.get()),
            claim_escalate_passes=(
                int(escalate) if escalate is not None else 2 * base.claim_iters
            ),
            arrival_order=bool(envs.SGLANG_LATE_D_ARRIVAL_ORDER.get()),
            walk_digest_every=int(envs.SGLANG_LATE_D_WALK_DIGEST_EVERY.get()),
            export_aux_reserve=(
                int(reserve) if reserve is not None else base.lead_max_reqs + 2
            ),
        )


def symmetric_bind_timeout_s() -> float:
    """Symmetric owner bind timer: from the first ADMIT
    to the reader's BEGIN agreed on every rank. Unset: max(900, bootstrap
    timeout, waiting timeout)."""
    bind = envs.SGLANG_LATE_D_BIND_TIMEOUT.get()
    if bind is not None:
        return float(bind)
    return float(
        max(
            900,
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get(),
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get(),
        )
    )


def admission_timeout_s() -> float:
    """P bound from sender creation to the first ADMIT."""
    admission = envs.SGLANG_LATE_D_ADMISSION_TIMEOUT.get()
    if admission is None:
        return float(max(1800, envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()))
    return float(admission)


def late_d_timer_verdict(
    created: float,
    first_admit: Optional[float],
    now: float,
    admission_s: float,
    bind_s: float,
) -> Optional[Tuple[str, float]]:
    """The owner's late-D timers, on one clock: the admission
    timer runs from creation to the first ADMIT, the bind timer from the
    first ADMIT. None while the running timer has time left, else
    ``(phase, elapsed)``."""
    if first_admit is None:
        elapsed, bound = now - created, admission_s
        phase = "no ADMIT yet, admission timer"
    else:
        elapsed, bound = now - first_admit, bind_s
        phase = "since the first ADMIT, bind timer"
    if elapsed < bound:
        return None
    return phase, elapsed


@dataclass(frozen=True)
class LateDIntent:
    d_ip: str
    d_port: int
    d_nnodes: int
    recv_time: float
    # Stage 2: the candidate index of the D leg (0 for a stage-1 INTENT).
    cand_idx: int = 0

    @property
    def d_key(self) -> Tuple[str, int]:
        return (self.d_ip, self.d_port)

    @property
    def label(self) -> str:
        return f"{self.d_ip}:{self.d_port}"


@dataclass
class LateDControl:
    """A reader-side control event that rides the request broadcast from
    attn-TP rank 0, so every reader rank applies it in the same iteration and
    order. ``attempt`` names the owner-reader pairing where the transport has
    one (0 = none); ``t_recv`` is the rank-0 monotonic receive time; ``detail``
    says why a ``fail`` control ends its entry (FAIL_*)."""

    kind: str
    room: int
    seq: int = 0
    attempt: int = 0
    t_recv: float = field(default=0.0, compare=False)
    detail: str = field(default="", compare=False)
    # Symmetric ADMIT only (SGLANG_KV_SESSION_ADMIT_TAG): the owner's P-leg
    # kv_session tag ``(session_id, version)`` for the reader's D leg.
    kv_session: Optional[Tuple[str, int]] = field(default=None, compare=False)


def intent_parts(
    room: int,
    d_ip: str,
    d_port: int,
    d_nnodes: int,
    cand_idx: Optional[int] = None,
) -> List[bytes]:
    """INTENT frames. A stage-2 D leg adds its candidate index (INTENT v2);
    without it the frames are the stage-1 INTENT, which parses as candidate
    0."""
    parts = [
        INTENT_HEADER,
        str(room).encode("ascii"),
        d_ip.encode("ascii"),
        str(d_port).encode("ascii"),
        str(d_nnodes).encode("ascii"),
    ]
    if cand_idx is not None:
        parts.append(str(cand_idx).encode("ascii"))
    return parts


def parse_intent(msg: List[bytes], recv_time: float) -> Tuple[int, LateDIntent]:
    return int(msg[1].decode("ascii")), LateDIntent(
        d_ip=msg[2].decode("ascii"),
        d_port=int(msg[3].decode("ascii")),
        d_nnodes=int(msg[4].decode("ascii")),
        recv_time=recv_time,
        cand_idx=int(msg[5].decode("ascii")) if len(msg) > 5 else 0,
    )


def admit_parts(room: int, seq: int) -> List[bytes]:
    return [ADMIT_HEADER, str(room).encode("ascii"), str(seq).encode("ascii")]


def parse_admit(msg: List[bytes]) -> LateDControl:
    return LateDControl(
        kind=CONTROL_ADMIT,
        room=int(msg[1].decode("ascii")),
        seq=int(msg[2].decode("ascii")),
    )


def release_parts(room: int, seq: int) -> List[bytes]:
    return [RELEASE_HEADER, str(room).encode("ascii"), str(seq).encode("ascii")]


def parse_release(msg: List[bytes]) -> LateDControl:
    return LateDControl(
        kind=CONTROL_RELEASE,
        room=int(msg[1].decode("ascii")),
        seq=int(msg[2].decode("ascii")),
    )


@dataclass(frozen=True)
class LateDChoice:
    """The P leg's stage-2 ``late_d`` field: K candidates, the default
    candidate (the router's arrival pick), the router epoch."""

    k: int
    default_idx: int
    router_epoch: int
    # Symmetric: the owner may run the P-leg itself (LOCAL,
    # WINNER_LOCAL); PD never sets it.
    local_ok: bool = False

    @classmethod
    def of(cls, field_value: Dict[str, Any]) -> LateDChoice:
        """From a field that passed ``late_d_request_error``."""
        return cls(
            k=field_value["k"],
            default_idx=field_value["default_idx"],
            router_epoch=field_value["router_epoch"],
            local_ok=field_value.get("local_ok") is True,
        )


def _uint_below(value: Any, bound: int) -> Optional[int]:
    if type(value) is int and 0 <= value < bound:
        return value
    return None


def late_d_request_error(
    field_value: Any, engine_mode: str, role: Optional[str], symmetric: bool = False
) -> Optional[str]:
    """Why a request's ``late_d`` field (stage 2, router-set) is refused with
    HTTP 400, or None. ``role``: "prefill" (the P leg: needs ``default_idx``)
    or "decode" (a D leg: needs ``cand_idx``); anything else refuses the
    field. A leg the engine could not bind as a candidate must fail at once:
    treated as a stage-1 leg it would wait DEFERRED until its TTL.
    ``symmetric``: the role comes from the leg's mark, and a P-leg may allow
    LOCAL (``local_ok``; ``default_idx`` 254 = LOCAL)."""
    if engine_mode != LATE_D_CHOOSE:
        return (
            f"late_d requires --disaggregation-late-d {LATE_D_CHOOSE}; this "
            f"server runs {engine_mode}"
        )
    if role not in ("prefill", "decode"):
        if symmetric:
            return "late_d is valid only on a kv_export or kv_import leg"
        return "late_d is valid only on a PD prefill or decode server"
    if not isinstance(field_value, dict):
        return "late_d must be an object"
    if _uint_below(field_value.get("v"), 2**8) != LATE_D_FIELD_VERSION:
        return f"late_d.v must be {LATE_D_FIELD_VERSION}"
    if field_value.get("mode") != LATE_D_CHOOSE:
        return f'late_d.mode must be "{LATE_D_CHOOSE}"'
    if _uint_below(field_value.get("router_epoch"), 2**32) is None:
        return "late_d.router_epoch must be an integer in [0, 2^32)"
    k = _uint_below(field_value.get("k"), LATE_D_MAX_CANDIDATES + 1)
    if not k:
        return f"late_d.k must be an integer in [1, {LATE_D_MAX_CANDIDATES}]"
    local_ok = field_value.get("local_ok", False)
    if not isinstance(local_ok, bool):
        return "late_d.local_ok must be a boolean"
    if local_ok and not (symmetric and role == "prefill"):
        return "late_d.local_ok is valid only on a symmetric P-leg"
    name = "default_idx" if role == "prefill" else "cand_idx"
    value = field_value.get(name)
    if local_ok and name == "default_idx" and value == WINNER_LOCAL:
        return None
    if _uint_below(value, k) is None:
        return f"late_d.{name} must be an integer in [0, k)"
    return None


class LateDLegEnded(Exception):
    """A choose leg (a request with the router's ``late_d`` field) ended by an
    abort before its first output chunk. The route answers with
    ``status_code`` and an error body before any stream byte, and with
    ``X-Late-D-Reason: <reason>`` for a late-D outcome."""

    def __init__(
        self, status_code: int, err_type: str, message: str, reason: Optional[str]
    ):
        super().__init__(message)
        self.status_code = status_code
        self.err_type = err_type
        self.message = message
        self.reason = reason


def late_d_leg_ended(finish_reason: Any) -> Optional[LateDLegEnded]:
    """The LateDLegEnded of an abort finish reason (FINISH_ABORT.to_json(), or
    the tokenizer manager's AbortReq echo), else None. The status is the
    abort's own; 503 if it has none, or none in [400, 600)."""
    if not isinstance(finish_reason, dict) or finish_reason.get("type") != "abort":
        return None
    try:
        status = int(finish_reason.get("status_code"))
    except (TypeError, ValueError):
        status = 0
    if not 400 <= status < 600:
        status = int(HTTPStatus.SERVICE_UNAVAILABLE)
    err_type = finish_reason.get("err_type")
    reason = None
    if isinstance(err_type, str) and err_type.startswith(LATE_D_ERR_PREFIX):
        reason = err_type[len(LATE_D_ERR_PREFIX) :] or None
    if not isinstance(err_type, str) or not err_type:
        try:
            err_type = HTTPStatus(status).name
        except ValueError:
            err_type = "ERROR"
    message = str(finish_reason.get("message") or "Aborted")
    return LateDLegEnded(status, err_type, message, reason)


def describe_late_d(server_args: Any) -> Optional[dict]:
    """The ``late_d`` block of /server_info: None when late D is
    off; else the mode and the KV IP (the address the KV connector puts in
    its bootstrap messages), and on a prefill the router link port. The
    router opens its link to tcp://<worker host>:<control_port>."""
    mode = _resolved(server_args, "disaggregation_late_d") or LATE_D_OFF
    if mode in (None, LATE_D_OFF):
        return None
    from decoupled_engine.srt.utils.network import get_local_ip_auto

    try:
        kv_ip = get_local_ip_auto()
    except Exception:  # detection failed; the block still names the mode
        kv_ip = None
    block = {"mode": mode, "kv_ip": kv_ip}
    if (_resolved(server_args, "disaggregation_late_d_allocation_trigger") or "lead") != "lead":
        block["allocation_trigger"] = _resolved(server_args, "disaggregation_late_d_allocation_trigger")
    port = _resolved(server_args, "disaggregation_late_d_control_port")
    if server_args.disaggregation_mode == "prefill" and port is not None:
        block["control_port"] = port
    return block


def registration_frame(mode: str) -> bytes:
    return REGISTRATION_TAG + mode.encode("ascii")


def registration_mode(msg: List[bytes]) -> str:
    """The late-D mode a decode KV-args registration declares ("off" when the
    decode sent no capability frame: flag off, or a build without late D)."""
    last = msg[-1] if msg else b""
    if last.startswith(REGISTRATION_TAG):
        return last[len(REGISTRATION_TAG) :].decode("ascii")
    return LATE_D_OFF


def prefill_leader_bootstrap_info(receiver: Any) -> Optional[dict]:
    """Bootstrap info of the prefill attn-TP rank 0 (CP rank 0) for a decode
    rank-0 receiver. Decode TP rank 0 always targets prefill TP rank 0 and
    CP rank 0 first (CommonKVManager._resolve_rank_mapping), so it is the
    first entry of ``bootstrap_infos``."""
    infos = getattr(receiver, "bootstrap_infos", None)
    if not infos:
        return None
    if receiver.target_tp_ranks[0] != 0 or receiver.target_cp_ranks[0] != 0:
        return None
    return infos[0]


# ----------------------------------------------------------------------
# Transport helpers (mode-neutral)
# ----------------------------------------------------------------------


class LateDInbox:
    """Controls from a connector thread to the scheduler thread. One writer
    thread and one reader thread: deque append / popleft are atomic."""

    def __init__(self):
        self._items: Deque[LateDControl] = deque()

    def put(self, control: LateDControl) -> None:
        self._items.append(control)

    def drain(self) -> List[LateDControl]:
        out = []
        while self._items:
            out.append(self._items.popleft())
        return out

    def __len__(self) -> int:
        return len(self._items)


class LateDSender:
    """Sends late-D frames on a daemon thread with its own ZMQ PUSH socket per
    endpoint, so a send never blocks the scheduler and a dead peer never
    stalls the others. A frame that cannot go out at once is dropped with a
    warning: the sender's resend and the timers cover it.

    ``frames``: the parts of one message (PD: ``admit_parts``), or one
    ``bytes`` frame (symmetric: an encoded pull-plane message, which the
    reader's PULL channel receives whole)."""

    def __init__(
        self,
        zmq_ctx: Optional[zmq.Context] = None,
        name: str = "LateDSender",
        label: str = "control",
    ):
        self._zmq_ctx = zmq_ctx
        self._name = name
        self._label = label
        self._outbox: Optional[queue.SimpleQueue] = None
        # send() may run on two threads (stage 2: the scheduler and the
        # bootstrap thread); the thread starts once.
        self._start_lock = threading.Lock()

    def send(self, endpoint: str, is_ipv6: bool, frames: Any) -> None:
        if isinstance(frames, (bytes, bytearray)):
            frames = [bytes(frames)]
        if self._outbox is None:
            with self._start_lock:
                if self._outbox is None:
                    outbox = queue.SimpleQueue()
                    threading.Thread(
                        target=self._loop, args=(outbox,), name=self._name, daemon=True
                    ).start()
                    self._outbox = outbox
        self._outbox.put((endpoint, is_ipv6, frames))

    def _loop(self, outbox: queue.SimpleQueue) -> None:
        sockets: Dict[str, zmq.Socket] = {}
        ctx = self._zmq_ctx if self._zmq_ctx is not None else zmq.Context.instance()
        while True:
            endpoint, is_ipv6, frames = outbox.get()
            sock = sockets.get(endpoint)
            if sock is None:
                sock = ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                sockets[endpoint] = sock
            try:
                # Never block: a lost ADMIT is resent once (T6), and the bind
                # timer bounds the rest.
                sock.send_multipart(frames, flags=zmq.NOBLOCK)
            except zmq.ZMQError as e:
                logger.warning(
                    "LATE_D %s send to %s failed: %s", self._label, endpoint, e
                )


# ----------------------------------------------------------------------
# PD (Mooncake) adapter: connector state
# ----------------------------------------------------------------------


class _Tombstone(NamedTuple):
    """A stage-2 room P already committed: a later INTENT of another
    candidate gets RELEASE at once."""

    expires: float  # time.monotonic()
    seq: int
    winner_idx: int


class LateDConnState:
    """Late-D state of one KV manager, shared by the scheduler thread and the
    connector's bootstrap (prefill) or decode thread."""

    INTENT_TTL_S = 600.0
    INTENT_CAP = 65536
    TOMBSTONE_CAP = 65536
    _ROLES = ("prefill", "decode")

    def __init__(
        self,
        mode: str,
        role: str,
        is_leader: bool,
        zmq_ctx: Optional[zmq.Context] = None,
        admission_timeout: float = 1800.0,
        nnodes: int = 1,
    ):
        assert role in self._ROLES, role
        self.mode = mode
        self.role = role
        self.is_leader = is_leader
        self.admission_timeout_s = admission_timeout
        # Nodes of this worker; a decode sends it in INTENT (destination check).
        self.nnodes = nnodes
        # Prefill: room -> {candidate index -> INTENT} (stage 1: candidate 0
        # only), in order of the room's latest INTENT; and the tombstones of
        # committed stage-2 rooms. Written by the bootstrap thread and the
        # committing thread, under the lock.
        self._intents: OrderedDict[int, Dict[int, LateDIntent]] = OrderedDict()
        self._tombstones: OrderedDict[int, _Tombstone] = OrderedDict()
        self._intents_lock = threading.Lock()
        # Prefill leader: room -> (D ip, D nnodes) of the ADMITted D (dest
        # check) and room -> wall time of the first ADMIT (bind timer).
        self.admitted_d: Dict[int, Tuple[str, int]] = {}
        self.first_admit: Dict[int, float] = {}
        # Decode leader: ADMITs and RELEASEs from the decode thread, drained
        # by the scheduler into the request broadcast.
        self.inbox = LateDInbox()
        # Prefill leader: ADMIT and RELEASE sends run on their own thread and
        # sockets, so a dead decode never stalls the scheduler, the bootstrap
        # thread or the other decodes.
        self._sender = LateDSender(
            zmq_ctx, name="LateDControlSender", label="ADMIT/RELEASE"
        )
        self._warned_mismatch: set = set()
        # Prefill leader: the binder's metrics (the bootstrap thread counts
        # the ABORTs it ignores); None elsewhere.
        self.metrics: Optional[LateDMetrics] = None

    @classmethod
    def create(cls, mgr: CommonKVManager, mode: str) -> Optional[LateDConnState]:
        if mode == LATE_D_OFF:
            return None
        from decoupled_engine.srt.disaggregation.utils import DisaggregationMode
        from decoupled_engine.srt.runtime_context import get_parallel

        if mgr.disaggregation_mode == DisaggregationMode.PREFILL:
            role = "prefill"
        elif mgr.disaggregation_mode == DisaggregationMode.DECODE:
            role = "decode"
        else:
            return None
        return cls(
            mode=mode,
            role=role,
            is_leader=(
                mgr.attn_tp_rank == 0 and mgr.attn_cp_rank == 0 and mgr.pp_rank == 0
            ),
            zmq_ctx=mgr._zmq_ctx,
            admission_timeout=admission_timeout_s(),
            nnodes=get_parallel().nnodes,
        )

    # ------------------------------------------------------------------
    # Prefill: INTENT table (bootstrap thread writes, binder reads)
    # ------------------------------------------------------------------

    def on_intent(self, msg: List[bytes]) -> None:
        room, intent = parse_intent(msg, time.monotonic())
        release_seq = None
        with self._intents_lock:
            tomb = self._tombstones.get(room)
            if tomb is None:
                cands = self._intents.get(room)
                if cands is None:
                    cands = self._intents[room] = {}
                # The same candidate index again (a resent INTENT, or a
                # stage-1 D leg sent again): the latest wins.
                cands[intent.cand_idx] = intent
                self._intents.move_to_end(room)
                while len(self._intents) > self.INTENT_CAP:
                    self._intents.popitem(last=False)
            elif intent.cand_idx != tomb.winner_idx:
                release_seq = tomb.seq
        if release_seq is not None:
            # A candidate whose INTENT came after the commit (stage 2).
            logger.info(
                "LATE_D release late INTENT room=%s cand=%d d=%s seq=%d",
                room,
                intent.cand_idx,
                intent.label,
                release_seq,
            )
            self.send_release(intent.d_key, room, release_seq)

    def peek_intent(self, room: int) -> Optional[LateDIntent]:
        """Stage 1: the room's INTENT (candidate 0)."""
        with self._intents_lock:
            cands = self._intents.get(room)
            return None if cands is None else cands.get(0)

    def peek_intents(self, room: int) -> Dict[int, LateDIntent]:
        """Stage 2: a copy of the room's INTENTs by candidate index."""
        with self._intents_lock:
            return dict(self._intents.get(room) or {})

    def prune_intents(
        self, now: Optional[float] = None, keep: Set[int] = frozenset()
    ) -> None:
        """Drop the INTENTs of rooms that are not in ``keep`` (the rooms in
        this prefill's bootstrap queue) once their latest INTENT is older than
        INTENT_TTL_S: their P leg never arrived, or the room ended before its
        ADMIT. A room that waits in the bootstrap queue keeps its INTENTs
        however long it waits (the admission timer, >= 1800 s, ends it).
        Also drop expired tombstones."""
        now = time.monotonic() if now is None else now
        with self._intents_lock:
            stale = []
            # In order of the latest INTENT: stop at the first fresh room.
            for room, cands in self._intents.items():
                if now - max(i.recv_time for i in cands.values()) <= self.INTENT_TTL_S:
                    break
                if room not in keep:
                    stale.append(room)
            for room in stale:
                del self._intents[room]
            while self._tombstones:
                room, tomb = next(iter(self._tombstones.items()))
                if tomb.expires > now:
                    break
                del self._tombstones[room]

    def note_admit(self, room: int, intent: LateDIntent, now: float) -> None:
        with self._intents_lock:
            self._intents.pop(room, None)
        self.admitted_d[room] = (intent.d_ip, intent.d_nnodes)
        self.first_admit.setdefault(room, now)

    def commit_choice(
        self, room: int, seq: int, winner_idx: int, now: float
    ) -> Optional[Tuple[LateDIntent, List[LateDIntent]]]:
        """Stage 2, the room's only commit: ADMIT to the winner, then RELEASE
        to every other candidate with an INTENT; the tombstone answers a later
        INTENT with RELEASE. None (nothing sent) if the winner has no INTENT."""
        with self._intents_lock:
            cands = self._intents.get(room) or {}
            winner = cands.get(winner_idx)
            if winner is None:
                return None
            del self._intents[room]
            self._tombstones[room] = _Tombstone(
                expires=time.monotonic() + self.admission_timeout_s,
                seq=seq,
                winner_idx=winner_idx,
            )
            self._tombstones.move_to_end(room)
            while len(self._tombstones) > self.TOMBSTONE_CAP:
                self._tombstones.popitem(last=False)
        losers = [cands[i] for i in sorted(cands) if i != winner_idx]
        self.admitted_d[room] = (winner.d_ip, winner.d_nnodes)
        self.first_admit.setdefault(room, now)
        self.send_admit(winner.d_key, room, seq)
        for loser in losers:
            self.send_release(loser.d_key, room, seq)
        return winner, losers

    def forget_room(self, room: int) -> None:
        self.admitted_d.pop(room, None)
        self.first_admit.pop(room, None)

    def accepts_destination(self, room: int, endpoint: str) -> bool:
        """A late room takes its destination only from the ADMITted D. A
        room this prefill never ADMITted (e.g. a decode-side rebootstrap,
        which binds without INTENT) keeps today's behavior. The check is by
        decode IP, so it applies only to single-node decode workers."""
        admitted = self.admitted_d.get(room)
        if admitted is None:
            return True
        d_ip, d_nnodes = admitted
        return d_nnodes != 1 or endpoint == d_ip

    def foreign_abort(
        self, infos: Optional[Dict[Any, Any]], ip: str, port: int
    ) -> bool:
        """Stage 2 ABORT filter, on every prefill rank: True for
        an ABORT from a decode rank (``ip``, ``port``) that is not among the
        destinations this rank recorded for the room (``infos``: session ->
        TransferInfo), once it has any. Only the winner sends metadata, so
        such an ABORT comes from a candidate that lost (its RELEASE was lost
        and the router dropped its leg): it must not fail the room. Before
        any destination, an ABORT fails the room as today."""
        if self.mode != LATE_D_CHOOSE or not infos:
            return False
        for info in list(infos.values()):
            if info.endpoint == ip and info.dst_port == port:
                return False
        return True

    def check_registration(self, session_id: str, decode_mode: str) -> None:
        """Log (once per decode session) a decode whose late-D mode differs.
        The decode refuses the mismatch itself at its first bootstrap
        (CommonKVManager.try_ensure_parallel_info); this catches older
        decode builds that do not check."""
        if decode_mode == self.mode or session_id in self._warned_mismatch:
            return
        self._warned_mismatch.add(session_id)
        logger.error(
            "LATE_D capability mismatch: decode %s registered with "
            "--disaggregation-late-d %s, this prefill runs %s; its requests "
            "will not bind late",
            session_id,
            decode_mode,
            self.mode,
        )

    # ------------------------------------------------------------------
    # Prefill leader: ADMIT and RELEASE sends
    # ------------------------------------------------------------------

    def use_sender(self, sender: Any) -> None:
        """Stage 2: send ADMIT and RELEASE through the router link's thread
        (any object with LateDSender.send)."""
        self._sender = sender

    def send_admit(self, d_key: Tuple[str, int], room: int, seq: int) -> None:
        na = NetworkAddress(d_key[0], d_key[1])
        self._sender.send(na.to_tcp(), na.is_ipv6, admit_parts(room, seq))

    def send_release(self, d_key: Tuple[str, int], room: int, seq: int) -> None:
        na = NetworkAddress(d_key[0], d_key[1])
        self._sender.send(na.to_tcp(), na.is_ipv6, release_parts(room, seq))

    # ------------------------------------------------------------------
    # Decode leader: control inbox (decode thread writes, scheduler drains)
    # ------------------------------------------------------------------

    def on_admit(self, msg: List[bytes]) -> None:
        self.inbox.put(parse_admit(msg))

    def on_release(self, msg: List[bytes]) -> None:
        self.inbox.put(parse_release(msg))

    def drain_inbox(self) -> List[LateDControl]:
        return self.inbox.drain()


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


class LateDMetrics:
    """Prometheus metrics of late D, recorded on the attn-TP leader only."""

    _BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 1800)
    _CHOICE_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5)
    _READY_BUCKETS = (1, 2, 3, 4, 6, 8, 16, 32, 64)

    def __init__(self, labels: Dict[str, str], role: str, symmetric: bool = False):
        from prometheus_client import Counter, Gauge, Histogram

        self.labels = labels
        names = list(labels.keys())
        self.symmetric = symmetric

        def counter(name, doc):
            return Counter(f"sglang:{name}", doc, labelnames=names).labels(**labels)

        def gauge(name, doc):
            return Gauge(
                f"sglang:{name}", doc, labelnames=names, multiprocess_mode="mostrecent"
            ).labels(**labels)

        def histogram(name, doc, buckets=self._BUCKETS):
            return Histogram(
                f"sglang:{name}", doc, labelnames=names, buckets=buckets
            ).labels(**labels)

        if role == "prefill":
            self.backlog_wait = histogram(
                "late_d_backlog_wait_seconds", "Sender creation to first ADMIT."
            )
            self.bind_latency = histogram(
                "late_d_bind_latency_seconds",
                "First ADMIT to bootstrap done, as the scheduler sees it.",
            )
            self.lead = histogram(
                "late_d_lead_seconds", "Bootstrap done to first-chunk launch."
            )
            self.admits = counter("late_d_admits_total", "ADMIT messages sent.")
            self.held = counter(
                "late_d_held_total", "ADMITs outstanding past hold_detect_ms."
            )
            self.held_reqs = gauge("late_d_held_reqs", "Rooms held at their D.")
            self.starved = counter(
                "late_d_p_starved_episodes_total",
                "Episodes with no runnable bound request while unbound rooms wait.",
            )
            self.r_ahead = gauge("late_d_r_ahead", "Lead window size R_ahead.")
            self.lead_target = gauge(
                "late_d_lead_target_tokens",
                "Lead mode work: uncached prompt tokens kept bound ahead.",
            )
            self.intent_missing = counter(
                "late_d_intent_missing_total",
                "Late rooms that waited over 30 s in the bootstrap queue without "
                "an INTENT (must be 0).",
            )
            self.intent_restored = counter(
                "late_d_intent_restored_total",
                "Late rooms whose INTENT arrived after it was missing (must be 0).",
            )
            self.releases = counter(
                "late_d_releases_total",
                "Stage 2: RELEASE messages sent at a commit (late INTENTs and "
                "resends not counted).",
            )
            self.rebinds = counter(
                "late_d_rebinds_total",
                "Stage 2: commits to a candidate other than the default.",
            )
            self.choice_latency = histogram(
                "late_d_choice_latency_seconds",
                "Stage 2: BIND_READY sent to the router's decision received.",
                buckets=self._CHOICE_BUCKETS,
            )
            self.choice_timeouts = counter(
                "late_d_choice_timeouts_total",
                "Stage 2: choices committed to the default at the timeout.",
            )
            self.choice_ready = histogram(
                "late_d_choice_ready_candidates",
                "Stage 2: candidates with an INTENT when a room is offered.",
                buckets=self._READY_BUCKETS,
            )
            self.link_restarts = counter(
                "late_d_link_restarts_total", "Stage 2: link thread restarts."
            )
            self.abort_filtered = counter(
                "late_d_abort_filtered_total",
                "Stage 2: ABORTs from a decode that is not a destination of the "
                "room, acknowledged and not applied (prefill rank 0).",
            )
            self.cap_overrun = counter(
                "late_d_cap_overrun_total",
                "Stage 2: commits to a decode at or above its outstanding cap.",
            )
            self.router_present = gauge(
                "late_d_router_present", "Stage 2: 1 while the router link is up."
            )
        else:
            self.deferred_reqs = gauge(
                "late_d_deferred_reqs", "Requests DEFERRED on this decode."
            )
            self.allocs = counter("late_d_alloc_total", "Late allocations.")
            self.skips = counter(
                "late_d_skips_total", "Preallocation passes a late entry did not fit."
            )
            self.held_d = counter(
                "late_d_held_entries_total", "Late entries that were HELD."
            )
            self.claims = counter("late_d_claims_total", "Reservation claims made.")
            self.held_seconds = histogram(
                "late_d_held_seconds", "First no-fit to allocation of a HELD entry."
            )
            self.ttl_aborts = counter(
                "late_d_ttl_aborts_total", "DEFERRED requests aborted by the TTL."
            )
            self.released = counter(
                "late_d_released_total",
                "Stage 2: DEFERRED candidates released (HTTP 409).",
            )
            self.intent_refreshes = counter(
                "late_d_intent_refreshes_total",
                "INTENTs sent again for entries still DEFERRED after 60 s.",
            )
        if not symmetric:
            return
        # Symmetric mode, on the worker's rank 0.
        if role == "prefill":
            self.late_timer_fail = counter(
                "late_d_late_timer_fail_total",
                "Held late P-legs failed by the admission or bind timer (503).",
            )
            self.held_aux_wait = counter(
                "late_d_held_aux_wait_total",
                "Rounds a begun late P-leg waited for an owner aux slot.",
            )
            self.bind_near_timeout = counter(
                "late_d_bind_near_timeout_total",
                "Late P-legs bound after more than 80% of the bind timer.",
            )
            self.reader_gone = counter(
                "late_d_reader_gone_total",
                "Held late P-legs whose reader gave up (503).",
            )
        else:
            self.early_held = counter(
                "late_d_early_controls_held_total",
                "Controls held back on rank 0 until their leg arrived.",
            )
            self.early_dropped = Counter(
                "sglang:late_d_early_controls_dropped_total",
                "Held-back controls dropped (TTL, cap or ended key), by kind.",
                labelnames=names + ["kind"],
            )
            self.owner_abort_fail = counter(
                "late_d_owner_abort_fail_total",
                "Never-admitted late entries ended by an owner ABORT.",
            )
            self.claim_skips = counter(
                "late_d_claim_skips_total",
                "Requests the PrefillAdder skipped for the claim's reserve.",
            )
            self.claim_escalations = counter(
                "late_d_claim_escalations_total",
                "Claims that started charging new P-leg starts (escalation).",
            )
            self.claim_wait_passes = histogram(
                "late_d_claim_wait_passes",
                "Walk passes from becoming the claim to its allocation.",
                buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 5120),
            )
            self.reader_bind = Histogram(
                "sglang:late_d_reader_bind_seconds",
                "ADMIT received on the reader IO thread to the import's BEGIN, by "
                "the reader's last iteration (decode, prefill or idle).",
                labelnames=names + ["iter"],
                buckets=self._BUCKETS,
            )


TM_LOOP_LAG_PERIOD_S = 0.05
_TM_LOOP_LAG_BUCKETS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5)


# One metric family per name in a process (a test builds the readout more
# than once; prometheus_client refuses a second registration).
_RECV_BROADCAST_FAMILIES: Dict[str, Any] = {}


def _recv_broadcast_counter(name: str, doc: str, labelnames: List[str]) -> Any:
    family = _RECV_BROADCAST_FAMILIES.get(name)
    if family is None:
        from prometheus_client import Counter

        family = Counter(name, doc, labelnames=labelnames)
        _RECV_BROADCAST_FAMILIES[name] = family
    return family


class RecvBroadcastReadout:
    """Overhead readout on the source of the request broadcast (the
    scheduler leader), once per scheduler iteration, by payload:

    * "empty": nothing to send. One size broadcast, and the wait until the
      other ranks reach it. Every server pays this share, with late D or not.
    * "data": requests or controls. Also the pickle and the payload broadcast.

    The late-D cost is the data share, or the difference to a late-D-off
    server that runs with SGLANG_RECV_BROADCAST_READOUT=1."""

    PAYLOADS = ("empty", "data")

    def __init__(self, labels: Dict[str, str]):
        names = list(labels)
        seconds = _recv_broadcast_counter(
            "sglang:late_d_recv_broadcast_seconds_total",
            "Seconds the leader spends in its request broadcast to the "
            "other ranks, by payload; the rate is the share of wall time.",
            names + ["payload"],
        )
        calls = _recv_broadcast_counter(
            "sglang:late_d_recv_broadcasts_total",
            "Request broadcasts of the leader, by payload.",
            names + ["payload"],
        )
        sent = _recv_broadcast_counter(
            "sglang:late_d_recv_broadcast_bytes_total",
            "Pickled bytes the leader sent in its request broadcasts.",
            names,
        )
        self._seconds = {p: seconds.labels(**labels, payload=p) for p in self.PAYLOADS}
        self._calls = {p: calls.labels(**labels, payload=p) for p in self.PAYLOADS}
        self._bytes = sent.labels(**labels)

    def observe(self, seconds: float, has_data: bool) -> None:
        payload = "data" if has_data else "empty"
        self._seconds[payload].inc(seconds)
        self._calls[payload].inc()

    def add_bytes(self, nbytes: int) -> None:
        self._bytes.inc(nbytes)


def make_recv_broadcast_readout(scheduler: Any) -> Optional[RecvBroadcastReadout]:
    """The readout on the source of the request broadcast (pp, attn-TP
    and attn-CP rank 0) of a scheduler that exports metrics, with late D on a
    decode, or with SGLANG_RECV_BROADCAST_READOUT=1 on any server (late D on
    or off). Else None: the request receiver times nothing."""
    collector = getattr(scheduler, "metrics_collector", None)
    ps = getattr(scheduler, "ps", None)
    if collector is None or ps is None:
        return None
    if ps.pp_rank != 0 or ps.attn_tp_rank != 0 or ps.attn_cp_rank != 0:
        return None
    prealloc = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    if (
        getattr(prealloc, "late_d", None) is None
        and not envs.SGLANG_RECV_BROADCAST_READOUT.get()
    ):
        return None
    return RecvBroadcastReadout(collector.labels)


def make_tm_loop_lag_histogram(labels: Dict[str, str]) -> Any:
    """The tokenizer manager's event-loop lag, in its own process."""
    from prometheus_client import Histogram

    return Histogram(
        "sglang:late_d_tm_loop_lag_seconds",
        "Late D overhead: how late a 50 ms sleep on the tokenizer "
        "manager's event loop wakes; its p99 is the head-of-line delay a "
        "request meets there.",
        labelnames=list(labels),
        buckets=_TM_LOOP_LAG_BUCKETS,
    ).labels(**labels)


async def tm_loop_lag_monitor(
    observe: Callable[[float], None],
    period: float = TM_LOOP_LAG_PERIOD_S,
    iterations: Optional[int] = None,
) -> None:
    """Sample the running event loop's lag every ``period`` seconds
    (forever by default). A failure stops the sampling; it never reaches the
    server."""
    loop = asyncio.get_running_loop()
    done = 0
    try:
        while iterations is None or done < iterations:
            start = loop.time()
            await asyncio.sleep(period)
            observe(max(0.0, loop.time() - start - period))
            done += 1
    except Exception as e:  # metrics only
        logger.warning("LATE_D tokenizer loop-lag sampling stopped: %r", e)


def make_late_d_metrics(scheduler: Any, conn: Any) -> Optional[LateDMetrics]:
    """Metrics on the leader of a scheduler that exports metrics, else None.
    ``conn`` is a LateDConnState or a port (both carry ``role``, ``is_leader``;
    a symmetric port also ``symmetric``)."""
    collector = getattr(scheduler, "metrics_collector", None)
    if not conn.is_leader or collector is None:
        return None
    return LateDMetrics(
        collector.labels, conn.role, symmetric=getattr(conn, "symmetric", False)
    )


def sym_metrics(metrics: Any) -> Optional[LateDMetrics]:
    """``metrics`` when it carries the symmetric set (LateDMetrics with
    symmetric=True), else None: PD, and stand-ins without the set."""
    return metrics if getattr(metrics, "symmetric", False) is True else None


def _input_len(req: Req) -> int:
    return len(req.origin_input_ids)


def _unlaunched_tokens(req: Req) -> int:
    """Prompt tokens of the chunked request not yet in any launched chunk.
    ``extend_range`` is the last launched chunk (set in lockstep by the
    PrefillAdder on every rank)."""
    total = len(req.origin_input_ids) + len(req.output_ids)
    extend_range = getattr(req, "extend_range", None)
    done = extend_range.end if extend_range is not None else len(req.prefix_indices)
    return max(0, total - done)


# ----------------------------------------------------------------------
# Ports: what the binder and the decode walk need from a mode
# ----------------------------------------------------------------------


class AheadAll:
    """The lead window's work ahead of an unbound room: every runnable bound
    request in the waiting queue, whatever its arrival (PD)."""

    __slots__ = ("reqs", "tokens")

    def __init__(self, waiting: List[Req], port: LateDBinderPort):
        reqs = tokens = 0
        for req in waiting:
            if port.is_bound_runnable(req):
                reqs += 1
                tokens += _input_len(req)
        self.reqs, self.tokens = reqs, tokens

    def before(self, req: Req) -> Tuple[int, int]:
        """(requests, tokens) of bound work queued ahead of ``req``."""
        return self.reqs, self.tokens


class AheadByArrival:
    """Arrival order (symmetric): only the runnable bound work that arrived
    before the room counts as ahead of it (``sym_arrival_seq``, identical on
    every rank; -1 sorts first), so later Local requests never starve a held
    P-leg."""

    __slots__ = ("reqs", "tokens", "_seqs", "_cum")

    def __init__(self, waiting: List[Req], port: LateDBinderPort):
        pts = sorted(
            (getattr(req, "sym_arrival_seq", -1), _input_len(req))
            for req in waiting
            if port.is_bound_runnable(req)
        )
        self._seqs = [seq for seq, _ in pts]
        self._cum = list(itertools.accumulate((t for _, t in pts), initial=0))
        self.reqs, self.tokens = len(pts), self._cum[-1]

    def before(self, req: Req) -> Tuple[int, int]:
        i = bisect.bisect_left(self._seqs, getattr(req, "sym_arrival_seq", -1))
        return i, self._cum[i]


def insert_by_arrival(queue: List[Req], req: Req) -> None:
    """Insert ``req`` at its arrival position (``sym_arrival_seq``; -1
    sorts first). FCFS keeps append order, so the position is the same on
    every rank."""
    seq = getattr(req, "sym_arrival_seq", -1)
    i = len(queue)
    while i > 0 and getattr(queue[i - 1], "sym_arrival_seq", -1) > seq:
        i -= 1
    queue.insert(i, req)


class LateDBinderPort(abc.ABC):
    """The owner side of late D as the binder sees it: PD P attn-TP rank 0,
    symmetric owner rank 0. Rank-0 data rule: a port reads only state that is
    identical on every owner rank or private to rank 0, runs no collective,
    no match_prefix, no prefetch check and no consensus round, and never
    blocks (sends go to a sender thread)."""

    role = "prefill"
    is_leader = True

    def key_of(self, req: Req) -> int:
        return req.bootstrap_room

    def is_bound_runnable(self, req: Req) -> bool:
        """A bound request in the waiting queue that will prefill."""
        return not req.pending_bootstrap and not is_aborted(req)

    def housekeeping(self, now: float, unbound: List[Req]) -> None:
        """Called at most once per second, with the requests that wait for
        their bind (PD: the bootstrap queue)."""

    def intent_missing(self, key: int) -> bool:
        """PD: no candidate's INTENT is held for ``key`` (the binder counts a
        room that stays so; LateDPrefillBinder._check_intents)."""
        return False

    def ahead(self, waiting: List[Req]) -> AheadAll:
        return AheadAll(waiting, self)

    # Stage 2 (``choose``). A port without stage 2 binds every room as in
    # stage 1: ``choice`` returns None and the rest is never called.

    def choice(self, req: Req) -> Optional[LateDChoice]:
        """The request's candidate set and default, or None (stage 1)."""
        return None

    def candidates(self, key: int) -> Dict[int, Any]:
        """Candidate index -> destination, for the candidates that announced
        themselves (PD: INTENT held)."""
        return {}

    def commit(
        self, key: int, seq: int, winner_idx: int, now: float
    ) -> Optional[Tuple[Any, List[Any]]]:
        """The key's only commit: ADMIT to the winner, RELEASE to the other
        candidates, and start the bind timer. Returns (winner, losers), or
        None if the winner is not a candidate (nothing sent)."""
        raise NotImplementedError("late D stage 2 is not available here")

    def send_release(self, dest: Any, key: int, seq: int) -> None:
        """Queue a RELEASE for the sender thread (the resend)."""
        raise NotImplementedError("late D stage 2 is not available here")

    def local_load(self, key: int) -> int:
        """Symmetric stage 2 (L4 ``local_load``): the owner's bound view
        without ``key``, for the router's LOCAL choice. 0 elsewhere."""
        return 0

    @abc.abstractmethod
    def is_late(self, req: Req) -> bool:
        """The request binds late (FAKE rooms and similar do not)."""

    @abc.abstractmethod
    def is_failed(self, key: int) -> bool:
        """The transfer of ``key`` already failed: do not admit it."""

    @abc.abstractmethod
    def destination(self, key: int, req: Req) -> Any:
        """The reader to admit ``key`` at (``.d_key`` for the per-reader cap,
        ``.label`` for logs), or None while it is not known yet (B3)."""

    @abc.abstractmethod
    def note_admit(self, key: int, dest: Any, now: float) -> None:
        """First ADMIT of ``key``: start its bind timer."""

    @abc.abstractmethod
    def send_admit(self, dest: Any, key: int, seq: int) -> None:
        """Queue an ADMIT for the sender thread (also used for the resend)."""

    @abc.abstractmethod
    def forget(self, key: int) -> None:
        """``key`` left the binder (launched, failed or aborted)."""

    @abc.abstractmethod
    def created_at(self, req: Req, now: float) -> float:
        """Start of the admission timer, on the binder's clock."""

    @abc.abstractmethod
    def bound_time(self, req: Req) -> float:
        """When the request was bound, on the request-time-stats clock; 0 if
        unknown."""

    @abc.abstractmethod
    def first_admit_time(self, key: int) -> Optional[float]:
        """Time of the first ADMIT of ``key`` (bind timer start), if any."""


class LateDReaderPort(abc.ABC):
    """The reader side of late D as the decode walk sees it: PD D ranks,
    symmetric reader ranks. Called on every rank in lockstep, except
    ``drain_inbox``, ``key_state`` and ``batch_import_keys`` (rank 0)."""

    role = "decode"
    is_leader = True
    # Rank 0 holds back a control whose key has no entry yet (a control can
    # beat its leg); False keeps "an unknown key is ignored" (PD: INTENT
    # proves the D leg before any ADMIT).
    hold_unknown = False
    # Rank 0 sends the INTENT of an entry still DEFERRED again every
    # LateDDecode.INTENT_REFRESH_S (refresh_intent; PD).
    refreshes_intents = False

    @abc.abstractmethod
    def drain_inbox(self) -> List[LateDControl]:
        """Rank 0: controls received since the last call, in order."""

    @abc.abstractmethod
    def expire(self, entry: Any, key: int, reason: str) -> None:
        """Every rank: end a DEFERRED entry whose TTL passed."""

    def send_intent(self, entry: Any) -> None:
        """Rank 0, after the entry's handshake: announce it to the owner."""

    def refresh_intent(self, entry: Any) -> None:
        """Rank 0: announce a still-DEFERRED entry again (a failed send does
        not fail the entry)."""

    def key_state(self, key: int) -> str:
        """Rank 0: "unknown", or "active" / "ended" for a key whose entry
        already left the late states (its controls are stale)."""
        return "unknown"

    def batch_import_keys(self, recv_reqs: Any) -> Set[int]:
        """Rank 0: keys of the reader legs in the batch about to be broadcast
        (their controls in the same broadcast apply after the add)."""
        return set()

    def release(self, entry: Any, key: int) -> None:
        """Every rank, stage 2: end a DEFERRED candidate that lost the bind
        (HTTP 409); it must never send ABORT to the owner."""
        raise NotImplementedError("late D stage 2 is not available here")

    def fail(self, entry: Any, key: int, detail: str) -> None:
        """Every rank, symmetric: end a never-allocated entry (DEFERRED,
        ADMITTED or HELD) with an error and no recompute (rule X): its owner
        aborted (no ABORT back) or runs another late-D mode (ABORT, so
        the owner's export does not wait for this reader)."""
        raise NotImplementedError("fail controls are symmetric-mode only")


# ----------------------------------------------------------------------
# P side: lead-window binder (mode-neutral)
# ----------------------------------------------------------------------


def _work_headroom(
    outstanding: Dict[Any, int], pend_share: Dict[Any, float], key: Any, cap: int
) -> bool:
    """Lead mode work, choose rule 4: a candidate has headroom while its
    outstanding ADMITs plus its expected share of the CHOICE_PENDING rooms
    (rounded up) stay below the cap."""
    return outstanding[key] + math.ceil(pend_share[key] - 1e-9) < cap


def _fallback_winner(default_idx: int, eligible: List[int]) -> int:
    """The commit without a router decision: the default candidate if it is
    eligible, else the lowest eligible index."""
    return default_idx if default_idx in eligible else eligible[0]


@dataclass
class _BindRoom:
    req: Any
    d_key: Any
    seq: int
    tokens: int
    admit_time: float  # time.time(), the clock of the sender's init_time
    admit_perf: float  # time.perf_counter(), the clock of req.time_stats
    state: str = ADMIT_SENT
    resent: bool = False
    dest: Any = None
    # Stage 2: the candidates released at the commit (resent with the ADMIT).
    losers: Tuple[Any, ...] = ()
    # Stage 2, CHOICE_PENDING: the choice and the log fields of its commit.
    choice: Any = None
    ready: int = 0
    lead_reqs: int = 0
    # The binder's decision (ADMIT, or BIND_READY in choose), perf_counter
    # clock; the start of T_bind in lead mode work.
    decide_perf: float = 0.0
    # CHOICE_PENDING: the D keys of the eligible candidates at BIND_READY.
    pend_keys: Tuple[Any, ...] = ()


class LeadWork:
    """Lead mode ``work`` (PD binder): size the lead window by prefill work,
    so the prefill does not idle while bound work waits for a bind.

    need_iters = ceil(T_bind / T_iter) + margin + boost, in [1, MAX_ITERS];
    target = need_iters x ``unit`` (chunked_prefill_size: the prefill's
    tokens per iteration). T_bind is the EWMA of the binder's decision (ADMIT,
    or BIND_READY in choose) to bootstrap done, plus 2 EWMA absolute
    deviations. T_iter is the EWMA of back-to-back busy iterations (a batch
    ran in this and in the previous iteration). Work is uncached prompt
    tokens: prompt x (1 - hit), where hit is the EWMA of cached / prompt over
    launched requests (``req.cached_tokens``, set at the first chunk).

    Rank-0 data rule: reads only the launched requests' ``cached_tokens``
    (identical on all ranks) and its own clock."""

    ALPHA = 0.2
    MAX_ITERS = 8
    # A bind slower than this waited for D memory (HELD); it counts as this.
    MAX_BIND_SAMPLE_S = 2.0
    LOG_EVERY_S = 10.0

    def __init__(self, config: LateDConfig, unit_tokens: int):
        self.unit = unit_tokens
        self.margin = config.lead_margin_iters
        self.kv_cap = config.lead_kv_max_tokens or math.inf
        self.bind_s = config.hold_detect_s
        self.bind_dev = 0.0
        self.binds = 0
        self.iter_s = 0.0
        self.iters = 0
        self.hit = 0.0
        self.launches = 0
        self.need_iters = 1 + self.margin
        self.target = self.need_iters * self.unit
        self._last_step: Optional[Tuple[float, bool]] = None
        self._last_log = 0.0

    def on_step(self, now_perf: float, batch_ran: bool) -> None:
        last = self._last_step
        if batch_ran and last is not None and last[1]:
            dt = max(0.0, now_perf - last[0])
            if self.iters == 0:
                self.iter_s = dt
            else:
                self.iter_s += self.ALPHA * (dt - self.iter_s)
            self.iters += 1
        self._last_step = (now_perf, batch_ran)

    def on_bound(self, seconds: float) -> None:
        seconds = min(seconds, self.MAX_BIND_SAMPLE_S)
        if self.binds == 0:
            self.bind_s, self.bind_dev = seconds, 0.0
        else:
            err = seconds - self.bind_s
            self.bind_s += self.ALPHA * err
            self.bind_dev += self.ALPHA * (abs(err) - self.bind_dev)
        self.binds += 1

    def on_launched(self, req: Req) -> None:
        total = _input_len(req)
        if total <= 0:
            return
        ratio = min(1.0, max(0.0, getattr(req, "cached_tokens", 0) / total))
        if self.launches == 0:
            self.hit = ratio
        else:
            self.hit += self.ALPHA * (ratio - self.hit)
        self.launches += 1

    def bind_budget_s(self) -> float:
        return self.bind_s + 2.0 * self.bind_dev

    def plan(self, boost: int) -> int:
        """Set need_iters and target for this step; returns the target."""
        base = 1
        if self.iters > 0:
            base = math.ceil(self.bind_budget_s() / max(self.iter_s, 1e-3))
        self.need_iters = max(1, min(self.MAX_ITERS, base + self.margin + boost))
        self.target = self.need_iters * self.unit
        return self.target

    def uncached(self, tokens: float) -> float:
        return tokens * (1.0 - self.hit)

    def maybe_log(self, now: float, work: float, bound_ahead: int) -> None:
        if now - self._last_log < self.LOG_EVERY_S:
            return
        self._last_log = now
        logger.info(
            "LATE_D lead mode=work need_iters=%d target=%d work=%d "
            "bound_ahead=%d t_bind_ms=%.1f t_bind_dev_ms=%.1f t_iter_ms=%.1f "
            "hit=%.3f binds=%d iters=%d",
            self.need_iters,
            self.target,
            work,
            bound_ahead,
            self.bind_s * 1000.0,
            self.bind_dev * 1000.0,
            self.iter_s * 1000.0,
            self.hit,
            self.binds,
            self.iters,
        )


class LateDPrefillBinder:
    """P attn-TP rank 0: decides when each late room gets its ADMIT. Runs
    once per scheduler iteration, before ``pop_bootstrapped``.

    Rank-0 data rule: reads only state that is identical on all P ranks (queue
    order, input lengths, the chunked request's launched range) or private to
    rank 0 (INTENTs, its own table). It never calls ``match_prefix`` or
    ``check_prefetch_progress`` and runs no collective; other ranks learn a
    bind only as KVPoll WaitingForInput through the existing MIN all-reduce.
    The mode reaches it only through ``self.port`` (a LateDBinderPort).
    """

    ADMIT_RESEND_S = 2.0
    EWMA_ALPHA = 0.2
    CALM_ITERS = 50
    # A late room without any INTENT this long after its P leg arrived is
    # logged and counted (normal INTENT lag: milliseconds).
    INTENT_MISSING_S = 30.0

    def __init__(
        self,
        source: Any,
        config: LateDConfig,
        chunked_prefill_size: Optional[int],
        metrics: Optional[LateDMetrics] = None,
    ):
        # ``source``: a LateDBinderPort, or a PD KV manager (PdBinderPort).
        self.port: LateDBinderPort = (
            source if isinstance(source, LateDBinderPort) else PdBinderPort(source)
        )
        self.cfg = config
        self.chunk_tokens = (
            chunked_prefill_size if (chunked_prefill_size or 0) > 0 else math.inf
        )
        self.metrics = metrics
        # Lead mode work (None: count). Its unit is the prefill's tokens per
        # iteration: the chunk, or 8192 without chunked prefill.
        self.work: Optional[LeadWork] = None
        if config.lead_mode in (LEAD_WORK, LEAD_NEED):
            unit = chunked_prefill_size if (chunked_prefill_size or 0) > 0 else 8192
            self.work = LeadWork(config, unit)
        # Rooms this prefill has ADMITted, until they launch or fail.
        self.rooms: Dict[int, _BindRoom] = {}
        self.seq = 0
        self.ewma = 0.0
        self.boost = 0
        self.r_ahead = config.lead_min_reqs
        self._calm_iters = 0
        self._starved = False
        self._last_prune = 0.0
        # Rooms counted as missing their INTENT: key -> (time, rid).
        self._missing: Dict[int, Tuple[float, str]] = {}
        # Stage 2: the router link (None: every choose room commits at once,
        # as with an absent router). Its thread commits CHOICE_PENDING rooms
        # through commit_choice; ``lock`` serializes that with step().
        self.link: Optional[LateDLink] = None
        self.lock = threading.Lock()
        self.need_plan = None
        self.need_gate_closed = False
        self.need_parked = set()
        self.need_rule4_overrides = 0


    def first_admit_time(self, room: int) -> Optional[float]:
        return self.port.first_admit_time(room)

    def step(
        self,
        bootstrap_queue: List[Req],
        waiting_queue: List[Req],
        chunked_req: Optional[Req],
        batch_ran: bool,
    ) -> None:
        """``bootstrap_queue``: the requests not bound yet, in arrival order
        (PD: the bootstrap queue)."""
        with self.lock:
            self._step(bootstrap_queue, waiting_queue, chunked_req, batch_ran)

    def _step(
        self,
        bootstrap_queue: List[Req],
        waiting_queue: List[Req],
        chunked_req: Optional[Req],
        batch_ran: bool,
    ) -> None:
        if self.cfg.lead_mode == LEAD_NEED:
            self._step_need(bootstrap_queue, waiting_queue, chunked_req, batch_ran)
            return
        now = time.time()
        cfg = self.cfg
        port = self.port
        work = self.work
        hold_s = cfg.hold_detect_s
        if work is not None:
            work.on_step(time.perf_counter(), batch_ran)
            # A bind slower than usual still counts as in flight (up to the
            # ADMIT resend age).
            hold_s = max(hold_s, min(work.bind_budget_s(), self.ADMIT_RESEND_S))
        if now - self._last_prune >= 1.0:
            self._last_prune = now
            port.housekeeping(now, bootstrap_queue)
            self._check_intents(now, bootstrap_queue)
            if self.link is not None:
                self.link.watchdog()

        in_bootstrap = {id(req) for req in bootstrap_queue}
        in_waiting = {id(req) for req in waiting_queue}
        outstanding: Dict[Any, int] = defaultdict(int)
        # Work mode: CHOICE_PENDING rooms reserve 1 / |eligible| on each of
        # their eligible candidates (count mode: 1 on every candidate).
        pend_share: Dict[Any, float] = defaultdict(float)
        in_flight = in_flight_tokens = held = launched = pending_choice = 0
        for room, st in list(self.rooms.items()):
            req = st.req
            if st.state == CHOICE_PENDING:
                if id(req) in in_bootstrap:
                    # Waits for the link's commit: in the window like an
                    # ADMIT in flight, and in every candidate's headroom.
                    pending_choice += 1
                    in_flight += 1
                    in_flight_tokens += st.tokens
                    for key in st.pend_keys:
                        pend_share[key] += 1.0 / len(st.pend_keys)
                else:  # aborted or failed before its commit
                    self.rooms.pop(room)
                    port.forget(room)
                continue
            if id(req) in in_bootstrap:
                age = now - st.admit_time
                if st.state == ADMIT_SENT and age >= hold_s:
                    # HELD_AT_D no longer counts in the lead window.
                    st.state = HELD_AT_D
                    if self.metrics is not None:
                        self.metrics.held.inc()
                if not st.resent and age >= self.ADMIT_RESEND_S:
                    # T6: ADMIT and RELEASE are idempotent on D; resend once.
                    st.resent = True
                    port.send_admit(st.dest, room, st.seq)
                    for loser in st.losers:
                        port.send_release(loser, room, st.seq)
                outstanding[st.d_key] += 1
                if st.state == ADMIT_SENT:
                    in_flight += 1
                    in_flight_tokens += st.tokens
                else:
                    held += 1
                continue
            aborted = is_aborted(req)
            if id(req) in in_waiting and not aborted:
                if st.state != BOUND:
                    self._on_bound(room, st)
                continue
            # Launched, or failed / aborted: the room leaves the binder.
            self.rooms.pop(room)
            port.forget(room)
            if not aborted:
                if st.state != BOUND:
                    self._on_bound(room, st)
                launched += 1
                self._on_launched(st)

        ahead = port.ahead(waiting_queue)

        if batch_ran:
            self.ewma += self.EWMA_ALPHA * (launched - self.ewma)
        self.r_ahead = max(
            cfg.lead_min_reqs,
            min(cfg.lead_max_reqs, math.ceil(2 * self.ewma) + self.boost),
        )
        # Rule 3: a long chunked prefill binds nothing until <= 1 chunk is left
        # (work mode: until <= need_iters chunks are left).
        chunk_left = 0 if chunked_req is None else _unlaunched_tokens(chunked_req)
        last_chunk = chunked_req is None or chunk_left <= self.chunk_tokens
        chunk_ok = last_chunk
        if work is not None:
            chunk_ok = chunk_left <= work.plan(self.boost)
        unbound = False
        submits: List[ChoiceItem] = []
        for req in bootstrap_queue:
            if not port.is_late(req) or is_aborted(req):
                continue
            room = port.key_of(req)
            st = self.rooms.get(room)
            if st is not None:
                unbound |= st.state in (ADMIT_SENT, CHOICE_PENDING)
                continue
            unbound = True
            # Rules 1 and 2 over the work ahead of this room: bound requests
            # not started yet plus the ADMITs in flight. Below R_min the token
            # cap does not apply, so the window always covers the overlap lag
            # (X5) even for long prompts.
            ahead_reqs, ahead_tokens = ahead.before(req)
            bound_not_started = ahead_reqs + in_flight
            lead_tokens = ahead_tokens + in_flight_tokens
            if work is not None:
                if not self._work_window_open(
                    chunk_ok, bound_not_started, chunk_left, lead_tokens
                ):
                    continue
            elif not chunk_ok or bound_not_started >= self.r_ahead:
                continue
            elif (
                bound_not_started >= cfg.lead_min_reqs
                and lead_tokens >= cfg.lead_max_tokens
            ):
                continue
            if port.is_failed(room):
                continue
            choice = port.choice(req)
            if choice is None:
                dest = port.destination(room, req)
                if dest is None:
                    # B3: the D leg is behind; bind when its INTENT arrives.
                    continue
                if outstanding[dest.d_key] >= cfg.max_outstanding_per_d:
                    # Rule 4: skip a room whose D already has enough outstanding.
                    continue
                tokens = _input_len(req)
                self._admit(req, room, dest, tokens, now, bound_not_started)
            else:
                # Stage 2: a candidate is eligible with its INTENT held and
                # rule-4 headroom (rooms in CHOICE_PENDING count against
                # every candidate); B3 when none is.
                cands = port.candidates(room)
                if work is None:
                    cap = cfg.max_outstanding_per_d - pending_choice
                    eligible = [
                        i
                        for i, cand in sorted(cands.items())
                        if i < choice.k and outstanding[cand.d_key] < cap
                    ]
                else:
                    cap = cfg.max_outstanding_per_d
                    eligible = [
                        i
                        for i, cand in sorted(cands.items())
                        if i < choice.k
                        and _work_headroom(outstanding, pend_share, cand.d_key, cap)
                    ]
                if choice.local_ok:
                    # Symmetric: the owner itself (LOCAL) has no reader and
                    # no rule-4 cap; it is the last eligible index.
                    eligible.append(WINNER_LOCAL)
                if not eligible:
                    continue
                tokens = _input_len(req)
                fallback = _fallback_winner(choice.default_idx, eligible)
                if self.link is not None and self.link.router_present():
                    item = self._pend(req, room, choice, eligible, fallback, len(cands))
                    pend = self.rooms[room]
                    pend.lead_reqs = bound_not_started
                    pend.pend_keys = tuple(
                        cands[i].d_key if i in cands else LATE_D_LOCAL_KEY
                        for i in eligible
                    )
                    for key in pend.pend_keys:
                        pend_share[key] += 1.0 / len(pend.pend_keys)
                    submits.append(item)
                    pending_choice += 1
                    in_flight += 1
                    in_flight_tokens += tokens
                    continue
                dest = self._commit(
                    req,
                    room,
                    choice,
                    fallback,
                    len(cands),
                    tokens,
                    now,
                    bound_not_started,
                    REASON_NO_ROUTER,
                )
                if dest is None:
                    continue
            outstanding[dest.d_key] += 1
            in_flight += 1
            in_flight_tokens += tokens

        if submits:
            self.link.submit(submits)

        # P-starved has one definition in both modes: nothing bound to run,
        # at most one chunk left, unbound rooms waiting.
        self._update_boost(ahead.reqs == 0 and last_chunk and unbound)
        if self.metrics is not None:
            self.metrics.held_reqs.set(held)
            self.metrics.r_ahead.set(self.r_ahead)
        if work is not None:
            if self.metrics is not None:
                self.metrics.lead_target.set(work.target)
            lead_work = chunk_left + work.uncached(ahead.tokens + in_flight_tokens)
            work.maybe_log(now, lead_work, ahead.reqs + in_flight)

    def need_start_code(self, req, queued_uncached=0):
        """Leader lane: INTENT, pending choice OR committed ADMIT permits start."""
        from decoupled_engine.srt.disaggregation.late_d_need import HOLD, START_OK
        room = self.port.key_of(req)
        with self.lock:
            st = self.rooms.get(room)
            ok = st is not None or bool(self.port.candidates(room))
            if not ok:
                ok = self.port.destination(room, req) is not None
            code = START_OK if ok else 0
            threshold = envs.SGLANG_LATE_D_WARM_HOLD_MIN_TOKENS.get()
            if threshold > 0 and queued_uncached < self.chunk_tokens:
                candidate = self.need_plan.candidate(req) if self.need_plan else None
                device = getattr(candidate, "device_tokens", 0)
                age = time.time() - self.port.created_at(req, time.time())
                wait = min(.250, self.work.bind_budget_s())
                if device >= threshold and age < wait:
                    code |= HOLD
            return code

    def _need_trigger(self, req):
        from decoupled_engine.srt.disaggregation.late_d_need import need_bind_lead
        if self.cfg.allocation_trigger == "first_chunk":
            return "first_chunk" if getattr(req, "late_d_first_chunk_started_perf", 0.0) > 0 else None
        room = self.port.key_of(req)
        planned = None
        candidate = None
        plan = getattr(self.need_plan, "plan", None)
        if self.need_plan is not None:
            candidate = self.need_plan.candidate(req, ignore_intent=True)
        if plan is not None:
            planned = plan.entries.get(req.rid)
        uncached = getattr(candidate, "uncached", _unlaunched_tokens(req))
        host = getattr(candidate, "host_tokens", 0)
        started = getattr(req, "launched_unbound", False)
        initial_uncached = getattr(req, "late_d_initial_uncached_tokens", None)
        if started and initial_uncached is not None:
            # h/u describe work at the room's first launch, not its shrinking
            # remaining suffix. Already computed KV still needs paced transfer.
            uncached = initial_uncached
            host = req.late_d_initial_host_tokens
        budget = envs.SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS.get()
        b, k = need_bind_lead(self.work.bind_budget_s(), self.work.iter_s or .45,
                              self.chunk_tokens, host, uncached, budget,
                              envs.SGLANG_LATE_D_BIND_MARGIN.get())
        e = getattr(planned, "end_pass", None)
        if room in self.need_parked or (started and (
                (e is not None and e <= 1) or _unlaunched_tokens(req) == 0)):
            # A final chunk leaves chunked_req and the waiting plan at launch.
            # It remains tracked in last_batch and must still override rule 4.
            return "T3"
        if started and self.need_gate_closed:
            return "T4"
        if getattr(req, "late_d_gate_blocked", False):
            p = getattr(planned, "gate_open_start", None)
            # Without a complete forecast, binding the gated head preserves
            # liveness: it removes the gate after D allocates the destination.
            if p is not None and p <= b + envs.SGLANG_LATE_D_BIND_MARGIN.get():
                return "T2"
        if e is not None and e <= k:
            return "T1"
        if planned is None and plan is not None and candidate is not None:
            tail = plan.tail_candidate(candidate)
            if tail is not None and tail.end_pass is not None and tail.end_pass <= k:
                return "T0"
        # No lookahead exists on the very first receive pass. T0 can still
        # admit a one-chunk room on an otherwise empty worker. A started
        # request uses its actual remaining chunks; no giant-size rule.
        if plan is None and uncached <= self.chunk_tokens:
            return "T0"
        if started and math.ceil(_unlaunched_tokens(req) / self.chunk_tokens) <= k:
            return "T1"
        return None

    def _step_need(self, tracked, waiting, chunked_req, batch_ran):
        """Same wire protocol as work mode; track pending rooms in every queue."""
        now = time.time()
        port, cfg = self.port, self.cfg
        self.work.on_step(time.perf_counter(), batch_ran)
        self.boost = 0
        pending_ids = {id(r) for r in tracked}
        if now - self._last_prune >= 1.0:
            self._last_prune = now
            port.housekeeping(now, tracked)
            self._check_intents(now, tracked)
            if self.link is not None:
                self.link.watchdog()
        outstanding = defaultdict(int)
        shares = defaultdict(float)
        hold_s = max(cfg.hold_detect_s, min(self.work.bind_budget_s(), self.ADMIT_RESEND_S))
        for room, st in list(self.rooms.items()):
            req = st.req
            if id(req) not in pending_ids or is_aborted(req):
                self.rooms.pop(room)
                if not is_aborted(req) and not req.pending_bootstrap:
                    self._on_bound(room, st)
                    self._on_launched(st)
                port.forget(room)
                continue
            if st.state == CHOICE_PENDING:
                for key in st.pend_keys:
                    shares[key] += 1.0 / len(st.pend_keys)
                continue
            age = now - st.admit_time
            if st.state == ADMIT_SENT and age >= hold_s:
                st.state = HELD_AT_D
                logger.info("LATE_D_NEED held rid=%s age_s=%.3f tokens=%d",
                            req.rid, age, st.tokens)
                if self.metrics is not None:
                    self.metrics.held.inc()
            if not st.resent and age >= self.ADMIT_RESEND_S:
                st.resent = True
                port.send_admit(st.dest, room, st.seq)
                for dest in st.losers:
                    port.send_release(dest, room, st.seq)
            outstanding[st.d_key] += 1
        self._submit_need(tracked, outstanding, shares, now)

    def first_chunk_started(self, requests):
        """Leader-only asynchronous signal at the CPU forward launch boundary.

        No poll, collective, bind wait, or GPU synchronization is added here.
        A full outstanding window cannot postpone this explicit trigger; the
        existing bounded P retention/park timer still limits unbound KV.
        """
        if self.cfg.allocation_trigger != "first_chunk":
            return
        with self.lock:
            outstanding, shares = defaultdict(int), defaultdict(float)
            for st in self.rooms.values():
                if st.state == CHOICE_PENDING:
                    for key in st.pend_keys:
                        shares[key] += 1.0 / len(st.pend_keys)
                else:
                    outstanding[st.d_key] += 1
            self._submit_need(requests, outstanding, shares, time.time())

    def _submit_need(self, tracked, outstanding, shares, now):
        from decoupled_engine.srt.disaggregation.late_d_need import admit_delay_for_smoke, force_bound_for_smoke
        port, cfg = self.port, self.cfg
        submits = []
        for req in tracked:
            if not port.is_late(req) or is_aborted(req):
                continue
            room = port.key_of(req)
            if room in self.rooms or port.is_failed(room):
                continue
            force_bound = force_bound_for_smoke(req)
            withheld_prefix = envs.SGLANG_TEST_LATE_D_WITHHOLD_ADMIT_RID_PREFIX.get()
            if withheld_prefix and req.rid.startswith(withheld_prefix) and not force_bound:
                continue  # Exercise parked-room failure while D stays DEFERRED.
            delay = 0.0 if force_bound else admit_delay_for_smoke(req)
            if delay > 0 and now - port.created_at(req, now) < delay:
                continue
            trigger = "test_bound" if force_bound else self._need_trigger(req)
            if trigger is None:
                continue
            override = trigger in ("T3", "first_chunk")
            choice = port.choice(req)
            tokens = _input_len(req)
            if choice is None:
                dest = port.destination(room, req)
                if dest is None:
                    continue
                at_cap = not _work_headroom(outstanding, shares, dest.d_key,
                                           cfg.max_outstanding_per_d)
                if at_cap and not override:
                    continue
                if at_cap:
                    self.need_rule4_overrides += 1
                self._admit(req, room, dest, tokens, now, 0)
            else:
                cands = port.candidates(room)
                eligible = [i for i, c in sorted(cands.items()) if i < choice.k and
                            (override or _work_headroom(outstanding, shares, c.d_key,
                                                        cfg.max_outstanding_per_d))]
                if not eligible:
                    continue
                if override and any(not _work_headroom(outstanding, shares, cands[i].d_key,
                                    cfg.max_outstanding_per_d) for i in eligible):
                    self.need_rule4_overrides += 1
                fallback = _fallback_winner(choice.default_idx, eligible)
                if self.link is not None and self.link.router_present():
                    item = self._pend(req, room, choice, eligible, fallback, len(cands))
                    st = self.rooms[room]
                    st.pend_keys = tuple(cands[i].d_key for i in eligible)
                    for key in st.pend_keys:
                        shares[key] += 1.0 / len(st.pend_keys)
                    submits.append(item)
                    logger.info("LATE_D_NEED trigger=%s rid=%s choice_pending=1", trigger, req.rid)
                    continue
                dest = self._commit(req, room, choice, fallback, len(cands), tokens,
                                    now, 0, REASON_NO_ROUTER)
                if dest is None:
                    continue
            outstanding[dest.d_key] += 1
            logger.info("LATE_D_NEED trigger=%s rid=%s rule4_overrides=%d", trigger,
                        req.rid, self.need_rule4_overrides)
        if submits:
            self.link.submit(submits)

    def _check_intents(self, now: float, bootstrap_queue: List[Req]) -> None:
        """Defense in depth, once per second. A late room that waits in the
        bootstrap queue without any INTENT for over INTENT_MISSING_S is
        logged once and counted (late_d_intent_missing_total, must be 0: the
        binder skips such a room, B3, until the admission timer ends it). If
        its INTENT arrives later (the D resends the INTENTs of DEFERRED
        entries), the room is counted as restored."""
        port = self.port
        waiting = set()
        for req in bootstrap_queue:
            if not port.is_late(req) or is_aborted(req):
                continue
            key = port.key_of(req)
            waiting.add(key)
            if key in self.rooms or key in self._missing:
                continue
            age = now - port.created_at(req, now)
            if age > self.INTENT_MISSING_S and port.intent_missing(key):
                self._missing[key] = (now, req.rid)
                if self.metrics is not None:
                    self.metrics.intent_missing.inc()
                logger.warning(
                    "LATE_D no INTENT rid=%s room=%s waited_s=%.1f", req.rid, key, age
                )
        for key, (since, rid) in list(self._missing.items()):
            if key in self.rooms or (key in waiting and not port.intent_missing(key)):
                del self._missing[key]
                if self.metrics is not None:
                    self.metrics.intent_restored.inc()
                logger.warning(
                    "LATE_D INTENT restored rid=%s room=%s missing_s=%.1f",
                    rid,
                    key,
                    now - since,
                )
            elif key not in waiting:
                del self._missing[key]  # the room ended without its INTENT

    def _work_window_open(
        self, chunk_ok: bool, bound_not_started: int, chunk_left: int, lead_tokens: int
    ) -> bool:
        """Lead mode work: whether the next unbound room may bind. The window
        is open below lead_min_reqs, and below the work target while the
        prompt tokens bound ahead stay under the D KV cap; never at
        lead_max_reqs or with more than need_iters chunks left."""
        cfg, work = self.cfg, self.work
        if not chunk_ok or bound_not_started >= cfg.lead_max_reqs:
            return False
        if bound_not_started < cfg.lead_min_reqs:
            return True
        lead_work = chunk_left + work.uncached(lead_tokens)
        return lead_work < work.target and lead_tokens < work.kv_cap


    def _update_boost(self, starved: bool) -> None:
        """P-starved: nothing bound to run while unbound rooms wait. Grow the
        window once per starved episode (the P loop spins without sleeping
        while it waits, so a per-iteration count would saturate at once);
        shrink it after CALM_ITERS iterations without starvation."""
        if starved:
            self._calm_iters = 0
            if not self._starved:
                self.boost = min(
                    self.boost + 1, self.cfg.lead_max_reqs - self.cfg.lead_min_reqs
                )
                if self.metrics is not None:
                    self.metrics.starved.inc()
        else:
            self._calm_iters += 1
            if self._calm_iters >= self.CALM_ITERS:
                self._calm_iters = 0
                self.boost = max(0, self.boost - 1)
        self._starved = starved

    def _admit(
        self,
        req: Req,
        room: int,
        dest: Any,
        tokens: int,
        now: float,
        lead_reqs: int,
    ) -> None:
        self.seq += 1
        admit_perf = time.perf_counter()
        st = _BindRoom(
            req=req,
            d_key=dest.d_key,
            seq=self.seq,
            tokens=tokens,
            admit_time=now,
            admit_perf=admit_perf,
            dest=dest,
            decide_perf=admit_perf,
        )
        self.rooms[room] = st
        self.port.note_admit(room, dest, now)
        self.port.send_admit(dest, room, st.seq)
        req.time_stats.late_d_admit_time = st.admit_perf
        init_time = self.port.created_at(req, now)
        if self.metrics is not None:
            self.metrics.admits.inc()
            self.metrics.backlog_wait.observe(max(0.0, now - init_time))
        logger.info(
            "LATE_D admit rid=%s room=%s d=%s seq=%d tokens=%d lead_reqs=%d "
            "t_ms=%.1f",
            req.rid,
            room,
            dest.label,
            st.seq,
            tokens,
            lead_reqs,
            (now - init_time) * 1000.0,
        )

    def _commit(
        self,
        req: Req,
        room: int,
        choice: LateDChoice,
        winner_idx: int,
        ready: int,
        tokens: int,
        now: float,
        lead_reqs: int,
        reason: str,
        seq: Optional[int] = None,
        decide_perf: Optional[float] = None,
    ) -> Any:
        """Stage 2: commit the room to ``winner_idx`` (ADMIT, RELEASE,
        bind timer). ``seq``: the room's bind_seq from CHOICE_PENDING; None
        takes the next one. ``decide_perf``: the room's BIND_READY time;
        None: now. Returns the winner's destination, or None if the port
        refused (nothing was sent)."""
        if seq is None:
            seq = self.seq + 1
        committed = self.port.commit(room, seq, winner_idx, now)
        if committed is None:
            return None
        self.seq = max(self.seq, seq)
        winner, losers = committed
        admit_perf = time.perf_counter()
        st = _BindRoom(
            req=req,
            d_key=winner.d_key,
            seq=seq,
            tokens=tokens,
            admit_time=now,
            admit_perf=admit_perf,
            dest=winner,
            losers=tuple(losers),
            decide_perf=admit_perf if decide_perf is None else decide_perf,
        )
        self.rooms[room] = st
        req.time_stats.late_d_admit_time = st.admit_perf
        init_time = self.port.created_at(req, now)
        if self.metrics is not None:
            self.metrics.admits.inc()
            self.metrics.releases.inc(len(losers))
            if winner_idx != choice.default_idx:
                self.metrics.rebinds.inc()
            self.metrics.backlog_wait.observe(max(0.0, now - init_time))
        logger.info(
            "LATE_D commit rid=%s room=%s winner=%d default=%d k=%d ready=%d "
            "d=%s released=%d seq=%d tokens=%d lead_reqs=%d reason=%s t_ms=%.1f",
            req.rid,
            room,
            winner_idx,
            choice.default_idx,
            choice.k,
            ready,
            winner.label,
            len(losers),
            seq,
            tokens,
            lead_reqs,
            reason,
            (now - init_time) * 1000.0,
        )
        return winner

    def _pend(
        self,
        req: Req,
        room: int,
        choice: LateDChoice,
        eligible: List[int],
        fallback: int,
        ready: int,
    ) -> ChoiceItem:
        """Stage 2 with a router: CHOICE_PENDING until the link commits.
        Returns the BIND_READY item for the link."""
        self.seq += 1
        tokens = _input_len(req)
        self.rooms[room] = _BindRoom(
            req=req,
            d_key=None,
            seq=self.seq,
            tokens=tokens,
            admit_time=0.0,
            admit_perf=0.0,
            state=CHOICE_PENDING,
            choice=choice,
            ready=ready,
            decide_perf=time.perf_counter(),
        )
        return ChoiceItem(
            key=room,
            bind_seq=self.seq,
            eligible=sum(1 << i for i in eligible if i < LATE_D_MAX_CANDIDATES),
            ready=ready,
            input_len=tokens,
            default_idx=choice.default_idx,
            fallback_idx=fallback,
            rid=req.rid,
            local_ok=WINNER_LOCAL in eligible,
            local_load=self.port.local_load(room) if WINNER_LOCAL in eligible else 0,
            first_chunk_start_unix_ns=getattr(req, "late_d_first_chunk_started_unix_ns", 0),
        )

    def commit_choice(
        self, room: int, bind_seq: int, winner_idx: int, reason: str
    ) -> Optional[int]:
        """The link thread: commit a CHOICE_PENDING room to ``winner_idx``.
        Returns the committed candidate, or None when the room is gone, was
        already committed, or its winner lost its INTENT (the room is then
        unbound again and the next step selects it anew)."""
        with self.lock:
            st = self.rooms.get(room)
            if st is None or st.state != CHOICE_PENDING or st.seq != bind_seq:
                return None
            del self.rooms[room]
            self._count_cap_overrun(room, winner_idx)
            dest = self._commit(
                st.req,
                room,
                st.choice,
                winner_idx,
                st.ready,
                st.tokens,
                time.time(),
                st.lead_reqs,
                reason,
                seq=bind_seq,
                decide_perf=st.decide_perf,
            )
            return None if dest is None else winner_idx

    def _count_cap_overrun(self, room: int, winner_idx: int) -> None:
        """A commit to a candidate that already has the outstanding cap
        (ADMITs sent or held since the room's BIND_READY). Rooms in
        CHOICE_PENDING reserve headroom on every candidate, so only stage-1
        rooms bound in between can cause it."""
        if self.metrics is None:
            return
        cand = self.port.candidates(room).get(winner_idx)
        if cand is None:
            return
        outstanding = sum(
            1
            for st in self.rooms.values()
            if st.state in (ADMIT_SENT, HELD_AT_D) and st.d_key == cand.d_key
        )
        if outstanding >= self.cfg.max_outstanding_per_d:
            self.metrics.cap_overrun.inc()

    def _on_bound(self, room: int, st: _BindRoom) -> None:
        st.state = BOUND
        done = self.port.bound_time(st.req)
        bind_s = max(0.0, done - st.admit_perf) if done > 0 else 0.0
        if self.metrics is not None and done > 0:
            self.metrics.bind_latency.observe(bind_s)
        # Finalize after compute can include the forward, not just the bind.
        # Need mode uses the specified conservative fallback: sample only
        # rooms bound before their first launch.
        sample_ok = not (self.cfg.lead_mode == LEAD_NEED and
                         getattr(st.req, "launched_unbound", False))
        if self.work is not None and done > 0 and sample_ok:
            self.work.on_bound(max(0.0, done - (st.decide_perf or st.admit_perf)))
        if self.cfg.lead_mode == LEAD_NEED:
            logger.info("LATE_D_NEED bind_sample rid=%s source=%s", st.req.rid,
                        "before_launch" if sample_ok else "excluded_after_launch")
        logger.info(
            "LATE_D bound rid=%s room=%s seq=%d bind_ms=%.1f",
            st.req.rid,
            room,
            st.seq,
            bind_s * 1000.0,
        )

    def _on_launched(self, st: _BindRoom) -> None:
        if self.work is not None:
            self.work.on_launched(st.req)
        stats = st.req.time_stats
        if self.metrics is None or not stats.forward_entry_time > 0:
            return
        done = self.port.bound_time(st.req)
        if done > 0:
            self.metrics.lead.observe(max(0.0, stats.forward_entry_time - done))


# ----------------------------------------------------------------------
# D side: late allocation walk (mode-neutral)
# ----------------------------------------------------------------------


class LateDDecode:
    """D side of late allocation. Every D rank runs it in
    lockstep: its decisions depend only on the request broadcast (ADMIT and
    EXPIRE controls, in order) and on the pool state, and its aging counts
    preallocation passes, never wall clock. The leader (attn-TP rank 0) also
    sends INTENTs, turns ADMITs and DEFERRED expiries into broadcast controls,
    and owns the metrics and log lines. The mode reaches it only through
    ``self.port`` (a LateDReaderPort)."""

    # Defense in depth: the leader sends the INTENT of an entry that is still
    # DEFERRED again after this long, in case the prefill lost it (it is
    # idempotent there).
    INTENT_REFRESH_S = 60.0

    def __init__(
        self,
        conn: Any,
        config: LateDConfig,
        metrics: Optional[LateDMetrics] = None,
    ):
        # ``conn``: a LateDReaderPort, or a PD LateDConnState (PdReaderPort).
        self.conn = conn
        self.port: LateDReaderPort = (
            conn if isinstance(conn, LateDReaderPort) else PdReaderPort(conn)
        )
        self.cfg = config
        self.is_leader = self.port.is_leader
        self.metrics = metrics
        self.pass_index = 0
        self.admit_counter = 0
        # The reservation claim: the oldest HELD entry, after claim_iters passes.
        self.claim: Optional[DecodeRequest] = None
        self._claim_need: Optional[Tuple[int, ...]] = None
        # Symmetric: why the claim missed in this pass
        # (the driver's reason), whether it missed on memory in the last
        # pass, how many passes in a row, and when it became the claim.
        self._claim_miss: Optional[str] = None
        self.claim_memory_blocked = False
        self.claim_blocked_passes = 0
        self._claim_since_pass = 0
        self._last_ttl_scan = 0.0
        # Leader, ports with hold_unknown: controls whose key has no entry yet.
        self._held_back: List[LateDControl] = []
        self.stats_early_held = 0
        self.stats_early_dropped: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Queue add (all ranks) and INTENT (leader)
    # ------------------------------------------------------------------

    def defer(self, decode_req: DecodeRequest) -> None:
        decode_req.late_d_state = DEFERRED
        decode_req.late_d_deferred_at = time.monotonic()

    def send_intent(self, decode_req: DecodeRequest) -> None:
        """Leader, after ``receiver.init``: tell the prefill leader this room
        waits DEFERRED here. A failed send fails the room (as send_metadata)."""
        if not self.is_leader:
            return
        self.port.send_intent(decode_req)
        if self.port.refreshes_intents:
            decode_req.late_d_intent_at = time.monotonic()

    # ------------------------------------------------------------------
    # Broadcast controls
    # ------------------------------------------------------------------

    def pull_controls(
        self, prealloc_queue: List[DecodeRequest], recv_reqs: Any = ()
    ) -> List[LateDControl]:
        """Leader, before the request broadcast: pending ADMITs, plus an
        EXPIRE for each DEFERRED entry past its TTL (N3). ``recv_reqs``: the
        raw batch about to be broadcast (for ports with hold_unknown)."""
        held_back, self._held_back = self._held_back, []
        controls = held_back + self.port.drain_inbox()
        if self.port.hold_unknown and controls:
            controls = self._release_known(
                controls, len(held_back), prealloc_queue, recv_reqs
            )
        now = time.monotonic()
        if now - self._last_ttl_scan >= 1.0:
            self._last_ttl_scan = now
            held_ttl = self.cfg.held_ttl_s
            for entry in prealloc_queue:
                state = entry.late_d_state
                if (
                    state == DEFERRED
                    and now - entry.late_d_deferred_at > self.cfg.deferred_ttl_s
                ):
                    entry.late_d_deferred_at = math.inf  # expire once
                    controls.append(
                        LateDControl(kind=CONTROL_EXPIRE, room=entry.req.bootstrap_room)
                    )
                elif (
                    held_ttl > 0
                    and state in (ADMITTED, HELD)
                    and now - entry.late_d_admitted_at > held_ttl
                ):
                    # Symmetric HELD TTL (from the ADMIT): a backstop for a
                    # dead owner or a lost message; the owner's bind timer
                    # fires first.
                    entry.late_d_admitted_at = math.inf  # expire once
                    controls.append(
                        LateDControl(kind=CONTROL_EXPIRE, room=entry.req.bootstrap_room)
                    )
                elif (
                    state == DEFERRED
                    and self.port.refreshes_intents
                    and math.isfinite(entry.late_d_deferred_at)
                    and 0 < entry.late_d_intent_at <= now - self.INTENT_REFRESH_S
                ):
                    entry.late_d_intent_at = now
                    self.port.refresh_intent(entry)
                    if self.metrics is not None:
                        self.metrics.intent_refreshes.inc()
        return controls

    def _release_known(
        self,
        controls: List[LateDControl],
        n_old: int,
        prealloc_queue: List[Any],
        recv_reqs: Any,
    ) -> List[LateDControl]:
        """Leader: pass the controls whose key has a late entry, or a leg in
        this broadcast; drop those of keys that already left the late states;
        hold back the rest (the first ``n_old`` were held before) for at most
        ``early_control_ttl_s`` and ``early_control_cap`` entries."""
        known = {
            entry.req.bootstrap_room
            for entry in prealloc_queue
            if entry.late_d_state is not None
        }
        known |= self.port.batch_import_keys(recv_reqs)
        now = time.monotonic()
        out = []
        for i, control in enumerate(controls):
            if control.room in known:
                out.append(control)
            elif self.port.key_state(control.room) != "unknown":
                self._drop_early(control)
            elif (
                now - control.t_recv <= self.cfg.early_control_ttl_s
                and len(self._held_back) < self.cfg.early_control_cap
            ):
                self._held_back.append(control)
                if i >= n_old:
                    self.stats_early_held += 1
                    metrics = sym_metrics(self.metrics)
                    if metrics is not None:
                        metrics.early_held.inc()
            else:
                self._drop_early(control)
        return out

    def _drop_early(self, control: LateDControl) -> None:
        self.stats_early_dropped[control.kind] += 1
        metrics = sym_metrics(self.metrics)
        if metrics is not None:
            metrics.early_dropped.labels(**metrics.labels, kind=control.kind).inc()

    @staticmethod
    def _applies(kind: str, state: Optional[str], held_ttl: bool) -> bool:
        """The entry states a control acts on: ``fail`` (symmetric) on every
        never-allocated state; ``expire`` also on ADMITTED / HELD with the
        symmetric HELD TTL; everything else on DEFERRED only."""
        if kind == CONTROL_FAIL or (kind == CONTROL_EXPIRE and held_ttl):
            return state in (DEFERRED, ADMITTED, HELD)
        return state == DEFERRED

    def apply_control(
        self, control: LateDControl, prealloc_queue: List[DecodeRequest]
    ) -> None:
        """All ranks, same iteration and order (from the request broadcast)."""
        entry = next(
            (
                e
                for e in prealloc_queue
                if e.late_d_state is not None and e.req.bootstrap_room == control.room
            ),
            None,
        )
        held_ttl = self.cfg.held_ttl_s > 0
        if entry is None or not self._applies(
            control.kind, entry.late_d_state, held_ttl
        ):
            # Unknown, already allocated, or a resent duplicate.
            if self.is_leader:
                state = None if entry is None else entry.late_d_state
                admitted = state in (ADMITTED, HELD, ALLOCATED)
                if control.kind == CONTROL_RELEASE and admitted:
                    # P releases only candidates it did not admit (V5).
                    logger.warning(
                        "LATE_D release ignored for an admitted entry rid=%s "
                        "room=%s seq=%d state=%s",
                        entry.req.rid,
                        control.room,
                        control.seq,
                        state,
                    )
                else:
                    logger.debug(
                        "LATE_D %s ignored room=%s seq=%d state=%s",
                        control.kind,
                        control.room,
                        control.seq,
                        state,
                    )
            return
        if control.kind == CONTROL_RELEASE:
            # Stage 2: another candidate won the room. Every rank ends the
            # leg in this iteration (HTTP 409); the sweep clears it without
            # an ABORT to P.
            entry.late_d_state = RELEASED
            self.port.release(entry, control.room)
            if self.is_leader:
                logger.info(
                    "LATE_D released rid=%s room=%s seq=%d",
                    entry.req.rid,
                    control.room,
                    control.seq,
                )
                if self.metrics is not None:
                    self.metrics.released.inc()
        elif control.kind == CONTROL_ADMIT:
            entry.late_d_state = ADMITTED
            self.admit_counter += 1
            entry.late_d_admit_order = self.admit_counter
            entry.req.time_stats.late_d_admit_time = time.perf_counter()
            if held_ttl:
                # Symmetric: the HELD TTL and the reader bind latency start
                # at rank 0's IO-thread receipt (only rank 0 reads them).
                entry.late_d_admitted_at = control.t_recv or time.monotonic()
        elif control.kind == CONTROL_EXPIRE:
            deferred = entry.late_d_state == DEFERRED
            ttl = self.cfg.deferred_ttl_s if deferred else self.cfg.held_ttl_s
            what = "DEFERRED" if deferred else "ADMITTED"
            self.port.expire(
                entry,
                control.room,
                f"Late D: request {what} for more than {ttl:.0f}s without "
                f"{'ADMIT' if deferred else 'allocation'}",
            )
            if self.is_leader:
                logger.error(
                    "LATE_D %s TTL abort rid=%s room=%s ttl_s=%.0f",
                    "deferred" if deferred else "held",
                    entry.req.rid,
                    control.room,
                    ttl,
                )
                if self.metrics is not None:
                    self.metrics.ttl_aborts.inc()
        elif control.kind == CONTROL_FAIL:
            state = entry.late_d_state
            # The reader IO thread's FAIL (an owner ABORT) carries no detail.
            detail = control.detail or FAIL_OWNER_ABORT
            self.port.fail(entry, control.room, detail)
            if self.is_leader:
                logger.warning(
                    "LATE_D fail rid=%s room=%s state=%s detail=%s",
                    entry.req.rid,
                    control.room,
                    state,
                    detail,
                )
                metrics = sym_metrics(self.metrics)
                if metrics is not None and detail == FAIL_OWNER_ABORT:
                    metrics.owner_abort_fail.inc()

    # ------------------------------------------------------------------
    # Preallocation walk (all ranks)
    # ------------------------------------------------------------------

    def walk_order(
        self, prealloc_queue: List[DecodeRequest], removed: Set[int] = frozenset()
    ) -> List[Tuple[int, DecodeRequest]]:
        """``(index, entry)`` pairs for pop_preallocated: entries that are not
        late keep today's order and come first; then the late entries D may
        allocate now, the claim first, then by ADMIT order. DEFERRED entries
        and the ``removed`` indices (failed this pass) are not visited."""
        self.pass_index += 1
        first, late = [], []
        deferred = 0
        for i, entry in enumerate(prealloc_queue):
            if i in removed:
                continue
            state = entry.late_d_state
            if state is None:
                first.append((i, entry))
            elif state == DEFERRED:
                deferred += 1
            elif state in (ADMITTED, HELD):
                late.append((i, entry))
        self._update_claim(late)
        claim = self.claim
        late.sort(key=lambda item: (item[1] is not claim, item[1].late_d_admit_order))
        if self.metrics is not None:
            self.metrics.deferred_reqs.set(deferred)
        return first + late

    def _update_claim(self, late: List[Tuple[int, DecodeRequest]]) -> None:
        if self.claim is not None and not any(
            entry is self.claim and entry.late_d_state == HELD for _, entry in late
        ):
            self.claim = None  # allocated or removed
        if self.claim is not None:
            return
        self.claim_blocked_passes = 0
        self.claim_memory_blocked = False
        held = [entry for _, entry in late if entry.late_d_state == HELD]
        if not held:
            return
        oldest = min(held, key=lambda entry: entry.late_d_admit_order)
        if self.pass_index - oldest.late_d_held_since < self.cfg.claim_iters:
            return
        self.claim = oldest
        self._claim_since_pass = self.pass_index
        if self.is_leader:
            logger.info(
                "LATE_D claim rid=%s room=%s held_passes=%d skips=%d",
                oldest.req.rid,
                oldest.req.bootstrap_room,
                self.pass_index - oldest.late_d_held_since,
                oldest.late_d_skips,
            )
            if self.metrics is not None:
                self.metrics.claims.inc()

    def admit_fits(
        self,
        entry: DecodeRequest,
        need: Tuple[int, ...],
        claim_need: Optional[Tuple[int, ...]],
        *budgets: Any,
    ) -> bool:
        """``need`` and ``budgets``: one value per admission dimension (PD:
        the full and SWA tokens of today's projected-growth checks). While a
        claim is active, another late entry must leave the claim's need free
        in every dimension: budget - claim_need >= need."""
        if claim_need is not None and entry is not self.claim:
            return all(
                n <= b - r for n, b, r in zip(need, budgets, claim_need, strict=True)
            )
        return all(n <= b for n, b in zip(need, budgets, strict=True))

    def on_no_fit(
        self,
        entry: DecodeRequest,
        need: Tuple[int, ...],
        claim_need: Optional[Tuple[int, ...]],
        reason: Optional[str] = None,
    ) -> Optional[Tuple[int, ...]]:
        """A late entry that does not fit is HELD and skipped. Returns the
        claim reservation for the rest of this pass. ``reason``: which
        admission dimension missed (symmetric: memory or budget)."""
        if entry.late_d_state == ADMITTED:
            entry.late_d_state = HELD
            entry.late_d_held_since = self.pass_index
            entry.late_d_held_at = time.perf_counter()
            if self.is_leader and self.metrics is not None:
                self.metrics.held_d.inc()
        entry.late_d_skips += 1
        if self.is_leader and self.metrics is not None:
            self.metrics.skips.inc()
        if entry is self.claim:
            self._claim_miss = reason
            return need
        return claim_need

    def end_pass(
        self,
        claim_need: Optional[Tuple[int, ...]],
        memory_misses: Tuple[str, ...] = (),
    ) -> None:
        """Walk driver, after each pass: the claim reservation the pass ended
        with (None when the claim fitted or there is none). Symmetric: a claim
        whose miss reason is in ``memory_misses`` is memory-blocked; the
        consecutive memory-blocked passes feed the escalation (6.4)."""
        self._claim_need = claim_need
        miss, self._claim_miss = self._claim_miss, None
        blocked = self.claim is not None and miss in memory_misses
        self.claim_blocked_passes = self.claim_blocked_passes + 1 if blocked else 0
        self.claim_memory_blocked = blocked
        escalate = self.cfg.claim_escalate_passes
        if (
            blocked
            and escalate > 0
            and self.claim_blocked_passes == escalate
            and self.is_leader
        ):
            logger.info(
                "LATE_D claim escalation rid=%s room=%s blocked_passes=%d",
                self.claim.req.rid,
                self.claim.req.bootstrap_room,
                self.claim_blocked_passes,
            )
            metrics = sym_metrics(self.metrics)
            if metrics is not None:
                metrics.claim_escalations.inc()

    def claim_reserve(self) -> Optional[Tuple[int, ...]]:
        """The reservation of the active claim as of the last pass, or None."""
        return self._claim_need if self.claim is not None else None

    def on_alloc(
        self,
        entry: DecodeRequest,
        need_full: int,
        budget_before: int,
        budget_after: int,
    ) -> None:
        was_claim = entry is self.claim
        if was_claim:
            self.claim = None
        entry.late_d_state = ALLOCATED
        if not self.is_leader:
            return
        if self.metrics is not None:
            self.metrics.allocs.inc()
            metrics = sym_metrics(self.metrics)
            if was_claim and metrics is not None:
                metrics.claim_wait_passes.observe(
                    self.pass_index - self._claim_since_pass
                )
            if entry.late_d_held_at > 0:
                self.metrics.held_seconds.observe(
                    time.perf_counter() - entry.late_d_held_at
                )
        logger.info(
            "LATE_D alloc rid=%s room=%s need=%d budget_before=%d budget_after=%d "
            "claim=%d skips=%d",
            entry.req.rid,
            entry.req.bootstrap_room,
            need_full,
            budget_before,
            budget_after,
            int(was_claim),
            entry.late_d_skips,
        )

    @staticmethod
    def deferred_load(prealloc_queue: List[DecodeRequest]) -> Tuple[int, int]:
        """(requests, tokens) DEFERRED on this decode, for /v1/loads."""
        reqs = tokens = 0
        for entry in prealloc_queue:
            if entry.late_d_state == DEFERRED:
                reqs += 1
                tokens += entry.req.seqlen
        return reqs, tokens


# ----------------------------------------------------------------------
# PD (Mooncake) adapter: ports
# ----------------------------------------------------------------------


class PdBinderPort(LateDBinderPort):
    """The P attn-TP rank 0 of a PD prefill: rooms wait in the bootstrap
    queue, INTENTs name their D, ADMITs (and stage-2 RELEASEs) leave on the
    LateDConnState sender."""

    def __init__(self, kv_mgr: CommonKVManager):
        self.kv_mgr = kv_mgr
        self.conn: LateDConnState = kv_mgr.late_d

    @property
    def is_leader(self) -> bool:
        return self.conn.is_leader

    def housekeeping(self, now: float, unbound: List[Req]) -> None:
        self.conn.prune_intents(keep={req.bootstrap_room for req in unbound})

    def intent_missing(self, key: int) -> bool:
        return not self.conn.peek_intents(key)

    def is_late(self, req: Req) -> bool:
        return getattr(req.disagg_kv_sender, "late_d", False)

    def is_failed(self, key: int) -> bool:
        return self.kv_mgr.request_status.get(key) == KVPoll.Failed

    def destination(self, key: int, req: Req) -> Optional[LateDIntent]:
        return self.conn.peek_intent(key)

    def note_admit(self, key: int, dest: LateDIntent, now: float) -> None:
        self.conn.note_admit(key, dest, now)

    def send_admit(self, dest: LateDIntent, key: int, seq: int) -> None:
        self.conn.send_admit(dest.d_key, key, seq)

    def forget(self, key: int) -> None:
        self.conn.forget_room(key)

    def created_at(self, req: Req, now: float) -> float:
        return getattr(req.disagg_kv_sender, "init_time", None) or now

    def bound_time(self, req: Req) -> float:
        return req.time_stats.bootstrap_done_time

    def first_admit_time(self, key: int) -> Optional[float]:
        return self.conn.first_admit.get(key)

    def choice(self, req: Req) -> Optional[LateDChoice]:
        if self.conn.mode != LATE_D_CHOOSE:
            return None
        field_value = getattr(req, "late_d", None)
        # A P leg without the field binds as in stage 1 (late-D-off traffic).
        return None if field_value is None else LateDChoice.of(field_value)

    def candidates(self, key: int) -> Dict[int, LateDIntent]:
        return self.conn.peek_intents(key)

    def commit(
        self, key: int, seq: int, winner_idx: int, now: float
    ) -> Optional[Tuple[LateDIntent, List[LateDIntent]]]:
        return self.conn.commit_choice(key, seq, winner_idx, now)

    def send_release(self, dest: LateDIntent, key: int, seq: int) -> None:
        self.conn.send_release(dest.d_key, key, seq)


class PdReaderPort(LateDReaderPort):
    """The ranks of a PD decode: ADMITs and RELEASEs come from the Mooncake
    decode thread (rank 0), an expiry aborts the room's receiver (ABORT to
    P), a release ends the leg with HTTP 409 and no ABORT."""

    def __init__(self, conn: LateDConnState):
        self.conn = conn

    @property
    def is_leader(self) -> bool:
        return self.conn.is_leader

    def drain_inbox(self) -> List[LateDControl]:
        return self.conn.drain_inbox()

    def expire(self, entry: DecodeRequest, key: int, reason: str) -> None:
        # The Failed poll ends the leg with 503 late_d_ttl.
        entry.late_d_err_type = LATE_D_TTL_ERR
        entry.kv_receiver.abort()  # ABORT to prefill; next pass drops it
        entry.kv_receiver.kv_mgr.record_failure(key, reason)

    refreshes_intents = True

    def send_intent(self, entry: DecodeRequest) -> None:
        receiver = entry.kv_receiver
        if receiver is None or receiver.conclude_state == KVPoll.Failed:
            return
        cand_idx = getattr(entry, "late_d_cand_idx", None)
        if cand_idx is None:
            receiver.send_late_d_intent()
        else:
            receiver.send_late_d_intent(cand_idx=cand_idx)

    def refresh_intent(self, entry: DecodeRequest) -> None:
        receiver = entry.kv_receiver
        if receiver is None or receiver.conclude_state == KVPoll.Failed:
            return
        receiver.send_late_d_intent(
            cand_idx=getattr(entry, "late_d_cand_idx", None), fail_room=False
        )

    def release(self, entry: DecodeRequest, key: int) -> None:
        # V6: set on every rank before the leg can end, so neither abort()
        # nor the waiting timeout sends ABORT to P for this room.
        if entry.kv_receiver is not None:
            entry.kv_receiver.released = True
        prepare_abort(
            entry.req,
            f"room {key} released",
            status_code=HTTPStatus.CONFLICT,
            err_type=LATE_D_RELEASED_ERR,
        )

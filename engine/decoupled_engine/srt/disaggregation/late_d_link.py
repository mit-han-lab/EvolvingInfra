"""Late D stage 2: the owner's control link to the router (LateDLink).

One daemon thread on the owner's attn-TP rank 0 (PD: the prefill;
symmetric: every worker). It owns

* a ZMQ ROUTER socket bound on ``--disaggregation-late-d-control-port``; the
  router connects one DEALER per owner;
* the PUSH sockets to the reader rank-0 endpoints, for ADMIT and RELEASE;
* an inproc PULL that wakes it for binder submissions and for sends queued by
  other threads.

It is the commit authority for the rooms the binder put in CHOICE_PENDING: on
the router's decision, at the choice timeout, or for a key the router does not
know, it calls ``binder.commit_choice`` under the binder lock. That commit
queues ADMIT and RELEASE on this thread, which sends them before it handles
anything else, with no second thread hop. While the router is absent (no
message for ``ROUTER_ABSENT_S``) the binder commits directly (stage-1 timing).

Messages: one ZMQ frame each, a
msgpack map with ``v`` (1) and ``t``; every message stays under 1 KB, so a
batch carries at most ``MAX_ITEMS`` items (8: a BIND_READY item is about 110
bytes, and the lead window offers at most 8 rooms per binder step).

  L1 hello      router -> owner  router_epoch, router_id, rules
  L2 hello_ack  owner -> router  router_epoch, engine_boot_ns, mode, kv_ip
                (PD) or owner_addr (symmetric)
  L3 ping/pong  router -> owner -> router  ts_ns
  L4 bind_ready owner -> router  router_epoch, p_seq, t_send_ns, items [key,
                bind_seq, eligible (bit i = candidate i), ready, input_len,
                default_idx, local_ok, local_load]
  L5 decision   router -> owner  router_epoch, items [key, bind_seq,
                winner_idx (254 LOCAL, 255 unknown key), rule, t_recv_ns,
                t_decide_ns]
  L6 final      owner -> router  router_epoch, items [key, bind_seq,
                admitted_idx, reason, t_admit_ns]

Watchdog: the thread beats ``heartbeat`` on every wake (at least once a
second). ``watchdog`` (binder, once a second) restarts a dead thread; the new
thread first sends the frames the dead one had queued but not sent.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import msgspec
import zmq

if TYPE_CHECKING:
    from decoupled_engine.srt.disaggregation.late_d import LateDMetrics, LateDPrefillBinder

logger = logging.getLogger(__name__)

LINK_VERSION = 1
WINNER_LOCAL = 254
WINNER_UNKNOWN = 255
MAX_ITEMS = 8
MAX_MESSAGE_BYTES = 1024
ROUTER_ABSENT_S = 3.0
SILENT_WARN_S = 3.0

REASON_DECIDED = "decided"
REASON_TIMEOUT = "timeout"
REASON_NO_ROUTER = "no_router"
REASON_UNKNOWN_KEY = "unknown_key"

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder()


@dataclass
class ChoiceItem:
    """One room in CHOICE_PENDING, as the binder hands it to the link."""

    key: int
    bind_seq: int
    eligible: int  # bit i = candidate i
    ready: int
    input_len: int
    default_idx: int
    fallback_idx: int  # the default if eligible, else the lowest eligible
    rid: str = ""
    local_ok: bool = False
    local_load: int = 0
    first_chunk_start_unix_ns: int = 0
    # Link thread only.
    deadline: float = 0.0
    t_send_ns: int = 0
    t_send: float = field(default=0.0, compare=False)

    def bind_ready_item(self) -> Dict[str, Any]:
        value = {
            "key": self.key,
            "bind_seq": self.bind_seq,
            "eligible": self.eligible,
            "ready": self.ready,
            "input_len": self.input_len,
            "default_idx": self.default_idx,
            "local_ok": self.local_ok,
            "local_load": self.local_load,
        }
        if self.first_chunk_start_unix_ns:
            value["first_chunk_start_unix_ns"] = self.first_chunk_start_unix_ns
        return value


def encode_message(t: str, **fields: Any) -> bytes:
    return _encoder.encode({"v": LINK_VERSION, "t": t, **fields})


def decode_message(frame: bytes) -> Dict[str, Any]:
    """A link message; ValueError for anything that is not a version-1 map."""
    try:
        msg = _decoder.decode(frame)
    except msgspec.DecodeError as e:
        raise ValueError(f"undecodable link message: {e}") from e
    if not isinstance(msg, dict) or msg.get("v") != LINK_VERSION:
        raise ValueError(f"not a version-{LINK_VERSION} link message")
    if not isinstance(msg.get("t"), str):
        raise ValueError("link message without a type")
    return msg


class LateDLink:
    """The owner side of the router link; see the module docstring."""

    def __init__(
        self,
        binder: LateDPrefillBinder,
        control_port: int,
        mode: str,
        kv_ip: str,
        choice_timeout_s: float,
        zmq_ctx: Optional[zmq.Context] = None,
        metrics: Optional[LateDMetrics] = None,
        bind_host: str = "*",
        owner_addr: Optional[str] = None,
    ):
        self._binder = binder
        self._requested_port = control_port
        self._bind_host = bind_host
        self.mode = mode
        self.kv_ip = kv_ip
        # Symmetric (L2): the owner's rank-0 pull-plane endpoint.
        self.owner_addr = owner_addr
        self.choice_timeout_s = choice_timeout_s
        self._ctx = zmq_ctx if zmq_ctx is not None else zmq.Context.instance()
        self.metrics = metrics
        self.engine_boot_ns = time.time_ns()
        self.port: Optional[int] = None  # the bound port (control_port 0: any)
        # Commands from other threads, and the inproc wake. Each thread
        # generation binds its own wake endpoint (inproc does not reconnect).
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._gen = 0
        self._local = threading.local()
        # Router state (written by the link thread; floats and references are
        # read by other threads without a lock).
        self._router_id: Optional[bytes] = None
        self._router_epoch: Optional[int] = None
        self._last_router_seen = -float("inf")
        # Link-thread state. The object outlives a dead thread: a restarted
        # thread keeps the pending choices and sends the unsent frames.
        self._pending: Dict[Tuple[int, int], ChoiceItem] = {}
        self._outbox: List[Tuple[str, bool, List[bytes]]] = []
        self._finals: List[Dict[str, Any]] = []
        self._p_seq = 0
        self._push: Dict[str, zmq.Socket] = {}
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self.heartbeat = time.monotonic()
        self.restarts = 0
        self._silent_warned = False
        self._present_reported: Optional[bool] = None
        self._stopping = False
        self.stats: Dict[str, int] = {
            "decided": 0,
            "timeouts": 0,
            "unknown_key": 0,
            "stale": 0,
            "bad_winner": 0,
            "bad_messages": 0,
        }

    # ------------------------------------------------------------------
    # Any thread
    # ------------------------------------------------------------------

    def _wake_endpoint(self, gen: int) -> str:
        return f"inproc://late-d-link-{id(self)}-{gen}"

    def start(self, timeout: float = 30.0) -> None:
        """Start the thread; returns once its sockets are bound."""
        self._ready.clear()
        self._start_error = None
        self._gen += 1
        self._thread = threading.Thread(
            target=self._run, args=(self._gen,), name="LateDLink", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("late-D link thread did not start")
        if self._start_error is not None:
            raise RuntimeError(
                f"late-D link could not bind port {self._requested_port}"
            ) from self._start_error

    def stop(self, timeout: float = 10.0) -> None:
        """End the thread (shutdown, tests); it closes its sockets."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        self._command(("stop",))
        thread.join(timeout)

    def router_present(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return (
            self._router_id is not None
            and now - self._last_router_seen < ROUTER_ABSENT_S
        )

    def submit(self, items: List[ChoiceItem]) -> None:
        """Binder: rooms that entered CHOICE_PENDING (one BIND_READY)."""
        self._command(("submit", items))

    def send(self, endpoint: str, is_ipv6: bool, frames: Any) -> None:
        """The sender interface of LateDConnState (and of the symmetric owner
        port): inline on the link thread (a commit), else queued for it
        (late-INTENT RELEASEs, resends, stage-1 ADMITs). ``frames``: the
        parts of one message, or one ``bytes`` frame (symmetric: an encoded
        pull-plane message), as LateDSender.send takes them."""
        if isinstance(frames, (bytes, bytearray)):
            frames = [bytes(frames)]
        if threading.current_thread() is self._thread:
            self._outbox.append((endpoint, is_ipv6, frames))
        else:
            self._command(("send", endpoint, is_ipv6, frames))

    def watchdog(self, now: Optional[float] = None) -> None:
        """Binder, at most once a second: restart a dead thread; warn about a
        silent live one."""
        now = time.monotonic() if now is None else now
        thread = self._thread
        if thread is None:
            return
        if not thread.is_alive():
            self.restarts += 1
            logger.error(
                "LATE_D link thread died; restarting it (restart %d, %d "
                "pending choices, %d unsent frames)",
                self.restarts,
                len(self._pending),
                len(self._outbox),
            )
            if self.metrics is not None:
                self.metrics.link_restarts.inc()
            try:
                self.start()
            except RuntimeError:
                # Retried at the next call; until then the binder commits
                # new rooms itself (the router looks absent).
                logger.exception("LATE_D link thread restart failed")
            return
        silent = now - self.heartbeat > SILENT_WARN_S
        if silent and not self._silent_warned:
            logger.warning(
                "LATE_D link thread alive but silent for %.1fs", now - self.heartbeat
            )
        self._silent_warned = silent

    def _command(self, cmd: Tuple) -> None:
        self._cmds.put(cmd)
        sock = getattr(self._local, "wake", None)
        if sock is None or self._local.gen != self._gen:
            if sock is not None:
                sock.close(linger=0)
            sock = self._ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._wake_endpoint(self._gen))
            self._local.wake, self._local.gen = sock, self._gen
        try:
            sock.send(b"", flags=zmq.NOBLOCK)
        except zmq.Again:
            pass  # the link is behind; the queued command waits for it

    # ------------------------------------------------------------------
    # Link thread
    # ------------------------------------------------------------------

    def _run(self, gen: int) -> None:
        try:
            router = self._ctx.socket(zmq.ROUTER)
            router.setsockopt(zmq.LINGER, 0)
            wake = self._ctx.socket(zmq.PULL)
            wake.setsockopt(zmq.LINGER, 0)
            if self.port is None:
                if self._requested_port == 0:
                    self.port = router.bind_to_random_port(f"tcp://{self._bind_host}")
                else:
                    router.bind(f"tcp://{self._bind_host}:{self._requested_port}")
                    self.port = self._requested_port
            else:  # a restart rebinds the same port
                router.bind(f"tcp://{self._bind_host}:{self.port}")
            wake.bind(self._wake_endpoint(gen))
        except BaseException as e:  # noqa: BLE001
            self._start_error = e
            self._ready.set()
            return
        self._router, self._wake = router, wake
        self._push = {}  # a restarted thread opens its own sockets
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        poller.register(wake, zmq.POLLIN)
        self._ready.set()
        self._stopping = False
        try:
            self._flush()  # frames a dead thread queued but did not send
            while not self._stopping:
                self._loop_once(poller)
        finally:
            for sock in (router, wake, *self._push.values()):
                sock.close(linger=0)

    def _loop_once(self, poller: zmq.Poller) -> None:
        self._drain_commands()  # also the ones queued before this thread ran
        if self._stopping:
            return
        now = time.monotonic()
        self.heartbeat = now
        events = dict(poller.poll(self._poll_timeout_ms(now)))
        if self._wake in events:
            while True:
                try:
                    self._wake.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
        self._drain_commands()
        # One clock reading for the due check and the expiry: an item expires
        # only if no decision for it was in the socket when it was due. A
        # stall (GIL) after the poll can pass a deadline while the decision
        # already waits in the socket, which the old poll result does not show.
        now = time.monotonic()
        if self._router in events or any(
            item.deadline <= now for item in self._pending.values()
        ):
            self._recv_router()
        self._expire(now)
        self._send_finals()
        present = self.router_present()
        if present != self._present_reported:
            self._present_reported = present
            if self.metrics is not None:
                self.metrics.router_present.set(int(present))
            logger.info("LATE_D link router_present=%d", int(present))

    def _recv_router(self) -> None:
        """Handle every router message now in the socket (no blocking)."""
        while True:
            try:
                ident, frame = self._router.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except ValueError:
                self.stats["bad_messages"] += 1
                continue
            self._on_router_message(ident, frame)

    def _drain_commands(self) -> None:
        new_items: List[ChoiceItem] = []
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                break
            if cmd[0] == "submit":
                new_items.extend(cmd[1])
            elif cmd[0] == "send":
                self._outbox.append(cmd[1:])
            else:  # "stop"
                self._stopping = True
        self._flush()
        if new_items:
            self._bind_ready(new_items)

    def _poll_timeout_ms(self, now: float) -> int:
        timeout = 1.0
        for item in self._pending.values():
            timeout = min(timeout, item.deadline - now)
        return max(0, int(timeout * 1000.0 + 0.999))

    def _flush(self) -> None:
        """Send the queued ADMIT / RELEASE frames, in order. A frame that
        cannot go out at once is dropped with a warning: the binder's resend
        and the timers cover it."""
        while self._outbox:
            endpoint, is_ipv6, frames = self._outbox[0]
            sock = self._push.get(endpoint)
            if sock is None:
                sock = self._ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                self._push[endpoint] = sock
            try:
                sock.send_multipart(frames, flags=zmq.NOBLOCK)
            except zmq.ZMQError as e:
                logger.warning("LATE_D control send to %s failed: %s", endpoint, e)
            self._outbox.pop(0)

    def _to_router(self, frame: bytes) -> None:
        if self._router_id is None:
            return
        try:
            self._router.send_multipart([self._router_id, frame], flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logger.warning("LATE_D link send to the router failed: %s", e)

    def _bind_ready(self, items: List[ChoiceItem]) -> None:
        now = time.monotonic()
        for item in items:
            item.deadline = now + self.choice_timeout_s
            self._pending[(item.key, item.bind_seq)] = item
            if self.metrics is not None:
                self.metrics.choice_ready.observe(item.ready)
        if self._router_id is None:
            return  # the timer commits them
        # The optional launch timestamp adds a map key/value. Four extended
        # items stay below the unchanged 1KiB protocol frame limit.
        limit = 4 if any(item.first_chunk_start_unix_ns for item in items) else MAX_ITEMS
        for i in range(0, len(items), limit):
            batch = items[i : i + limit]
            self._p_seq += 1
            t_send_ns = time.time_ns()
            for item in batch:
                item.t_send_ns, item.t_send = t_send_ns, now
            self._to_router(
                encode_message(
                    "bind_ready",
                    router_epoch=self._router_epoch,
                    p_seq=self._p_seq,
                    t_send_ns=t_send_ns,
                    items=[item.bind_ready_item() for item in batch],
                )
            )
            # The choice timeout is the router's round-trip budget: it starts
            # when the offer has left, not before a send that a stall delayed.
            deadline = time.monotonic() + self.choice_timeout_s
            for item in batch:
                item.deadline = deadline

    def _on_router_message(self, ident: bytes, frame: bytes) -> None:
        try:
            msg = decode_message(frame)
        except ValueError as e:
            self.stats["bad_messages"] += 1
            logger.warning("LATE_D link dropped a message: %s", e)
            return
        kind = msg["t"]
        if kind == "hello":
            self._router_id = ident
            self._router_epoch = msg.get("router_epoch")
            self._last_router_seen = time.monotonic()
            logger.info(
                "LATE_D link hello router_id=%s router_epoch=%s rules=%s",
                msg.get("router_id"),
                self._router_epoch,
                msg.get("rules"),
            )
            where = (
                {"owner_addr": self.owner_addr}
                if self.owner_addr is not None
                else {"kv_ip": self.kv_ip}
            )
            self._to_router(
                encode_message(
                    "hello_ack",
                    router_epoch=self._router_epoch,
                    engine_boot_ns=self.engine_boot_ns,
                    mode=self.mode,
                    **where,
                )
            )
            return
        if ident != self._router_id:
            self.stats["bad_messages"] += 1
            return  # not the router that said hello
        self._last_router_seen = time.monotonic()
        if kind == "ping":
            self._to_router(encode_message("pong", ts_ns=msg.get("ts_ns", 0)))
        elif kind == "decision":
            if msg.get("router_epoch") != self._router_epoch:
                self.stats["stale"] += 1
                return
            for d in msg.get("items") or ():
                self._on_decision(d)
        else:
            self.stats["bad_messages"] += 1

    def _on_decision(self, d: Any) -> None:
        try:
            key, bind_seq, winner = d["key"], d["bind_seq"], d["winner_idx"]
        except (KeyError, TypeError):
            self.stats["bad_messages"] += 1
            return
        item = self._pending.get((key, bind_seq))
        if item is None:
            self.stats["stale"] += 1  # late, repeated, or already timed out
            return
        if winner == WINNER_UNKNOWN:
            del self._pending[(key, bind_seq)]
            self.stats["unknown_key"] += 1
            self._commit(item, item.fallback_idx, REASON_UNKNOWN_KEY)
            return
        local = winner == WINNER_LOCAL and item.local_ok
        if not local and (
            not isinstance(winner, int)
            or not (0 <= winner < 64 and item.eligible >> winner & 1)
        ):
            # LOCAL (254) for an item sent with local_ok false, or a
            # candidate that was not eligible: ignored; the timer commits.
            self.stats["bad_winner"] += 1
            logger.warning(
                "LATE_D link ignored decision key=%s bind_seq=%s winner=%s "
                "eligible=%#x",
                key,
                bind_seq,
                winner,
                item.eligible,
            )
            return
        del self._pending[(key, bind_seq)]
        self.stats["decided"] += 1
        wait = time.monotonic() - item.t_send if item.t_send > 0 else 0.0
        if self.metrics is not None and item.t_send > 0:
            self.metrics.choice_latency.observe(wait)
        logger.info(
            "LATE_D decision rid=%s room=%s winner=%d default=%d ready=%d "
            "rule=%s t_us=%d",
            item.rid,
            key,
            winner,
            item.default_idx,
            item.ready,
            d.get("rule"),
            int(wait * 1e6),
        )
        self._commit(item, winner, REASON_DECIDED)

    def _expire(self, now: float) -> None:
        expired = [item for item in self._pending.values() if item.deadline <= now]
        for item in expired:
            del self._pending[(item.key, item.bind_seq)]
            self.stats["timeouts"] += 1
            if self.metrics is not None:
                self.metrics.choice_timeouts.inc()
            self._commit(item, item.fallback_idx, REASON_TIMEOUT)

    def _commit(self, item: ChoiceItem, winner: int, reason: str) -> None:
        admitted = self._binder.commit_choice(item.key, item.bind_seq, winner, reason)
        self._flush()  # ADMIT, then RELEASE, before anything else
        if admitted is None:
            return  # the room left the binder (aborted or failed)
        self._finals.append(
            {
                "key": item.key,
                "bind_seq": item.bind_seq,
                "admitted_idx": admitted,
                "reason": reason,
                "t_admit_ns": time.time_ns(),
            }
        )

    def _send_finals(self) -> None:
        while self._finals:
            batch, self._finals = self._finals[:MAX_ITEMS], self._finals[MAX_ITEMS:]
            self._to_router(
                encode_message("final", router_epoch=self._router_epoch, items=batch)
            )

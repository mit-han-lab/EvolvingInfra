from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from trie.phased_workload import current_slot


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClientLifecycleRequest:
    """Privacy-safe identity for one model request in a replay trace."""

    experiment_request_id: str
    trace_id_hash: str
    cycle_id: int
    event_index: int
    routing_key_hash: str | None

    @property
    def rid(self) -> str:
        return self.experiment_request_id

    def as_event_request(self) -> dict[str, str | int | None]:
        return {
            "experiment_request_id": self.experiment_request_id,
            "trace_id": None,
            "trace_id_hash": self.trace_id_hash,
            "cycle_id": self.cycle_id,
            "event_index": self.event_index,
            "rid": self.rid,
            "bootstrap_room": None,
            "routing_key_hash": self.routing_key_hash,
        }


@dataclass(frozen=True)
class _TimingPoint:
    ts_ns: int
    mono_ns: int

    @classmethod
    def now(cls) -> "_TimingPoint":
        return cls(ts_ns=time.time_ns(), mono_ns=time.monotonic_ns())


@dataclass
class RequestTiming:
    """Client-observed timing for one streamed model request.

    Wall-clock timestamps support joins with lifecycle events from other
    processes. Durations are derived only from the monotonic timestamps.
    ``generate_prepare_start`` is sampled when a Generate event begins, so its
    interval includes prompt materialization and any client admission wait.
    ``request_start`` is sampled alongside the existing streamed-request TTFT
    origin, after those pre-admission costs.
    ``sdk_call_start`` is the entry to ``completions.create``, before the SDK's
    synchronous request transformation; it is not a socket-submit timestamp.
    ``headers_or_first_byte`` records response headers with the current OpenAI
    SDK transport; the serialized observation label makes that boundary
    explicit. ``event_loop_lag_ns`` is the largest delay between an SDK call
    entry and the event loop servicing a ready ``call_soon`` callback. It can
    expose synchronous SDK preparation work, but is not a network-latency
    measurement.
    """

    generate_prepare_start: _TimingPoint
    request_start: _TimingPoint | None = None
    sdk_call_start: _TimingPoint | None = None
    headers_or_first_byte: _TimingPoint | None = None
    first_parsed_token: _TimingPoint | None = None
    sdk_call_attempt_count: int = 0
    event_loop_lag_ns: int | None = None

    @classmethod
    def start(cls) -> "RequestTiming":
        return cls(generate_prepare_start=_TimingPoint.now())

    def mark_request_start(self) -> None:
        # A migrated request keeps the start of its first attempt.
        if self.request_start is None:
            self.request_start = _TimingPoint.now()

    def mark_sdk_call_start(self) -> int:
        point = _TimingPoint.now()
        self.sdk_call_start = point
        self.sdk_call_attempt_count += 1
        return point.mono_ns

    def observe_event_loop_turn(self, scheduled_mono_ns: int) -> None:
        lag_ns = max(time.monotonic_ns() - scheduled_mono_ns, 0)
        if self.event_loop_lag_ns is None or lag_ns > self.event_loop_lag_ns:
            self.event_loop_lag_ns = lag_ns

    def mark_headers_or_first_byte(self) -> None:
        self.headers_or_first_byte = _TimingPoint.now()

    def mark_first_parsed_token(self) -> tuple[int, int]:
        if self.first_parsed_token is None:
            self.first_parsed_token = _TimingPoint.now()
        return self.first_parsed_token.ts_ns, self.first_parsed_token.mono_ns

    @staticmethod
    def _duration_ns(
        start: _TimingPoint | None,
        end: _TimingPoint | None,
    ) -> int | None:
        if start is None or end is None:
            return None
        return max(end.mono_ns - start.mono_ns, 0)

    def as_lifecycle_attrs(self) -> dict[str, object]:
        return {
            "generate_prepare_start_ts_ns": self.generate_prepare_start.ts_ns,
            "generate_prepare_start_mono_ns": (
                self.generate_prepare_start.mono_ns
            ),
            "request_start_ts_ns": (
                self.request_start.ts_ns if self.request_start is not None else None
            ),
            "request_start_mono_ns": (
                self.request_start.mono_ns
                if self.request_start is not None
                else None
            ),
            "sdk_call_start_ts_ns": (
                self.sdk_call_start.ts_ns
                if self.sdk_call_start is not None
                else None
            ),
            "sdk_call_start_mono_ns": (
                self.sdk_call_start.mono_ns
                if self.sdk_call_start is not None
                else None
            ),
            "headers_or_first_byte_ts_ns": (
                self.headers_or_first_byte.ts_ns
                if self.headers_or_first_byte is not None
                else None
            ),
            "headers_or_first_byte_mono_ns": (
                self.headers_or_first_byte.mono_ns
                if self.headers_or_first_byte is not None
                else None
            ),
            "first_parsed_token_ts_ns": (
                self.first_parsed_token.ts_ns
                if self.first_parsed_token is not None
                else None
            ),
            "first_parsed_token_mono_ns": (
                self.first_parsed_token.mono_ns
                if self.first_parsed_token is not None
                else None
            ),
            "sdk_call_attempt_count": self.sdk_call_attempt_count,
            "event_loop_lag_ns": self.event_loop_lag_ns,
            "event_loop_lag_scope": "sdk_call_start_to_next_loop_turn",
            "headers_or_first_byte_observation": "response_headers",
            "generate_prepare_to_request_start_ns": self._duration_ns(
                self.generate_prepare_start,
                self.request_start,
            ),
            "request_start_to_sdk_call_ns": self._duration_ns(
                self.request_start,
                self.sdk_call_start,
            ),
            "sdk_call_to_headers_or_first_byte_ns": self._duration_ns(
                self.sdk_call_start,
                self.headers_or_first_byte,
            ),
            "headers_or_first_byte_to_first_parsed_token_ns": self._duration_ns(
                self.headers_or_first_byte,
                self.first_parsed_token,
            ),
            "sdk_call_to_first_parsed_token_ns": self._duration_ns(
                self.sdk_call_start,
                self.first_parsed_token,
            ),
            "request_start_to_first_parsed_token_ns": self._duration_ns(
                self.request_start,
                self.first_parsed_token,
            ),
        }


@dataclass(frozen=True)
class ClientLifecycleWriter:
    """Append privacy-safe client lifecycle events as canonical OPX JSONL."""

    output_path: Path
    run_id: str
    cell_id: str
    arm: Literal["off", "on"]
    concurrency: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("lifecycle run_id must be non-empty")
        if not self.cell_id:
            raise ValueError("lifecycle cell_id must be non-empty")
        if self.concurrency < 1:
            raise ValueError("lifecycle concurrency must be at least 1")
        self.output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.output_path.parent, 0o700)

    @staticmethod
    def request_identity(
        *,
        trace_id: str,
        cycle_id: int,
        event_index: int,
        routing_key: str | None,
    ) -> ClientLifecycleRequest:
        if cycle_id < 0 or event_index < 0:
            raise ValueError("lifecycle cycle_id and event_index must be non-negative")
        slot = current_slot.get()
        if slot is not None:
            trace_id = json.dumps([slot.label, trace_id])
        trace_id_hash = _sha256(trace_id)
        request_digest = _sha256(f"{trace_id_hash}:{cycle_id}:{event_index}")
        return ClientLifecycleRequest(
            experiment_request_id=f"opx_{request_digest}",
            trace_id_hash=trace_id_hash,
            cycle_id=cycle_id,
            event_index=event_index,
            routing_key_hash=_sha256(routing_key) if routing_key else None,
        )

    def emit(
        self,
        event: str,
        request: ClientLifecycleRequest,
        *,
        attrs: dict[str, object] | None = None,
        ts_ns: int | None = None,
        mono_ns: int | None = None,
    ) -> None:
        slot = current_slot.get()
        row = {
            "schema_version": "opx.lifecycle-event.v1",
            "ts_ns": time.time_ns() if ts_ns is None else ts_ns,
            "mono_ns": time.monotonic_ns() if mono_ns is None else mono_ns,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "arm": self.arm,
            "concurrency": self.concurrency,
            "source": {
                "component": "client",
                "worker_id": "strict-replay",
                "dp_rank": None,
                "engine_epoch": None,
            },
            "request": request.as_event_request(),
            "event": event,
            "attrs": {**(slot.attrs() if slot is not None else {}), **(attrs or {})},
        }
        payload = (
            json.dumps(row, separators=(",", ":"), sort_keys=True).encode("utf-8")
            + b"\n"
        )
        fd = os.open(
            self.output_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
        finally:
            os.close(fd)

# Adapted from SGLang python/sglang/srt/disaggregation/kv_events.py (commit 297e54fd10), modified for evolving-infra.
"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
KV caching events
"""

import atexit
import enum
import hashlib
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from itertools import count
from queue import Queue
from typing import Any, Callable, Literal, Optional, Union

import msgspec
import zmq
from pydantic import BaseModel

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.runtime_context import get_parallel
from decoupled_engine.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


def select_kv_publisher_dp_rank(
    attn_dp_size: int, attn_dp_rank: int, dp_rank: Optional[int]
) -> int:
    """Index used to offset this scheduler's KV-event publisher port.

    Each independent KV cache must publish on its own port so a consumer can
    subscribe per replica. There are always ``dp_size`` such publishers; which
    rank distinguishes them depends on the parallelism mode:

    - DP-attention (``attn_dp_size > 1``): each attention-DP rank owns a KV
      cache shard, so distinguish by ``attn_dp_rank``.
    - Pure DP (``attn_dp_size == 1``): every worker has ``attn_dp_rank == 0``,
      so distinguish by ``dp_rank`` (the data-parallel replica index).

    Both span ``0..dp_size-1``, matching the ``dp_size`` advertised in
    ``/server_info`` and the per-rank ports the router subscribes to.
    """
    if attn_dp_size > 1:
        return attn_dp_rank
    return dp_rank or 0


def is_kv_publisher_rank(kv_events_config: Optional[str]) -> bool:
    """Whether this scheduler owns a KV-event publisher slot: one per
    independent KV cache (pp/attn-TP/attn-CP rank 0). Shared by
    `SchedulerKvEventsPublisher` and `SchedulerLoadPublisher`, which must
    gate identically or their /server_info-derived ports disagree.
    """
    parallel = get_parallel()
    return bool(
        kv_events_config
        and parallel.pp_rank == 0
        and parallel.attn_tp_rank == 0
        and parallel.attn_cp_rank == 0
    )


# Advertised as `load_topic` in /server_info; the load socket carries only
# load, so subscribers can subscribe-all.
LOAD_TOPIC = "load"

# Hosts a PUB socket binds rather than connects to. Matched on the parsed
# host, not a substring: "::" appears inside every IPv6 address, so a
# substring test would wrongly call a concrete remote host bindable.
_BIND_WILDCARD_HOSTS = frozenset({"*", "0.0.0.0", "::"})


def parse_tcp_port(endpoint: Optional[str]) -> Optional[int]:
    """Legal port of a tcp:// endpoint regardless of host, or None.

    Host-agnostic: answers "which ports does something else occupy" for the
    collision checks (the replay ROUTER binds any host spelling).
    """
    if not endpoint or not endpoint.startswith("tcp://"):
        return None
    try:
        port = NetworkAddress.parse(endpoint[len("tcp://") :]).port
    except ValueError:
        return None
    return port if 0 < port <= 65535 else None


def parse_advertisable_tcp(endpoint: Optional[str]) -> Optional[tuple[str, int]]:
    """``(host, port)`` of a tcp:// endpoint fit for /server_info, else None.

    Any host (KV events work connect-style); IPv6 re-bracketed so consumers
    can splice ``tcp://{host}:{port}``. Bare unbracketed IPv6 is rejected —
    same parse as the resolver, so descriptor and bind agree.
    """
    if not endpoint or not endpoint.startswith("tcp://"):
        return None
    try:
        addr = NetworkAddress.parse(endpoint[len("tcp://") :])
    except ValueError:
        return None
    if not addr.host or not (0 < addr.port <= 65535):
        return None
    host = f"[{addr.host}]" if addr.is_ipv6 else addr.host
    return host, addr.port


def parse_bindable_tcp(endpoint: Optional[str]) -> Optional[tuple[str, int]]:
    """``(host, port)`` if a PUB socket can BIND this tcp:// endpoint, else
    None. A concrete host is connect-style here, so a load PUB there would
    reach nobody while reporting no error."""
    if not endpoint or not endpoint.startswith("tcp://"):
        return None
    try:
        addr = NetworkAddress.parse(endpoint[len("tcp://") :])
    except ValueError:
        return None
    if addr.host not in _BIND_WILDCARD_HOSTS or not (0 < addr.port <= 65535):
        return None
    return addr.host, addr.port


def resolve_load_pub_range(
    *,
    kv_endpoint: Optional[str],
    replay_endpoint: Optional[str],
    dp_size: int,
    load_publish_endpoint: Optional[str] = None,
) -> tuple[Optional[tuple[str, int]], Optional[str]]:
    """``((host, base), reason)`` for the load PUB range — exactly one is None.

    Rank ``r`` binds ``base + r`` and ``/server_info`` advertises ``base``.
    Single source of truth for both the bind (`SchedulerLoadPublisher`) and
    the advertisement (`describe_kv_events_publisher`), so they cannot drift.

    Opt-in via ``--load-publish-endpoint``: unset (or ``off``) disables it, so
    an upgrade never reserves a port a co-hosted neighbor's KV publisher would
    bind. ``auto`` packs the range after the KV-event range, bumping past an
    overlapping replay ROUTER range (with the conventional replay = kv + 1,
    always); an explicit ``tcp://`` address sets it outright.

    ``reason`` is set when an operator would want to know why publishing is
    off (unusable endpoint, collision, u16 overflow) and None when the decline
    is unremarkable (feature off). Callers log it once; /server_info calls
    this per request, so it must not log here.

    Two inherited limits, both from the KV-event discovery structure: with
    ``page_size`` <= 0 `describe_kv_events_publisher` suppresses the whole
    block, so the range binds unadvertised; and with DP-attention across
    ``nnodes`` > 1 the single advertised base is paired with one worker-URL
    host, so ranks on other nodes are unreachable at that host.
    """
    # Opt-in: off unless the operator sets `auto` (derive) or an address, so an
    # upgrade never claims a port a co-hosted neighbor's KV publisher binds.
    mode = (load_publish_endpoint or "").strip()
    if dp_size < 1 or not mode or mode.lower() == "off":
        return None, None

    if mode.lower() == "auto":
        resolved = parse_bindable_tcp(kv_endpoint)
        if resolved is None:
            why = (
                "--kv-events-config is not set"
                if kv_endpoint is None
                else f"{kv_endpoint!r} is not one"
            )
            return None, (
                f"--load-publish-endpoint=auto needs a bindable wildcard-host "
                f"tcp:// --kv-events-config endpoint to pack after; {why}"
            )
        host, kv_base = resolved
        base = kv_base + dp_size
        replay_base = parse_tcp_port(replay_endpoint)
        if (
            replay_base is not None
            and base < replay_base + dp_size
            and replay_base < base + dp_size
        ):
            # Overlap implies kv < replay < kv + 2*dp_size, so packing after
            # the replay range also clears the KV range.
            base = replay_base + dp_size
    else:
        # Explicit address. Discovery still needs the kv_events block, absent
        # for a non-tcp KV endpoint — so the range would bind but never
        # advertise.
        if parse_tcp_port(kv_endpoint) is None:
            absent = (
                "without --kv-events-config"
                if kv_endpoint is None
                else f"for endpoint {kv_endpoint!r}"
            )
            return None, (
                f"--load-publish-endpoint={mode!r} needs a routable tcp:// "
                f"--kv-events-config endpoint: routers discover the load range "
                f"through /server_info's kv_events block, absent {absent}, so "
                f"the socket would be bound but never advertised"
            )
        resolved = parse_bindable_tcp(mode)
        if resolved is None:
            return None, (
                f"--load-publish-endpoint={mode!r} is not a bindable tcp:// "
                f"address (a concrete host would be connected to, not bound)"
            )
        host, base = resolved
        for port in (parse_tcp_port(kv_endpoint), parse_tcp_port(replay_endpoint)):
            if port is not None and base < port + dp_size and port < base + dp_size:
                return None, (
                    f"--load-publish-endpoint range [{base}, {base + dp_size}) "
                    f"overlaps the kv-events range [{port}, {port + dp_size})"
                )
    if base + dp_size - 1 > 65535:
        return None, f"load port range from {base} would run past the u16 ceiling"
    return (host, base), None


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    attn_dp_rank: Optional[int] = None


class KVCacheEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events.

    Events are tagged msgpack maps: ``type`` carries the class name and every
    other key is a field name. Optional fields left at ``None`` are omitted, so
    adding an optional field never changes the shape an older consumer sees.
    This is the same encoding vLLM uses for its ``KVCacheEvent``, so a consumer
    such as Dynamo decodes both engines with one code path.

    ``EventBatch`` stays a positional array ``[ts, events, attn_dp_rank]``.
    """


class StorageMedium(str, enum.Enum):
    """Storage tier for KV cache events."""

    GPU = "GPU"  # L1: device HBM
    CPU = "CPU_PINNED"  # L2: host pinned memory
    DISK = "DISK"  # L3: SSD / NVMe
    EXTERNAL = "EXTERNAL"  # L4: shared / remote pool (e.g. Mooncake)


class OffloadedState(msgspec.Struct):
    """Decode-side offload progress for one request, keyed by Req in the manager."""

    # Decode-incremental length already submitted for D2H offload.
    inc_len: int = 0
    # Tail of the page hash chain, extended as each offloaded chunk is backed up.
    last_hash: Optional[str] = None


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None
    # Salt of the request that stored these blocks. Block hashes are already
    # namespaced by it; consumers index the emitted hashes rather than
    # recompute them.
    cache_salt: Optional[str] = None
    # Session that triggered this store. Attribution only: the blocks may be
    # shared with other sessions, and the hash does not depend on it.
    session_id: Optional[str] = None
    # Public adapter name for router lookup. Published hashes are opaque IDs
    # scoped to the internal cache namespace, including the adapter load.
    lora_name: Optional[str] = None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None


class AllBlocksCleared(KVCacheEvent):
    pass


# The dynamo format publishes namespaced hashes as block hashes. If you change
# this tag, all of these hashes change. unified_tree_core.rs has a Rust copy.
_NAMESPACE_SEED_TAG = b"sglang-kv-event-namespace-v1"


def kv_event_namespace_seed(
    *, extra_key: Optional[str], cache_salt: Optional[str]
) -> Optional[bytes]:
    """Return the seed for namespaced_block_hash, or None for no namespace."""
    if extra_key is None and cache_salt is None:
        return None
    digest = hashlib.sha256(_NAMESPACE_SEED_TAG)
    # The presence byte and the length prefix make ("a", "bc") and ("ab", "c") differ.
    for part in (extra_key, cache_salt):
        if part is None:
            digest.update(b"\x00")
            continue
        encoded = part.encode("utf-8")
        digest.update(b"\x01" + len(encoded).to_bytes(8, "little") + encoded)
    return digest.digest()


def namespaced_block_hash(block_hash: int, *, namespace_seed: Optional[bytes]) -> int:
    """Return the namespaced hash of a published block hash.

    If namespace_seed is None, the result is ``block_hash``. If not, the result is
    the first 8 bytes of SHA-256(seed, block_hash as 8 big-endian bytes), as int64.
    """
    if namespace_seed is None:
        return block_hash
    digest = hashlib.sha256(
        namespace_seed + block_hash.to_bytes(8, "big", signed=True)
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


# ---- EI wire format (SGLANG_KV_EVENTS_WIRE_FORMAT) ----
#
# EI's consumers (the Rust router's kv_events/wire.rs, the Morphosis router
# and the KV indexer) decode only EI's legacy positional layout: each event
# is an array ``[tag, fields...]``, BlockStored is ``[block_hashes,
# parent_block_hash, token_ids, block_size, lora_id, medium]``, and a cache
# salt rides only on BlockStoredWithMetadata (tag "BlockStored", trailing
# ``metadata``). The publisher converts upstream's event structs (above) to
# these at encode time: "ei_array" (the default) sends this layout and
# "upstream_map" sends upstream's tagged maps. The legacy structs keep EI's
# msgspec tags; only their Python names are private.
KV_EVENTS_WIRE_EI_ARRAY = "ei_array"
KV_EVENTS_WIRE_UPSTREAM_MAP = "upstream_map"


class _LegacyKVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class of EI's legacy (positional) KV cache events."""


class _LegacyBlockStoredMetadata(msgspec.Struct, omit_defaults=True, gc=False):
    """Typed request metadata attached to a stored KV block."""

    cache_salt: str


class _LegacyBlockStored(_LegacyKVCacheEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None


class _LegacyBlockStoredWithMetadata(
    _LegacyBlockStored, tag="BlockStored", kw_only=True
):
    """BlockStored wire extension used only when typed metadata is present.

    A separate struct keeps unsalted events at their legacy array length; an
    optional field on BlockStored would still serialize a trailing null.
    """

    metadata: _LegacyBlockStoredMetadata


class _LegacyBlockRemoved(_LegacyKVCacheEvent, tag="BlockRemoved"):
    block_hashes: list[int]
    medium: Optional[str] = None


class _LegacyAllBlocksCleared(_LegacyKVCacheEvent, tag="AllBlocksCleared"):
    pass


def kv_events_wire_format() -> str:
    """SGLANG_KV_EVENTS_WIRE_FORMAT: "ei_array" (default) or "upstream_map"."""
    fmt = (envs.SGLANG_KV_EVENTS_WIRE_FORMAT.get() or KV_EVENTS_WIRE_EI_ARRAY).strip()
    fmt = fmt.lower()
    if fmt not in (KV_EVENTS_WIRE_EI_ARRAY, KV_EVENTS_WIRE_UPSTREAM_MAP):
        raise ValueError(
            f"SGLANG_KV_EVENTS_WIRE_FORMAT={fmt!r}: expected "
            f"{KV_EVENTS_WIRE_EI_ARRAY!r} or {KV_EVENTS_WIRE_UPSTREAM_MAP!r}"
        )
    return fmt


def to_legacy_event(event: Any) -> Any:
    """``event`` as EI's legacy wire struct. A salted BlockStored becomes
    BlockStoredWithMetadata; session_id and lora_name are not sent. Events of
    other types pass through unchanged."""
    if isinstance(event, BlockStored):
        fields = dict(
            block_hashes=event.block_hashes,
            parent_block_hash=event.parent_block_hash,
            token_ids=event.token_ids,
            block_size=event.block_size,
            lora_id=event.lora_id,
            medium=event.medium,
        )
        if event.cache_salt is not None:
            return _LegacyBlockStoredWithMetadata(
                **fields,
                metadata=_LegacyBlockStoredMetadata(cache_salt=event.cache_salt),
            )
        return _LegacyBlockStored(**fields)
    if isinstance(event, BlockRemoved):
        return _LegacyBlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
    if isinstance(event, AllBlocksCleared):
        return _LegacyAllBlocksCleared()
    return event


def to_wire_batch(batch: EventBatch, wire_format: str) -> EventBatch:
    """The batch whose encoding goes on the wire: ``batch`` itself for
    "upstream_map", else a positional EventBatch of EI's legacy events (the
    batch itself is ``[ts, events, attn_dp_rank]`` in both formats)."""
    if wire_format != KV_EVENTS_WIRE_EI_ARRAY:
        return batch
    return EventBatch(
        ts=batch.ts,
        events=[to_legacy_event(event) for event in batch.events],
        attn_dp_rank=batch.attn_dp_rank,
    )


# Event types the batched-publication caps can cut at block boundaries, in
# both wire formats.
_STORED_EVENT_TYPES = (BlockStored, _LegacyBlockStored)
_REMOVED_EVENT_TYPES = (BlockRemoved, _LegacyBlockRemoved)


# ---- Message caps of batched publication (SGLANG_KV_EVENTS_BATCH_PUBLISH) ----

# Floor of SGLANG_KV_EVENTS_BATCH_MAX_BYTES: a single block of a large page
# still fits in one message.
MIN_BATCH_MAX_BYTES = 4096
# Bytes the events-array header of a message can grow by over the header of
# an empty batch (msgpack fixarray, 1 byte, up to array32, 5 bytes).
_ARRAY_HEADER_SLACK = 4


def kv_events_batch_caps() -> tuple[int, int]:
    """(max encoded message bytes, max events per message) of batched
    publication; 0 means no cap. (0, 0) unless SGLANG_KV_EVENTS_BATCH_PUBLISH
    is on."""
    if not envs.SGLANG_KV_EVENTS_BATCH_PUBLISH.get():
        return 0, 0
    max_bytes = max(0, envs.SGLANG_KV_EVENTS_BATCH_MAX_BYTES.get())
    if 0 < max_bytes < MIN_BATCH_MAX_BYTES:
        logger.warning(
            "SGLANG_KV_EVENTS_BATCH_MAX_BYTES=%d is below %d; using %d",
            max_bytes,
            MIN_BATCH_MAX_BYTES,
            MIN_BATCH_MAX_BYTES,
        )
        max_bytes = MIN_BATCH_MAX_BYTES
    return max_bytes, max(0, envs.SGLANG_KV_EVENTS_BATCH_MAX_EVENTS.get())


def _with_fields(event: Any, **changes: Any) -> Any:
    """A copy of the msgspec event ``event`` (same type, so the same tag and
    metadata) with ``changes``."""
    fields = {name: getattr(event, name) for name in event.__struct_fields__}
    fields.update(changes)
    return type(event)(**fields)


def split_event_by_blocks(
    event: Any, budget: int, encode: Callable[[Any], bytes]
) -> list[tuple[Any, int]]:
    """``event`` as consecutive events of its kind that each encode to at most
    ``budget`` bytes, cut at block boundaries: [(event, encoded size)].

    A BlockStored piece takes its blocks' token ids and links to the last
    block of the piece before it (the first keeps the parent), which is the
    form the recorder queues before it coalesces pages: the same blocks, in
    the same order. BlockRemoved pieces keep the medium. Other events, events
    that fit, and events that cannot be cut (one block, or token ids that are
    not block_size per block) come back whole."""
    size = len(encode(event))
    if size <= budget or not isinstance(
        event, _STORED_EVENT_TYPES + _REMOVED_EVENT_TYPES
    ):
        return [(event, size)]
    hashes = event.block_hashes
    n_blocks = len(hashes)
    stored = isinstance(event, _STORED_EVENT_TYPES)
    block_size = event.block_size if stored else 0
    if n_blocks < 2 or (stored and len(event.token_ids) != n_blocks * block_size):
        return [(event, size)]
    n_parts = min(n_blocks, max(2, -(-size // budget)))
    while True:
        per = -(-n_blocks // n_parts)
        parts = []
        parent = event.parent_block_hash if stored else None
        for start in range(0, n_blocks, per):
            piece_hashes = hashes[start : start + per]
            if stored:
                piece = _with_fields(
                    event,
                    block_hashes=piece_hashes,
                    parent_block_hash=parent,
                    token_ids=event.token_ids[
                        start * block_size : (start + len(piece_hashes)) * block_size
                    ],
                )
                parent = piece_hashes[-1]
            else:
                piece = _with_fields(event, block_hashes=piece_hashes)
            parts.append((piece, len(encode(piece))))
        if per == 1 or all(piece_size <= budget for _, piece_size in parts):
            return parts
        n_parts = min(n_blocks, n_parts + max(1, n_parts // 4))


def cap_event_batch(
    batch: EventBatch,
    max_bytes: int,
    max_events: int,
    encode: Callable[[Any], bytes],
) -> tuple[list[bytes], int]:
    """The encoded messages that carry ``batch`` within the caps, and the
    number of its events that were split (SGLANG_KV_EVENTS_BATCH_PUBLISH).

    A batch within the caps is one message, encoded as before. Otherwise its
    events go out in order, as many per message as fit, every message with
    the batch's ts and attn_dp_rank. Each event keeps its bytes, except one
    that alone exceeds the byte budget: it is split at block boundaries
    (``split_event_by_blocks``). A message exceeds ``max_bytes`` only when a
    single block does."""
    payload = encode(batch)
    events = batch.events
    if (max_bytes <= 0 or len(payload) <= max_bytes) and (
        max_events <= 0 or len(events) <= max_events
    ):
        return [payload], 0

    def make(group: list) -> EventBatch:
        return type(batch)(ts=batch.ts, events=group, attn_dp_rank=batch.attn_dp_rank)

    budget = None
    if max_bytes > 0:
        budget = max(1, max_bytes - len(encode(make([]))) - _ARRAY_HEADER_SLACK)
    pieces: list[tuple[Any, int]] = []
    n_split = 0
    for event in events:
        if budget is None:
            pieces.append((event, 0))
            continue
        parts = split_event_by_blocks(event, budget, encode)
        if len(parts) > 1:
            n_split += 1
        pieces.extend(parts)
    messages: list[bytes] = []
    group: list = []
    group_bytes = 0
    for event, size in pieces:
        if group and (
            (budget is not None and group_bytes + size > budget)
            or (max_events > 0 and len(group) >= max_events)
        ):
            messages.append(encode(make(group)))
            group, group_bytes = [], 0
        group.append(event)
        group_bytes += size
    if group:
        messages.append(encode(make(group)))
    return messages, n_split


class _WireStats:
    """ZMQ-thread counters (SGLANG_KV_EVENTS_PUBLISH_STATS), read by the
    scheduler's 60-s line through ``ZmqEventPublisher.take_wire_stats``."""

    KEYS = ("batches", "messages", "bytes", "split_batches", "split_events", "over_cap")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._window = dict.fromkeys(self.KEYS, 0)
        self._window_max_bytes = 0
        self._window_s = 0.0
        self._totals = dict.fromkeys(self.KEYS, 0)
        self._max_bytes = 0

    def add(
        self, payloads: list[bytes], n_split: int, cap: int, seconds: float
    ) -> None:
        sizes = [len(p) for p in payloads]
        largest = max(sizes) if sizes else 0
        delta = {
            "batches": 1,
            "messages": len(payloads),
            "bytes": sum(sizes),
            "split_batches": int(len(payloads) > 1),
            "split_events": n_split,
            "over_cap": sum(s > cap for s in sizes) if cap > 0 else 0,
        }
        with self._lock:
            for key, value in delta.items():
                self._window[key] += value
                self._totals[key] += value
            self._window_max_bytes = max(self._window_max_bytes, largest)
            self._max_bytes = max(self._max_bytes, largest)
            self._window_s += seconds

    def take(self) -> dict:
        """The window's counters (then reset) and the totals."""
        with self._lock:
            out = dict(self._window)
            out["max_bytes"] = self._window_max_bytes
            out["encode_send_ms"] = self._window_s * 1e3
            out["total_messages"] = self._totals["messages"]
            out["total_bytes"] = self._totals["bytes"]
            out["total_split_events"] = self._totals["split_events"]
            out["total_over_cap"] = self._totals["over_cap"]
            out["total_max_bytes"] = self._max_bytes
            self._window = dict.fromkeys(self.KEYS, 0)
            self._window_max_bytes = 0
            self._window_s = 0.0
        return out


class EventPublisher(ABC):
    """
    Lightweight publisher for EventBatch batches with
    support for DP attention.

    In DP attention - each rank has its own Scheduler and
    KV cache instance in order to avoid duplicate events
    and ensure proper event attribution. In our implementation

    - Each DP rank has its own EventPublisher
    - Publishers annotate events with the dp rank
    - This allows consumers to distinguish events from different DP ranks
    """

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""

    def describe_local_source(self, block_size: int) -> Optional[dict[str, Any]]:
        """Describe a source accessible to a separate local process, if any."""
        return None


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return

    def shutdown(self) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

    Spawns a separate thread to handle publishing from a queue.

    Parameters
    ----------
    endpoint:
        PUB address. Use ``tcp://*:5557`` to bind or ``tcp://host:5557`` to
        connect, unless overridden by ``bind``.
    replay_endpoint:
        Optional ROUTER address for replay requests. When given, subscribers can
        request missed batches by sending the starting sequence number as an
        8-byte big-endian integer.
    buffer_steps:
        Number of past batches to keep for replay.
    hwm:
        ZeroMQ high-water-mark for PUB socket.
    max_queue_size:
        Maximum number of events to buffer in memory.
    topic:
        Topic to publish events to.
    bind:
        Whether to bind the PUB socket. When unset, preserve the endpoint-based
        heuristic: bind wildcard, IPC and inproc addresses; connect otherwise.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        attn_dp_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
        bind: Optional[bool] = None,
    ) -> None:
        # Storage
        self._event_queue = Queue[Optional[EventBatch]](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._local_pub_endpoint: Optional[str] = None
        self._local_replay_endpoint: Optional[str] = None
        self._replay: Optional[zmq.Socket] = None
        self._dp_rank = attn_dp_rank
        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._bind = bind
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        self._hwm = hwm
        self._socket_setup()

        # Payload
        self._seq_gen = count()
        self._topic_bytes = topic.encode("utf-8")
        # Message caps (SGLANG_KV_EVENTS_BATCH_PUBLISH) and wire counters
        # (SGLANG_KV_EVENTS_PUBLISH_STATS). With both off the thread encodes
        # and sends each batch as one message, as before.
        self._cap_bytes, self._cap_events = kv_events_batch_caps()
        self._wire_stats = (
            _WireStats() if envs.SGLANG_KV_EVENTS_PUBLISH_STATS.get() else None
        )
        self._capped_path = bool(
            self._cap_bytes or self._cap_events or self._wire_stats is not None
        )
        # SGLANG_KV_EVENTS_WIRE_FORMAT: what goes on the wire (and into the
        # replay buffer, which stores the bytes sent).
        self._wire_format = kv_events_wire_format()

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="zmq-publisher"
        )
        self._thread.start()

        atexit.register(self.shutdown)

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.attn_dp_rank is None:
            events.attn_dp_rank = self._dp_rank
        self._event_queue.put(events)

    def describe_local_source(self, block_size: int) -> Optional[dict[str, Any]]:
        """Report the bound socket, not a port reconstructed from global DP size.

        Connect-style publishers and inproc sockets have no subscribable local
        endpoint. They keep working as before, but are not advertised to external
        processes. Wildcard binds are reachable on loopback from this node.
        """

        def local_endpoint(endpoint: Optional[str]) -> Optional[str]:
            if endpoint is None:
                return None
            if endpoint.startswith("ipc://"):
                return endpoint
            if parse_advertisable_tcp(endpoint) is None:
                return None
            address = NetworkAddress.parse(endpoint[len("tcp://") :])
            host = address.host
            if host in ("*", "0.0.0.0"):
                host = "127.0.0.1"
            elif host == "::":
                host = "::1"
            return NetworkAddress(host, address.port).to_tcp()

        endpoint = local_endpoint(self._local_pub_endpoint)
        if endpoint is None:
            return None
        source = {
            "dp_rank": self._dp_rank,
            "endpoint": endpoint,
            "topic": self._topic_bytes.decode("utf-8"),
            "block_size": block_size,
        }
        replay_endpoint = local_endpoint(self._local_replay_endpoint)
        if replay_endpoint is not None:
            source["replay_endpoint"] = replay_endpoint
        return source

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        self._running = False
        self._event_queue.put_nowait(None)

        start = time.time()
        pending_items = True
        while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
            pending_items = not self._event_queue.empty()
            if pending_items:
                time.sleep(0.1)

        if pending_items:
            logger.warning(
                "Warning: Queue still has %s items after %s seconds timeout",
                self._event_queue.qsize(),
                self.SHUTDOWN_TIMEOUT,
            )

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        # Clean up ZMQ resources
        try:
            if self._pub is not None:
                self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            pass  # Do not terminate context; other sockets may use it

    def _socket_setup(self) -> None:
        """Initialize sockets
        https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
        """
        if self._pub is None:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.set_hwm(self._hwm)
            # Default heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention.
            # ``0.0.0.0`` is the IPv4 bind-all wildcard alongside ``*``
            # and ``::``; ``/server_info`` advertises it as a wildcard,
            # so the publisher must bind it for the advertised endpoint
            # to actually be listening.
            should_bind = self._bind
            if should_bind is None:
                should_bind = (
                    "*" in self._endpoint
                    or "::" in self._endpoint
                    or "0.0.0.0" in self._endpoint
                    or self._endpoint.startswith("ipc://")
                    or self._endpoint.startswith("inproc://")
                )
            if should_bind:
                logger.debug(
                    f"ZmqEventPublisher socket publisher_endpoint bind to {self._endpoint}"
                )
                self._pub.bind(self._endpoint)
                self._local_pub_endpoint = self._bound_endpoint(self._pub)
            else:
                self._pub.connect(self._endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            logger.debug(
                f"ZmqEventPublisher socket replay_endpoint bind to {self._replay_endpoint}"
            )
            self._replay.bind(self._replay_endpoint)
            self._local_replay_endpoint = self._bound_endpoint(self._replay)

    @staticmethod
    def _bound_endpoint(socket: zmq.Socket) -> str:
        endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        if endpoint.startswith("ipc://"):
            path = endpoint[len("ipc://") :]
            # Capture the publisher's working directory at bind time so a
            # separate process can use the address. Abstract IPC names are
            # independent of the filesystem and must remain unchanged.
            if not path.startswith("@"):
                endpoint = f"ipc://{os.path.abspath(path)}"
        return endpoint

    def _publisher_thread(self) -> None:
        """Background thread that processes the event queue."""
        self._pack = msgspec.msgpack.Encoder()

        assert self._pub is not None  # narrows type for mypy

        while self._running or self._event_queue.qsize() > 0:
            # --- replay (non-critical) ---------------------------------
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.exception("Error in replay: %s", e)

            # --- main queue (critical) ---------------------------------
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:
                    break  # Sentinel received, exit thread
            except queue.Empty:
                continue

            try:
                if self._capped_path:
                    self._send_capped(event)
                    continue

                seq = next(self._seq_gen)

                payload = self._pack.encode(to_wire_batch(event, self._wire_format))
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))

                self._buffer.append((seq, payload))
                self._event_queue.task_done()

            except Exception as e:
                # Publishing failed;  back-off a bit to avoid a tight error loop
                logger.exception("Error in publisher thread: %s", e)
                time.sleep(0.1)

    def _send_capped(self, event: EventBatch) -> None:
        """Send ``event`` as the messages ``cap_event_batch`` gives (one when
        it fits), each with its own sequence number and replay entry, and
        count them (SGLANG_KV_EVENTS_PUBLISH_STATS)."""
        t0 = time.perf_counter()
        # Caps and splits measure the bytes actually sent: the wire batch.
        payloads, n_split = cap_event_batch(
            to_wire_batch(event, self._wire_format),
            self._cap_bytes,
            self._cap_events,
            self._pack.encode,
        )
        for payload in payloads:
            seq = next(self._seq_gen)
            self._pub.send_multipart(
                (self._topic_bytes, seq.to_bytes(8, "big"), payload)
            )
            self._buffer.append((seq, payload))
        self._event_queue.task_done()
        if self._wire_stats is not None:
            self._wire_stats.add(
                payloads, n_split, self._cap_bytes, time.perf_counter() - t0
            )

    def take_wire_stats(self) -> Optional[dict]:
        """The ZMQ thread's counters since the previous call, with totals
        (SGLANG_KV_EVENTS_PUBLISH_STATS); None when off."""
        return None if self._wire_stats is None else self._wire_stats.take()

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        assert self._replay is not None  # narrows type for mypy

        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning("Invalid replay request: %s", frame)
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")
        # The EI wire keeps the replay framing EI's consumers read (the router's
        # KV-event bridge): (seq, payload) after the delimiter, no topic frame.
        legacy = self._wire_format == KV_EVENTS_WIRE_EI_ARRAY

        for seq, buf in self._buffer:
            if seq >= start_seq:
                if legacy:
                    self._replay.send_multipart(
                        (client_id, b"", seq.to_bytes(8, "big"), buf)
                    )
                    continue
                # Subscribers receive the same topic, sequence, and payload as live events.
                self._replay.send_multipart(
                    (client_id, b"", self._topic_bytes, seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        if legacy:
            self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))
        else:
            self._replay.send_multipart((client_id, b"", b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(
        endpoint: Optional[str], data_parallel_rank: int
    ) -> Optional[str]:
        """Helper function to offset the port in an endpoint by
            the data parallel rank.

        Args:
            endpoint: The endpoint string
                (e.g., "tcp://*:5557" or "inproc://cache")
            data_parallel_rank: The data parallel rank to offset by

        Returns:
            The endpoint with the port offset by data_parallel_rank
                or suffix appended
        """
        # Do nothing if input is None or data_parallel_rank is 0
        if not endpoint or data_parallel_rank == 0:
            return endpoint

        if "inproc" in endpoint:
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint:
            if endpoint and ":" in endpoint:
                # Get everything after the last colon (the port)
                last_colon_idx = endpoint.rfind(":")
                base_addr = endpoint[:last_colon_idx]
                base_port = int(endpoint[last_colon_idx + 1 :])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


class KVEventsConfig(BaseModel):
    """Configuration for KV event publishing."""

    publisher: str = "null"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    bind: Optional[bool] = None
    """Bind (true) or connect (false) the PUB socket. When unset, infer from
    the endpoint as before. Use true to bind a concrete address such as loopback.
    """

    replay_endpoint: Optional[str] = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    format: Literal["default", "dynamo"] = "default"
    """The event format. "default" is the SGLang format. "dynamo" is the format
    the Dynamo KV router reads: block hashes also depend on extra_key and
    cache_salt (see ``namespaced_block_hash``), and BlockStored has lora_name.
    """

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVEventsConfig":
        """Parse the CLI value for the event publisher config."""
        return KVEventsConfig.model_validate_json(cli_value)


def uses_dynamo_format(kv_events_config: Optional[str]) -> bool:
    if not kv_events_config:
        return False
    return KVEventsConfig.from_cli(kv_events_config).format == "dynamo"


class EventPublisherFactory:
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str, ctor: Callable[..., EventPublisher]) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(cls, config: Optional[str], attn_dp_rank: int = 0) -> EventPublisher:
        """Create publisher from a config mapping."""
        if not config:
            return NullEventPublisher()
        config = KVEventsConfig.from_cli(config)
        config_dict = config.model_dump()
        # The cache event recorder uses the format. The publisher does not accept it.
        config_dict.pop("format")

        kind = config_dict.pop("publisher", "null")
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(attn_dp_rank=attn_dp_rank, **config_dict)

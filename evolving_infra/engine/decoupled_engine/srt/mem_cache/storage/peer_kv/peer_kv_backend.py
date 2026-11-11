"""PeerKVBackend — hint-driven peer-fetch HiCache L3 storage backend (kv-p2p).

Reader side of the cross-engine KV sharing data plane. The wire contract
with the owner-side PeerKVServicer is the kv-p2p wire protocol v1: ZMQ REQ/REP
msgpack LOOKUP/RELEASE for leases + per-pool (ptr, nbytes) page segments, then
one-sided mooncake TransferEngine READs from the owner's pinned host pool into
the local pinned host pool.

Router-directed v1 semantics:
- ``batch_exists`` / ``batch_exists_v2`` return zero hits IMMEDIATELY
  when the request carries no ``kv_pull_hint`` (delivered via
  ``HiCacheStorageExtraInfo.extra_info``) — no socket is touched. This
  keeps the HiCache probe gate (which taxes every enqueued request once
  a storage backend is mounted) at in-process cost for ordinary traffic.
- With a hint, exists = one LOOKUP RPC to the hinted owner; the reply's
  confirmed prefix is shortened client-side per the pools' hit policies
  (TRAILING_PAGES pools must cover the trailing window of the adopted
  prefix); the lease is stashed for the subsequent get calls.
- GET failure of any segment fails the whole call (the engine's
  prefix/all-or-nothing verdicts then discard and recompute) and
  releases the lease. Otherwise the lease lives across every get call
  of the prefetch operation (multiple KV batches + KV-derived sidecars
  + the trailing SWA/state fetch) and is released when all expected
  pools are in — see ``_PeerLease``.

Write paths are acknowledged no-ops: this backend serves the OWNER's
host tier and stores nothing itself. That is safe because exists never
consults any local store (hint-only), so nothing ever claims these
pages are retrievable later; host eviction behavior is unchanged (the
radix tree does not track storage residency — a future miss simply
recomputes).

v1 scope cuts (deliberate): no mooncake-store fallback tier
(peer-only), no cross-rank fan-out (rank-symmetric fetch: rank i pulls
from owner rank i = servicer base port + tp_rank).

Deployment:
    --hicache-storage-backend dynamic
    --hicache-storage-backend-extra-config '{
        "backend_name": "peer_kv",
        "module_path": "decoupled_engine.srt.mem_cache.storage.peer_kv.peer_kv_backend",
        "class_name": "PeerKVBackend",
        "interface_v1": 1,              # required: zero-copy get/set surface
        "trace_dir": "/results/trace",  # optional per-fetch JSONL spans
        "lookup_timeout_ms": 1000,      # optional
        "lease_ttl_ms": 30000,          # optional
        "ib_device": null,              # optional NIC override
        "mc_slice_timeout_s": 10,       # optional, exported as MC_SLICE_TIMEOUT
        "share_transfer_engine": true   # optional; false = dedicated reader
                                        # TransferEngine (_shared_transfer_engine)
    }'
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
import uuid
import zlib
from collections import Counter, OrderedDict
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import torch
import zmq

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from decoupled_engine.srt.mem_cache.peer_kv_demote import (
    LAG_EDGES_S,
    NOTICE_MAX_AGE_S,
    NOTICE_MAX_KEYS,
    NOTICE_SKIP_CAUSES,
    RecordWriter,
    _age_line,
    lag_bucket,
)
from decoupled_engine.srt.mem_cache.peer_kv_namespace import peer_page_keys
from decoupled_engine.srt.mem_cache.peer_kv_servicer import SWA_RANGE_CAPABILITY
from decoupled_engine.srt.mem_cache.pool_host.common import host_buffer_registration_ranges

logger = logging.getLogger(__name__)

_msgpack_encoder = msgspec.msgpack.Encoder()
_msgpack_decoder = msgspec.msgpack.Decoder()
_json_encoder = msgspec.json.Encoder()

# One (ptr, nbytes) pair; a page maps to 1+ contiguous segments per pool.
Segment = Tuple[int, int]

# Poison-on-release debug detector (owner side: peer_kv_servicer.POISON_*,
# enabled by env or the servicer's runtime "debug" op; the LOOKUP reply then
# carries "poison": true and the reader checks 64 bytes per pulled segment).
_POISON_BYTE = 0xDE
_POISON_LEN = 64


class _PeerLease:
    """Reader-side record of one owner lease (mutable bookkeeping).

    Lifecycle: the lease is stashed at LOOKUP (batch_exists_v2) and must
    stay alive across EVERY get call the controller issues for the same
    prefetch operation. HybridCacheController drives, per operation:

        for each STORAGE_BATCH_SIZE slice of the KV prefix:
            batch_get_v1(kv slice)                       # KV anchor
            batch_get_v2(KV-derived sidecars, slice)     # ALL_PAGES pools
        batch_get_v2(trailing pools)                     # TRAILING_PAGES pools
                                                          # (SWA + SWA-derived
                                                          # state), keys = the
                                                          # last N hit pages

    so a multi-batch pull sees several v1/v2 calls and even a single-batch
    pull sees two v2 calls. The lease is complete — and released — only
    when the KV pages and every ALL_PAGES pool cover ``usable_pages`` and
    every TRAILING_PAGES pool has been fetched at least once (or when any
    read fails; the engine then discards the whole prefetch). Expected
    pools are recorded at LOOKUP time from the ``pool_transfers`` the
    controller passed to batch_exists_v2; a v1 ``batch_exists`` (no
    pool_transfers) expects KV only. Leases that never complete (prefetch
    revoked / truncated / terminated) fall back to TTL expiry on both
    sides.
    """

    def __init__(
        self,
        *,
        lease_id: str,
        te_endpoint: str,
        servicer_addr: str,
        pools: Dict[str, List[Optional[List[Segment]]]],
        keys: List[str],
        usable_pages: int,
        expiry_mono: float,
        expected_all_pages: Optional[set] = None,
        expected_trailing: Optional[set] = None,
    ) -> None:
        self.lease_id = lease_id
        self.te_endpoint = te_endpoint
        self.servicer_addr = servicer_addr
        self.pools = pools
        self.keys = keys
        self.usable_pages = usable_pages
        self.exact_endpoint = False
        self.expiry_mono = expiry_mono
        # Pool name -> page indices fetched so far (KV anchor under "kv").
        self.fetched_pages: Dict[str, set] = {}
        # Sidecar pools the controller will fetch page-aligned with KV
        # (must cover usable_pages) vs. trailing-window pools fetched once
        # after KV completes (must be touched at least once).
        self.expected_all_pages: set = set(expected_all_pages or ())
        self.expected_trailing: set = set(expected_trailing or ())
        self.released = False
        # Requesting rid: set from HiCacheStorageExtraInfo.extra_info at
        # lookup time; gets of the same request match only their own lease.
        self.req_id: str = ""
        self.attempt_id: str = ""
        self.trace_context: dict = {}
        # Optional owner-side crc32 per (pool, page) (servicer
        # --peer-kv-servicer-verify-checksums); verified after each read.
        self.sums: Dict[str, List[Optional[int]]] = {}
        # Owner runs poison-on-release: look for the pattern after each read.
        self.poison_check: bool = False

    def record_fetched(self, pool_name: str, page_idxs: List[int]) -> None:
        self.fetched_pages.setdefault(pool_name, set()).update(page_idxs)

    def is_complete(self) -> bool:
        kv = str(PoolName.KV)
        if len(self.fetched_pages.get(kv, ())) < self.usable_pages:
            return False
        for name in self.expected_all_pages:
            if len(self.fetched_pages.get(name, ())) < self.usable_pages:
                return False
        for name in self.expected_trailing:
            if not self.fetched_pages.get(name):
                return False
        return True


class _NoticeRecord:
    """NOTICE, reader TP0: the request-scoped NOTICE record of one
    hinted request. The owner, the keys up to the owner's confirmed end (at
    most NOTICE_MAX_KEYS), the smallest lookup tick, confirmed (pages from the
    first LOOKUP key), the most pages committed from the owner, whether a
    commit reached the confirmed end, and per prefetch attempt the index of
    the confirmed end in that attempt's keys (None: not among them)."""

    __slots__ = (
        "addr",
        "keys",
        "tick",
        "confirmed",
        "committed",
        "covered",
        "end_pos",
        "refused",
    )

    def __init__(self, addr, keys, tick, confirmed, refused):
        self.addr = addr
        self.keys = keys
        self.tick = tick
        self.confirmed = confirmed
        self.committed = 0
        self.covered = False
        self.end_pos: Dict[str, Optional[int]] = {}
        self.refused = refused


class _NoticeItem:
    """A NOTICE on the release queue; born: the reader's prefill end."""

    __slots__ = (
        "addr",
        "rid",
        "keys",
        "tick",
        "confirmed",
        "committed",
        "refused",
        "born",
    )

    def __init__(self, record: _NoticeRecord, rid: str, born: float):
        self.addr = record.addr
        self.rid = rid
        self.keys = record.keys
        self.tick = record.tick
        self.confirmed = record.confirmed
        self.committed = record.committed
        self.refused = record.refused
        self.born = born


class _BornMark(tuple):
    """A DEMOTE_OK on the release queue: ("demote_ok", owner addr, lease id,
    pages) and born, its commit time. It still equals the
    plain tuple."""

    def __new__(cls, items, born: float):
        mark = super().__new__(cls, items)
        mark.born = born
        return mark


def _key_index(keys: List[str], key: str) -> Optional[int]:
    try:
        return keys.index(key)
    except ValueError:
        return None


def _flatten_meta(ptrs: List[Any], sizes: List[Any]) -> Tuple[List[int], List[int]]:
    """Flatten get_page_buffer_meta output (may nest per-page segment lists)."""
    if ptrs and isinstance(ptrs[0], (list, tuple)):
        flat_ptrs = [p for group in ptrs for p in group]
        flat_sizes = [s for group in sizes for s in group]
        return flat_ptrs, flat_sizes
    return list(ptrs), list(sizes)


def _iter_host_pool_buffers(host_pool: Any):
    # Mirrors mooncake_store._iter_host_pool_buffers: heterogeneous host-pool
    # classes expose their physical tensors via get_hybrid_pool_buffer() when
    # they have more than a kv_buffer.
    get_buffers = getattr(
        host_pool,
        "get_hybrid_pool_buffer",
        lambda: [getattr(host_pool, "kv_buffer", None)],
    )
    for buf in get_buffers():
        if buf is not None:
            yield buf


class _ReaderTransferEngine:
    """Thin injectable wrapper around a MooncakeTransferEngine (read side).

    ``engine`` wraps an existing engine (the process-shared one, see
    PeerKVBackend._shared_transfer_engine); otherwise a dedicated engine is
    created from ``hostname`` / ``gpu_id`` / ``ib_device``."""

    def __init__(
        self,
        *,
        hostname: Optional[str] = None,
        gpu_id: Optional[int] = None,
        ib_device: Optional[str] = None,
        engine: Any = None,
    ):
        if engine is None:
            from decoupled_engine.srt.distributed.device_communicators.mooncake_transfer_engine import (
                MooncakeTransferEngine,
            )

            engine = MooncakeTransferEngine(
                hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
            )
        self._te = engine
        self._registered_buffers = set()

    def register(self, ptr: int, nbytes: int) -> None:
        if (ptr, nbytes) in self._registered_buffers:
            return
        # The generic wrapper only logs failures at debug level. A peer must
        # never advertise host pages whose RDMA registration failed. On a
        # shared engine the span may already be registered by another user
        # (the owner servicer): register_memory_once skips it.
        register_once = getattr(self._te, "register_memory_once", None)
        if register_once is not None:
            result = register_once(ptr, nbytes)
        else:
            result = self._te.engine.register_memory(ptr, nbytes)
        if result != 0:
            raise RuntimeError(
                f"peer_kv host registration failed: bytes={nbytes}, rc={result}"
            )
        self._registered_buffers.add((ptr, nbytes))

    def get_session_id(self):
        return self._te.get_session_id()

    def batch_read(
        self,
        *,
        session_id: str,
        local_ptrs: List[int],
        remote_ptrs: List[int],
        lengths: List[int],
    ) -> bool:
        try:
            ret = self._te.engine.batch_transfer_sync_read(
                session_id, local_ptrs, remote_ptrs, lengths
            )
        except Exception:
            logger.warning("peer_kv: TE batch read raised", exc_info=True)
            return False
        return ret >= 0


class PeerKVBackend(HiCacheStorage):
    supports_complete_endpoints = True
    # batch_exists/_v2 return zero hits without a kv_pull_hint, so the
    # scheduler skips hint-less prefetches and hit queries altogether
    # (hicache_storage.skip_hintless_storage_lookup).
    requires_kv_pull_hint = True
    # Write paths are acknowledged no-ops (module docstring): the caches skip
    # storage backups for this backend (hicache_storage.skip_noop_storage_backup).
    stores_backups = False
    # Seconds before an owner found incapable of LOOKUP_SWA_RANGE (or
    # unreachable) is probed again.
    _SWA_RANGE_REPROBE_S = 30.0
    # read_swa_range starts a pool read only while the range lease has at
    # least this long left (>= the expected read time), and fails a read that
    # ends past the expiry: the owner's cooling covers only reads that
    # started before the TTL ran out.
    _SWA_RANGE_READ_MARGIN_S = 2.0

    def __init__(self, storage_config: HiCacheStorageConfig, kwargs: dict):
        cfg = storage_config.extra_config or {}
        self.supports_complete_endpoints = storage_config.pp_size == 1
        self._storage_config = storage_config
        self._rank = storage_config.tp_rank
        self._lookup_timeout_ms = int(cfg.get("lookup_timeout_ms", 1000))
        # F5 (SGLANG_PEER_KV_LOOKUP_DEADLINE): a LOOKUP carries its reply
        # deadline, so the owner does not grant (and pin until the TTL) after
        # this reader gave up.
        self._lookup_deadline = bool(envs.SGLANG_PEER_KV_LOOKUP_DEADLINE.get())
        self._gpu_export_timeout_s = float(cfg.get("gpu_export_timeout_s", 30.0))
        self._release_timeout_ms = int(cfg.get("release_timeout_ms", 200))
        self._lease_ttl_ms = int(cfg.get("lease_ttl_ms", 30000))
        self._ib_device = cfg.get("ib_device")
        # Read through the process-wide Mooncake engine instead of starting a
        # dedicated one (see _shared_transfer_engine); false = always dedicated.
        self._share_transfer_engine = bool(cfg.get("share_transfer_engine", True))
        os.environ.setdefault(
            "MC_SLICE_TIMEOUT", str(int(cfg.get("mc_slice_timeout_s", 10)))
        )

        self._zmq_ctx = zmq.Context.instance()
        self._sockets: Dict[str, zmq.Socket] = {}
        # One lock per peer address: REQ sockets require strictly alternating
        # send->recv from a single caller at a time. Unlocked reuse across
        # concurrent prefetch threads produced EFSM double-sends plus
        # close-during-use races at c>=32 (steady ~5% lookup error rate).
        self._socket_locks: Dict[str, threading.Lock] = {}
        # key -> [(lease, page_idx), ...], newest first. A key CAN belong to
        # several live leases: concurrent prefetches for requests sharing a
        # prefix (same session, common system prompt) each LOOKUP the same
        # pages. Earlier v1 kept one lease per key and released the older
        # one on collision, which killed in-flight pulls.
        self._lease_by_key: Dict[str, List[Tuple[_PeerLease, int]]] = {}
        self._leases: Dict[str, _PeerLease] = {}
        self._lock = threading.Lock()
        # RELEASE RPCs issued from the scheduler / prefetch threads (hit
        # reconciliation, termination) go through this worker so those threads
        # never block on a peer round-trip; get-path releases stay inline.
        self._release_queue: Queue[_PeerLease] = Queue()
        # SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT: a lease read in full sends a
        # plain RELEASE marked demote_after_commit; the scheduler's commit of
        # the pulled pages then sends DEMOTE_OK (on_prefetch_commit). Under
        # self._lock: req_id -> [(owner addr, lease id, pages, expiry,
        # attempt id)]. A pending entry that ends without DEMOTE_OK is counted
        # by cause (COMMIT_DROP_CAUSES); its owner never demotes.
        self._demote_after_commit = bool(envs.SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT.get())
        self._commit_ttl_s = float(envs.SGLANG_PEER_KV_DEMOTE_PENDING_TTL_S.get())
        self._commit_pending: OrderedDict[str, list] = OrderedDict()
        self.stats_demote_ok_sent = 0
        self.stats_commit_no_rid = 0
        self.stats_commit_dropped: Dict[str, int] = {}
        self._init_notice(self._rank)
        self._release_thread = threading.Thread(
            target=self._release_worker, daemon=True, name="peer-kv-release"
        )
        self._release_thread.start()
        self._te: Optional[_ReaderTransferEngine] = None
        # Owner rank address -> (swa_range capable, probe time); backfill IO
        # thread only.
        self._swa_range_caps: Dict[str, Tuple[bool, float]] = {}
        self._warned_set = False
        self._warned_bad_hint_ts = 0.0
        # True when the primary host pool is a logical anchor with no bytes
        # (DeepSeek V4 "kv" entry): get_v1 then counts pages without a TE
        # read and the bytes travel via the batch_get_v2 sidecar transfers
        # (mooncake_store precedent). Set at register_mem_pool_host time.
        self._anchor_is_logical = False

        trace_dir = cfg.get("trace_dir")
        self._trace_file = None
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            self._trace_file = open(
                os.path.join(
                    trace_dir,
                    f"peer_fetch_trace.rank{self._rank}.pid{os.getpid()}.jsonl",
                ),
                "ab",
                buffering=0,
            )
        self._open_records(trace_dir, self._rank)

        logger.info(
            "PeerKVBackend init: rank=%s lookup_timeout_ms=%s lease_ttl_ms=%s",
            self._rank,
            self._lookup_timeout_ms,
            self._lease_ttl_ms,
        )

    # ------------------------------------------------------------------
    # registration (local pools; also TE-registers their pinned buffers)
    # ------------------------------------------------------------------

    def _make_te(self) -> _ReaderTransferEngine:
        from decoupled_engine.srt.utils.network import get_local_ip_auto

        shared = self._shared_transfer_engine()
        if shared is not None:
            logger.info(
                "PeerKVBackend rank %s: peer reads use the process Mooncake "
                "TransferEngine %s (no dedicated reader engine)",
                self._rank,
                shared.get_session_id(),
            )
            return _ReaderTransferEngine(engine=shared)
        te = _ReaderTransferEngine(
            hostname=get_local_ip_auto(),
            gpu_id=self._rank,
            ib_device=self._ib_device,
        )
        logger.info(
            "PeerKVBackend rank %s: dedicated reader Mooncake TransferEngine %s",
            self._rank,
            te.get_session_id(),
        )
        return te

    @staticmethod
    def _device_set(devices: Optional[str]) -> frozenset:
        return frozenset(d.strip() for d in (devices or "").split(",") if d.strip())

    def _local_gpu_id(self) -> int:
        """This scheduler's CUDA device (what the servicer passes as gpu_id
        for the process engine); the TP rank when CUDA is not initialized."""
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                return int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 - best effort, falls back to the rank
            pass
        return self._rank

    def _shared_transfer_engine(self):
        """The process-wide Mooncake engine to read through, or None for a
        dedicated reader engine (the previous behavior).

        Every TransferEngine runs its own completion pollers (on EFA a thread
        per engine spinning on sched_yield), so a dedicated reader engine is a
        per-rank cost on every scheduler step even when no peer read happens.
        The reader only issues one-sided READs into host spans it registers,
        so any engine of the process can carry them, and the owner servicer
        registers the same host spans on the process engine anyway
        (register_memory_once dedups them):

        - the process engine exists (P/D or symmetric Mooncake, or created
          earlier) and serves the same NICs (the configured ``ib_device`` is
          unset or resolves to the same devices): share it;
        - it does not exist and ``ib_device`` is set: create it with this
          reader's parameters; the servicer's init_mooncake_transfer_engine
          then reuses it instead of starting a second engine;
        - otherwise (other NICs, or no device list: an auto-topology process
          engine could change the owner's NICs), or with
          ``share_transfer_engine: false``: a dedicated engine.
        """
        if not getattr(self, "_share_transfer_engine", True):
            return None
        try:
            from decoupled_engine.srt.distributed.device_communicators import (
                mooncake_transfer_engine as mte,
            )
        except ImportError:
            return None
        engine = mte.get_mooncake_transfer_engine()
        if engine is None:
            if self._ib_device is None:
                return None
            from decoupled_engine.srt.utils.network import get_local_ip_auto

            return mte.init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self._local_gpu_id(),
                ib_device=self._ib_device,
            )
        if self._ib_device is None:
            return engine
        if os.environ.get("MC_FORCE_TCP") == "1":
            wanted = ""
        else:
            try:
                wanted = mte.get_ib_devices_for_gpu(self._ib_device, engine.gpu_id)
            except (ValueError, RuntimeError):
                wanted = None
        if wanted is not None and self._device_set(wanted) == self._device_set(
            engine.get_ib_device()
        ):
            return engine
        logger.info(
            "PeerKVBackend rank %s: process Mooncake engine serves %r, reader "
            "ib_device resolves to %r; using a dedicated reader engine",
            self._rank,
            engine.get_ib_device(),
            wanted,
        )
        return None

    def _ensure_te(self) -> _ReaderTransferEngine:
        if self._te is None:
            self._te = self._make_te()
        return self._te

    def get_registered_transfer_engine(self):
        """Reuse the reader's host registrations for a source-only owner."""
        return self._ensure_te()

    def _te_register_pool(self, host_pool: Any) -> None:
        te = self._ensure_te()
        for buf in _iter_host_pool_buffers(host_pool):
            for ptr, nbytes in host_buffer_registration_ranges(buf):
                te.register(ptr=ptr, nbytes=nbytes)

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        if getattr(self.mem_pool_host, "kv_buffer", None) is None:
            # Hybrid logical anchor (HostPoolGroup): physical tensors arrive
            # via register_mem_host_pool_v2 per pool.
            self._anchor_is_logical = True
            return
        self._te_register_pool(self.mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._te_register_pool(host_pool)

    # ------------------------------------------------------------------
    # hint + LOOKUP
    # ------------------------------------------------------------------

    def _hint_from(
        self, extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> Optional[dict]:
        if extra_info is None or extra_info.extra_info is None:
            return None
        hint = extra_info.extra_info.get("kv_pull_hint")
        if hint is None:
            return None
        if not isinstance(hint, dict) or not isinstance(
            hint.get("source_servicer_addr"), str
        ):
            self._warn_bad_hint(hint)
            return None
        return hint

    @staticmethod
    def _req_id_from(extra_info: Optional[HiCacheStorageExtraInfo]) -> str:
        if extra_info is None or extra_info.extra_info is None:
            return ""
        rid = extra_info.extra_info.get("req_id")
        return rid if isinstance(rid, str) else ""

    def _warn_bad_hint(self, hint: Any) -> None:
        now = time.monotonic()
        if now - self._warned_bad_hint_ts > 10.0:
            self._warned_bad_hint_ts = now
            logger.warning("peer_kv: malformed kv_pull_hint ignored: %r", hint)

    def _target_addr(self, hint: dict) -> Optional[str]:
        addr = hint["source_servicer_addr"]
        host, sep, port = addr.rpartition(":")
        if not sep or not port.isdigit():
            self._warn_bad_hint(hint)
            return None
        return f"{host}:{int(port) + self._rank}"

    def _socket_for(self, addr: str) -> zmq.Socket:
        sock = self._sockets.get(addr)
        if sock is None:
            sock = self._zmq_ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{addr}")
            self._sockets[addr] = sock
        return sock

    def _rpc(
        self,
        *,
        addr: str,
        msg: dict,
        timeout_ms: int,
        deadline_field: Optional[str] = None,
    ) -> Optional[dict]:
        lock = self._socket_locks.setdefault(addr, threading.Lock())
        with lock:
            sock = self._socket_for(addr)
            try:
                if deadline_field is not None:
                    # F5: the wall clock when this reader stops waiting (the
                    # poll below starts at the send, after the socket lock).
                    msg[deadline_field] = time.time() + timeout_ms / 1e3
                sock.send(_msgpack_encoder.encode(msg))
                if sock.poll(timeout_ms, zmq.POLLIN) == 0:
                    # REQ socket is now stuck mid-state; drop and rebuild.
                    sock.close(linger=0)
                    self._sockets.pop(addr, None)
                    logger.warning("peer_kv: RPC timeout to %s (%s)", addr, msg["op"])
                    return None
                return _msgpack_decoder.decode(sock.recv())
            except Exception:
                sock.close(linger=0)
                self._sockets.pop(addr, None)
                logger.warning("peer_kv: RPC to %s failed", addr, exc_info=True)
                return None

    def _lookup(
        self,
        *,
        hint: dict,
        keys: List[str],
        require_complete_endpoint: bool = False,
        require_full_swa_prefix: bool = False,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Optional[dict]:
        addr = self._target_addr(hint)
        if addr is None:
            return None
        message = {"op": "lookup", "hashes": keys, "lease_ttl_ms": self._lease_ttl_ms}
        # Opaque request attribution only; the owner never uses it for a grant.
        message["request_metadata"] = {
            "req_id": self._req_id_from(extra_info),
            **self._trace_context_from(extra_info),
        }
        if require_complete_endpoint:
            message["require_complete_endpoint"] = True
        if require_full_swa_prefix:
            message["require_full_swa_prefix"] = True
        started = time.monotonic()
        reply = self._rpc(
            addr=addr,
            msg=message,
            timeout_ms=self._lookup_timeout_ms,
            **(
                {"deadline_field": "deadline_unix_s"}
                if getattr(self, "_lookup_deadline", False)
                else {}
            ),
        )
        if reply is None:
            return None
        if not reply.get("ok"):
            # Refusals (no_pages/budget) come back ok=True with num_pages=0;
            # ok=False is a servicer-side exception — surface it.
            logger.warning(
                "peer_kv: LOOKUP error from %s: %s", addr, reply.get("error")
            )
            return None
        reply["_servicer_addr"] = addr
        reply["_lookup_started_mono"] = started
        return reply

    @staticmethod
    def _attempt_id_from(extra_info: Optional[HiCacheStorageExtraInfo]) -> str:
        extra = getattr(extra_info, "extra_info", None)
        return (
            str(extra.get("prefetch_attempt_id", "")) if isinstance(extra, dict) else ""
        )

    @classmethod
    def _trace_context_from(cls, extra_info: Optional[HiCacheStorageExtraInfo]) -> dict:
        extra = getattr(extra_info, "extra_info", None)
        extra = extra if isinstance(extra, dict) else {}
        context = {"prefetch_attempt_id": cls._attempt_id_from(extra_info)}
        for name in ("endpoint_round", "initial_query_pages"):
            value = extra.get(name)
            context[name] = value if type(value) is int and value >= 0 else None
        return context

    @staticmethod
    def _lease_trace_context(lease: _PeerLease) -> dict:
        return {"prefetch_attempt_id": lease.attempt_id, **lease.trace_context}

    # ------------------------------------------------------------------
    # lease bookkeeping
    # ------------------------------------------------------------------

    def _stash_lease(
        self,
        *,
        reply: dict,
        keys: List[str],
        usable_pages: int,
        expected_all_pages: Optional[set] = None,
        expected_trailing: Optional[set] = None,
        req_id: str = "",
        attempt_id: str = "",
        exact_endpoint: bool = False,
        trace_context: Optional[dict] = None,
    ) -> Optional[_PeerLease]:
        if not reply.get("lease_id"):
            return None
        if usable_pages <= 0:
            # Owner leased pages we cannot use (e.g. trailing shortening):
            # give them back immediately instead of waiting for TTL expiry.
            self._rpc(
                addr=reply["_servicer_addr"],
                msg={"op": "release", "lease_id": reply["lease_id"]},
                timeout_ms=self._release_timeout_ms,
            )
            return None
        lease = _PeerLease(
            lease_id=reply["lease_id"],
            te_endpoint=reply["te_endpoint"],
            servicer_addr=reply["_servicer_addr"],
            pools={
                name: [
                    [(int(p), int(n)) for p, n in page] if page is not None else None
                    for page in pages
                ]
                for name, pages in (reply.get("pools") or {}).items()
            },
            keys=keys[: reply["num_pages"]],
            usable_pages=usable_pages,
            expiry_mono=reply.get("_lookup_started_mono", time.monotonic())
            + min(
                self._lease_ttl_ms, int(reply.get("lease_ttl_ms", self._lease_ttl_ms))
            )
            / 1e3,
            expected_all_pages=expected_all_pages,
            expected_trailing=expected_trailing,
        )
        lease.req_id = req_id
        lease.attempt_id = attempt_id
        lease.exact_endpoint = exact_endpoint
        lease.trace_context = dict(trace_context or {})
        sums = reply.get("sums")
        if isinstance(sums, dict):
            lease.sums = {str(k): list(v) for k, v in sums.items()}
        lease.poison_check = bool(reply.get("poison", False))
        with self._lock:
            self._sweep_expired_locked(time.monotonic())
            for idx, key in enumerate(lease.keys):
                # Newest first so a re-LOOKUP of the same prefix is matched
                # before an older, still-running lease on the same pages.
                self._lease_by_key.setdefault(key, []).insert(0, (lease, idx))
            self._leases[lease.lease_id] = lease
        return lease

    def _sweep_expired_locked(self, now_mono: float) -> None:
        """Drop expired leases from the key index (owner expires its side
        on its own TTL; no RELEASE RPC needed). Caller holds ``_lock``."""
        expired = [
            lease for lease in self._leases.values() if now_mono > lease.expiry_mono
        ]
        for lease in expired:
            lease.released = True
            self._unindex_locked(lease)

    def _unindex_locked(self, lease: _PeerLease) -> None:
        for key in lease.keys:
            entries = self._lease_by_key.get(key)
            if not entries:
                continue
            entries[:] = [e for e in entries if e[0] is not lease]
            if not entries:
                self._lease_by_key.pop(key, None)
        self._leases.pop(lease.lease_id, None)

    def _match_lease_pages(
        self, keys: List[str], req_id: str = "", attempt_id: str = ""
    ) -> Optional[Tuple[_PeerLease, List[int]]]:
        """Find a live lease whose confirmed, usable pages cover ``keys``
        contiguously starting at ``keys[0]``.

        A lease that carries a request id matches only gets of that same
        request: concurrent operations over the same pages each hold their
        own lease, and attributing one operation's reads to another's lease
        would complete/release the wrong one — the owner unpins on RELEASE
        immediately, so a foreign completion could free pages under a read
        that is still in flight. Leases without a request id (foreign
        controllers) stay matchable by anyone."""
        now = time.monotonic()
        with self._lock:
            candidates = self._lease_by_key.get(keys[0]) or ()
            for lease, start in candidates:
                if lease.released or now > lease.expiry_mono:
                    continue
                if lease.req_id and lease.req_id != req_id:
                    continue
                if lease.attempt_id and lease.attempt_id != attempt_id:
                    continue
                idxs = []
                for j, key in enumerate(keys):
                    idx = start + j
                    if idx >= lease.usable_pages or lease.keys[idx] != key:
                        idxs = None
                        break
                    idxs.append(idx)
                if idxs is not None:
                    return lease, idxs
            return None

    def _release(self, lease: _PeerLease, complete: bool = False) -> None:
        """RELEASE the owner's pins. ``complete``: every leased page of every
        expected pool was read (``is_complete``); the owner may then demote
        its own copy of those pages (--peer-kv-demote-source; owners without
        it ignore the extra fields)."""
        if lease.released:
            return
        lease.released = True
        with self._lock:
            self._unindex_locked(lease)
        msg = {"op": "release", "lease_id": lease.lease_id}
        if complete and getattr(self, "_demote_after_commit", False):
            # The owner demotes on DEMOTE_OK, after this tree committed the
            # pages (on_prefetch_commit); an abort sends nothing more.
            msg["demote_after_commit"] = True
            msg["pages"] = int(lease.usable_pages)
            if not getattr(self, "_notice_on", False) or not getattr(self, "_rank", 0):
                # With NOTICE on, DEMOTE_OK leaves TP0 only:
                # only the owner's TP0 (the demote source) knows the op.
                self._note_commit_pending(lease)
        elif complete:
            msg["complete"] = True
            msg["pages"] = int(lease.usable_pages)
        self._rpc(
            addr=lease.servicer_addr,
            msg=msg,
            timeout_ms=self._release_timeout_ms,
        )

    _COMMIT_PENDING_MAX = 4096
    # Why a pending demote (a lease read in full, SGLANG_PEER_KV_DEMOTE_AFTER_
    # COMMIT) ended without DEMOTE_OK. No demote follows: the owner's entry
    # expires after its TTL (owner counter demote_commit_expired).
    COMMIT_DROP_CAUSES = (
        "abort",  # the request ended after its reads, before the commit
        "hybrid_discard",  # the hybrid all-or-nothing check discarded the pull
        "insert_dropped",  # insert_host dropped the pulled pages
        "buffer_mode",  # buffer_only host memory: no commit into a tree
        "other_attempt",  # a lease of another prefetch attempt of the request
        "no_pages",  # the commit kept no page
        "expired",  # no commit within SGLANG_PEER_KV_DEMOTE_PENDING_TTL_S
        "overflow",  # more than _COMMIT_PENDING_MAX requests pending
    )

    def _count_commit_drop_locked(self, cause: str, n: int = 1) -> None:
        dropped = self.stats_commit_dropped
        dropped[cause] = dropped.get(cause, 0) + n

    def _note_commit_pending(self, lease: _PeerLease) -> None:
        req_id = getattr(lease, "req_id", "")
        if not req_id:
            self.stats_commit_no_rid += 1  # no commit can name it: no demote
            return
        now = time.monotonic()
        with self._lock:
            pending = self._commit_pending
            while pending:
                oldest, entries = next(iter(pending.items()))
                if (
                    any(e[3] > now for e in entries)
                    and len(pending) < self._COMMIT_PENDING_MAX
                ):
                    break
                del pending[oldest]
                expired = sum(1 for e in entries if e[3] <= now)
                if expired:
                    self._count_commit_drop_locked("expired", expired)
                if len(entries) > expired:
                    self._count_commit_drop_locked("overflow", len(entries) - expired)
            pending.setdefault(req_id, []).append(
                (
                    lease.servicer_addr,
                    lease.lease_id,
                    int(lease.usable_pages),
                    now + self._commit_ttl_s,
                    str(getattr(lease, "attempt_id", "") or ""),
                )
            )
            pending.move_to_end(req_id)

    def _open_records(self, trace_dir: Optional[str], rank) -> None:
        """SGLANG_PEER_KV_DEMOTE_RECORDS (R3; needs COUNTERS
        and a trace dir): one record per queued DEMOTE_OK, written by a
        thread (put never blocks; a full queue drops, counted)."""
        self._records = None
        if not (
            trace_dir
            and envs.SGLANG_PEER_KV_DEMOTE_COUNTERS.get()
            and envs.SGLANG_PEER_KV_DEMOTE_RECORDS.get()
        ):
            return
        os.makedirs(trace_dir, exist_ok=True)
        self._records = RecordWriter(
            os.path.join(
                trace_dir, f"peer_reader_commit.rank{rank}.pid{os.getpid()}.jsonl"
            )
        )
        logger.info("PeerKV demote: reader commit records %s", self._records.path)

    def on_prefetch_commit(
        self, req_id: str, pages: int, attempt_id: Optional[str] = None
    ) -> int:
        """SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT, scheduler thread: ``pages``
        pulled pages of ``req_id`` (from the first leased page on) are now in
        the reader's tree (insert_host at the prefetch completion). Queue
        DEMOTE_OK for each lease of this prefetch attempt that was read in
        full; the release worker sends it. A lease of another attempt of the
        request (a stopped pull whose reads finished), an expired entry and a
        commit of no page send nothing (counted). Returns the number queued."""
        now = time.monotonic()
        born = self._now()
        attempt = str(attempt_id) if attempt_id else ""
        records = getattr(self, "_records", None)  # RECORDS (R3)
        wall = time.time() if records is not None else 0.0
        nmap = getattr(self, "_notice_map", None)
        with self._lock:
            record = nmap.get(req_id) if nmap is not None else None
            if record is not None:  # what reached this tree
                record.committed = max(record.committed, int(pages))
                pos = record.end_pos.get(attempt)
                if pos is not None and pages > pos:
                    record.covered = True
            entries = self._commit_pending.pop(req_id, [])
            queued = 0
            for addr, lease_id, leased, expiry, lease_attempt in entries:
                if expiry <= now:
                    self._count_commit_drop_locked("expired")
                elif attempt and lease_attempt and lease_attempt != attempt:
                    self._count_commit_drop_locked("other_attempt")
                elif pages <= 0:
                    self._count_commit_drop_locked("no_pages")
                else:
                    self._release_queue.put(
                        _BornMark(
                            ("demote_ok", addr, lease_id, min(leased, pages)), born
                        )
                    )
                    queued += 1
                    if records is not None:
                        records.put(
                            {
                                "event": "reader_commit",
                                "ts": wall,
                                "rid": req_id,
                                "owner": addr,
                                "lease_id": lease_id,
                                "pages": min(leased, pages),
                                "attempt": lease_attempt,
                            }
                        )
            self.stats_demote_ok_sent += queued
        return queued

    def on_prefetch_discard(self, req_id: str, cause: str) -> int:
        """SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT, scheduler thread: the pull of
        ``req_id`` ended without a commit (``cause``: abort after its reads,
        hybrid discard, dropped insert, buffer mode). Its leases read in full
        get no DEMOTE_OK, so their owners never demote them; count them.
        Returns the number dropped. A lease whose reads finish after this call
        expires the same way (cause "expired")."""
        with self._lock:
            entries = self._commit_pending.pop(req_id, [])
            if entries:
                self._count_commit_drop_locked(cause, len(entries))
        return len(entries)

    def commit_stats_line(self) -> str:
        """The reader's commit-gated demote counters as one log line."""
        with self._lock:
            pending = sum(len(v) for v in self._commit_pending.values())
            dropped = dict(self.stats_commit_dropped)
            sent = self.stats_demote_ok_sent
        parts = [
            f"rank={getattr(self, '_rank', -1)}",
            f"demote_ok_sent={sent}",
            f"pending={pending}",
            f"no_rid={self.stats_commit_no_rid}",
        ]
        parts += [f"dropped_{c}={dropped.get(c, 0)}" for c in self.COMMIT_DROP_CAUSES]
        if getattr(self, "_notice_map", None) is not None:
            parts.append(self._notice_line())
        return " ".join(parts)

    # ------------------------------------------------------------------
    # NOTICE (SGLANG_PEER_KV_DEMOTE_NOTICE)
    # ------------------------------------------------------------------

    _NOTICE_MAP_MAX = 8192
    # The counters of the reader line, in this order (notice_<name>).
    NOTICE_LINE_KEYS = (
        "eligible",
        "queued",
        "sent",
        "unknown",
        "rpc_fail",
        "covered",
        "skipped_flag_off",
        *(f"skipped_no_insert_{c}" for c in NOTICE_SKIP_CAUSES),
        "dropped_age_notice",
        "dropped_age_notice_pages",
        "dropped_age_demote_ok",
        "dropped_age_demote_ok_pages",
        "aborted",
        "overflow",
    )

    def _init_notice(self, rank) -> None:
        """Reader side. Records live on TP0 only (with NOTICE on, DEMOTE_OK
        and NOTICE leave TP0 only), need AFTER_COMMIT (its commit hook tells
        what this tree committed), and are kept with NOTICE or COUNTERS on
        (the control arm counts the eligible NOTICEs as skipped_flag_off).
        The map has no TTL: a record ends at the prefill end or at an abort
        (bounded by _NOTICE_MAP_MAX, the oldest dropped and counted)."""
        after_commit = bool(getattr(self, "_demote_after_commit", False))
        notice = bool(envs.SGLANG_PEER_KV_DEMOTE_NOTICE.get())
        self._notice_on = notice and after_commit
        track = (
            after_commit
            and rank == 0
            and (notice or bool(envs.SGLANG_PEER_KV_DEMOTE_COUNTERS.get()))
        )
        self._notice_map: Optional[OrderedDict] = OrderedDict() if track else None
        self._notice_stats_lock = threading.Lock()
        self.notice_stats: Counter = Counter()
        self.notice_send_age = [0] * (len(LAG_EDGES_S) + 1)
        if notice and not after_commit and rank == 0:
            logger.warning(
                "SGLANG_PEER_KV_DEMOTE_NOTICE needs "
                "SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT on the reader: no NOTICE"
            )

    @property
    def notice_tracking(self) -> bool:
        """This reader keeps NOTICE records (TP0)."""
        return getattr(self, "_notice_map", None) is not None

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _notice_bump(self, key: str, n: int = 1) -> None:
        lock = getattr(self, "_notice_stats_lock", None)
        if lock is None:
            return
        with lock:
            self.notice_stats[key] += n

    def _notice_note_lookup(
        self, rid: str, reply: dict, lookup_keys: List[str], attempt: Any
    ) -> None:
        """Prefetch thread, TP0: one hinted LOOKUP reply. The first reply with
        confirmed pages opens the record; a later reply of the same owner
        keeps the smallest lookup tick. Each reply notes the index of the
        confirmed end in that attempt's keys (for the commit check)."""
        nmap = getattr(self, "_notice_map", None)
        confirmed = reply.get("confirmed")
        tick = reply.get("lookup_tick")
        if (
            nmap is None
            or not rid
            or type(confirmed) is not int
            or confirmed <= 0
            or isinstance(tick, bool)
            or not isinstance(tick, (int, float))
        ):
            return
        addr = reply.get("_servicer_addr", "")
        attempt = str(attempt or "")
        overflow = False
        with self._lock:
            record = nmap.get(rid)
            if record is None:
                confirmed = min(confirmed, len(lookup_keys))
                if confirmed <= 0:
                    return
                if len(nmap) >= self._NOTICE_MAP_MAX:
                    nmap.popitem(last=False)
                    overflow = True
                keys = list(
                    lookup_keys[max(0, confirmed - NOTICE_MAX_KEYS) : confirmed]
                )
                record = _NoticeRecord(
                    addr, keys, float(tick), confirmed, reply.get("refused_reason")
                )
                record.end_pos[attempt] = confirmed - 1
                nmap[rid] = record
            elif record.addr == addr:
                record.tick = min(record.tick, float(tick))
                record.end_pos[attempt] = _key_index(lookup_keys, record.keys[-1])
        if overflow:
            self._notice_bump("overflow")

    def on_prefill_end(self, rid: str, cause: Optional[str] = None) -> bool:
        """Scheduler thread, TP0: the final prefill end of ``rid`` (right
        after maybe_cache_unfinished_req). ``cause``: the insert gate
        (notice_insert_cause; None = inserted with write-through). Queue one
        NOTICE when the record exists, no commit from the owner reached its
        confirmed end, the gate passes and NOTICE is on; else count why.
        Returns True when a NOTICE was queued. The record ends here."""
        nmap = getattr(self, "_notice_map", None)
        if nmap is None or not rid:
            return False
        with self._lock:
            record = nmap.pop(rid, None)
        if record is None:
            return False
        if record.covered:
            self._notice_bump("covered")
            self._notice_record(rid, record, "covered")
            return False
        self._notice_bump("eligible")
        if cause is not None:
            self._notice_bump(f"skipped_no_insert_{cause}")
            self._notice_record(rid, record, f"skipped_no_insert_{cause}")
            return False
        if not self._notice_on:
            self._notice_bump("skipped_flag_off")
            self._notice_record(rid, record, "skipped_flag_off")
            return False
        self._release_queue.put(_NoticeItem(record, rid, self._now()))
        self._notice_bump("queued")
        return True

    def notice_forget(self, rid: str) -> None:
        """Scheduler thread, TP0: ``rid`` was aborted; no NOTICE for it."""
        nmap = getattr(self, "_notice_map", None)
        if nmap is None or not rid:
            return
        with self._lock:
            record = nmap.pop(rid, None)
        if record is not None:
            self._notice_bump("aborted")
            self._notice_record(rid, record, "aborted")

    def _send_notice(self, item: _NoticeItem) -> None:
        """Release worker: send the NOTICE with its age since the prefill
        end, or drop it when it is older than NOTICE_MAX_AGE_S."""
        age_s = max(0.0, self._now() - item.born)
        age_ms = int(age_s * 1000)
        if age_s > NOTICE_MAX_AGE_S:
            self._notice_bump("dropped_age_notice")
            self._notice_bump("dropped_age_notice_pages", item.confirmed)
            self._notice_record(item.rid, item, "dropped_age", age_ms)
            return
        reply = self._rpc(
            addr=item.addr,
            msg={
                "op": "notice",
                "rid": item.rid,
                "keys": item.keys,
                "lookup_tick": item.tick,
                "age_ms": age_ms,
            },
            timeout_ms=self._release_timeout_ms,
        )
        if reply is None:
            outcome = "rpc_fail"
        elif reply.get("ok"):
            outcome = "sent"
            with self._notice_stats_lock:
                self.notice_send_age[lag_bucket(age_s)] += 1
        else:
            outcome = "unknown"  # the owner does not take NOTICEs
        self._notice_bump(outcome)
        self._notice_record(item.rid, item, outcome, age_ms)

    def _notice_record(
        self, rid: str, src: Any, outcome: str, age_ms: Optional[int] = None
    ) -> None:
        """R4 (SGLANG_PEER_KV_DEMOTE_RECORDS): one reader_notice
        record per record end (covered, skipped, aborted) or send."""
        records = getattr(self, "_records", None)
        if records is None:
            return
        records.put(
            {
                "event": "reader_notice",
                "ts": time.time(),
                "rid": rid,
                "owner": src.addr,
                "confirmed": src.confirmed,
                "committed": src.committed,
                "lookup_tick": src.tick,
                "refused": src.refused,
                "keys": len(src.keys),
                "age_ms": age_ms,
                "outcome": outcome,
            }
        )

    def _notice_line(self) -> str:
        with self._notice_stats_lock:
            stats = dict(self.notice_stats)
            ages = list(self.notice_send_age)
        nmap = self._notice_map
        parts = [f"notice_{k}={stats.get(k, 0)}" for k in self.NOTICE_LINE_KEYS]
        parts.append(f"notice_open={len(nmap) if nmap is not None else 0}")
        parts.append(_age_line("notice_send_age_s", ages, LAG_EDGES_S))
        return " ".join(parts)

    def _release_async(self, lease: _PeerLease) -> None:
        """Unindex now (no further get can match it), RELEASE on the worker."""
        with self._lock:
            if lease.released:
                return
            lease.released = True
            self._unindex_locked(lease)
        self._release_queue.put(lease)

    def _release_worker(self) -> None:
        while True:
            item = self._release_queue.get()
            try:
                self._send_queued(item)
            except Exception:  # noqa: BLE001
                logger.warning("peer_kv: async release failed", exc_info=True)

    def _send_queued(self, item: Any) -> None:
        """One RPC of the release worker: a plain RELEASE of a lease, a
        ("demote_ok", owner addr, lease id, pages) tuple, or a NOTICE. With
        NOTICE on, a mark older than NOTICE_MAX_AGE_S since its birth (the
        commit, the prefill end) is dropped here."""
        if isinstance(item, _NoticeItem):
            self._send_notice(item)
            return
        if isinstance(item, tuple):
            _, addr, lease_id, pages = item
            born = getattr(item, "born", None)
            if (
                born is not None
                and getattr(self, "_notice_on", False)
                and self._now() - born > NOTICE_MAX_AGE_S
            ):
                self._notice_bump("dropped_age_demote_ok")
                self._notice_bump("dropped_age_demote_ok_pages", int(pages))
                return
            self._rpc(
                addr=addr,
                msg={"op": "demote_ok", "lease_id": lease_id, "pages": pages},
                timeout_ms=self._release_timeout_ms,
            )
            return
        self._rpc(
            addr=item.servicer_addr,
            msg={"op": "release", "lease_id": item.lease_id},
            timeout_ms=self._release_timeout_ms,
        )

    def release_for_endpoint_retry(self, req_id: str, attempt_id: str) -> bool:
        """Release this not-yet-submitted prefetch before changing its endpoint.

        A synchronous ACK prevents the old lease from consuming the reserve
        needed by the next query. A timeout makes the caller abandon the retry.
        Cancellation can concurrently unindex a lease; it also terminates the
        operation, which the controller checks before issuing another LOOKUP.
        """
        if not req_id or not attempt_id:
            return False
        with self._lock:
            leases = [
                lease
                for lease in self._leases.values()
                if lease.req_id == req_id
                and lease.attempt_id == attempt_id
                and not lease.released
            ]
            for lease in leases:
                lease.released = True
                self._unindex_locked(lease)
        acknowledged = True
        for lease in leases:
            reply = self._rpc(
                addr=lease.servicer_addr,
                msg={"op": "release", "lease_id": lease.lease_id},
                timeout_ms=self._release_timeout_ms,
            )
            acknowledged &= reply is not None and reply.get("ok") is True
            self._trace(
                event="release",
                lease_id=lease.lease_id,
                req_id=req_id,
                pages=0,
                bytes=0,
                dur_ms=0.0,
                verdict="endpoint_retry",
                release_ack=reply is not None and reply.get("ok") is True,
                **self._lease_trace_context(lease),
            )
        return bool(acknowledged)

    def on_hit_reconciled(self, req_id: str, kv_pages: int, *, attempt_id=None) -> None:
        """The controller MIN-reduced this request's storage hit across TP
        ranks to ``kv_pages`` (prefetch thread). Pages beyond it will never be
        fetched here: trim the lease so it can complete, or give it back at
        once when the reduced hit is zero (a partial-grant abort otherwise
        leaves the owner's pins live until TTL + cooling)."""
        if not req_id:
            return
        with self._lock:
            leases = [
                lease
                for lease in self._leases.values()
                if lease.req_id == req_id
                and not lease.released
                and (attempt_id is None or lease.attempt_id == attempt_id)
            ]
        for lease in leases:
            if kv_pages <= 0 or (
                lease.exact_endpoint and kv_pages != lease.usable_pages
            ):
                self._trace(
                    event="release",
                    lease_id=lease.lease_id,
                    req_id=req_id,
                    pages=0,
                    bytes=0,
                    dur_ms=0.0,
                    verdict="reconciled_zero",
                    **self._lease_trace_context(lease),
                )
                self._release_async(lease)
            elif kv_pages < lease.usable_pages:
                self._trace(
                    event="trim",
                    lease_id=lease.lease_id,
                    req_id=req_id,
                    pages=kv_pages,
                    bytes=0,
                    dur_ms=0.0,
                    verdict="reconciled",
                    **self._lease_trace_context(lease),
                )
                lease.usable_pages = kv_pages

    # ------------------------------------------------------------------
    # exists
    # ------------------------------------------------------------------

    def _prepare_gpu_lookup(self, hint, keys, reply, extra_info):
        """Prepare on owner rank zero; only this IO worker waits, never REP.

        Drop any partial grant before waiting. Normal LOOKUP obtains the RDMA
        lease while the export ticket still holds the source host backing.
        Cancellation drops only this waiter's preparation interest.
        """
        owner = str(hint["source_servicer_addr"])
        addr = self._target_addr(hint)
        if reply.get("lease_id"):
            released = self._rpc(
                addr=addr,
                msg={"op": "release", "lease_id": reply["lease_id"]},
                timeout_ms=self._release_timeout_ms,
            )
            if not released or not released.get("ok"):
                return None
        waiter = uuid.uuid4().hex
        export_id = None
        state, reason = "error", "prepare_rpc"
        started = time.monotonic()
        cancelled = (extra_info.extra_info or {}).get(
            "peer_is_cancelled", lambda: False
        )
        try:
            result = self._rpc(
                addr=owner,
                msg={"op": "prepare_gpu", "hashes": keys, "waiter": waiter},
                timeout_ms=self._lookup_timeout_ms,
            )
            while result and result.get("ok"):
                export_id = result.get("export_id", export_id)
                state, reason = result.get("state"), result.get("reason")
                if cancelled():
                    state, reason = "cancelled", "request_cancelled"
                    return None
                if state == "ready":
                    grant = self._lookup(
                        hint=hint,
                        keys=keys,
                        require_complete_endpoint=True,
                        require_full_swa_prefix=True,
                        extra_info=extra_info,
                    )
                    wanted = int(result.get("pages", 0))
                    if grant is None or int(grant.get("num_pages", 0)) >= wanted:
                        return grant
                    capacity = grant.get("gpu_export_pin_capacity_pages")
                    if capacity is not None and wanted > int(capacity):
                        state, reason = "infeasible", "source_pin_capacity"
                        return grant
                    if grant.get("lease_id"):
                        released = self._rpc(
                            addr=addr,
                            msg={"op": "release", "lease_id": grant["lease_id"]},
                            timeout_ms=self._release_timeout_ms,
                        )
                        if not released or not released.get("ok"):
                            return None
                    reason = "peer_pin_wait"
                if state not in ("queued", "preparing", "ready"):
                    break
                if time.monotonic() - started >= self._gpu_export_timeout_s:
                    state, reason = "expired", "reader_prepare_deadline"
                    break
                time.sleep(0.02)
                result = self._rpc(
                    addr=owner,
                    msg={
                        "op": "prepare_gpu_status",
                        "export_id": export_id,
                        "waiter": waiter,
                    },
                    timeout_ms=self._lookup_timeout_ms,
                )
            # Capacity/admission/unavailable outcomes preserve a fresh host-only
            # fallback. They are logged separately from absent GPU backing.
            if cancelled():
                return None
            return self._lookup(
                hint=hint,
                keys=keys,
                require_complete_endpoint=True,
                require_full_swa_prefix=True,
                extra_info=extra_info,
            )
        finally:
            if export_id is not None:
                self._rpc(
                    addr=owner,
                    msg={
                        "op": "prepare_gpu_release",
                        "export_id": export_id,
                        "waiter": waiter,
                    },
                    timeout_ms=self._release_timeout_ms,
                )
            self._trace(
                event="gpu_export_prepare",
                req_id=self._req_id_from(extra_info),
                export_id=export_id,
                state=state,
                reason=reason,
                dur_ms=(time.monotonic() - started) * 1000,
                **self._trace_context_from(extra_info),
            )

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        result = self.batch_exists_v2(keys, None, extra_info)
        return result.kv_hit_pages

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        hint = self._hint_from(extra_info)
        req_id = self._req_id_from(extra_info)
        if hint is None or not keys:
            return PoolTransferResult.empty()
        t0 = time.monotonic()
        namespace = (extra_info.extra_info or {}).get("peer_namespace") or {}
        lookup_keys = peer_page_keys(
            keys, namespace.get("cache_salt"), namespace.get("extra_key")
        )
        full_swa_prefix = any(
            transfer.name == PoolName.SWA
            and transfer.hit_policy == PoolHitPolicy.ALL_PAGES
            for transfer in pool_transfers or []
        )
        exact_endpoint = full_swa_prefix or any(
            transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES
            for transfer in pool_transfers or []
        )
        if exact_endpoint and not self.supports_complete_endpoints:
            # PP stages do not share a complete cache-allocation vote yet.
            # Preserve FULL-only storage behavior; hybrid peer pulls miss.
            return PoolTransferResult.empty()
        reply = self._lookup(
            hint=hint,
            keys=lookup_keys,
            require_complete_endpoint=exact_endpoint,
            **({"require_full_swa_prefix": True} if full_swa_prefix else {}),
            extra_info=extra_info,
        )
        if reply is not None and getattr(self, "_notice_map", None) is not None:
            self._notice_note_lookup(
                req_id, reply, lookup_keys, self._attempt_id_from(extra_info)
            )
        if (
            full_swa_prefix
            and hint.get("allow_gpu_export", True) is not False
            and reply
            and reply.get("gpu_export")
            and int(reply.get("num_pages", 0)) < len(lookup_keys)
        ):
            reply = self._prepare_gpu_lookup(hint, lookup_keys, reply, extra_info)
        t_lookup_ms = (time.monotonic() - t0) * 1e3
        if reply is None:
            # Timeout / RPC failure / servicer exception: record the attempt
            # so failed pulls are visible in the fetch trace, not just logs.
            self._trace(
                event="lookup",
                lease_id="",
                req_id=req_id,
                pages=0,
                dur_ms=t_lookup_ms,
                verdict="error",
                query_pages=len(keys),
                require_complete_endpoint=exact_endpoint,
                granted_exact_endpoint=None,
                owner_refused_reason=None,
                **self._trace_context_from(extra_info),
            )
            return PoolTransferResult.empty()

        kv_entries = reply.get("pools", {}).get(str(PoolName.KV)) or []
        limit = min(int(reply["num_pages"]), len(keys), len(kv_entries))
        kv_pages = 0
        for i in range(limit):
            if kv_entries[i] is None:
                break
            kv_pages += 1

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        # Per-pool adoptable boundary (trace only): tells apart "owner has
        # no FULL host copy" from "owner has FULL but lost the SWA trailing
        # page" when the adopted prefix shrinks.
        bounds: Dict[str, int] = {}
        expected_all_pages: set = set()
        expected_trailing: set = set()
        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = str(transfer.name)
            entries = reply.get("pools", {}).get(name) or []
            page_exists = [
                i < len(entries) and entries[i] is not None for i in range(kv_pages)
            ]
            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = (
                    page_exists.index(False) if False in page_exists else kv_pages
                )
                expected_all_pages.add(name)
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                if exact_endpoint and kv_pages < trailing:
                    bounds[name] = 0
                    final_pages = 0
                    break
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        page_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
                expected_trailing.add(name)
            bounds[name] = boundary
            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)

        if exact_endpoint and (
            reply.get("exact_endpoint") is not True
            or kv_pages != int(reply["num_pages"])
            or final_pages != kv_pages
        ):
            # A different pool boundary is not a valid shorter checkpoint:
            # the owner pinned the tail at its advertised exact endpoint.
            # Old owners without protocol support safely fall back to compute.
            final_pages = 0
            hit_count = {}

        lease = self._stash_lease(
            reply=reply,
            keys=keys,
            usable_pages=final_pages,
            expected_all_pages=expected_all_pages,
            expected_trailing=expected_trailing,
            req_id=req_id,
            attempt_id=self._attempt_id_from(extra_info),
            exact_endpoint=exact_endpoint,
            trace_context={
                **self._trace_context_from(extra_info),
                "query_pages": len(keys),
            },
        )
        self._trace(
            event="lookup",
            lease_id=lease.lease_id if lease is not None else "",
            req_id=req_id,
            pages=final_pages,
            kv_pages=kv_pages,
            bounds=bounds,
            dur_ms=t_lookup_ms,
            verdict="hit" if final_pages > 0 else "miss",
            query_pages=len(keys),
            require_complete_endpoint=exact_endpoint,
            granted_exact_endpoint=reply.get("exact_endpoint"),
            owner_refused_reason=reply.get("refused_reason"),
            **self._trace_context_from(extra_info),
        )
        return PoolTransferResult(final_pages, hit_count)

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def _read_pages(
        self,
        *,
        lease: _PeerLease,
        pool_name: str,
        page_idxs: List[int],
        local_ptrs: List[int],
        local_sizes: List[int],
    ) -> bool:
        pages = lease.pools.get(pool_name)
        if pages is None:
            return False
        remote: List[Segment] = []
        for idx in page_idxs:
            if idx >= len(pages) or pages[idx] is None:
                return False
            remote.extend(pages[idx])
        if len(remote) != len(local_ptrs):
            logger.warning(
                "peer_kv: segment count mismatch pool=%s remote=%d local=%d "
                "(asymmetric layout?)",
                pool_name,
                len(remote),
                len(local_ptrs),
            )
            return False
        remote_ptrs = [seg[0] for seg in remote]
        lengths = [seg[1] for seg in remote]
        if lengths != local_sizes:
            logger.warning(
                "peer_kv: segment size mismatch pool=%s (asymmetric layout?)",
                pool_name,
            )
            return False
        ok = self._ensure_te().batch_read(
            session_id=lease.te_endpoint,
            local_ptrs=local_ptrs,
            remote_ptrs=remote_ptrs,
            lengths=lengths,
        )
        if ok and lease.sums:
            ok = self._verify_checksums(
                lease, pool_name, page_idxs, pages, local_ptrs, local_sizes
            )
        if ok and lease.poison_check:
            ok = self._check_poison(lease, pool_name, local_ptrs, local_sizes)
        return ok

    def _check_poison(
        self,
        lease: _PeerLease,
        pool_name: str,
        local_ptrs: List[int],
        local_sizes: List[int],
    ) -> bool:
        """Debug detector counterpart of the owner's poison-on-release hook."""
        for p, n in zip(local_ptrs, local_sizes):
            k = min(_POISON_LEN, int(n))
            if k > 0 and ctypes.string_at(p, k) == bytes([_POISON_BYTE]) * k:
                logger.error(
                    "peer_kv: POISONED page pulled pool=%s lease=%s req=%s",
                    pool_name,
                    lease.lease_id,
                    lease.req_id,
                )
                self._trace(
                    event="get_pool",
                    lease_id=lease.lease_id,
                    req_id=lease.req_id,
                    pool=pool_name,
                    pages=1,
                    bytes=0,
                    dur_ms=0.0,
                    verdict="poisoned",
                    **self._lease_trace_context(lease),
                )
                return False
        return True

    def _verify_checksums(
        self,
        lease: _PeerLease,
        pool_name: str,
        page_idxs: List[int],
        pages: List[Optional[List[Segment]]],
        local_ptrs: List[int],
        local_sizes: List[int],
    ) -> bool:
        """Debug detector: compare crc32 of the pulled bytes with the owner's
        per-page crc32 taken while the page was pinned. A mismatch means a
        page was reused under the read (a data-safety violation)."""
        sums = lease.sums.get(pool_name)
        if not sums:
            return True
        cursor = 0
        for idx in page_idxs:
            nseg = len(pages[idx])
            expected = sums[idx] if idx < len(sums) else None
            if expected is not None:
                crc = 0
                for k in range(cursor, cursor + nseg):
                    crc = zlib.crc32(
                        ctypes.string_at(local_ptrs[k], local_sizes[k]), crc
                    )
                if (crc & 0xFFFFFFFF) != expected:
                    logger.error(
                        "peer_kv: CHECKSUM MISMATCH pool=%s page=%d lease=%s req=%s "
                        "(owner %08x, pulled %08x)",
                        pool_name,
                        idx,
                        lease.lease_id,
                        lease.req_id,
                        expected,
                        crc & 0xFFFFFFFF,
                    )
                    self._trace(
                        event="get_pool",
                        lease_id=lease.lease_id,
                        req_id=lease.req_id,
                        pool=pool_name,
                        pages=1,
                        bytes=0,
                        dur_ms=0.0,
                        verdict="checksum_mismatch",
                    )
                    return False
            cursor += nseg
        return True

    def on_prefetch_terminated(self, req_id: str, *, attempt_id=None) -> None:
        """The controller abandoned this request's prefetch (timeout /
        revoke): RELEASE its leases now instead of holding owner pins for
        TTL + cooling."""
        if not req_id:
            return
        with self._lock:
            victims = [
                lease
                for lease in self._leases.values()
                if lease.req_id == req_id
                and not lease.released
                and (attempt_id is None or lease.attempt_id == attempt_id)
            ]
        for lease in victims:
            self._trace(
                event="release",
                lease_id=lease.lease_id,
                req_id=req_id,
                pages=0,
                bytes=0,
                dur_ms=0.0,
                verdict="terminated",
                **self._lease_trace_context(lease),
            )
            self._release_async(lease)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        n = len(keys)
        req_id = self._req_id_from(extra_info)
        match = self._match_lease_pages(keys, req_id, self._attempt_id_from(extra_info))
        if match is None:
            # No live lease covers this batch (expired, released by an
            # earlier failure, or never leased). Trace it — a silent
            # [False]*n here is what the engine logs as "failed to
            # retrieve page".
            self._trace(
                event="get_kv",
                lease_id="",
                req_id=req_id,
                pages=n,
                bytes=0,
                dur_ms=0.0,
                verdict="nolease",
                **self._trace_context_from(extra_info),
            )
            return [False] * n
        lease, idxs = match
        t0 = time.monotonic()
        if self._anchor_is_logical:
            # Logical KV anchor: zero bytes to move; the page count still
            # advances so the controller proceeds to the sidecar transfers.
            local_sizes: List[int] = []
            ok = True
        else:
            local_ptrs, local_sizes = _flatten_meta(
                *self.mem_pool_host.get_page_buffer_meta(host_indices)
            )
            ok = self._read_pages(
                lease=lease,
                pool_name=str(PoolName.KV),
                page_idxs=idxs,
                local_ptrs=local_ptrs,
                local_sizes=local_sizes,
            )
        self._trace(
            event="get_kv",
            lease_id=lease.lease_id,
            req_id=lease.req_id,
            pages=n,
            bytes=sum(local_sizes),
            dur_ms=(time.monotonic() - t0) * 1e3,
            verdict="ok" if ok else "fail",
            **self._lease_trace_context(lease),
        )
        if not ok:
            self._release(lease)
            return [False] * n
        lease.record_fetched(str(PoolName.KV), idxs)
        if lease.is_complete():
            self._release(lease, complete=True)
        return [True] * n

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        results: dict = {}
        lease_seen: Optional[_PeerLease] = None
        failed = False
        req_id = self._req_id_from(extra_info)
        for transfer in transfers:
            keys = transfer.keys or []
            n = len(keys)
            match = (
                self._match_lease_pages(keys, req_id, self._attempt_id_from(extra_info))
                if n
                else None
            )
            if match is None:
                self._trace(
                    event="get_pool",
                    lease_id="",
                    req_id=req_id,
                    pool=str(transfer.name),
                    pages=n,
                    bytes=0,
                    dur_ms=0.0,
                    verdict="nolease",
                    **self._trace_context_from(extra_info),
                )
                results[transfer.name] = [False] * n
                continue
            lease, idxs = match
            lease_seen = lease
            host_pool = self.registered_pools[transfer.name]
            t0 = time.monotonic()
            if not any(True for _ in _iter_host_pool_buffers(host_pool)):
                # Logical-anchor transfer (dsv4 "swa"): the local pool owns
                # indices but no bytes and its get_page_buffer_meta returns
                # None; the owner attested these pages with [] segments.
                # Count them as fetched — the bytes live in sidecar pools.
                local_sizes = []
                ok = True
            else:
                local_ptrs, local_sizes = _flatten_meta(
                    *host_pool.get_page_buffer_meta(transfer.host_indices)
                )
                ok = self._read_pages(
                    lease=lease,
                    pool_name=str(transfer.name),
                    page_idxs=idxs,
                    local_ptrs=local_ptrs,
                    local_sizes=local_sizes,
                )
            self._trace(
                event="get_pool",
                lease_id=lease.lease_id,
                req_id=lease.req_id,
                pool=str(transfer.name),
                pages=n,
                bytes=sum(local_sizes),
                dur_ms=(time.monotonic() - t0) * 1e3,
                verdict="ok" if ok else "fail",
                **self._lease_trace_context(lease),
            )
            results[transfer.name] = [ok] * n
            if ok:
                lease.record_fetched(str(transfer.name), idxs)
            else:
                failed = True
        # The controller calls batch_get_v2 once per KV batch (KV-derived
        # sidecars) and once more for the trailing pools after KV completes.
        # Release only when every expected pool is in (or on failure — the
        # engine's all-or-nothing verdict discards the prefetch anyway).
        # NOT after the first call: that killed every multi-batch pull at
        # batch 2 and every single-batch pull at the trailing SWA fetch.
        if lease_seen is not None and (failed or lease_seen.is_complete()):
            self._release(lease_seen, complete=not failed)
        return results

    # ------------------------------------------------------------------
    # SWA range backfill (DSv4 window residency, mem_cache/swa_backfill.py).
    # Called from the backfill manager's IO thread only, never from the
    # scheduler thread: every method may block on a peer round trip.
    # ------------------------------------------------------------------

    def _swa_range_addr(self, source_addr: str) -> Optional[str]:
        return self._target_addr({"source_servicer_addr": source_addr})

    def swa_range_capable(self, source_addr: str) -> bool:
        """Whether the owner rank paired with this reader answers
        LOOKUP_SWA_RANGE with this reader's page size. The PING result is
        cached; an incapable or unreachable owner is probed again after
        ``_SWA_RANGE_REPROBE_S``."""
        addr = self._swa_range_addr(source_addr)
        if addr is None:
            return False
        now = time.monotonic()
        cached = self._swa_range_caps.get(addr)
        if cached is not None and (
            cached[0] or now - cached[1] < self._SWA_RANGE_REPROBE_S
        ):
            return cached[0]
        reply = self._rpc(
            addr=addr, msg={"op": "ping"}, timeout_ms=self._lookup_timeout_ms
        )
        swa_pool = getattr(self, "registered_pools", {}).get(PoolName.SWA)
        local_page = getattr(swa_pool, "page_size", None)
        capable = bool(
            reply is not None
            and reply.get("ok")
            and SWA_RANGE_CAPABILITY in (reply.get("capabilities") or ())
            and (local_page is None or reply.get("page_size") == local_page)
        )
        self._swa_range_caps[addr] = (capable, now)
        self._trace(
            event="swa_range_ping",
            addr=addr,
            capable=capable,
            reachable=reply is not None,
        )
        return capable

    def lookup_swa_range(
        self,
        source_addr: str,
        page_keys: List[str],
        *,
        request_metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        """LOOKUP_SWA_RANGE on the paired owner rank. ``page_keys`` are the
        peer page keys (chain hashes in the request's RadixKey namespace) of
        the wanted range. Returns the grant (``num_pages`` may be 0) or None
        when the RPC failed."""
        addr = self._swa_range_addr(source_addr)
        if addr is None or not page_keys:
            return None
        metadata = dict(request_metadata or {})
        message = {
            "op": "lookup_swa_range",
            "hashes": list(page_keys),
            "lease_ttl_ms": self._lease_ttl_ms,
            "request_metadata": metadata,
        }
        started = time.monotonic()
        reply = self._rpc(addr=addr, msg=message, timeout_ms=self._lookup_timeout_ms)
        dur_ms = (time.monotonic() - started) * 1e3
        if reply is None or not reply.get("ok"):
            # Probe the owner again before the next attempt.
            self._swa_range_caps.pop(addr, None)
            if reply is not None:
                logger.warning(
                    "peer_kv: LOOKUP_SWA_RANGE error from %s: %s",
                    addr,
                    reply.get("error"),
                )
            self._trace(
                event="lookup_swa_range",
                lease_id="",
                req_id=metadata.get("req_id", ""),
                pages=0,
                query_pages=len(page_keys),
                dur_ms=dur_ms,
                verdict="error",
            )
            return None
        reply["_servicer_addr"] = addr
        reply["_lookup_started_mono"] = started
        if reply.get("refused_reason") == "unsupported":
            # The owner cannot serve ranges (no complete host SWA): cache it
            # as incapable until the re-probe interval.
            self._swa_range_caps[addr] = (False, time.monotonic())
        self._trace(
            event="lookup_swa_range",
            lease_id=reply.get("lease_id", ""),
            req_id=metadata.get("req_id", ""),
            pages=int(reply.get("num_pages", 0)),
            query_pages=len(page_keys),
            dur_ms=dur_ms,
            verdict="hit" if int(reply.get("num_pages", 0)) else "miss",
            owner_refused_reason=reply.get("refused_reason"),
        )
        return reply

    def _swa_range_lease(self, grant: dict, num_pages: int) -> _PeerLease:
        """Reader-side view of a range grant, never indexed for prefetch gets."""
        lease = _PeerLease(
            lease_id=grant.get("lease_id", ""),
            te_endpoint=grant.get("te_endpoint", ""),
            servicer_addr=grant.get("_servicer_addr", ""),
            pools={
                str(name): [
                    [(int(p), int(n)) for p, n in page] if page is not None else None
                    for page in pages
                ]
                for name, pages in (grant.get("pools") or {}).items()
            },
            keys=[],
            usable_pages=num_pages,
            expiry_mono=grant.get("_lookup_started_mono", time.monotonic())
            + min(
                self._lease_ttl_ms, int(grant.get("lease_ttl_ms", self._lease_ttl_ms))
            )
            / 1e3,
        )
        sums = grant.get("sums")
        if isinstance(sums, dict):
            lease.sums = {str(k): list(v) for k, v in sums.items()}
        lease.poison_check = bool(grant.get("poison", False))
        return lease

    def read_swa_range(
        self,
        grant: dict,
        num_pages: int,
        pool_names: List[Any],
        host_indices: torch.Tensor,
    ) -> bool:
        """RDMA-read the first ``num_pages`` granted pages of every
        SWA-anchored pool in ``pool_names`` into the local host slots
        ``host_indices`` (SWA-pool indices; the state pools share them).
        Any missing pool or segment mismatch fails the whole read.

        Lease expiry is checked like ``_match_lease_pages`` does for gets: a
        pool read starts only while the lease has more than
        ``_SWA_RANGE_READ_MARGIN_S`` left, and a read that ends past the
        expiry fails (the owner may have unpinned and reused the pages).
        A deterministic layout mismatch (a pool the owner does not serve or
        this reader did not register, different segment geometry) also marks
        the owner incapable, so the next lookup ends the backfill as
        no_capability instead of retrying until its deadline."""
        lease = self._swa_range_lease(grant, num_pages)
        page_idxs = list(range(num_pages))
        t0 = time.monotonic()
        nbytes = 0

        def fail(verdict: str, pool_name: str = "") -> bool:
            if verdict == "layout_mismatch" and lease.servicer_addr:
                self._swa_range_caps[lease.servicer_addr] = (False, time.monotonic())
            self._trace(
                event="get_swa_range",
                lease_id=lease.lease_id,
                pool=pool_name,
                pages=num_pages,
                bytes=0,
                dur_ms=(time.monotonic() - t0) * 1e3,
                verdict=verdict,
            )
            return False

        # Validate every pool before any transfer: a mismatch is deterministic.
        plan = []
        for name in pool_names:
            pool_name = str(name)
            pages = lease.pools.get(pool_name)
            host_pool = getattr(self, "registered_pools", {}).get(name)
            if pages is None or host_pool is None:
                return fail("layout_mismatch", pool_name)
            if len(pages) < num_pages or any(pages[i] is None for i in page_idxs):
                return fail("short_grant", pool_name)
            if not any(True for _ in _iter_host_pool_buffers(host_pool)):
                continue  # logical pool: attested, no bytes to move
            local_ptrs, local_sizes = _flatten_meta(
                *host_pool.get_page_buffer_meta(host_indices)
            )
            remote_sizes = [int(n) for i in page_idxs for _, n in pages[i]]
            if remote_sizes != [int(n) for n in local_sizes]:
                logger.warning(
                    "peer_kv: SWA range layout mismatch pool=%s owner=%s: remote "
                    "%d segments / %d bytes, local %d / %d; backfill off for "
                    "this owner",
                    pool_name,
                    lease.servicer_addr,
                    len(remote_sizes),
                    sum(remote_sizes),
                    len(local_sizes),
                    sum(int(n) for n in local_sizes),
                )
                return fail("layout_mismatch", pool_name)
            plan.append((pool_name, local_ptrs, local_sizes))
        for pool_name, local_ptrs, local_sizes in plan:
            if time.monotonic() >= lease.expiry_mono - self._SWA_RANGE_READ_MARGIN_S:
                return fail("expired", pool_name)
            ok = self._read_pages(
                lease=lease,
                pool_name=pool_name,
                page_idxs=page_idxs,
                local_ptrs=local_ptrs,
                local_sizes=local_sizes,
            )
            if not ok:
                return fail("fail", pool_name)
            if time.monotonic() > lease.expiry_mono:
                return fail("expired", pool_name)
            nbytes += sum(local_sizes)
        self._trace(
            event="get_swa_range",
            lease_id=lease.lease_id,
            pages=num_pages,
            bytes=nbytes,
            dur_ms=(time.monotonic() - t0) * 1e3,
            verdict="ok",
        )
        return True

    def release_swa_range(self, grant: Optional[dict]) -> None:
        """Give a range grant back (asynchronously, on the release worker)."""
        if not grant or not grant.get("lease_id"):
            return
        self._release_queue.put(self._swa_range_lease(grant, 0))

    # ------------------------------------------------------------------
    # write paths: acknowledged no-ops (see module docstring)
    # ------------------------------------------------------------------

    def _warn_set_once(self) -> None:
        if not self._warned_set:
            self._warned_set = True
            logger.warning(
                "peer_kv: storage write-through is an acknowledged NO-OP for "
                "this backend (peer serving tier is the owner's host pool; "
                "exists never consults a local store, so this is safe)."
            )

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        self._warn_set_once()
        return [True] * len(keys)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        self._warn_set_once()
        return {t.name: [True] * len(t.keys or []) for t in transfers}

    # ------------------------------------------------------------------
    # abstract single-key surface (unused by the zero-copy paths)
    # ------------------------------------------------------------------

    def get(self, key, target_location=None, target_sizes=None):
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)

    def set(self, key, value=None, target_location=None, target_sizes=None) -> bool:
        self._warn_set_once()
        return True

    def batch_set(
        self, keys, values=None, target_locations=None, target_sizes=None
    ) -> bool:
        self._warn_set_once()
        return True

    def exists(self, key: str) -> bool:
        return False

    def close(self) -> None:
        with self._lock:
            leases = list(self._leases.values())
        for lease in leases:
            self._release(lease)
        for sock in self._sockets.values():
            sock.close(linger=0)
        self._sockets.clear()
        if self._trace_file is not None:
            self._trace_file.close()

    # ------------------------------------------------------------------
    # trace
    # ------------------------------------------------------------------

    def _trace(self, **fields) -> None:
        if self._trace_file is None:
            return
        fields["ts"] = time.time()
        self._trace_file.write(_json_encoder.encode(fields) + b"\n")

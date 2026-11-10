"""Wire protocol of the symmetric P/D pull plane (owner <-> reader ZMQ side channel).

Covers chunk publication, the registration blob, reader address math, and late /
duplicate handling. The Owner publishes chunks, the Reader pulls them under the
Owner's Lease.

Serialization: msgspec msgpack dicts, the same encoding the existing P2P owner
Lease servicer and reader use (mem_cache/peer_kv_servicer.py,
mem_cache/storage/peer_kv/peer_kv_backend.py). Every message carries
``transfer_id`` and ``attempt_id`` so late or duplicated messages are dropped by
identity. Page lists are counted in pages, not tokens.

Message flow per TP rank (rank i <-> rank i, homogeneous TP):

    reader ---- REGISTER_REQ {reader_addr} ----------------------> owner
    reader <--- REGISTER     {owner registration blob} ----------- owner
    reader <--- CHUNK_READY  {chunk_seq, token range, pools} ----- owner   (xN)
    reader <--- FINAL        {aux_index, state_indices, gens} ---- owner
    reader ---- DONE | ABORT {reason} ---------------------------> owner

With SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER the reader also sends BEGIN {} when the
scheduler admits the import (pages allocated); the owner queues the P-leg for
prefill only after it (reader-first exports). BEGIN may overtake the owner's
begin_export and is then parked there. Owners that predate it reject the type,
so set the flag on every worker.

Late D (``--disaggregation-late-d``, disaggregation/late_d.py): owner rank 0
sends ADMIT {seq} to reader rank 0 when the held P-leg enters its lead window;
the reader then allocates the import and answers with BEGIN. ADMIT is sent by
the late-D sender thread, not by the OwnerPlane IO thread. Readers that
predate it fail to decode the type and drop it with a warning, so every
symmetric worker must run the same late-D mode (Registration.late_d).

REGISTER happens once per (owner, reader, tp_rank) pair and is cached by the
reader. It is the reverse of today's decode-side one-time ZMQ KVArgs
registration (mooncake/conn.py MooncakeKVReceiver._register_kv_args): for a
one-sided READ the READER needs the OWNER's base pointers.

No torch / zmq dependency, so protocol tests run anywhere.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

from decoupled_engine.srt.disaggregation.pull.api import ChunkReady, FinalMeta, PoolPages

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

MSG_REGISTER_REQ = "REGISTER_REQ"  # reader -> owner
MSG_REGISTER = "REGISTER"  # owner -> reader
MSG_CHUNK_READY = "CHUNK_READY"  # owner -> reader
MSG_FINAL = "FINAL"  # owner -> reader
MSG_DONE = "DONE"  # reader -> owner
MSG_ABORT = "ABORT"  # either direction
MSG_BEGIN = "BEGIN"  # reader -> owner: import admitted (reader-first exports)
MSG_ADMIT = "ADMIT"  # owner rank 0 -> reader rank 0: late D, allocate now
# owner rank 0 -> reader rank 0: late D stage 2, another candidate won (409)
MSG_RELEASE = "RELEASE"

OWNER_TO_READER = frozenset(
    {MSG_REGISTER, MSG_CHUNK_READY, MSG_FINAL, MSG_ABORT, MSG_ADMIT, MSG_RELEASE}
)
READER_TO_OWNER = frozenset({MSG_REGISTER_REQ, MSG_DONE, MSG_ABORT, MSG_BEGIN})
ALL_TYPES = OWNER_TO_READER | READER_TO_OWNER

# REGISTER_REQ / REGISTER belong to an (owner, reader, rank) pair, not a transfer.
REGISTRATION_TRANSFER_ID = 0

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder()


class ProtocolError(ValueError):
    """Malformed or unexpected wire message."""


def _validated_owner_cache_epoch(value: Optional[str]) -> Optional[str]:
    """Return the optional FINAL lifetime unchanged, or reject malformed metadata."""
    if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 128):
        raise ProtocolError(
            "owner_cache_epoch must be a non-empty string of at most 128 characters"
        )
    return value


# ---------------------------------------------------------------------------
# Registration blob: the owner's per-rank base pointers
# ---------------------------------------------------------------------------


class PoolRegion(msgspec.Struct, frozen=True):
    """One KV pool as the transfer engine sees it, parallel per-entry lists.

    ``ptrs[i]`` is the registered base pointer of entry i (a layer, or a K/V
    half of a layer for MHA pools; entry order is KVArgs.kv_data_ptrs order).
    ``page_nbytes[i]`` is the byte stride of one page index in entry i, so page
    ``p`` lives at ``ptrs[i] + p * page_nbytes[i]``. Page-granular form of
    mooncake/conn.py _send_kvcache_generic's ``ptr + token_index * item_len``;
    page_nbytes = item_len * page_size for token-major pools, DeepSeek-V4
    compressed pools override it via PoolLayout.page_nbytes.
    """

    ptrs: List[int]
    page_nbytes: List[int]
    layer_ids: List[int] = []

    def to_wire(self) -> Dict[str, Any]:
        return {
            "p": list(self.ptrs),
            "s": list(self.page_nbytes),
            "l": list(self.layer_ids),
        }

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> PoolRegion:
        return cls(
            ptrs=[int(x) for x in d["p"]],
            page_nbytes=[int(x) for x in d["s"]],
            layer_ids=[int(x) for x in d.get("l", [])],
        )


class ItemRegion(msgspec.Struct, frozen=True):
    """Aux (disagg metadata buffers) or per-request state tensors addressed by
    one item index: ``ptrs[i] + index * item_lens[i]`` (mooncake/conn.py
    send_aux / send_state address math)."""

    ptrs: List[int]
    item_lens: List[int]

    def to_wire(self) -> Dict[str, Any]:
        return {"p": list(self.ptrs), "i": list(self.item_lens)}

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> ItemRegion:
        return cls(ptrs=[int(x) for x in d["p"]], item_lens=[int(x) for x in d["i"]])


class PoolLayout(msgspec.Struct, frozen=True):
    """How a named pool ('full', 'swa', 'c4', 'c128', ...) maps onto entries of
    KVArgs.kv_data_ptrs. Supplied by the scheduler half, which owns the pool
    configuration (mem_cache/kv_cache_configurator.py). ``page_nbytes``
    overrides the derived ``kv_item_lens[e] * page_size`` stride."""

    name: str
    entry_indices: List[int]
    page_nbytes: Optional[List[int]] = None


class Registration(msgspec.Struct):
    """What a reader needs to turn the owner's page indices into remote
    addresses. Exchanged once per (owner, reader, tp_rank)."""

    session_id: str  # transfer-engine session (Mooncake "host:rpc_port")
    tp_rank: int
    attn_tp_size: int
    page_size: int
    pools: Dict[str, PoolRegion]
    aux: ItemRegion
    states: Dict[str, ItemRegion] = {}  # keyed by str(state_type)
    worker: str = ""
    version: int = PROTOCOL_VERSION
    # The owner's --disaggregation-late-d mode. Additive (same
    # PROTOCOL_VERSION): an older peer ignores the key, and a blob without it
    # decodes as "off". A mismatch never makes the pair incompatible
    # (check_compatible); the reader logs it (late_d_mismatch).
    late_d: str = "off"

    def to_wire(self) -> Dict[str, Any]:
        return {
            "v": self.version,
            "session_id": self.session_id,
            "tp_rank": self.tp_rank,
            "attn_tp_size": self.attn_tp_size,
            "page_size": self.page_size,
            "pools": {k: v.to_wire() for k, v in self.pools.items()},
            "aux": self.aux.to_wire(),
            "states": {k: v.to_wire() for k, v in self.states.items()},
            "worker": self.worker,
            "late_d": self.late_d,
        }

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> Registration:
        return cls(
            version=int(d.get("v", PROTOCOL_VERSION)),
            session_id=str(d["session_id"]),
            tp_rank=int(d["tp_rank"]),
            attn_tp_size=int(d["attn_tp_size"]),
            page_size=int(d["page_size"]),
            pools={str(k): PoolRegion.from_wire(v) for k, v in d["pools"].items()},
            aux=ItemRegion.from_wire(d["aux"]),
            states={
                str(k): ItemRegion.from_wire(v) for k, v in d.get("states", {}).items()
            },
            worker=str(d.get("worker", "")),
            late_d=str(d.get("late_d", "off")),
        )

    @classmethod
    def from_kv_args(
        cls,
        *,
        kv_args: Any,
        session_id: str,
        attn_tp_size: int,
        pool_layouts: Optional[Sequence[PoolLayout]] = None,
        worker: str = "",
        default_pool: str = "full",
        late_d: str = "off",
    ) -> Registration:
        """Build from a disaggregation ``KVArgs`` (base/conn.py). Without
        ``pool_layouts`` every kv_data_ptrs entry belongs to one pool named
        ``default_pool`` (plain MHA/MLA); multi-component
        pools pass explicit layouts."""
        page_size = int(kv_args.page_size)
        kv_ptrs = [int(p) for p in kv_args.kv_data_ptrs]
        kv_item_lens = [int(n) for n in kv_args.kv_item_lens]
        kv_layer_ids = [int(lid) for lid in kv_args.kv_layer_ids]
        if pool_layouts is None:
            pool_layouts = [
                PoolLayout(name=default_pool, entry_indices=list(range(len(kv_ptrs))))
            ]
        pools = {
            layout.name: _pool_region_from_layout(
                layout=layout,
                kv_ptrs=kv_ptrs,
                kv_item_lens=kv_item_lens,
                kv_layer_ids=kv_layer_ids,
                page_size=page_size,
            )
            for layout in pool_layouts
        }
        aux = ItemRegion(
            ptrs=[int(p) for p in kv_args.aux_data_ptrs],
            item_lens=[int(n) for n in kv_args.aux_item_lens],
        )
        states = {
            str(st): ItemRegion(
                ptrs=[int(p) for p in kv_args.state_data_ptrs[i]],
                item_lens=[int(n) for n in kv_args.state_item_lens[i]],
            )
            for i, st in enumerate(kv_args.state_types)
        }
        return cls(
            session_id=session_id,
            tp_rank=int(kv_args.engine_rank),
            attn_tp_size=int(attn_tp_size),
            page_size=page_size,
            pools=pools,
            aux=aux,
            states=states,
            worker=worker,
            late_d=late_d,
        )


def _pool_region_from_layout(
    *,
    layout: PoolLayout,
    kv_ptrs: List[int],
    kv_item_lens: List[int],
    kv_layer_ids: List[int],
    page_size: int,
) -> PoolRegion:
    idx = list(layout.entry_indices)
    if layout.page_nbytes is not None:
        strides = [int(s) for s in layout.page_nbytes]
        if len(strides) != len(idx):
            raise ProtocolError(
                f"PoolLayout {layout.name}: {len(strides)} strides for {len(idx)} entries"
            )
    else:
        strides = [kv_item_lens[e] * page_size for e in idx]
    return PoolRegion(
        ptrs=[kv_ptrs[e] for e in idx],
        page_nbytes=strides,
        layer_ids=[kv_layer_ids[e] for e in idx if e < len(kv_layer_ids)],
    )


def check_compatible(*, local: Registration, remote: Registration) -> Optional[str]:
    """Homogeneous-TP / identical-layout precondition.
    Returns a failure reason or None."""
    if local.version != remote.version:
        return f"protocol_version {local.version}!={remote.version}"
    if local.tp_rank != remote.tp_rank:
        return f"tp_rank {local.tp_rank}!={remote.tp_rank} (homogeneous TP only)"
    if local.attn_tp_size != remote.attn_tp_size:
        return f"attn_tp_size {local.attn_tp_size}!={remote.attn_tp_size}"
    if local.page_size != remote.page_size:
        return f"page_size {local.page_size}!={remote.page_size}"
    for name, lp in local.pools.items():
        rp = remote.pools.get(name)
        if rp is None:
            return f"pool {name} missing on owner"
        if len(lp.ptrs) != len(rp.ptrs) or lp.page_nbytes != rp.page_nbytes:
            return f"pool {name} layout mismatch"
    if local.aux.item_lens != remote.aux.item_lens:
        return "aux layout mismatch"
    for name, ls in local.states.items():
        rs = remote.states.get(name)
        if rs is None or ls.item_lens != rs.item_lens:
            return f"state {name} layout mismatch"
    # Registration.late_d is not a layout property: late_d_mismatch reports it.
    return None


def late_d_mismatch(*, local: Registration, remote: Registration) -> Optional[str]:
    """The two sides run different --disaggregation-late-d modes ("mixed
    late-D flags"): the reason, or None. Never a reason to
    refuse the pair; the reader logs it."""
    if local.late_d == remote.late_d:
        return None
    return f"late_d {local.late_d}!={remote.late_d}"


# ---------------------------------------------------------------------------
# Address math, pure functions unit-tested on CPU
# ---------------------------------------------------------------------------


def _contiguous_runs(
    src: Sequence[int], dst: Sequence[int]
) -> List[Tuple[int, int, int]]:
    """Runs contiguous on both sides as (src0, dst0, n); numpy-free
    group_concurrent_contiguous (disaggregation/utils.py)."""
    if len(src) != len(dst):
        raise ProtocolError(f"{len(src)} remote pages vs {len(dst)} local pages")
    runs: List[Tuple[int, int, int]] = []
    if not src:
        return runs
    s0, d0, n = int(src[0]), int(dst[0]), 1
    for k in range(1, len(src)):
        s, d = int(src[k]), int(dst[k])
        if s == s0 + n and d == d0 + n:
            n += 1
        else:
            runs.append((s0, d0, n))
            s0, d0, n = s, d, 1
    runs.append((s0, d0, n))
    return runs


def build_page_read_plan(
    *,
    local: PoolRegion,
    remote: PoolRegion,
    local_pages: Sequence[int],
    remote_pages: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    """READ plan for one pool: (local_ptrs, remote_ptrs, lengths), one triple per
    (entry, contiguous run). local = READ destination (reader), remote = READ
    source (owner): the mirror of mooncake/conn.py _send_kvcache_generic."""
    if len(local.ptrs) != len(remote.ptrs):
        raise ProtocolError(
            f"entry count mismatch: local {len(local.ptrs)} vs remote {len(remote.ptrs)}"
        )
    runs = _contiguous_runs(remote_pages, local_pages)
    local_ptrs: List[int] = []
    remote_ptrs: List[int] = []
    lengths: List[int] = []
    for e in range(len(local.ptrs)):
        stride = int(local.page_nbytes[e])
        if stride != int(remote.page_nbytes[e]):
            raise ProtocolError(
                f"entry {e}: page stride {stride}!={remote.page_nbytes[e]}"
            )
        lbase, rbase = int(local.ptrs[e]), int(remote.ptrs[e])
        for r0, l0, n in runs:
            local_ptrs.append(lbase + l0 * stride)
            remote_ptrs.append(rbase + r0 * stride)
            lengths.append(n * stride)
    return local_ptrs, remote_ptrs, lengths


def build_item_read_plan(
    *,
    local: ItemRegion,
    remote: ItemRegion,
    local_index: int,
    remote_index: int,
) -> Tuple[List[int], List[int], List[int]]:
    """READ plan for one aux/state slot: every tensor i of the region."""
    if len(local.ptrs) != len(remote.ptrs) or local.item_lens != remote.item_lens:
        raise ProtocolError("item region layout mismatch")
    local_ptrs, remote_ptrs, lengths = [], [], []
    for i in range(len(local.ptrs)):
        n = int(local.item_lens[i])
        local_ptrs.append(int(local.ptrs[i]) + n * int(local_index))
        remote_ptrs.append(int(remote.ptrs[i]) + n * int(remote_index))
        lengths.append(n)
    return local_ptrs, remote_ptrs, lengths


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Message(msgspec.Struct):
    """Decoded wire message; ``payload`` holds the type-specific fields."""

    type: str
    transfer_id: int
    attempt_id: int
    payload: Dict[str, Any] = {}

    def chunk(self) -> ChunkReady:
        if self.type != MSG_CHUNK_READY:
            raise ProtocolError(f"not a CHUNK_READY: {self.type}")
        p = self.payload
        return ChunkReady(
            transfer_id=self.transfer_id,
            chunk_seq=int(p["chunk_seq"]),
            token_start=int(p["token_start"]),
            token_end=int(p["token_end"]),
            pools=[
                PoolPages(
                    pool=str(q["pool"]),
                    page_indices=[int(x) for x in q["pages"]],
                    generations=[int(x) for x in q["gens"]],
                )
                for q in p["pools"]
            ],
            is_last=bool(p.get("is_last", False)),
            tier=str(p.get("tier", "gpu")),
        )

    def final(self) -> FinalMeta:
        if self.type != MSG_FINAL:
            raise ProtocolError(f"not a FINAL: {self.type}")
        p = self.payload
        addr = p.get("owner_peer_kv_addr")
        return FinalMeta(
            transfer_id=self.transfer_id,
            total_tokens=int(p["total_tokens"]),
            aux_index=int(p["aux_index"]),
            state_indices={
                str(k): [int(x) for x in v]
                for k, v in p.get("state_indices", {}).items()
            },
            # Optional: owners that predate it (or run no servicer) omit it.
            owner_peer_kv_addr=str(addr) if addr else None,
            owner_cache_epoch=_validated_owner_cache_epoch(p.get("owner_cache_epoch")),
        )

    def final_generations(self) -> Dict[str, Dict[int, int]]:
        """FINAL's owner-side generation snapshot {pool: {page: generation}};
        the reader compares it with CHUNK_READY (a mismatch means
        the owner evicted or reused the page)."""
        if self.type != MSG_FINAL:
            raise ProtocolError(f"not a FINAL: {self.type}")
        out: Dict[str, Dict[int, int]] = {}
        for pool, pairs in self.payload.get("gens", {}).items():
            out[str(pool)] = {int(pg): int(g) for pg, g in zip(pairs[0], pairs[1])}
        return out

    def registration(self) -> Registration:
        if self.type != MSG_REGISTER:
            raise ProtocolError(f"not a REGISTER: {self.type}")
        return Registration.from_wire(self.payload["reg"])

    @property
    def reason(self) -> str:
        return str(self.payload.get("reason", ""))

    @property
    def owner_addr(self) -> str:
        return str(self.payload.get("owner_addr", ""))

    @property
    def reader_addr(self) -> str:
        return str(self.payload.get("reader_addr", ""))

    @property
    def seq(self) -> int:
        """ADMIT: the owner's send sequence (a resend repeats it)."""
        return int(self.payload.get("seq", 0))


def encode(msg: Message) -> bytes:
    if msg.type not in ALL_TYPES:
        raise ProtocolError(f"unknown message type {msg.type!r}")
    d: Dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "t": msg.type,
        "tid": int(msg.transfer_id),
        "aid": int(msg.attempt_id),
    }
    d.update(msg.payload)
    return _encoder.encode(d)


def decode(raw: bytes) -> Message:
    try:
        d = _decoder.decode(raw)
    except Exception as e:
        raise ProtocolError(f"undecodable message: {e}") from e
    if not isinstance(d, dict):
        raise ProtocolError("message is not a map")
    if int(d.get("v", -1)) != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {d.get('v')} != {PROTOCOL_VERSION}")
    t = d.get("t")
    if t not in ALL_TYPES:
        raise ProtocolError(f"unknown message type {t!r}")
    try:
        tid = int(d["tid"])
        aid = int(d["aid"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProtocolError("missing transfer_id/attempt_id") from e
    payload = {k: v for k, v in d.items() if k not in ("v", "t", "tid", "aid")}
    msg = Message(type=t, transfer_id=tid, attempt_id=aid, payload=payload)
    _validate(msg)
    return msg


def _validate(msg: Message) -> None:
    p = msg.payload
    if msg.type == MSG_CHUNK_READY:
        for k in ("chunk_seq", "token_start", "token_end", "pools"):
            if k not in p:
                raise ProtocolError(f"CHUNK_READY missing {k}")
        if int(p["token_end"]) < int(p["token_start"]):
            raise ProtocolError("CHUNK_READY token range inverted")
        for q in p["pools"]:
            if len(q["pages"]) != len(q["gens"]):
                raise ProtocolError(
                    f"CHUNK_READY pool {q.get('pool')}: "
                    f"{len(q['pages'])} pages, {len(q['gens'])} gens"
                )
    elif msg.type == MSG_FINAL:
        for k in ("total_tokens", "aux_index"):
            if k not in p:
                raise ProtocolError(f"FINAL missing {k}")
    elif msg.type == MSG_REGISTER:
        if "reg" not in p or "owner_addr" not in p:
            raise ProtocolError("REGISTER missing reg/owner_addr")
    elif msg.type == MSG_REGISTER_REQ:
        if "reader_addr" not in p:
            raise ProtocolError("REGISTER_REQ missing reader_addr")
    elif msg.type in (MSG_ADMIT, MSG_RELEASE):
        seq = p.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ProtocolError(
                f"{msg.type} needs a non-negative integer seq, got {seq!r}"
            )


def make_register_req(
    *, reader_addr: str, tp_rank: int, reader_worker: str = ""
) -> Message:
    return Message(
        type=MSG_REGISTER_REQ,
        transfer_id=REGISTRATION_TRANSFER_ID,
        attempt_id=0,
        payload={
            "reader_addr": reader_addr,
            "tp_rank": int(tp_rank),
            "worker": reader_worker,
        },
    )


def make_register(*, reg: Registration, owner_addr: str) -> Message:
    """``owner_addr`` is the owner's advertised channel (ImportMark.owner_addr),
    the key under which the reader caches this registration."""
    return Message(
        type=MSG_REGISTER,
        transfer_id=REGISTRATION_TRANSFER_ID,
        attempt_id=0,
        payload={"reg": reg.to_wire(), "owner_addr": owner_addr},
    )


def make_chunk_ready(*, chunk: ChunkReady, attempt_id: int) -> Message:
    return Message(
        type=MSG_CHUNK_READY,
        transfer_id=chunk.transfer_id,
        attempt_id=attempt_id,
        payload={
            "chunk_seq": int(chunk.chunk_seq),
            "token_start": int(chunk.token_start),
            "token_end": int(chunk.token_end),
            "is_last": bool(chunk.is_last),
            "tier": chunk.tier,
            "pools": [
                {
                    "pool": pp.pool,
                    "pages": [int(x) for x in pp.page_indices],
                    "gens": [int(x) for x in pp.generations],
                }
                for pp in chunk.pools
            ],
        },
    )


def make_final(
    *,
    final: FinalMeta,
    attempt_id: int,
    generations: Optional[Mapping[str, Mapping[int, int]]] = None,
) -> Message:
    """``generations`` is the owner's current {pool: {page: gen}} for every page it
    published; the reader verifies it against CHUNK_READY."""
    gens_wire: Dict[str, List[List[int]]] = {}
    for pool, m in (generations or {}).items():
        pages = [int(k) for k in m.keys()]
        gens_wire[str(pool)] = [pages, [int(m[k]) for k in m.keys()]]
    payload = {
        "total_tokens": int(final.total_tokens),
        "aux_index": int(final.aux_index),
        "state_indices": {
            str(k): [int(x) for x in v] for k, v in final.state_indices.items()
        },
        "gens": gens_wire,
    }
    if final.owner_peer_kv_addr:
        # Additive and optional (same PROTOCOL_VERSION): readers that predate
        # it ignore unknown payload keys; owners without a servicer omit it.
        payload["owner_peer_kv_addr"] = str(final.owner_peer_kv_addr)
    epoch = _validated_owner_cache_epoch(final.owner_cache_epoch)
    if epoch is not None:
        payload["owner_cache_epoch"] = epoch
    return Message(
        type=MSG_FINAL,
        transfer_id=final.transfer_id,
        attempt_id=attempt_id,
        payload=payload,
    )


def make_done(*, transfer_id: int, attempt_id: int) -> Message:
    return Message(
        type=MSG_DONE, transfer_id=transfer_id, attempt_id=attempt_id, payload={}
    )


def begin_reuse_prefix(payload: Dict[str, Any]) -> int:
    """Optional capability; malformed/old messages conservatively advertise zero."""
    value = payload.get("reusable_prefix_tokens", 0)
    return value if type(value) is int and 0 <= value < (1 << 31) else 0


def make_begin(*, transfer_id: int, attempt_id: int, reusable_prefix_tokens: int = 0) -> Message:
    """Reader -> owner: the import is admitted. ``attempt_id`` 0 when the
    reader has no attempt bound yet (legacy marks); the owner accepts it."""
    return Message(
        type=MSG_BEGIN, transfer_id=transfer_id, attempt_id=attempt_id,
        payload=({"reusable_prefix_tokens": reusable_prefix_tokens}
                 if attempt_id > 0 and reusable_prefix_tokens > 0 else {}),
    )


def make_admit(
    *, transfer_id: int, attempt_id: int, seq: int, kv_session: Any = None
) -> Message:
    """Owner rank 0 -> reader rank 0 (late D): allocate the import now.
    ``attempt_id`` is the owner's attempt for the transfer (stage 1: the
    export's attempt; stage 2: the winner's ``a_w``); ``seq`` is the owner's
    send sequence, repeated by a resend. ``kv_session`` (optional,
    SGLANG_KV_SESSION_ADMIT_TAG): the P leg's ``{session_id, version}`` tag
    for the reader's D-leg registration; absent, the payload is unchanged."""
    payload = {"seq": int(seq)}
    if kv_session is not None:
        payload["kv_session"] = dict(kv_session)
    return Message(
        type=MSG_ADMIT,
        transfer_id=transfer_id,
        attempt_id=attempt_id,
        payload=payload,
    )


def admit_kv_session(msg: Message) -> Optional[Tuple[str, int]]:
    """The optional ADMIT ``kv_session`` tag as ``(session_id, version)``.
    None when absent or invalid: a bad tag never drops the ADMIT (the D leg
    then runs untagged)."""
    if msg.type != MSG_ADMIT:
        return None
    value = msg.payload.get("kv_session")
    if value is None:
        return None
    from decoupled_engine.srt.mem_cache.unified_cache.session_ref_tracker import (
        parse_kv_session,
    )

    try:
        return parse_kv_session(value)
    except ValueError:
        logger.warning(
            "pull reader: ADMIT for transfer %d has an invalid kv_session tag; "
            "the D leg runs untagged",
            msg.transfer_id,
        )
        return None


def make_release(*, transfer_id: int, attempt_id: int, seq: int) -> Message:
    """Owner rank 0 -> reader rank 0 (late D stage 2): another candidate won
    the transfer (or it runs Local); the reader ends its DEFERRED import with
    409 and sends no ABORT. ``attempt_id`` is this candidate's ``a_i``;
    ``seq`` is the owner's send sequence, repeated by a resend."""
    return Message(
        type=MSG_RELEASE,
        transfer_id=transfer_id,
        attempt_id=attempt_id,
        payload={"seq": int(seq)},
    )


def make_abort(*, transfer_id: int, attempt_id: int, reason: str) -> Message:
    return Message(
        type=MSG_ABORT,
        transfer_id=transfer_id,
        attempt_id=attempt_id,
        payload={"reason": str(reason)},
    )


# ---------------------------------------------------------------------------
# Late / duplicate handling
# ---------------------------------------------------------------------------


class AttemptGate:
    """Per-transfer attempt binding shared by owner and reader.

    A transfer_id may be retried by the router with a new attempt (new
    begin_export, new attempt_id). Each side binds a transfer to the first
    attempt it sees and drops everything else: unknown transfer_id (late
    CHUNK_READY after the reader gave up, duplicate DONE after the owner
    released) and attempt mismatches are counted and ignored.
    """

    ACCEPT = "accept"
    DROP_UNKNOWN = "drop_unknown_transfer"
    DROP_ATTEMPT = "drop_attempt_mismatch"
    # A reader that gives up before the owner's first message has no attempt to
    # quote; its ABORT carries this and matches whatever attempt is bound.
    WILDCARD_ATTEMPT = 0

    def __init__(self) -> None:
        self._bound: Dict[int, int] = {}
        self.dropped: Dict[str, int] = {self.DROP_UNKNOWN: 0, self.DROP_ATTEMPT: 0}

    def bind(self, *, transfer_id: int, attempt_id: int) -> None:
        self._bound[int(transfer_id)] = int(attempt_id)

    def unbind(self, transfer_id: int) -> None:
        self._bound.pop(int(transfer_id), None)

    def attempt_of(self, transfer_id: int) -> Optional[int]:
        return self._bound.get(int(transfer_id))

    def check(self, msg: Message) -> str:
        bound = self._bound.get(int(msg.transfer_id))
        if bound is None:
            self.dropped[self.DROP_UNKNOWN] += 1
            return self.DROP_UNKNOWN
        if msg.type == MSG_ABORT and int(msg.attempt_id) == self.WILDCARD_ATTEMPT:
            return self.ACCEPT
        if bound != int(msg.attempt_id):
            self.dropped[self.DROP_ATTEMPT] += 1
            return self.DROP_ATTEMPT
        return self.ACCEPT

"""NCCL-free host-dedup fan-out over CUDA IPC (opt-in).

With host dedup only the source rank (TP0) owns a host pool. Today each
load-back layer is fanned out with an NCCL broadcast, which forces every model
forward to wait for the whole load (the forward's TP collectives are not NCCL,
so their order against the broadcast cannot be guaranteed otherwise).

This module replaces the broadcast with a *pull* over CUDA IPC:

- The source gathers host rows into its own device rows (unchanged) and, per
  layer, records an interprocess CUDA event.
- Each follower's load stream waits on that event (a stream-level wait, no
  spinning kernel) and then reads the source's rows directly from the source's
  IPC-mapped device layer buffers (peer ``index_select`` over NVLink into a
  local staging buffer) and scatters them into its own rows (``index_copy_``).
- The source's row indices travel through a source-owned, IPC-shared rows
  buffer (one region per slot) written on the source's load stream before the
  first layer, so physical page IDs may differ per rank.
- ``cudaStreamWaitEvent`` binds to the most recent record at call time, so the
  source enqueues its whole op (rows copy, gathers, records) before one CPU
  barrier on the dedup gloo group; followers enqueue their waits after it.
  Slots are reused only after at least one more barrier.
- Before reusing a slot's rows region, the source stream-waits on each
  follower's per-slot "done" event (normally already complete).

Source rows of an op stay allocated and locked until the MIN-reduced load
acknowledgement, i.e. until every follower finished reading them.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# Producer-ring slots 0..2 serve admission loads (LayerDoneCounter), slot 3 the
# bootstrap-window prefetch, slot 4 everything else (e.g. retraction restore).
NUM_SLOTS = 5
SLOT_BOOTSTRAP_PREFETCH = 3
SLOT_MISC = 4
# Source row ids shared with followers (device token indices < 2**31).
ROWS_DTYPE = torch.int32


def ipc_handle_bytes(handle: bytes) -> bytes:
    """Return the raw 64-byte cudaIpcMemHandle_t from torch's shareable handle.

    torch's caching allocator prefixes the CUDA handle with a version byte and
    a type byte ('c' = cudaMalloc segment); expandable segments use a different
    scheme and are rejected.
    """
    if len(handle) == 64:
        return handle
    if len(handle) == 65:
        kind, raw = handle[0:1], handle[1:]
    elif len(handle) == 66:
        kind, raw = handle[1:2], handle[2:]
    else:
        raise ValueError(f"Unexpected CUDA IPC handle length {len(handle)}")
    if kind != b"c":
        raise ValueError(
            "Dedup IPC fan-out requires cudaMalloc-backed KV buffers "
            f"(handle type {kind!r}; expandable segments are unsupported)"
        )
    return raw


def export_tensor(t: torch.Tensor) -> dict:
    """Describe a CUDA tensor for IPC: segment handle + byte offset + shape."""
    if not t.is_cuda or not t.is_contiguous():
        raise ValueError("Dedup IPC fan-out requires contiguous CUDA buffers")
    storage = t.untyped_storage()
    shared = storage._share_cuda_()
    handle, storage_offset_bytes = shared[1], shared[3]
    return dict(
        handle=ipc_handle_bytes(bytes(handle)),
        offset=int(storage_offset_bytes) + t.storage_offset() * t.element_size(),
        nbytes=t.numel() * t.element_size(),
        rows=int(t.shape[0]),
        row_bytes=(t.numel() * t.element_size()) // max(1, int(t.shape[0])),
        data_ptr=t.data_ptr(),
    )


SIDECAR_DRAFT_KIND = "draft_kv"


def layer_buffers(
    kv_pool, draft_pools: Sequence, sidecar_pools: Sequence = ()
) -> Dict[Tuple[str, int], torch.Tensor]:
    """(kind, host-layer id) -> device buffer, mirroring DSAHostDedup._layer_buffer.

    ``draft_pools``: packed MTP draft pools (host layers after the target's).
    ``"idx"`` entries exist only for pools with a DSA indexer (plain MLA has
    none). ``sidecar_pools``: separate HiCache draft pools (EAGLE3) with their
    own host pool; their layers are ``("draft_kv", layer)`` so they never
    collide with target layer ids (see ce_2d_load.peer_plan)."""
    out: Dict[Tuple[str, int], torch.Tensor] = {}
    pools = [(kv_pool, layer) for layer in range(kv_pool.layer_num)]
    pools += [(pool, 0) for pool in draft_pools]
    for host_layer, (pool, layer) in enumerate(pools):
        out[("kv", host_layer)] = pool.kv_buffer[layer]
        idx = getattr(pool, "index_k_with_scale_buffer", None)
        if idx is not None and len(idx):
            out[("idx", host_layer)] = idx[layer]
    for pool in sidecar_pools:
        for layer in range(pool.layer_num):
            out[(SIDECAR_DRAFT_KIND, layer)] = pool.kv_buffer[layer]
    return out


class DedupIPCFanout:
    def __init__(
        self,
        *,
        cpu_group,
        src_global_rank: int,
        device,
        kv_pool,
        draft_pools: Sequence,
        page_size: int,
        staging_bytes: int,
        num_layer_events: Optional[int] = None,
        cudart=None,
    ):
        self.group = cpu_group
        self.src_global_rank = src_global_rank
        self.is_src = torch.distributed.get_rank() == src_global_rank
        self.device = torch.device(device)
        self.device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.page_size = page_size
        self.staging_bytes = staging_bytes
        self.ce_loader = None  # source only, SGLANG_DSA_HOST_DEDUP_CE_LOAD
        memory_saver = getattr(kv_pool, "memory_saver_adapter", None)
        if memory_saver is not None and getattr(memory_saver, "enabled", False):
            raise ValueError("Dedup IPC fan-out is incompatible with memory saver")

        self.local_layers = layer_buffers(kv_pool, draft_pools)
        self.num_layer_events = num_layer_events or (
            kv_pool.layer_num + len(tuple(draft_pools))
        )
        self.rows_families, self.rows_capacity = self.capacity_for(
            self.local_layers, kv_pool.size, page_size
        )
        self._slot_used = [False] * NUM_SLOTS
        self._opened: Dict[bytes, int] = {}
        self._keepalive: List[object] = []
        self._cudart = cudart
        self.peer_layers: Dict[Tuple[str, int], torch.Tensor] = {}
        self._staging: Dict[int, torch.Tensor] = {}
        self._first_data_ptr = self.local_layers[("kv", 0)].data_ptr()

        # Source-owned objects: rows buffer and per-slot, per-layer events.
        if self.is_src:
            self.rows = torch.zeros(
                (NUM_SLOTS, self.rows_capacity), dtype=ROWS_DTYPE, device=self.device
            )
            self.layer_events = [
                [
                    torch.cuda.Event(
                        enable_timing=False, blocking=False, interprocess=True
                    )
                    for _ in range(self.num_layer_events)
                ]
                for _ in range(NUM_SLOTS)
            ]
            # An interprocess event must be recorded before its handle is used.
            for slot_events in self.layer_events:
                for ev in slot_events:
                    ev.record()
            payload = dict(
                layers={key: export_tensor(t) for key, t in self.local_layers.items()},
                rows=export_tensor(self.rows),
                events=[[ev.ipc_handle() for ev in s] for s in self.layer_events],
            )
            torch.cuda.current_stream().synchronize()
        else:
            payload = None
        box = [payload]
        torch.distributed.broadcast_object_list(
            box, src=self.src_global_rank, group=self.group
        )
        payload = box[0]

        # Followers map the source's buffers and events; each follower creates
        # its own per-slot "done" events, which the source opens.
        if not self.is_src:
            for key, spec in payload["layers"].items():
                self.peer_layers[key] = self._map(spec)
            self.peer_rows = self._map(payload["rows"], dtype=ROWS_DTYPE).view(
                NUM_SLOTS, self.rows_capacity
            )
            self.layer_events = [
                [
                    torch.cuda.Event.from_ipc_handle(self.device_index, handle)
                    for handle in slot
                ]
                for slot in payload["events"]
            ]
            self.done_events = [
                torch.cuda.Event(enable_timing=False, blocking=False, interprocess=True)
                for _ in range(NUM_SLOTS)
            ]
            for ev in self.done_events:
                ev.record()
            torch.cuda.current_stream().synchronize()
            mine = [ev.ipc_handle() for ev in self.done_events]
        else:
            mine = None
        gathered: List[Optional[list]] = [
            None for _ in range(torch.distributed.get_world_size(self.group))
        ]
        torch.distributed.all_gather_object(gathered, mine, group=self.group)
        if self.is_src:
            self.follower_done_events = [
                [
                    torch.cuda.Event.from_ipc_handle(self.device_index, handle)
                    for handle in handles
                ]
                for handles in gathered
                if handles is not None
            ]
        self._verify(payload)
        logger.info(
            "Host dedup IPC fan-out ready: role=%s layers=%d slots=%d rows_capacity=%d",
            "source" if self.is_src else "follower",
            len(self.local_layers),
            NUM_SLOTS,
            self.rows_capacity,
        )

    # ------------------------------------------------------------ mapping
    def _rt(self):
        if self._cudart is None:
            from decoupled_engine.srt.distributed.device_communicators.cuda_wrapper import (
                CudaRTLibrary,
            )

            self._cudart = CudaRTLibrary()
        return self._cudart

    def _open(self, raw: bytes) -> int:
        base = self._opened.get(raw)
        if base is None:
            from decoupled_engine.srt.distributed.device_communicators.cuda_wrapper import (
                cudaIpcMemHandle_t,
            )

            handle = cudaIpcMemHandle_t()
            ctypes.memmove(ctypes.addressof(handle), raw, len(raw))
            with torch.cuda.device(self.device_index):
                ptr = self._rt().cudaIpcOpenMemHandle(handle)
            base = int(ptr.value if hasattr(ptr, "value") else ptr)
            self._opened[raw] = base
        return base

    def _map(self, spec: dict, dtype=torch.uint8) -> torch.Tensor:
        from decoupled_engine.srt.utils.cuda_vmm_utils import tensor_from_pointer

        ptr = self._open(spec["handle"]) + spec["offset"]
        view = tensor_from_pointer(ptr, spec["nbytes"], device_id=self.device_index)
        if dtype is torch.uint8:
            return view.view(spec["rows"], spec["row_bytes"])
        return view.view(dtype)

    def _verify(self, payload) -> None:
        """Source writes a pattern into its rows buffer; followers read it back."""
        if self.is_src:
            self.rows.copy_(
                torch.arange(
                    self.rows.numel(), dtype=ROWS_DTYPE, device=self.device
                ).view_as(self.rows)
            )
            torch.cuda.current_stream().synchronize()
        torch.distributed.barrier(group=self.group)
        if not self.is_src:
            probe = self.peer_rows[:, :8].clone().cpu()
            expect = torch.arange(
                NUM_SLOTS * self.rows_capacity, dtype=ROWS_DTYPE
            ).view(NUM_SLOTS, self.rows_capacity)[:, :8]
            if not torch.equal(probe, expect):
                raise RuntimeError("Dedup IPC fan-out: rows buffer mapping mismatch")
            for key, t in self.local_layers.items():
                peer = self.peer_layers[key]
                if (
                    peer.shape[0] != t.shape[0]
                    or peer.numel() != t.numel() * t.element_size()
                ):
                    raise RuntimeError(f"Dedup IPC fan-out: layer {key} shape mismatch")
        torch.distributed.barrier(group=self.group)

    def _staging_for(self, stream) -> torch.Tensor:
        key = stream.cuda_stream if hasattr(stream, "cuda_stream") else id(stream)
        buf = self._staging.get(key)
        if buf is None:
            buf = torch.empty(self.staging_bytes, dtype=torch.uint8, device=self.device)
            self._staging[key] = buf
        return buf

    # ------------------------------------------------------------ per op
    def check_buffers(self) -> None:
        if self.local_layers[("kv", 0)].data_ptr() != self._first_data_ptr:
            raise RuntimeError("Dedup IPC fan-out: KV buffers were reallocated")

    @staticmethod
    def kind(host_pool) -> str:
        from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

        return "idx" if isinstance(host_pool, DSAIndexerPoolHost) else "kv"

    @staticmethod
    def capacity_for(local_layers, pool_size: int, page_size: int) -> Tuple[int, int]:
        """(families, rows per slot). One rows segment per distinct
        device-index tensor of an op: draft (MTP) transfers reuse their target
        transfer's tensor, but the DSA indexer transfer carries its own
        (equal-valued) tensor, so a T-token op needs T rows per family
        ("kv", "idx") and at most pool_size + page_size per family."""
        families = len({kind for kind, _ in local_layers}) or 1
        return families, families * (int(pool_size) + int(page_size))

    @staticmethod
    def distinct_indices(transfers) -> List[Tuple[torch.Tensor, int]]:
        """(device_indices, offset) per distinct index tensor, in transfer order."""
        out: List[Tuple[torch.Tensor, int]] = []
        offset = 0
        for t in transfers:
            di = t.device_indices
            if di is None or any(di is seen for seen, _ in out):
                continue
            out.append((di, offset))
            offset += di.numel()
        return out

    def src_begin(self, slot: int, transfers, stream) -> None:
        self.check_buffers()
        if self._slot_used[slot]:
            for follower in self.follower_done_events:
                stream.wait_event(follower[slot])
        self._slot_used[slot] = True
        distinct = self.distinct_indices(transfers)
        for di, offset in distinct:
            n = di.numel()
            if offset + n > self.rows_capacity:
                raise RuntimeError(
                    "Dedup IPC fan-out: op exceeds rows capacity "
                    f"({len(distinct)} index tensors, {offset + n} rows > "
                    f"{self.rows_capacity})"
                )
            self.rows[slot, offset : offset + n].copy_(
                di.to(device=self.device, dtype=ROWS_DTYPE), non_blocking=True
            )

    def src_layer_done(self, slot: int, layer_id: int, stream) -> None:
        self.layer_events[slot][layer_id].record(stream)

    def order(self) -> None:
        """CPU barrier: every source record of this op precedes follower waits."""
        torch.distributed.barrier(group=self.group)

    def dst_begin(self, slot: int, transfers, stream) -> dict:
        self.check_buffers()
        if self.num_layer_events <= 0:
            return {}
        stream.wait_event(self.layer_events[slot][0])
        src_rows = {}
        for di, offset in self.distinct_indices(transfers):
            src_rows[id(di)] = self.peer_rows[
                slot, offset : offset + di.numel()
            ].clone()
        return src_rows

    def dst_layer(
        self,
        slot: int,
        layer_id: int,
        layer_transfers,
        src_rows: dict,
        stream,
        local_layer_buffer,
        local_layer_rows,
    ) -> None:
        stream.wait_event(self.layer_events[slot][layer_id])
        staging = self._staging_for(stream)
        for transfer, local_layer_id in layer_transfers:
            di = transfer.device_indices
            if di is None or di.numel() == 0:
                continue
            kind = self.kind(transfer.host_pool)
            src_tok = src_rows[id(di)]
            if kind == "idx":
                src = src_tok.reshape(-1, self.page_size)[:, 0] // self.page_size
            else:
                src = src_tok
            dst = local_layer_rows(transfer.host_pool, di)
            peer = self.peer_layers[(kind, local_layer_id)]
            layer = local_layer_buffer(transfer.host_pool, local_layer_id)
            layer = layer.view(torch.uint8).reshape(layer.shape[0], -1)
            self.copy_rows(peer, src, layer, dst, staging)

    @staticmethod
    def copy_rows(peer, src_rows, layer, dst_rows, staging) -> None:
        row_bytes = layer.shape[1]
        per_chunk = staging.numel() // row_bytes
        if per_chunk == 0:
            raise ValueError("Dedup IPC fan-out staging is smaller than one row")
        for start in range(0, src_rows.numel(), per_chunk):
            s = src_rows[start : start + per_chunk]
            d = dst_rows[start : start + per_chunk]
            chunk = staging[: s.numel() * row_bytes].view(-1, row_bytes)
            torch.index_select(peer, 0, s, out=chunk)
            layer.index_copy_(0, d, chunk)

    def dst_end(self, slot: int, stream) -> None:
        self.done_events[slot].record(stream)

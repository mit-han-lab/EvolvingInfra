"""One CPU copy of replicated DSA KV and indexer, including packed MTP layers."""

import logging
import time

import torch

from decoupled_engine.srt.mem_cache.dsv4_host_dedup import (
    DeepSeekV4HostDedup,
    build_host_dedup,
)

logger = logging.getLogger(__name__)

_DIGEST_MOD = (1 << 61) - 1
_DIGEST_MUL = 1_000_003
_OP_SEP = 0x5EED


class DSAHostDedup(DeepSeekV4HostDedup):
    # Host-dedup kind for the pool NUMA-split auto policy
    # (pool_host/numa_split_policy.py).
    dedup_kind = "dsa"
    # C2g (FANOUT=ce_peer); class defaults so partially built objects
    # (tests, other fan-outs) read as "off".
    ce_peer = False
    peer = None
    # Separate HiCache draft pools mapped by C2g (MLA EAGLE3; mla_host_dedup.py).
    _sidecar_pools = ()

    def __init__(
        self,
        group,
        src_global_rank,
        device,
        fanout: str = "nccl",
        kv_pool=None,
        draft_pools=(),
        page_size=None,
    ):
        super().__init__(group, src_global_rank, device, prewarm=fanout == "nccl")
        self._symm_handle = None
        self.ipc_fanout = None
        self.shared = fanout == "shared"
        # C2g: the source loads its private pool with copy engines and
        # followers pull its rows with copy-engine peer copies, CPU-gated.
        self.ce_peer = fanout == "ce_peer"
        from decoupled_engine.srt.environ import envs

        self.loader = None
        self.verify_group = None
        self.peer = None
        if self.ce_peer:
            from decoupled_engine.srt.mem_cache.ce_2d_load import COST
            from decoupled_engine.srt.mem_cache.ce_peer import CePeer
            from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import layer_buffers

            # Every rank logs its costs: followers' peer gate / pull costs too.
            COST.enabled = True
            self.peer = CePeer(
                group=group,
                src_global_rank=src_global_rank,
                is_src=self.is_src,
                local_layers=layer_buffers(
                    kv_pool, draft_pools, getattr(self, "_sidecar_pools", ())
                ),
                topology=str(envs.SGLANG_DSA_HOST_DEDUP_PEER_TOPOLOGY.get()).lower(),
            )
            self._start_loader(device)
            if self.loader is None:
                raise RuntimeError(
                    "SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer needs the ce_submit native "
                    "helper on every rank (followers have no other load path)"
                )
        elif self.shared:
            # Every rank maps the source's pool (pool_host/shared_dedup.py) and
            # loads its own rows with copy engines: nothing to fan out.
            from decoupled_engine.srt.mem_cache.ce_2d_load import COST

            COST.enabled = self.is_src
            if envs.SGLANG_DSA_HOST_DEDUP_SHARED_LOADER.get():
                self._start_loader(device)
        elif fanout == "ipc":
            from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import DedupIPCFanout

            # `group` is a gloo group here: CPU ordering and handle exchange.
            self.ipc_fanout = DedupIPCFanout(
                cpu_group=group,
                src_global_rank=src_global_rank,
                device=device,
                kv_pool=kv_pool,
                draft_pools=draft_pools,
                page_size=page_size,
                staging_bytes=self.STAGING_BYTES,
            )
            if self.is_src and envs.SGLANG_DSA_HOST_DEDUP_CE_LOAD.get():
                from decoupled_engine.srt.mem_cache.staged_ce_load import StagedCELoader

                self.ipc_fanout.ce_loader = StagedCELoader(
                    page_size,
                    envs.SGLANG_DSA_HOST_DEDUP_CE_STAGING_MB.get() * 1024**2,
                    device,
                )
        elif envs.SGLANG_DSA_HOST_DEDUP_SYMMETRIC_BROADCAST.get():
            from torch._C._distributed_c10d import _SymmetricMemory

            # Reuse PyTorch's mapped peer buffer. Each follower keeps its own
            # 32 MiB destination; rank 0 remains the only CPU cache owner.
            device = torch.device("cuda", torch.cuda.current_device())
            self.staging = _SymmetricMemory.empty_strided_p2p(
                (self.STAGING_BYTES,), (1,), torch.uint8, device, group.group_name
            )
            self._symm_handle = _SymmetricMemory.rendezvous(self.staging)
            source = torch.distributed.get_process_group_ranks(group).index(
                src_global_rank
            )
            self._source_staging = self._symm_handle.get_buffer(
                source, (self.STAGING_BYTES,), torch.uint8
            )
            self.staging.zero_()
            self._symm_handle.barrier(channel=0, timeout_ms=30000)
            torch.cuda.current_stream().synchronize()

    def _start_loader(self, device) -> None:
        """v0.2: a per-rank loader thread with its own gloo group for the
        row check (the scheduler thread keeps ``self.group``)."""
        from decoupled_engine.srt.distributed.parallel_state import create_custom_parallel_group
        from decoupled_engine.srt.environ import envs
        from decoupled_engine.srt.mem_cache import ce_2d_load
        from decoupled_engine.srt.mem_cache.ce_loader import SharedLoader
        from decoupled_engine.srt.utils import get_device_module

        ranks = torch.distributed.get_process_group_ranks(self.group)
        # Build the native submit helper now (not on the first load). Without
        # it a loader thread would loop per copy in Python and pay a GIL
        # hand-off per call: keep v0.1 (scheduler thread) instead. The
        # decision is all-reduced so every rank takes the same path.
        ok = torch.tensor([int(ce_2d_load.native() is not None)], dtype=torch.int64)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=self.group
        )
        if not int(ok.item()):
            logger.error(
                "Shared host dedup: ce_submit native helper unavailable on some "
                "rank; loader thread disabled (v0.1 scheduler-thread submission)"
            )
            return
        self.verify_group = create_custom_parallel_group(ranks, backend="gloo")
        dm = get_device_module()
        self.loader = SharedLoader(
            dm.current_device(),
            self,
            device_module=dm,
            chunk_pages=envs.SGLANG_DSA_HOST_DEDUP_LOADER_CHUNK_PAGES.get(),
        )
        logger.info(
            "Shared host dedup: loader thread on device %s, %d pages per chunk",
            dm.current_device(),
            self.loader.chunk_pages,
        )

    def check_host_indices(
        self, transfers, group=None, field: str = "host_indices"
    ) -> None:
        """Shared pool: every rank must load the same host rows, in order.

        Followers read the source's pages by their own host indices; those are
        TP-consistent by construction (lockstep tree and allocator decisions),
        and this check makes a violation fail loudly instead of reading the
        wrong pages. One gloo all-reduce of a few int64 per load op, or, with
        ``SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY_EVERY=N`` > 1, one per N ops over
        a running digest (a divergence then surfaces up to N-1 ops late).
        """
        from decoupled_engine.srt.environ import envs

        if (
            not (self.shared or self.ce_peer)
            or not envs.SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY.get()
        ):
            return
        t0 = time.perf_counter()
        vals = [len(transfers)]
        for transfer in transfers:
            h = getattr(transfer, field)
            if h is None:
                vals += [-1, 0, 0]
                continue
            n = h.numel()
            page = int(
                getattr(getattr(transfer, "host_pool", None), "page_size", 1) or 1
            )
            h = h.detach().reshape(-1)
            if page > 1 and n % page == 0:
                # Page-granular allocation: page starts identify the rows
                # (64x less work per op than hashing every token).
                h = h[::page]
            h = h.to("cpu", torch.int64)
            w = torch.arange(1, h.numel() + 1, dtype=torch.int64)
            vals += [n, int(h.sum()), int((h * w).sum())]
        every = max(1, envs.SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY_EVERY.get())
        st = self.__dict__.setdefault(
            "_verify",
            dict(ops=0, digest=0, n=0, sum_ms=0.0, max_ms=0.0, t=time.monotonic()),
        )
        st["ops"] += 1
        if every > 1:
            d = st["digest"]
            for v in vals + [_OP_SEP]:
                d = (d * _DIGEST_MUL + v) % _DIGEST_MOD
            st["digest"] = d
            if st["ops"] % every:
                self._verify_account(st, t0)
                return
            vals = [st["ops"], d]
        x = torch.tensor(vals, dtype=torch.int64)
        both = torch.cat([x, -x])
        torch.distributed.all_reduce(
            both, op=torch.distributed.ReduceOp.MAX, group=group or self.group
        )
        hi, lo = both[: x.numel()], -both[x.numel() :]
        if not torch.equal(hi, lo):
            what = (
                "rows"
                if every == 1
                else f"rows within the last {every} ops (op count, digest)"
            )
            raise RuntimeError(
                f"Shared host-dedup pool: TP ranks would load different host {what} "
                f"(local {x.tolist()}, max {hi.tolist()}, min {lo.tolist()})"
            )
        self._verify_account(st, t0)

    def check_part_digest(self, digests, n_parts: int, group=None) -> None:
        """Overlap M1 (M-i), with SHARED_VERIFY: every rank splits a layer-wise
        op into the same parts with the same copy composition, which the
        follower's per-chunk gate assumes. ``digests``: per part (n2d, n1d,
        first page, last page, run heights), or None on a rank that loads the
        op on the per-layer fallback (it matches anything). One gloo all-reduce
        per layer-wise op, on the loader thread in op order on every rank."""
        from decoupled_engine.srt.environ import envs

        if (
            not (self.shared or self.ce_peer)
            or not envs.SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY.get()
        ):
            return
        n = int(n_parts) * 5
        big = 1 << 62
        if digests is None:
            both = torch.full((2 * n,), -big, dtype=torch.int64)
        else:
            x = torch.tensor([v for d in digests for v in d], dtype=torch.int64)
            if x.numel() != n:
                raise RuntimeError(
                    f"Shared host-dedup: {x.numel() // 5} layer parts, expected {n // 5}"
                )
            both = torch.cat([x, -x])
        torch.distributed.all_reduce(
            both, op=torch.distributed.ReduceOp.MAX, group=group or self.group
        )
        hi, lo = both[:n], -both[n:]
        if digests is not None and not torch.equal(hi, lo):
            raise RuntimeError(
                "Shared host-dedup: TP ranks split a layer-wise op differently "
                f"(local {digests}, max {hi.tolist()}, min {lo.tolist()})"
            )

    def _verify_account(self, st, t0) -> None:
        """Scheduler-thread cost of the check (CPU work plus the cross-rank wait)."""
        from decoupled_engine.srt.mem_cache.ce_2d_load import COST

        dt = (time.perf_counter() - t0) * 1e3
        st["n"] += 1
        st["sum_ms"] += dt
        st["max_ms"] = max(st["max_ms"], dt)
        COST.add("verify", dt)

    def _broadcast_staging(self, chunk) -> None:
        if self._symm_handle is None:
            return super()._broadcast_staging(chunk)
        # All callers pass a contiguous view at the start of staging. Barriers
        # and the peer read use the load stream, independent of model NCCL.
        self._symm_handle.barrier(channel=0, timeout_ms=30000)
        if not self.is_src:
            source = self._source_staging[: chunk.numel()]
            if torch.autograd.profiler._is_profiler_enabled:
                with torch.profiler.record_function("kv_symm_remote_read"):
                    chunk.view(-1).copy_(source)
            else:
                chunk.view(-1).copy_(source)
        # Rank 0 must not gather the next layer until all readers have finished.
        self._symm_handle.barrier(channel=1, timeout_ms=30000)

    @staticmethod
    def _layer_buffer(host_pool, layer_id):
        device_pool = host_pool.device_pool
        if layer_id >= device_pool.layer_num:
            device_pool = host_pool.mtp_draft_device_pools[
                layer_id - device_pool.layer_num
            ]
            layer_id = 0
        from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

        if isinstance(host_pool, DSAIndexerPoolHost):
            return device_pool.index_k_with_scale_buffer[layer_id]
        return device_pool.kv_buffer[layer_id]

    def _layer_rows(self, host_pool, device_indices):
        from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

        if isinstance(host_pool, DSAIndexerPoolHost):
            rows = device_indices.reshape(-1, host_pool.page_size)[:, 0]
            rows = rows // host_pool.page_size
        else:
            rows = device_indices
        return rows.to(device=self.device, dtype=torch.int64)

    def broadcast_layer(self, host_pool, device_indices, layer_id: int) -> None:
        if (
            self.shared
            or self.ce_peer
            or device_indices is None
            or device_indices.numel() == 0
        ):
            return
        self.broadcast_rows(
            self._layer_buffer(host_pool, layer_id),
            self._layer_rows(host_pool, device_indices),
        )

    def prepare_layer_broadcast(self, transfers):
        """Pack a layer's KV and indexer only when both fit the staging buffer."""
        from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
        from decoupled_engine.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        if self.shared or self.ce_peer:
            return None
        if len(transfers) != 2 or not (
            isinstance(transfers[0][0].host_pool, MLATokenToKVPoolHost)
            and isinstance(transfers[1][0].host_pool, DSAIndexerPoolHost)
        ):
            return None
        parts, offset = [], 0
        for transfer, layer_id in transfers:
            indices = transfer.device_indices
            if indices is None or indices.numel() == 0:
                return None
            layer = self._layer_buffer(transfer.host_pool, layer_id)
            if not layer.is_contiguous():
                raise ValueError("Host dedup requires contiguous device rows")
            count = indices.numel()
            if isinstance(transfer.host_pool, DSAIndexerPoolHost):
                count //= transfer.host_pool.page_size
            row_bytes = layer.numel() * layer.element_size() // layer.shape[0]
            size = count * row_bytes
            if offset + size > self.staging.numel():
                return None
            layer = layer.view(torch.uint8).reshape(layer.shape[0], -1)
            parts.append((layer, offset, size))
            offset += size
        return [
            (
                layer,
                self._layer_rows(transfer.host_pool, transfer.device_indices),
                self.staging[start : start + size].view(-1, layer.shape[1]),
            )
            for (transfer, _), (layer, start, size) in zip(transfers, parts)
        ]

    def broadcast_prepared_layer(self, parts):
        if self.is_src:
            for layer, rows, chunk in parts:
                torch.index_select(layer, 0, rows, out=chunk)
        size = sum(chunk.numel() for _, _, chunk in parts)
        self._broadcast_staging(self.staging[:size])
        if not self.is_src:
            for layer, rows, chunk in parts:
                layer.index_copy_(0, rows, chunk)


def maybe_build_dsa_host_dedup(params, kvcache, storage_backend, extra_config):
    from decoupled_engine.srt.runtime_context import get_memory

    if not get_memory().enable_dsa_hicache_host_dedup:
        return None
    from decoupled_engine.srt.mem_cache.memory_pool import DSATokenToKVPool

    if not isinstance(kvcache, DSATokenToKVPool) or kvcache.layer_shard_enabled:
        raise ValueError("DSA host dedup requires replicated DSA device pools")
    from decoupled_engine.srt.environ import envs

    fanout = envs.SGLANG_DSA_HOST_DEDUP_FANOUT.get()
    if fanout != "ipc" and envs.SGLANG_DSA_HOST_DEDUP_CE_LOAD.get():
        raise ValueError("SGLANG_DSA_HOST_DEDUP_CE_LOAD requires the IPC fan-out")
    if fanout == "shared":
        if envs.SGLANG_DSA_HOST_DEDUP_SYMMETRIC_BROADCAST.get():
            raise ValueError(
                "SGLANG_DSA_HOST_DEDUP_FANOUT=shared has no broadcast to replace"
            )
        # One pool mapped by every rank; each rank loads with copy engines.
        return build_host_dedup(
            params,
            kvcache,
            storage_backend,
            extra_config,
            DSAHostDedup,
            backend="gloo",
            io_layout=("direct", "page_first_direct"),
            fanout="shared",
        )
    if fanout == "nccl":
        return build_host_dedup(
            params, kvcache, storage_backend, extra_config, DSAHostDedup
        )
    if fanout == "ce_peer":
        from decoupled_engine.srt.arg_groups.symmetric_defaults import apply_gb300_dedup_tuning

        apply_gb300_dedup_tuning("SGLANG_DSA_HOST_DEDUP_FANOUT")
        if envs.SGLANG_DSA_HOST_DEDUP_SYMMETRIC_BROADCAST.get():
            raise ValueError(
                "SGLANG_DSA_HOST_DEDUP_FANOUT=ce_peer has no broadcast to replace"
            )
        return build_host_dedup(
            params,
            kvcache,
            storage_backend,
            extra_config,
            DSAHostDedup,
            backend="gloo",
            io_layout=("direct", "page_first_direct"),
            fanout="ce_peer",
            kv_pool=kvcache,
            draft_pools=_draft_pools(params),
            page_size=params.page_size,
        )
    if fanout != "ipc":
        raise ValueError(
            f"SGLANG_DSA_HOST_DEDUP_FANOUT must be nccl, ipc, shared or ce_peer: {fanout!r}"
        )
    if envs.SGLANG_DSA_HOST_DEDUP_SYMMETRIC_BROADCAST.get():
        raise ValueError(
            "SGLANG_DSA_HOST_DEDUP_FANOUT=ipc replaces the symmetric broadcast"
        )
    draft_pools = _draft_pools(params)
    return build_host_dedup(
        params,
        kvcache,
        storage_backend,
        extra_config,
        DSAHostDedup,
        backend="gloo",
        fanout="ipc",
        kv_pool=kvcache,
        draft_pools=draft_pools,
        page_size=params.page_size,
    )


def _draft_pools(params) -> tuple:
    return tuple(
        pool
        for pool in getattr(params, "mtp_draft_device_pools", ())
        if pool.index_k_with_scale_buffer
    )

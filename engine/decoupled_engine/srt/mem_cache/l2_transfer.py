# Adapted from SGLang python/sglang/srt/mem_cache/l2_transfer.py (commit 9a489f8d2f), modified for evolving-infra.
from __future__ import annotations

import logging
import time
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    kwargs = {"enable_timing": True} if timing_enabled else {}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), timing_enabled


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool
    # Shared host-dedup loader job (ce_loader) when the events are recorded
    # later by the loader thread; the events above are then GatedEvents.
    job: Any = None


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    def submit_device_to_host(
        self, transfers: list[L2Transfer], *, sm_copy: bool = False
    ) -> TransferCompletion:
        """``sm_copy``: the caller checked that every host pool here supports
        an SM-kernel backup (``supports_sm_copy_backup``) and moved the
        indices for it (host indices on the device)."""
        start_event = self._start_event(None)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        with device_module.stream(self.device_to_host_stream):
            start_event.wait(self.device_to_host_stream)
            ack_start.record()
            for transfer in transfers:
                if getattr(transfer.host_pool, "_is_dummy", False):
                    continue
                if sm_copy:
                    transfer.host_pool.backup_from_device_all_layer(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        self.io_backend,
                        sm_copy=True,
                    )
                    continue
                transfer.host_pool.backup_from_device_all_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
            ack_finish.record()
            self._record_stream(transfers, self.device_to_host_stream)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        layer_num: int,
        start_event=None,
        on_layer_done=None,
        wait_on_caller: bool = True,
        defer_broadcast: bool = False,
        stream=None,
        ipc_slot: Optional[int] = None,
        host_indices_cpu: Optional[dict] = None,
        fence_event=None,
        background: bool = False,
        layer_events=None,
        q3: bool = False,
        q3_dirty_event=None,
    ) -> TransferCompletion:
        """``q3`` (M4 load quarantine): the destination pages are clean, or
        clean once ``q3_dirty_event`` (forward_done of their release epoch)
        completes; the shared loader then skips ``start_event`` and the fence.
        Only the shared loader may take such an op (L-g)."""
        start_event = self._start_event(start_event)
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        primary = transfers[0] if transfers else None
        dedup = getattr(primary.host_pool, "host_dedup", None) if primary else None
        fanout = getattr(dedup, "ipc_fanout", None) if dedup is not None else None
        if q3 and fanout is not None:
            raise AssertionError("M4 Q3 load routed to the IPC fan-out")
        if fanout is not None:
            return self._submit_host_to_device_ipc(
                transfers,
                dedup=dedup,
                fanout=fanout,
                layer_num=layer_num,
                start_event=start_event,
                ack=(ack_start, ack_finish, timing_enabled),
                on_layer_done=on_layer_done,
                wait_on_caller=wait_on_caller,
                stream=stream if stream is not None else self.host_to_device_stream,
                ipc_slot=ipc_slot,
                host_indices_cpu=host_indices_cpu,
            )
        shared = getattr(dedup, "shared", False)
        ce_peer = getattr(dedup, "ce_peer", False)
        loader = getattr(dedup, "loader", None) if (shared or ce_peer) else None
        if loader is not None:
            return self._submit_shared_loader(
                transfers,
                loader,
                layer_num=layer_num,
                start_event=start_event,
                fence_event=fence_event,
                ack=(ack_start, ack_finish, timing_enabled),
                on_layer_done=on_layer_done,
                background=background,
                layer_events=layer_events,
                q3=q3,
                q3_dirty_event=q3_dirty_event,
            )
        if q3:
            raise AssertionError("M4 Q3 load without the shared loader")
        h2d = self.host_to_device_stream
        if stream is not None and stream is not h2d:
            if dedup is not None and not shared:
                # NCCL broadcasts must stay in one stream order on every rank.
                raise ValueError("A custom H2D stream requires the dedup IPC fan-out")
            h2d = stream
        if shared:
            t_submit = time.perf_counter()
            dedup.check_host_indices(transfers)
            plans = self._shared_2d_plans(transfers, layer_num, primary)
            if plans is not None:
                return self._submit_shared_2d(
                    transfers,
                    plans,
                    layer_num=layer_num,
                    h2d=h2d,
                    start_event=start_event,
                    ack=(ack_start, ack_finish, timing_enabled),
                    on_layer_done=on_layer_done,
                    wait_on_caller=wait_on_caller,
                    t_submit=t_submit,
                )
        prepare = getattr(dedup, "prepare_layer_broadcast", None)
        if not all(
            getattr(t.host_pool, "host_dedup", None) is dedup for t in transfers
        ):
            prepare = None
        with device_module.stream(h2d):
            start_event.wait(h2d)
            ack_start.record()
            deferred_layers = []
            for layer_id in range(layer_num):
                layer_transfers = []
                for transfer in transfers:
                    local_layer_id = (
                        transfer.layer_mapper(layer_id)
                        if transfer.layer_mapper is not None
                        else layer_id
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and layer_id >= transfer.host_pool.layer_num
                    ):
                        continue
                    layer_transfers.append((transfer, local_layer_id))
                packed = (
                    prepare(layer_transfers)
                    if prepare is not None and not defer_broadcast
                    else None
                )
                for transfer, local_layer_id in layer_transfers:
                    if not getattr(transfer.host_pool, "_is_dummy", False) or getattr(
                        transfer.host_pool, "_shared_reader", False
                    ):
                        transfer.host_pool.load_to_device_per_layer(
                            transfer.device_pool,
                            transfer.host_indices,
                            transfer.device_indices,
                            local_layer_id,
                            self.io_backend,
                            is_draft=transfer.is_draft,
                        )
                    transfer_dedup = getattr(transfer.host_pool, "host_dedup", None)
                    if (
                        transfer_dedup is not None
                        and not getattr(transfer_dedup, "shared", False)
                        and packed is None
                        and not defer_broadcast
                    ):
                        transfer_dedup.broadcast_layer(
                            transfer.host_pool, transfer.device_indices, local_layer_id
                        )
                if packed is not None:
                    dedup.broadcast_prepared_layer(packed)
                if defer_broadcast:
                    deferred_layers.append((layer_id, layer_transfers))
                elif on_layer_done is not None:
                    on_layer_done(layer_id)
            # NCCL launch ordering can delay the first broadcast behind the
            # current model's TP collectives. For lookahead, copy every layer
            # first so that wait cannot serialize the remaining H2D copies.
            for layer_id, layer_transfers in deferred_layers:
                packed = prepare(layer_transfers) if prepare is not None else None
                if packed is not None:
                    dedup.broadcast_prepared_layer(packed)
                else:
                    for transfer, local_layer_id in layer_transfers:
                        transfer_dedup = getattr(transfer.host_pool, "host_dedup", None)
                        if transfer_dedup is not None and not getattr(
                            transfer_dedup, "shared", False
                        ):
                            transfer_dedup.broadcast_layer(
                                transfer.host_pool,
                                transfer.device_indices,
                                local_layer_id,
                            )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
            ack_finish.record()
            self._record_stream(transfers, h2d)
        if wait_on_caller and any(
            getattr(t.host_pool, "host_dedup", None) is not None for t in transfers
        ):
            # Reuse the MLA collective-order fix: subsequent model TP collectives
            # must wait for the dedicated load communicator, without blocking CPU.
            # Shared pool (v0): the wait is on this rank's own copy-engine op
            # only (no cross-rank dependency). Layer-wise overlap needs the MTP
            # draft layer to wait too (its pool has no layer counter); v1.
            from decoupled_engine.srt.managers.cache_controller import _FWD_WAIT

            # M0 (LOAD_TIMING): the caller-stream stall this wait causes; on D
            # every running decode request's next step waits for it.
            t = _FWD_WAIT.begin()
            ack_finish.wait(device_module.current_stream())
            _FWD_WAIT.end(t, "caller_load_wait_gpu_ms")
            _FWD_WAIT.resolve()
        if shared:
            from decoupled_engine.srt.mem_cache.ce_2d_load import COST

            COST.add("submit_per_layer", (time.perf_counter() - t_submit) * 1e3)
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def _submit_shared_loader(
        self,
        transfers,
        loader,
        *,
        layer_num,
        start_event,
        fence_event,
        ack,
        on_layer_done,
        background: bool = False,
        layer_events=None,
        q3: bool = False,
        q3_dirty_event=None,
    ) -> TransferCompletion:
        """Shared pool, v0.2: hand the op to the per-rank loader thread.

        Nothing waits here: the consuming forward gates on the job at launch
        (LayerDoneCounter.set_consumer) and every returned event is a
        GatedEvent until the loader has submitted the op.

        Layer-wise (overlap M1, ``SGLANG_DSA_HOST_DEDUP_LAYER_PARTS`` > 1): a
        ring op (``layer_events`` = its producer slot) is copied in layer
        parts, and the forward waits only for the part of each layer it reads.
        Every rank derives the same parts from env + layer_num (I4).
        """
        from decoupled_engine.srt.mem_cache.ce_loader import (
            GatedEvent,
            LoadJob,
            layer_part_bounds,
        )

        ack_start, ack_finish, timing_enabled = ack
        parts = None
        part_plan = "page"
        # Ring ops (M1); background ops stay whole unless the caller handed a
        # detached part slot (M5 3.3: a bootstrap-prefetch op that a forward
        # may adopt in flight and then waits for per part).
        if layer_events is not None and (
            not background or getattr(layer_events, "detached", False)
        ):
            parts = layer_part_bounds(
                layer_num,
                envs.SGLANG_DSA_HOST_DEDUP_LAYER_PARTS.get(),
                envs.SGLANG_DSA_HOST_DEDUP_LAYER_PART_BOUNDS.get(),
            )
            # Copy shapes of the parts (same env on every rank, I4).
            part_plan = str(envs.SGLANG_DSA_HOST_DEDUP_PART_PLAN.get()).lower()
        part_events = (
            layer_events.set_parts(parts) if layer_events is not None else None
        )

        def plan_or_load(trs, plan_only):
            if plan_only:
                return self._shared_2d_plans(
                    trs, layer_num, trs[0] if trs else None, parts=parts
                )
            self._shared_per_layer_loads(trs, layer_num)
            return None

        def peer_plans(trs):
            return self._peer_2d_plans(
                trs, layer_num, loader.dedup.peer.ptrs, parts=parts
            )

        job = LoadJob(
            transfers,
            layer_num,
            start_event,
            fence_event,
            ack_start,
            ack_finish,
            on_layer_done,
            plan_or_load,
            peer_plans=peer_plans,
            background=background,
            parts=parts,
            part_events=part_events,
            gate_timeout_s=getattr(loader, "gate_timeout_s", 120.0),
            part_plan=part_plan,
            q3=q3,
            dirty_event=q3_dirty_event,
        )
        loader.submit(job)
        return TransferCompletion(
            GatedEvent(ack_start, job), GatedEvent(ack_finish, job), timing_enabled, job
        )

    def _shared_per_layer_loads(self, transfers, layer_num) -> None:
        """Per-(page, layer) fallback, run on the loader thread's stream."""
        if not getattr(self, "_loader_fallback_logged", False):
            self._loader_fallback_logged = True
            logger.warning(
                "Shared host dedup loader: per-layer fallback (sgl_kernel direct "
                "op; may hold the GIL while it blocks)"
            )
        primary = transfers[0] if transfers else None
        for _layer_id, layer_transfers in self._layer_plan(
            transfers, layer_num, primary
        ):
            for transfer, local_layer_id in layer_transfers:
                if getattr(transfer.host_pool, "_is_dummy", False) and not getattr(
                    transfer.host_pool, "_shared_reader", False
                ):
                    continue
                transfer.host_pool.load_to_device_per_layer(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    local_layer_id,
                    self.io_backend,
                    is_draft=transfer.is_draft,
                )

    def _part_local_ids(self, transfers, layer_num, primary, parts=None):
        """Per part, ``{id(transfer): [local layer ids]}``; one dict (the
        whole op) when ``parts`` is None. Layer-wise, a transfer's rows of
        global layers [l0, l1) go to that part (KV and indexer alike, I6), and
        MTP draft transfers go to the last part: ``_layer_plan`` lists the
        draft under layer 0 (its layer_mapper maps only layer 0), but the draft
        forward reads it after every target layer, so stream order covers it."""
        n = len(parts) if parts else 1
        out = [{} for _ in range(n)]
        part_of = None
        if parts:
            part_of = [0] * layer_num
            for p, (l0, l1) in enumerate(parts):
                for layer in range(l0, l1):
                    part_of[layer] = p
        for layer_id, layer_transfers in self._layer_plan(
            transfers, layer_num, primary
        ):
            for transfer, local_layer_id in layer_transfers:
                if part_of is None:
                    p = 0
                elif transfer.is_draft:
                    p = n - 1
                else:
                    p = part_of[layer_id]
                out[p].setdefault(id(transfer), []).append(local_layer_id)
        return out

    def _peer_2d_plans(self, transfers, layer_num, peer_ptrs, parts=None):
        """C2g follower: one D2D plan per transfer from the source's GPU, or
        None. Layer-wise (``parts``): one plan list per part."""
        from decoupled_engine.srt.mem_cache import ce_2d_load

        primary = transfers[0] if transfers else None
        by_part = []
        for local_ids in self._part_local_ids(transfers, layer_num, primary, parts):
            plans = []
            for transfer in transfers:
                ids = local_ids.get(id(transfer))
                if not ids:
                    continue
                if parts and not ce_2d_load.has_rows(transfer, ids):
                    continue  # only skip-topk placeholders in this part
                p = ce_2d_load.peer_plan(transfer, ids, peer_ptrs)
                if p is None:
                    logger.error(
                        "C2g follower: no peer-copy plan for %s -> %s",
                        type(transfer.host_pool).__name__,
                        type(transfer.device_pool).__name__,
                    )
                    return None
                plans.append(p)
            by_part.append(plans)
        return by_part if parts else by_part[0]

    def _shared_2d_plans(self, transfers, layer_num, primary, parts=None):
        """One 2D plan per loaded transfer (ce_2d_load), or None -> per-layer
        path. Layer-wise (``parts``): one plan list per part."""
        if (
            self.io_backend != "direct"
            or not envs.SGLANG_DSA_HOST_DEDUP_SHARED_2D.get()
        ):
            return None
        from decoupled_engine.srt.mem_cache import ce_2d_load

        by_part = []
        for local_ids in self._part_local_ids(transfers, layer_num, primary, parts):
            plans = []
            for transfer in transfers:
                if getattr(transfer.host_pool, "_is_dummy", False) and not getattr(
                    transfer.host_pool, "_shared_reader", False
                ):
                    continue
                ids = local_ids.get(id(transfer))
                if not ids:
                    continue
                if parts and not ce_2d_load.has_rows(transfer, ids):
                    continue  # only skip-topk placeholders in this part
                p = ce_2d_load.plan(transfer, ids)
                if p is None:
                    if not getattr(self, "_shared_2d_fallback_logged", False):
                        self._shared_2d_fallback_logged = True
                        logger.warning(
                            "Shared host dedup: 2D page loads unavailable for %s -> %s "
                            "(layout, strides or registration); using per-layer copies",
                            type(transfer.host_pool).__name__,
                            type(transfer.device_pool).__name__,
                        )
                    return None
                plans.append(p)
            by_part.append(plans)
        return by_part if parts else by_part[0]

    def _submit_shared_2d(
        self,
        transfers,
        plans,
        *,
        layer_num,
        h2d,
        start_event,
        ack,
        on_layer_done,
        wait_on_caller,
        t_submit,
    ) -> TransferCompletion:
        """Shared pool, v0.1: one 2D copy per page and layer run, copy engines only.

        Every layer event is recorded after all copies (the whole op), so a
        consumer waiting on any layer, the MTP draft included, sees the data.
        """
        from decoupled_engine.srt.mem_cache import ce_2d_load

        ack_start, ack_finish, timing_enabled = ack
        with device_module.stream(h2d):
            start_event.wait(h2d)
            ack_start.record()
            handle = getattr(h2d, "cuda_stream", 0)
            paced = getattr(self, "_paced_2d", None)
            draft_batch = int(envs.SGLANG_DSA_HOST_DEDUP_DRAFT_BATCH.get())
            if paced is None:
                paced = self._paced_2d = ce_2d_load.PacedSubmitter(
                    chunk_ops=envs.SGLANG_DSA_HOST_DEDUP_LOADER_CHUNK_PAGES.get(),
                    chunk_1d=max(draft_batch, 1),
                )
            calls, _, _ = paced.submit(
                ce_2d_load.op_arrays(plans, batch_1d=draft_batch > 0), handle
            )
            if on_layer_done is not None:
                for layer_id in range(layer_num):
                    on_layer_done(layer_id)
            ack_finish.record()
            self._record_stream(transfers, h2d)
        if wait_on_caller:
            ack_finish.wait(device_module.current_stream())
        ce_2d_load.COST.add("submit_2d", (time.perf_counter() - t_submit) * 1e3)
        ce_2d_load.COST.add("calls_2d", float(calls))
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def _layer_plan(self, transfers, layer_num, primary):
        plan = []
        for layer_id in range(layer_num):
            layer_transfers = []
            for transfer in transfers:
                local_layer_id = (
                    transfer.layer_mapper(layer_id)
                    if transfer.layer_mapper is not None
                    else layer_id
                )
                if local_layer_id is None or (
                    transfer is not primary
                    and transfer.layer_mapper is None
                    and layer_id >= transfer.host_pool.layer_num
                ):
                    continue
                layer_transfers.append((transfer, local_layer_id))
            plan.append((layer_id, layer_transfers))
        return plan

    def _submit_host_to_device_ipc(
        self,
        transfers: list[L2Transfer],
        *,
        dedup,
        fanout,
        layer_num: int,
        start_event,
        ack,
        on_layer_done,
        wait_on_caller: bool,
        stream,
        ipc_slot: Optional[int],
        host_indices_cpu: Optional[dict] = None,
    ) -> TransferCompletion:
        """Host-dedup load without NCCL: the source gathers, followers pull its
        rows over CUDA IPC after a per-layer interprocess event. Only the
        caller's own (local) completion is awaited, and only on request."""
        from decoupled_engine.srt.mem_cache.dedup_ipc_fanout import SLOT_MISC

        ack_start, ack_finish, timing_enabled = ack
        slot = SLOT_MISC if ipc_slot is None else ipc_slot
        primary = transfers[0]
        plan = self._layer_plan(transfers, layer_num, primary)
        if len(plan) > fanout.num_layer_events:
            raise RuntimeError("Dedup IPC fan-out has fewer layer events than layers")
        with device_module.stream(stream):
            start_event.wait(stream)
            ack_start.record()
            if fanout.is_src:
                fanout.src_begin(slot, transfers, stream)
                handled = set()
                ce_loader = getattr(fanout, "ce_loader", None)
                if ce_loader is not None and host_indices_cpu:
                    # Whole host pages via copy engines for every layer at once.
                    handled = ce_loader.load(plan, host_indices_cpu, stream)
                for layer_id, layer_transfers in plan:
                    for transfer, local_layer_id in layer_transfers:
                        if id(transfer) in handled:
                            continue
                        if not getattr(transfer.host_pool, "_is_dummy", False):
                            transfer.host_pool.load_to_device_per_layer(
                                transfer.device_pool,
                                transfer.host_indices,
                                transfer.device_indices,
                                local_layer_id,
                                self.io_backend,
                                is_draft=transfer.is_draft,
                            )
                    fanout.src_layer_done(slot, layer_id, stream)
                    if on_layer_done is not None:
                        on_layer_done(layer_id)
                ack_finish.record()
                fanout.order()
            else:
                fanout.order()
                src_rows = fanout.dst_begin(slot, transfers, stream)
                for layer_id, layer_transfers in plan:
                    fanout.dst_layer(
                        slot,
                        layer_id,
                        layer_transfers,
                        src_rows,
                        stream,
                        dedup._layer_buffer,
                        dedup._layer_rows,
                    )
                    if on_layer_done is not None:
                        on_layer_done(layer_id)
                fanout.dst_end(slot, stream)
                ack_finish.record()
            self._record_stream(transfers, stream)
        if wait_on_caller and (
            on_layer_done is None or not envs.SGLANG_DSA_HOST_DEDUP_IPC_LAYERWISE.get()
        ):
            # The caller's stream (and so the next forward) waits for this
            # rank's whole op, as on the NCCL path. Layer-wise overlap (the
            # forward waits per layer; followers' rows come from the source's
            # gather kernels over IPC events) hung under load on GB300 1P1D
            # (three P GPUs spinning in a TP collective, one parked). The exact
            # cycle is not established, so it is opt-in.
            ack_finish.wait(device_module.current_stream())
        return TransferCompletion(ack_start, ack_finish, timing_enabled)

    def bootstrap_prefetch_stream(self):
        stream = getattr(self, "_bootstrap_prefetch_stream", None)
        if stream is None:
            stream = device_module.Stream()
            self._bootstrap_prefetch_stream = stream
        return stream

    @staticmethod
    def _start_event(start_event):
        if start_event is None:
            start_event = device_module.Event()
            start_event.record()
        return start_event

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream) -> None:
        for transfer in transfers:
            for indices in (transfer.host_indices, transfer.device_indices):
                if indices.is_cuda:
                    indices.record_stream(stream)

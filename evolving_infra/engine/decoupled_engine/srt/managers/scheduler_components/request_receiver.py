# Adapted from SGLang python/sglang/srt/managers/scheduler_components/request_receiver.py (commit 297e54fd10), modified for evolving-infra.
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import torch
import zmq
from torch.distributed import ReduceOp, all_reduce, barrier

from decoupled_engine.srt.disaggregation.utils import prepare_abort
from decoupled_engine.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from decoupled_engine.srt.managers.mm_utils import (
    discard_shm_features,
    has_shm_features,
    unwrap_shm_features,
)
from decoupled_engine.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_RECV_REQUESTS,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from decoupled_engine.srt.runtime_context import get_disagg, get_exec, get_parallel
from decoupled_engine.srt.utils import (
    broadcast_pyobj,
    broadcast_pyobj_with_flag,
    point_to_point_pyobj,
)

if TYPE_CHECKING:
    from decoupled_engine.srt.configs.model_config import ModelConfig
    from decoupled_engine.srt.rust_server.server import RustServer
    from decoupled_engine.srt.server_args import ServerArgs
    from decoupled_engine.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from decoupled_engine.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy, RustServer]
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_batch: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None
    scheduler_stage_metrics: Optional[SchedulerStageMetricsRecorder] = None
    # SGLANG_HICACHE_SKIP_IDLE_SYNC (plain TP only): the source rank's
    # per-step HiCache sync decision rides the request broadcast
    # (broadcast_pyobj_with_flag); every rank hands it to the sink (None on a
    # step the recv skipper skipped: no broadcast, reuse the last decision).
    step_sync_flag_source: Optional[Callable[[], bool]] = None
    step_sync_flag_sink: Optional[Callable[[Optional[bool]], None]] = None
    # --peer-kv-demote-source, source rank only: returns a PeerKVDemoteReq (or
    # None). It goes first in the broadcast list, so every rank applies it
    # before any tree operation of the step.
    peer_kv_demote_source: Optional[Callable[[], Any]] = None
    # SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS, source rank only: returns a
    # PeerKVTouchReq (or None). It goes in the broadcast list before the
    # requests (after a demote request), so every rank applies it at the
    # same point of the step.
    peer_kv_touch_source: Optional[Callable[[], Any]] = None
    # Late D (decode, --disaggregation-late-d): ADMIT / EXPIRE controls that
    # the source rank appends before the broadcast, so every rank applies
    # them in the same iteration and order. The source gets the raw batch it
    # is about to broadcast: a control for a leg in that batch applies after
    # the leg is added, in the same iteration.
    late_d_control_source: Optional[Callable[[List], List]] = None
    # --enable-kv-session-retention, source rank only: given the list it is
    # about to broadcast, returns a KvSessionClockStamp (its time.monotonic())
    # or None. The stamp goes first in the list, so every rank applies it
    # before any request of the step: the retention clock of
    # SGLANG_KV_SESSION_RETENTION_ORDER is the same on every rank
    # (mem_cache/unified_cache/retention_order.py).
    kv_session_clock_source: Optional[Callable[[List], Any]] = None
    # Late-D overhead readout on the broadcast source (late_d.RecvBroadcastReadout:
    # late D on a decode, or SGLANG_RECV_BROADCAST_READOUT=1 on any server):
    # the seconds of each request broadcast by payload, and the pickled bytes
    # the source sent.
    broadcast_readout: Optional[Any] = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0:
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_stage_method(SCHEDULER_STAGE_RECV_REQUESTS)
    def recv_requests(
        self, local_reqs: Optional[List[AbortReq]] = None
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks.

        local_reqs are aborts the caller decided on this rank; they ride the
        same broadcast as the pulled requests.
        """

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_batch()):
                if self.step_sync_flag_sink is not None:
                    # No broadcast this step (the skip schedule is the same on
                    # every rank): the cache reuses the last broadcast decision.
                    self.step_sync_flag_sink(None)
                return []

        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.late_d_control_source is not None and recv_reqs is not None:
            recv_reqs.extend(self.late_d_control_source(recv_reqs))

        if self.kv_session_clock_source is not None and recv_reqs is not None:
            stamp = self.kv_session_clock_source(recv_reqs)
            if stamp is not None:
                recv_reqs.insert(0, stamp)

        readout = self.broadcast_readout
        if readout is None:
            recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs, local_reqs)
        else:
            # Payload "empty": nothing to send, one size broadcast; "data":
            # requests or controls (the pickle and a second broadcast too).
            has_data = bool(recv_reqs) or bool(local_reqs)
            start = time.perf_counter()
            recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs, local_reqs)
            readout.observe(time.perf_counter() - start, has_data)

        if get_parallel().pp_rank == 0:
            self.unwrap_pickle_wrapper(recv_reqs)

        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _pull_raw_reqs(self) -> Optional[List]:
        if get_parallel().pp_rank == 0:
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                recv_reqs = []
                if self.peer_kv_demote_source is not None:
                    demote = self.peer_kv_demote_source()
                    if demote is not None:
                        recv_reqs.append(demote)
                if self.peer_kv_touch_source is not None:
                    touch = self.peer_kv_touch_source()
                    if touch is not None:
                        recv_reqs.append(touch)

                # Rust ringbuffer backend: drain the in-process ring fed by the
                # embedded Rust TokenizerManager instead of a zmq socket. Same
                # non-blocking, msgpack-decoded contract as the zmq path below.
                if envs.SGLANG_RUST_SERVER.get():
                    recv_reqs.extend(
                        self.recv_from_tokenizer.drain(self.max_recv_per_poll)
                    )
                    return recv_reqs

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        recv_rpc = sock_recv(self.recv_from_rpc, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                dp_offset = (
                    get_parallel().attn_dp_rank
                    * get_parallel().attn_cp_size
                    * get_parallel().attn_tp_size
                )
                recv_reqs = point_to_point_pyobj(
                    [],
                    get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                    get_parallel().world_group.cpu_group,
                    (get_parallel().pp_rank - 1) * get_parallel().tp_size + dp_offset,
                    get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                )
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_size_kw(self) -> Dict[str, Callable[[int], None]]:
        """The broadcasts add the pickled bytes the source sends to the
        readout (no argument without one, so the call is unchanged)."""
        readout = self.broadcast_readout
        return {} if readout is None else {"size_sink": readout.add_bytes}

    def _broadcast_reqs_across_ranks(
        self, recv_reqs: Optional[List], local_reqs: Optional[List] = None
    ) -> List:
        """local_reqs ride the work channel, which is scoped to the ranks
        sharing one waiting queue; the control channel fans out from global
        rank 0 and would overwrite every DP group's aborts but the first.
        """
        local_reqs = local_reqs or []
        size_kw = self._broadcast_size_kw()
        if get_parallel().attn_dp_enabled:
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
                work_reqs.extend(local_reqs)
            else:
                work_reqs = None
                control_reqs = None

            # TODO: pass size_sink through attn_cp_tp_broadcast_pyobj
            # (distributed/communication_op.py); broadcast_readout counts the
            # seconds of its broadcasts below but not their bytes.
            work_reqs = attn_cp_tp_broadcast_pyobj(work_reqs)

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                get_parallel().enable_dp_attention_local_control_broadcast
                or get_exec().moe.is_ep_scale_joiner
            )
            if _local_ctrl:
                control_reqs = attn_cp_tp_broadcast_pyobj(control_reqs)
            elif get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                    **size_kw,
                )
            recv_reqs = work_reqs + control_reqs
        else:
            if recv_reqs is not None:
                recv_reqs = [*recv_reqs, *local_reqs]
            tp_size = get_parallel().tp_size
            if self.step_sync_flag_sink is not None:
                # Plain TP with SGLANG_HICACHE_SKIP_IDLE_SYNC (wired only without DP
                # attention or PP, see Scheduler._hicache_step_sync_hooks).
                tp_group = get_parallel().tp_group if tp_size != 1 else None
                src = tp_group.ranks[0] if tp_size != 1 else None
                is_src = tp_size == 1 or tp_group.rank == src
                flag = self.step_sync_flag_source() if is_src else True
                if tp_size != 1:
                    recv_reqs, flag = broadcast_pyobj_with_flag(
                        recv_reqs,
                        flag,
                        tp_group.rank,
                        tp_group.cpu_group,
                        src=src,
                        **size_kw,
                    )
                self.step_sync_flag_sink(flag)
            elif tp_size != 1:
                tp_group = get_parallel().tp_group
                recv_reqs = broadcast_pyobj(
                    recv_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                    **size_kw,
                )
        return recv_reqs

    def unwrap_pickle_wrapper(self, recv_reqs: Optional[List]) -> None:
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                req.unwrap_pickle_fields()
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                for sub_req in req:
                    sub_req.unwrap_pickle_fields()

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        if (
            get_parallel().pp_rank == 0
            and get_disagg().language_only
            and get_disagg().encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                if error_code is None:
                    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                elif isinstance(error_code, HTTPStatus):
                    status_code = error_code
                else:
                    status_code = HTTPStatus(int(error_code))
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        """Materialize SHM features or mark the request failed on every rank."""
        if not recv_reqs or not self.model_config.is_multimodal:
            return

        tokenized_reqs = []
        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                tokenized_reqs.append(req)
            elif isinstance(
                req,
                (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
            ):
                tokenized_reqs.extend(req.batch)
        if not tokenized_reqs or not has_shm_features(tokenized_reqs):
            return

        # 1. wait until every rank has opened the shared feature segments
        parallel = get_parallel()
        if parallel.attn_dp_enabled:
            if parallel.attn_tp_size > 1:
                barrier(group=parallel.attn_tp_group.cpu_group)
            if parallel.attn_cp_size > 1:
                barrier(group=parallel.attn_cp_group.cpu_group)
        elif parallel.tp_size > 1:
            barrier(group=parallel.tp_group.cpu_group)

        # 2. materialize independently so one bad VLM request does not stop the loop
        failed = torch.zeros(len(tokenized_reqs), dtype=torch.int32)
        for index, req in enumerate(tokenized_reqs):
            if not has_shm_features([req]):
                continue
            try:
                unwrap_shm_features(req)
            except Exception:
                logger.exception(
                    "Failed to materialize shared-memory multimodal features for rid=%s",
                    req.rid,
                )
                discard_shm_features(req)
                failed[index] = 1

        # 3. all ranks reject the same requests before entering model collectives
        if parallel.attn_dp_enabled:
            if parallel.attn_tp_size > 1:
                all_reduce(
                    failed, op=ReduceOp.MAX, group=parallel.attn_tp_group.cpu_group
                )
            if parallel.attn_cp_size > 1:
                all_reduce(
                    failed, op=ReduceOp.MAX, group=parallel.attn_cp_group.cpu_group
                )
        elif parallel.tp_size > 1:
            all_reduce(failed, op=ReduceOp.MAX, group=parallel.tp_group.cpu_group)

        error = MMInputsProcessError(
            "Failed to materialize shared-memory multimodal features on a scheduler rank."
        )
        for index, req in enumerate(tokenized_reqs):
            if failed[index].item():
                discard_shm_features(req)
                req.mm_inputs = error

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs

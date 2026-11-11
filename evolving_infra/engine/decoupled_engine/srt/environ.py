# Adapted from SGLang python/sglang/srt/environ.py (commit 9a489f8d2f), modified for evolving-infra.
import base64
import functools
import json
import os
import warnings
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Dict, Optional


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


_NON_UTF8_PREFIX = "base64:"


def _default_cache_subdir(name: str) -> str:
    """A directory under SGLANG_CACHE_DIR, for env defaults that track it.

    Pass as a callable default: SGLANG_CACHE_DIR is declared further down the
    Envs body, and resolving late also lets tests override it.
    """
    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), name)


def _default_tree_cache_sanity_check() -> bool:
    """Enable the expensive tree-cache sanity check by default in CI."""
    return envs.SGLANG_IS_IN_CI.get()


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any, secret: bool = False):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False
        self.secret = secret

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{default}"'
            )
            return default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvJSON(EnvField):
    def parse(self, value: str | None) -> list | dict | None:
        if not value:
            return None
        if os.path.exists(value):
            with open(value) as f:
                return json.load(f)
        return json.loads(value)


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def __init__(self, default: Any, deprecated_name: str, secret: bool = False):
        super().__init__(default, secret=secret)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class GateGemvMode(IntEnum):
    """Small-batch Inkling gate linear implementation.

    OFF: always the cublas GEMM
    PAIR: PDL-chained GEMV and gate JIT kernels
    FUSED: single-launch GEMV + gate epilogue (last-block ticket)
    """

    OFF = 0
    PAIR = 1
    FUSED = 2


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class InvariantCheckLevel(IntEnum):
    """Signal level for value/index validity checks (see invariants.py).

    OFF: data layer only (sanitize/containment); no detection, no signal.
    WARN: detect + throttled log/count; degrade, never crash (prod on-demand).
    STRICT: detect + crash on GUARD/FATAL violations (CI default).

    The data layer is unconditional and independent of this level; only the
    detection + signal layer is gated here.
    """

    OFF = 0
    WARN = 1
    STRICT = 2


class DsparkFoldedSampling(IntEnum):
    """Sampling support in the graph-folded DSpark draft proposal: OFF =
    greedy-only folding, AUTO = on when its buffers fit in free GPU memory,
    FORCE = always."""

    OFF = 0
    AUTO = 1
    FORCE = 2


class Envs:
    # Organization principles for this registry:
    # - Put every field in exactly one topical section. Prefer an existing
    #   section; add a new one only when no current section is a clear fit.
    # - Group by the behavior and owning call sites, not by name similarity
    #   alone. Keep closely related lifecycle or feature knobs adjacent.
    # - Keep each section focused and below 30 fields. Split growing sections
    #   by subsystem or lifecycle instead of creating catch-all groups.
    # - Order broad runtime subsystems before shared storage and backends; keep
    #   platform- and model-specific integrations in dedicated later sections.
    # - Use the same three-line section header everywhere; do not add ad hoc
    #   one-line headings or append unrelated fields at the end of a section.
    # - Keep vendor-specific aliases with their owning integration, and keep
    #   test/debug knobs with the feature or test workflow they exercise.
    # - Keep explanatory comments attached to their field when moving it.
    # - For organization-only changes, AST-check that field names, descriptor
    #   types, and defaults are unchanged and that only field order moved.

    # ===================================================================
    # Runtime configuration and process identity
    # ===================================================================
    # Per-role config-namespace bookkeeping: off / record / enforce (value is
    # validated fail-loud in runtime_context, which resolves it once at import
    # so the read stays dynamo-prunable).
    SGLANG_ROLE_NAMESPACES = EnvStr("off")
    # Record mode: append each newly observed (role, namespace) pair to this
    # file so the audit survives signal-killed workers.
    SGLANG_ROLE_NAMESPACES_OUT = EnvStr(None)
    IS_H200 = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)

    # ===================================================================
    # Model configuration, discovery, and weight loading
    # ===================================================================
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    # Controls weight-file ordering for load-time I/O optimization.
    #   -1 : no sorting, no staggering; preserves original file order.
    #    0 : sort files only; maximizes ordering but may reduce cross-rank I/O concurrency.
    #   k>0: sort files and stagger per-rank order with factor k.
    #        Files are processed in groups of (tp_size * k), and rank r starts each
    #        group at offset (r * k), improving multi-rank I/O concurrency while
    #        keeping access relatively ordered.
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
    SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)
    # Per-rank V4 tensor-copy executor; None preserves the Python default.
    SGLANG_DSV4_WEIGHT_LOADER_MAX_WORKERS = EnvInt(None)
    # Copy rank-local MoE slices into independent CPU storage before H2D when
    # they reference a larger mmap-backed checkpoint storage.
    SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D = EnvBool(False)
    SGLANG_LOAD_SNAPSHOT_SLOT_BYTES = EnvInt(16384)
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)
    # Optional native decode allocation/capacity telemetry; no GPU reads.
    SGLANG_DECODE_SELECTION_METRICS = EnvBool(False)
    # Optional CPU metadata only, sampled after prefill result processing.
    SGLANG_PREFILL_ROUTING_SNAPSHOT = EnvBool(False)
    # The same snapshot on a symmetric worker (the PD flag has no effect
    # there): attention TP0 samples it every pass of the overlap loop; held
    # P-legs are its bootstrap rows and computed exports its inflight rows
    # (prefill_routing_snapshot.py). The settings below apply. With
    # SGLANG_PREFILL_STATE_PUSH_PORT / _ENDPOINT it also starts the state push
    # there, for extend passes only. Default off.
    SGLANG_SYMMETRIC_ROUTING_SNAPSHOT = EnvBool(False)
    SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_RECORDS = EnvInt(32)
    SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_BYTES = EnvInt(4096)
    SGLANG_PREFILL_TIMELINE_SAMPLES = EnvBool(False)
    SGLANG_PREFILL_ROUTING_SNAPSHOT_INTERVAL_S = EnvFloat(0.1)
    SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_AGE_S = EnvFloat(1.0)
    # Request-trace D-local prefix observations; reuse current match scalars only.
    SGLANG_DECODE_PREFIX_OBSERVATION = EnvBool(False)
    # Opt-in push of the routing snapshot after each prefill launch and result, on a
    # ZMQ PUB socket (prefill_state_push.py).
    # Off when PORT is 0 and ENDPOINT is empty. PORT binds tcp://*:PORT+rank.
    # On a symmetric worker it needs SGLANG_SYMMETRIC_ROUTING_SNAPSHOT=1.
    SGLANG_PREFILL_STATE_PUSH_PORT = EnvInt(0)
    SGLANG_PREFILL_STATE_PUSH_ENDPOINT = EnvStr("")
    SGLANG_PREFILL_STATE_PUSH_HWM = EnvInt(16)
    SGLANG_PREFILL_STATE_PUSH_MAX_BYTES = EnvInt(1 << 20)
    SGLANG_PREFILL_STATE_PUSH_HEARTBEAT_S = EnvFloat(1.0)
    # When a pass launch goes out: "busy" sends it at
    # the launch only while the previous pass still runs on the GPU, else
    # with the end-of-iteration message; "always" sends it at every launch.
    SGLANG_PREFILL_STATE_PUSH_LAUNCH_MODE = EnvStr("busy")
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    # In seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging.
    # Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1)
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # ===================================================================
    # HTTP server and health
    # ===================================================================
    # Decompress request bodies tagged with `x-body-compressed`.
    SGLANG_ENABLE_REQUEST_DECOMPRESSION = EnvBool(False)
    # Override parsed request fields from headers.
    SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)
    SGLANG_EXPOSE_OWN_ENV_VARS = EnvBool(False)
    SGLANG_DIAG_BYPASS_HEALTH_GENERATE = EnvBool(False)

    # ===================================================================
    # Logging
    # ===================================================================
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_DECODE_GRAPH_KEY = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)
    SGLANG_ENABLE_RANK_CONSENSUS_CHECKER = EnvBool(False)
    # HiCache TP structural digest (detects host-tier divergence across TP ranks): "off" |
    # "warn" | "assert". Empty = "off" (opt in; each check is one extra CPU all-reduce across
    # the TP group per scheduler step).
    SGLANG_HICACHE_TP_DIGEST_MODE = EnvStr("")
    # Unified radix cache, two TP lockstep checks at every HiCache step sync,
    # piggybacked on its existing [x, -x] reduce (no extra collective). Both
    # are off by default; 1 = on (recommended for validation and benchmark
    # runs of a new configuration). On a mismatch each raises on every rank
    # after a CRITICAL "HICACHE_LOCKSTEP ... diverged across TP ranks" line;
    # while one is on, TP0 logs "HICACHE_LOCKSTEP checks=..
    # settled_write_acks=.." every 60 s. writing_check / loading_check settle
    # acks by a MIN-reduced COUNT, which settles different nodes on ranks
    # whose ops merged differently (per-rank gate-pending reorder x 6h
    # flushes).
    # - ACK_LOCKSTEP: all ranks hold the same write / load ack queues (acks
    #   per queue; node count and tokens per ack). One Python pass over the
    #   queues per step.
    # - SETTLE_DIGEST: all ranks settled the same write acks so far (a running
    #   digest of position, node count, tokens and host rows per settled ack;
    #   device rows instead where a peer-KV servicer's lease withholding makes
    #   host row ids rank-local and no rank reads another's host rows, i.e.
    #   any fan-out but shared). About 1 us per settled ack. Works
    #   with any SHARED_VERIFY.
    SGLANG_HICACHE_ACK_LOCKSTEP_CHECK = EnvBool(False)
    SGLANG_HICACHE_SETTLE_DIGEST_CHECK = EnvBool(False)
    # Eviction-order check, on the same step-sync reduce (no
    # extra collective): the step-sync digest also folds the tree's host
    # release digest, its device free digest (node id, component and demote
    # class of every device free), the flagged-id digest and the next node
    # id. "off" (default; empty = off), "log" (one CRITICAL line when only the
    # eviction order diverged, then the fold stops) or "stop" (raise on every
    # rank, as the two checks above). While it folds, every rank logs
    # "HICACHE_EVICT_ORDER rank=.. checks=.." every 60 s.
    SGLANG_HICACHE_EVICT_ORDER_CHECK = EnvStr("")
    # Skip the per-step HiCache ready-count all-reduce on steps where no HiCache
    # operation can be outstanding. The TP source rank decides from its own
    # queues plus the all-ranks in-flight count of the previous sync, and sends
    # the decision with the request broadcast it already makes every step, so
    # every rank skips the same steps (no extra collective). Plain TP only (no
    # PP / DP attention); with --scheduler-recv-interval the last decision is
    # held over the skipped receives. Digest mode must be off. Window
    # residency default (unset, digest off): on.
    SGLANG_HICACHE_SKIP_IDLE_SYNC = EnvBool(False)
    # While skipping, still sync at least once every this many steps (bounds
    # the delay of an op issued on some ranks only, e.g. a rank-local retry).
    SGLANG_HICACHE_IDLE_SYNC_INTERVAL = EnvInt(16)
    # Pipeline the per-step HiCache ready-count all-reduce (and, under DSv4
    # window residency, the SWA backfill poll's) by one scheduler step: each
    # step applies the previous step's reduced vector and starts its own as an
    # async all-reduce on a dedicated Gloo group, so the collective overlaps
    # the step instead of blocking it (mem_cache/hicache_pipelined_sync.py).
    # Plain TP only (no PP / DP attention); composes with SKIP_IDLE_SYNC.
    # Window residency default (unset): on.
    SGLANG_HICACHE_PIPELINED_SYNC = EnvBool(False)
    # Backfill job rows per pipelined vector (fixed, so diverged job lists
    # cannot size-mismatch the reduce); jobs beyond it wait for a later step.
    SGLANG_HICACHE_PIPELINED_SYNC_BACKFILL_SLOTS = EnvInt(64)
    # The pipelined reduction: "allreduce" (Gloo ring, 2(N-1) sequential
    # messages) or "alltoall" (each rank sends its vector to every rank, one
    # message latency, MIN taken locally).
    SGLANG_HICACHE_PIPELINED_SYNC_ALGO = EnvStr("allreduce")
    # kv-p2p debug detector (owner side; also toggled at runtime through the
    # servicer's "debug" op): overwrite the first 64 B of every host page
    # segment when the page enters the free list (never while pinned). LOOKUP
    # replies then carry "poison": true and readers flag pulled segments that
    # carry the pattern ("poisoned" verdict).
    SGLANG_PEER_KV_POISON_ON_RELEASE = EnvBool(False)
    # kv-p2p owner side, scheduler thread: publish (D2H ack, prefetch commit)
    # and retire (host eviction / release) a node's pages in the servicer's
    # shadow index as one index range per node (one generation gather, one
    # snapshot clone) instead of one tensor slice + generation lookup per
    # page; the servicer thread splits the range. Same shadow content, same
    # call points and order. Off: per-page pushes, as before.
    SGLANG_PEER_KV_PUSH_FAST = EnvBool(False)
    # F4 (owner side): a legacy LOOKUP over the
    # lease byte budget gets the longest prefix that fits (at least
    # MIN_PAGES pages) instead of a whole refusal. Off: refuse as before.
    SGLANG_PEER_KV_BUDGET_PARTIAL_GRANT = EnvBool(False)
    SGLANG_PEER_KV_BUDGET_PARTIAL_MIN_PAGES = EnvInt(4)
    # F5 (both sides; set it on every
    # prefill worker): a destination LOOKUP carries its reply deadline (the
    # wall clock at the send + lookup_timeout_ms). The owner refuses ("late")
    # a LOOKUP that it starts, or would grant, after that deadline + SLACK_S,
    # so a grant that its reader has given up on does not hold pins and lease
    # bytes until the lease TTL. Needs host clocks within SLACK_S of each
    # other. Off: no deadline on the wire, no check.
    SGLANG_PEER_KV_LOOKUP_DEADLINE = EnvBool(False)
    SGLANG_PEER_KV_LOOKUP_DEADLINE_SLACK_S = EnvFloat(0.02)
    # F6 (owner side): a legacy
    # LOOKUP checks the lease byte budget before it pins, with its confirmed
    # pages x the smallest KV page bytes of an earlier grant, so a refusal no
    # longer holds the single servicer thread for the pins and the address
    # lists (0.25-0.9 s at 5-11k pages). It refuses what the check after the
    # pins would refuse; the one difference: when the pins stop short (a page
    # reused, the pin reserve full) and only that shorter prefix fits, it
    # refuses where the check after the pins granted the shorter prefix.
    # Off: the check after the pins only, as before.
    SGLANG_PEER_KV_BUDGET_PRECHECK = EnvBool(False)
    # F3 (destination side): the storage
    # prefetch timeout does not count the LOOKUP phase (its per-page clock
    # starts when the TP-synced hit is known; before that the op is bounded
    # by LOOKUP_PHASE_MAX_S), and the JIT pull does not settle / re-pull a
    # request while its last pull's LOOKUP has not reached the hit drain
    # (bounded by LOOKUP_HOLD_MAX_PASSES scheduler passes). Off: as before.
    SGLANG_JIT_PULL_LOOKUP_AWARE_HOLD = EnvBool(False)
    SGLANG_JIT_PULL_LOOKUP_PHASE_MAX_S = EnvFloat(10.0)
    SGLANG_JIT_PULL_LOOKUP_HOLD_MAX_PASSES = EnvInt(400)
    # Optional lookahead lead for peer pulls; count preserves the old policy.
    SGLANG_JIT_PULL_LEAD = EnvStr("count")
    SGLANG_JIT_PULL_SPAN_FIXED_S = EnvFloat(0.78)
    SGLANG_JIT_PULL_SPAN_S_PER_100K = EnvFloat(0.22)
    SGLANG_JIT_PULL_NOMINAL_PASS_S = EnvFloat(0.38)
    SGLANG_JIT_PULL_ISSUE_PASSES = EnvInt(1)
    SGLANG_JIT_PULL_LEAD_MARGIN = EnvInt(1)
    # F1 (both sides; set it on every prefill
    # worker): while a hinted pull is deferred (JIT pull), the destination
    # sends TOUCH to the owner's servicer at the deferral and every
    # INTERVAL_S, and the owner refreshes the hinted prefix in its host LRU
    # on every TP rank (mem_cache/peer_kv_touch.py). Off: as before.
    SGLANG_PEER_KV_TOUCH_DEFERRED_PULLS = EnvBool(False)
    SGLANG_PEER_KV_TOUCH_INTERVAL_S = EnvFloat(30.0)
    SGLANG_PEER_KV_TOUCH_MAX_AGE_S = EnvFloat(900.0)
    SGLANG_PEER_KV_TOUCH_MAX_KEYS = EnvInt(512)
    # Demote hardening (--peer-kv-demote-source).
    # AFTER_COMMIT, reader side: a pull whose reads are complete sends a plain
    # RELEASE marked demote_after_commit, and DEMOTE_OK only after its pages
    # are committed to the reader's tree; the owner demotes on DEMOTE_OK (a
    # pending record expires after PENDING_TTL_S). Abort, hybrid discard and
    # dropped inserts send nothing more, so the owner keeps its copy. Off: the
    # complete RELEASE demotes at once, as before.
    SGLANG_PEER_KV_DEMOTE_AFTER_COMMIT = EnvBool(False)
    SGLANG_PEER_KV_DEMOTE_PENDING_TTL_S = EnvFloat(300.0)
    # LOCK_AWARE, owner side: the demote walk stops at a node with a host lock
    # (a queued local request, a prefetch anchor, a backup) and counts the
    # outcome "locked". Off: host locks do not stop the walk, as before.
    SGLANG_PEER_KV_DEMOTE_LOCK_AWARE = EnvBool(False)
    # COUNTERS, owner side (every TP rank: the TP digest needs it on all):
    # flag age at eviction and at clear, the no_node causes, outbox drops, one
    # trace record per flagged eviction (servicer trace dir), and the flagged
    # node ids in the TP structural digest. Off: as before.
    SGLANG_PEER_KV_DEMOTE_COUNTERS = EnvBool(False)
    # RECORDS (needs COUNTERS and a trace dir): JSON-line
    # records through a non-blocking writer thread: one per host eviction
    # on the owner (resolver rank) and one per queued DEMOTE_OK on the
    # reader. Off: no record, no thread.
    SGLANG_PEER_KV_DEMOTE_RECORDS = EnvBool(False)
    # NOTICE (the reader also needs AFTER_COMMIT, the owner
    # --peer-kv-demote-source). Reader TP0: at the prefill end of a hinted
    # request whose owner confirmed more pages than this reader committed
    # from it (and the request inserted its KV with write-through), send one
    # NOTICE; the owner walks it as a commit, from the deepest node end among
    # its keys. A mark older than 30 s at the send is dropped. After a shorter
    # host alloc the reader trims the lease. The LOOKUP reply carries
    # confirmed / refused_reason / lookup_tick with NOTICE or COUNTERS on.
    # Off: no NOTICE, as before.
    SGLANG_PEER_KV_DEMOTE_NOTICE = EnvBool(False)
    # NOTICE_GRACE_S (owner TP0): hold a NOTICE until G s
    # after the reader's prefill end, then walk it (a local use since the
    # lookup stops the walk: notice_grace_access). 0 (default): walk at once.
    SGLANG_PEER_KV_DEMOTE_NOTICE_GRACE_S = EnvFloat(0.0)
    # GPU_ORDER (acts only with --peer-kv-demote-source): the
    # device eviction heap takes class-2 (demoted) D-leaves first (key
    # (class, priority, id); a worker with no class-2 node keeps the plain
    # key, so the victims are the same); a class stays on a host-only release
    # while the GPU copy stays; the duplicate-reclaim fallback takes an
    # unlocked class-2 D-leaf whole (both copies) before any class-1 host
    # copy. Off: as before.
    SGLANG_PEER_KV_DEMOTE_GPU_ORDER = EnvBool(False)
    # SCOPE (acts only with --peer-kv-demote-source): the
    # scope walk of a mark (covered start, E1 dead siblings, E2 locks and
    # queued requests, E3, E4 off-path use, dormant marks). "off" (default;
    # empty = off); "shadow": evaluate and count it ("scope.*" in the
    # "PeerKV demote:" line), apply today's walk; "on": apply it, and a split
    # keeps the tail's tick, the insert walk and insert_host touch after the
    # split, insert_host clears the anchor's ancestors.
    SGLANG_PEER_KV_DEMOTE_SCOPE = EnvStr("")

    # ===================================================================
    # IPC, broadcasters, and ports
    # ===================================================================
    SGLANG_USE_PICKLE_IPC = EnvBool(True)
    # Log top-level PickleWrapper frames unwrapped on msgpack IPC decode.
    SGLANG_LOG_PICKLE_IPC_OBJECTS = EnvBool(False)
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)
    # Base port hint for ephemeral sockets (ZMQ, SHM broadcaster, etc.).
    # When set, get_open_port() and shm_broadcast search upwards from this
    # value instead of asking the OS for a random port.  Useful to keep all
    # SGLang ports in a predictable range behind a firewall.
    SGLANG_PORT = EnvInt(None)
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # ===================================================================
    # CI and test execution
    # ===================================================================
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    # Set to true by the check-changes CI job when a PR touches nothing under
    # rust/; default false so local and scheduled runs never skip the cargo tests.
    SGLANG_SKIP_RUST_TESTS = EnvBool(False)
    SGLANG_TEST_MAX_RETRY = EnvInt(None)
    # Expand jit_kernel test grids to their full parameter ranges (nightly).
    SGLANG_JIT_KERNEL_RUN_FULL_TESTS = EnvBool(False)
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # ===================================================================
    # Crash diagnostics and shutdown
    # ===================================================================
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    # None = unset, letting get_dump_dir() resolve the base (RUNNER_TEMP in CI,
    # else /tmp); see debug_utils/cuda_coredump.py.
    SGLANG_CUDA_COREDUMP_DIR = EnvStr(None)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)

    # ===================================================================
    # Constrained decoding and grammar
    # ===================================================================
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # ===================================================================
    # Fault injection and regression tests
    # ===================================================================
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)
    # Test only (Mooncake prefill): raise inside a transfer worker for the
    # chunks of one room, or on every Nth chunk, at most _MAX times per
    # process (0: no limit). Exercises the per-room worker failure path.
    SGLANG_TEST_MOONCAKE_TRANSFER_FAULT_ROOM = EnvInt(None)
    SGLANG_TEST_MOONCAKE_TRANSFER_FAULT_EVERY_N = EnvInt(0)
    SGLANG_TEST_MOONCAKE_TRANSFER_FAULT_MAX = EnvInt(1)
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2**31)
    # Scheduler: force lazy extra_buffer prealloc to fail at decode boundaries
    SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL = EnvBool(False)
    # KL tests: skip the cache-hit count assertion (e.g. when alloc failure reduces hits)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)

    # ===================================================================
    # PD and scripted-runtime tests
    # ===================================================================
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)
    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)

    # ===================================================================
    # Profiling, tracing, and metrics
    # ===================================================================
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_ENABLE_NVTX"
    )
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPERATIONS_ENABLE_PROFILE"
    )
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    # Opt-in: emit one CUDA-graph capture trace per captured batch size (per-bs).
    # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE (single combined trace) takes
    # precedence when both are set.
    SGLANG_GRAPH_BATCH_CAPTURE = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    # Allocator-history buffer for /start_profile activities=["MEM"]; the
    # default truncates long windows (each entry is one alloc/free event).
    SGLANG_MEM_PROFILE_MAX_ENTRIES = EnvInt(100000)
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_TRACE_ASYNC = EnvBool(False)
    SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD = EnvInt(100)
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    # Optional per-batch CUDA timing for --sched-trace-dir calibration runs.
    SGLANG_SCHED_TRACE_GPU_TIME = EnvBool(False)
    # Overlap load-back M0: measurement only.
    # Scheduler phase timers (perf_counter around admission, load-back,
    # result processing; summed into the sched trace's phase_ms and a 60 s
    # TP0 summary line).
    SGLANG_SCHED_PHASE_TIMERS = EnvBool(False)
    # With the phase timers on: log a forward launch gap (previous launch
    # return -> next launch start) above this many ms with the phases timed
    # inside it, on every rank, at most LOG_PER_MIN lines per 60 s.
    SGLANG_SCHED_LAUNCH_GAP_LOG_MS = EnvFloat(30.0)
    SGLANG_SCHED_LAUNCH_GAP_LOG_PER_MIN = EnvInt(20)
    # GIL switch interval of every scheduler process (seconds; CPython default
    # 0.005): a thread waiting for the GIL forces a hand-off only after this
    # long. Unset: no change. Logged once per scheduler process.
    SGLANG_SCHED_GIL_SWITCH_INTERVAL_S = EnvFloat(None)
    # torch.cuda sync-debug 'warn' inside the scheduler's admission, run_batch
    # and result-processing sites: logs the first N distinct stacks of every
    # synchronizing CUDA call per site (correctness runs only, not for timing).
    SGLANG_SCHED_SYNC_DEBUG = EnvBool(False)
    SGLANG_SCHED_SYNC_DEBUG_STACKS = EnvInt(8)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E_SYNC = EnvBool(False)
    SGLANG_TRACE_SAMPLER_E2E = EnvBool(False)
    SGLANG_TRACE_QWEN_MOE_DEEPEP_E2E = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_CONTIG = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_MASKED = EnvBool(False)

    # ===================================================================
    # Debugging and invariant checks
    # ===================================================================
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS = EnvBool(False)
    # NaN-fill the unified memory pool at boot (debug repro switch).
    SGLANG_DEBUG_POISON_POOL = EnvBool(False)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    # The explicit environment variable still takes precedence over this CI
    # default, so production remains opt-in and CI remains opt-out if needed.
    SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK = EnvBool(_default_tree_cache_sanity_check)
    # Physical KV-page checks: committed<=allocated + no page alias.
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)
    SGLANG_TBO_DEBUG = EnvBool(False)
    # Timing probe: run the swap-in fully but skip the host->device KV bytes,
    # measuring the "IO is free" floor. GARBAGE OUTPUT -- benchmarking only.
    SGLANG_DEBUG_HISPARSE_SKIP_IO = EnvBool(False)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    # Signal level for value/index validity checks (nan/inf/oob/...); see
    # invariants.py. OFF (prod default) runs only the free data layer, WARN
    # adds throttled logging, STRICT (CI default) crashes on violations.
    # Supersedes SGLANG_ENABLE_ASYNC_ASSERT, which is bridged as STRICT until
    # every callsite migrates.
    SGLANG_INVARIANT_CHECK = EnvInt(InvariantCheckLevel.OFF)

    # ===================================================================
    # Runtime simulations
    # ===================================================================
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_ACC_TOKEN_MODE = EnvStr("fixed")
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)

    # ===================================================================
    # DSpark speculative decoding
    # ===================================================================
    SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_DUMP = EnvTuple(tuple())
    SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_STS_COLLECT_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_ENABLE_SPS_RECORD = EnvBool(False)
    SGLANG_DSPARK_FAST_KERNEL = EnvBool(True)
    SGLANG_DSPARK_FP32_LM_HEAD = EnvBool(False)
    SGLANG_DSPARK_FAST_SAMPLING = EnvBool(True)
    SGLANG_DSPARK_FOLDED_SAMPLING = EnvInt(DsparkFoldedSampling.AUTO)
    SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)
    SGLANG_DSPARK_STACKED_CTX_KV = EnvBool(True)
    SGLANG_DSPARK_EMBED_IN_GRAPH = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)
    SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV = EnvBool(False)
    SGLANG_DSPARK_ENABLE_MULTI_STREAM = EnvBool(True)
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS = EnvInt(2)

    # ===================================================================
    # Memory pools and KV-cache sizing
    # ===================================================================
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    # Disable lazy compaction in the unified memory pool allocator and
    # fall back to the per-free eager compaction. Used for production
    # A/B and quick rollback. Default False (lazy compaction on).
    SGLANG_DISABLE_LAZY_COMPACTION = EnvBool(False)
    # Sort the multi-ended allocator's free list after a merge (perf A/B knob).
    SGLANG_SORT_FREE_LIST_AFTER_MERGE = EnvBool(False)
    # Periodically log lazy-compaction stats per sub-pool (observability only).
    SGLANG_LOG_LAZY_COMPACTION_STATS = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC = EnvInt(30)
    # HND KV layout folds (page, head) into one paged index for per-kv-head sparse
    # page tables (DP attn); paged backends like trtllm_mha consume it directly.
    SGLANG_USE_HND_KVCACHE = EnvBool(False)

    # Attention (aiter, ROCm): route NEXTN spec draft_extend (EAGLE-v2 KV
    # catch-up) through aiter unified_attention (GQA-packed + split-KV) instead
    # of the occupancy-starved mha_batch_prefill FMHA. Independent kill-switch
    # for the new path; pairs with SGLANG_AITER_UNIFIED_VERIFY. Default on.
    SGLANG_AITER_UNIFIED_DRAFT_EXTEND = EnvBool(True)
    # size the KV pool after CUDA-graph capture
    SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)

    # ===================================================================
    # Scheduler token budgeting and admission
    # ===================================================================
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    # PD decode retraction with the cpu_tensor backup (the KV pool's
    # get_cpu_copy). MAX_GB: the most host memory, in GiB (2**30 B), that the
    # retraction backups held on one TP rank may take, the new one included;
    # projected from token counts and pool shapes as get_cpu_copy allocates
    # (mem_cache/host_copy_bytes.py). A retraction that would pass it aborts
    # that request instead (RETRACT_BACKUP_CAP log lines, mem_cache/common.py).
    # Unset or negative: no cap. PAGEABLE: copy into pageable host memory with
    # blocking copies (exact size, freed on release) instead of pinned memory
    # from torch's caching host allocator (each chunk rounded up to a power of
    # two and kept cached after release).
    SGLANG_RETRACT_BACKUP_MAX_GB = EnvFloat(None)
    SGLANG_RETRACT_BACKUP_PAGEABLE = EnvBool(False)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)
    # PD decode admission (DecodePreallocQueue.pop_preallocated), decode growth
    # reserve: each active request (running, transfer queue, waiting queue, and
    # the requests a pass admits) and the candidate keep max(
    # num_reserved_decode_tokens, ceil(ratio x min(max_new_tokens - output
    # tokens, SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION))) free instead of
    # num_reserved_decode_tokens. Unset, 0 or negative: off. The SWA-tail
    # prealloc path does not use it.
    SGLANG_DECODE_ADMIT_GROWTH_RATIO = EnvFloat(0.0)
    # PD decode admission, retractable KV: with the cpu_tensor retraction
    # backup and SGLANG_RETRACT_BACKUP_MAX_GB set, a running request that a
    # retraction now would abort (its backup over the cap) does not count as
    # retractable. Off: every running request counts. The SWA-tail prealloc
    # path does not use it.
    SGLANG_DECODE_ADMIT_CAP_SAFE_RETRACT = EnvBool(False)
    SGLANG_MAX_NEW_TOKENS_LIMIT = EnvInt(None)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    # Window for the token-weighted recent cache-hit rate used to estimate
    # waiting-queue prefill load.
    SGLANG_CACHE_HIT_RATE_WINDOW_SECONDS = EnvFloat(15.0)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    # Compact extend-attention scheduler tile-budget admission (AMD/HIP-only).
    # Budget <= 0 disables; >0 sets the max prefix-extend tiles per batch.
    SGLANG_PREFILL_TILE_BUDGET = EnvInt(0)
    # Tile-budget mode: "compact" (default, counts actual per-request tiles) or
    # "legacy" (rectangular grid, max_extend_len-shaped).
    # Internal/testing only - users should not need to change this.
    SGLANG_PREFILL_TILE_BUDGET_MODE = EnvStr("compact")
    SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE = EnvInt(16)

    # ===================================================================
    # Scheduler polling, timeouts, and output
    # ===================================================================
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)
    # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    # For non-streaming requests, the scheduler still flushes intermediate
    # output batches to the tokenizer manager every N decoded tokens so that
    # `first_token_time`/TTFT can be recorded. Lower this (e.g. to 1) to get
    # an accurate TTFT for benchmarking; the upstream default of 50 trades
    # off some TTFT-metric accuracy for less IPC overhead.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)

    # ===================================================================
    # Overlap scheduler and pipeline parallelism
    # ===================================================================
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    # Force delay_sample_func for all overlap decode (not just grammar mode),
    # allowing CPU result processing to overlap with subsequent forward computation
    # and reducing the impact of sampling overhead on the critical path.
    SGLANG_ENABLE_DELAY_SAMPLE = EnvBool(False)
    # Force-enable the WAR (write-after-read) barrier for the overlap scheduler
    # even when is_cuda() is False (e.g. AMD/ROCm). On CUDA the barrier is
    # already enabled regardless of this flag (see start_event_loop).
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    # Force the WAR barrier to wait for the whole forward instead of the
    # read-done fastpath event.
    SGLANG_FORCE_COARSE_WAR_BARRIER = EnvBool(False)
    # Enable prefill read-done publication after compliant metadata initialization.
    SGLANG_ENABLE_PREFILL_WAR_READ_DONE = EnvBool(False)
    # PD prefill overlap loop: remove the CPU waits that sit behind the coarse
    # WAR barrier (the barrier makes schedule_stream wait for the forward just
    # launched, so any blocking device read there waits for that forward and
    # serializes the loop). GPU order is unchanged. The umbrella turns on every
    # sub-flag that is not set explicitly; a sub-flag set to 0 stays off.
    # See disaggregation/prefill_loop_unserialize.py.
    SGLANG_PREFILL_UNSERIALIZE_LOOP = EnvBool(False)
    # 6a: copy the EAGLE draft seed (hidden states, top-k p/index, DSA top-k)
    # with the async result D2H instead of blocking reads in result processing.
    SGLANG_PREFILL_ASYNC_DRAFT_D2H = EnvBool(False)
    # 6b: read KV page ids for the P->D send on a side stream ordered after
    # the scheduler's own writes but not behind the barrier.
    SGLANG_PREFILL_KV_PAGE_SNAPSHOT = EnvBool(False)
    # Debug for 6b: also compute the legacy page ids and compare (blocking).
    SGLANG_PREFILL_KV_PAGE_SNAPSHOT_CHECK = EnvBool(False)
    # 6c / 6c-ev: fixed-shape frees (free_segment) for tree eviction and the
    # load-failure path instead of torch.unique (a device sync).
    SGLANG_PREFILL_FIXED_SHAPE_FREE = EnvBool(False)
    # 6d: early-send of the cached prefix without a blocking read inside
    # run_batch; the send is issued after the launch at a fixed loop point.
    SGLANG_PREFILL_EARLY_SEND_ASYNC = EnvBool(False)
    # 6h: write-through backups started during result processing submit their
    # index D2H + copies after the next launch instead of behind the barrier.
    # Also the aggregated overlap loop (colocated / symmetric P/D symmetric) and the
    # PD decode overlap loop (scheduler_components/deferred_write_submit.py).
    SGLANG_PREFILL_DEFER_WRITE_SUBMIT = EnvBool(False)
    # Aggregated overlap loop (colocated / symmetric P/D symmetric): when a launch
    # will wait for the shared host-dedup loader to submit its load (consumer
    # gate pending), process the previous result first, as the PD prefill
    # loop does (scheduler_components/gate_pending_reorder.py). Also turned
    # on by the umbrella; an explicit 0 keeps it off under the umbrella.
    SGLANG_OVERLAP_GATE_PENDING_REORDER = EnvBool(False)
    # 6h with the gate-pending reorder (PD prefill loop, and the aggregated
    # loop's T4): after a launch whose previous result this rank processed
    # early, the flush keeps that result's writes for the next flush. Every TP
    # rank then merges the same writes into each op, which writing_check's
    # count-based settle needs. 0 = the previous per-rank merge (A/B).
    # The M2_LOCKSTEP hazard counters (m2_lockstep_stats.py) count either way.
    SGLANG_M2_WRITE_MERGE_LOCKSTEP = EnvBool(True)
    # DEBUG positive control for the item above, default off: a
    # comma-separated TP-rank list, e.g. "2,3". Those ranks take the
    # gate-pending reorder's early-result path at every launch that has a
    # previous result, with 6h on or off (mem_cache/ce_loader.py
    # gate_pending_for_batch). PD prefill loop, and the aggregated loop when
    # its reorder is on. Collective order unchanged. Logged on every rank at
    # loop start (DEBUG_FORCE_GATE_REORDER).
    SGLANG_DEBUG_FORCE_GATE_REORDER_RANKS = EnvStr("")
    # PD prefill overlap loop, default off: process the previous result before
    # the launch also when its forward has already finished (its copy_done
    # event is complete; a non-blocking query), not only when the launch's
    # shared-loader gate is pending. That result's prefill_finished stamps and
    # KV sends then do not wait behind the next forward; the next launch
    # starts after that processing. The early path is the gate-pending
    # reorder's (6h write-merge lockstep, phase "reorder_prev_result"),
    # decided per rank; collective order unchanged. The aggregated loop is
    # unchanged. Every TP rank logs "RESULT_FIRST loop=pd ..." every 60 s
    # (disaggregation/prefill_result_first.py).
    SGLANG_PREFILL_RESULT_FIRST_WHEN_READY = EnvBool(False)
    # PD prefill overlap loop: run_batch launches the forward block on one
    # worker thread while the scheduler thread processes the previous result
    # (it no longer waits behind the launch, which blocks on a full CUDA
    # launch queue for about one forward). Taken at every launch with a
    # previous result queued, instead of the result-first decision. Every TP
    # rank logs "LAUNCH_THREAD ..." every 60 s
    # (disaggregation/prefill_launch_thread.py).
    SGLANG_PREFILL_LAUNCH_THREAD = EnvBool(False)
    # 6g: the Mooncake transfer worker waits for an early send's
    # _early_send_wait_event (the forward in flight when it was issued) on its
    # own thread before posting the RDMA read. Also turned on by the umbrella.
    SGLANG_MOONCAKE_EARLY_SEND_WAIT = EnvBool(False)
    # M4 Q4: the PD prefill worker keeps every
    # page id of the full KV pool on the CPU (mem_cache/cpu_page_book.py): CPU
    # free lists, a CPU page row per request slot, CPU page ids on every tree
    # value. No device read for KV sends, loads, write-through or frees.
    SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING = EnvBool(False)
    # M4: page quarantine for HiCache loads in the
    # overlap PD prefill loop (mem_cache/load_quarantine.py). Releases are
    # tagged with pass-level epochs; loads get clean pages first and do not
    # wait for the in-flight forward. Requires SGLANG_PREFILL_CPU_PAGE_BOOKKEEPING.
    SGLANG_DSA_HOST_DEDUP_LOAD_QUARANTINE = EnvBool(False)
    # Debug: every N loop iterations compare the CPU page bookkeeping (rows,
    # tree values, free lists) with the device (blocking reads). 0 = off.
    SGLANG_LOAD_QUARANTINE_SHADOW_CHECK_EVERY = EnvInt(0)
    # Restore only the next prefill batch's candidates alongside this forward.
    SGLANG_HICACHE_PREFETCH_NEXT_BATCH = EnvBool(False)
    SGLANG_HICACHE_PREFETCH_MAX_TOKENS = EnvInt(131072)
    SGLANG_DSA_HOST_DEDUP_SYMMETRIC_BROADCAST = EnvBool(False)
    # DSA host-dedup fan-out: "nccl" (layerwise NCCL broadcast; loads block the
    # forward), "ipc" (followers pull the source's rows over CUDA IPC after
    # interprocess events) or "shared" (every TP rank maps the source's host
    # pool and loads its own rows with copy engines: no fan-out, no SM kernel,
    # no cross-rank dependency; needs --hicache-io-backend direct and
    # --hicache-mem-layout page_first_direct).
    SGLANG_DSA_HOST_DEDUP_FANOUT = EnvStr("nccl")
    # MLA host-dedup fan-out (--enable-mla-hicache-host-dedup, e.g. Kimi-K2.6):
    # "nccl" (default, layerwise NCCL broadcast) or "ce_peer" (the C2g
    # copy-engine fan-out of the DSA path; needs --hicache-io-backend direct and
    # --hicache-mem-layout page_first_direct). A separate word, so a DSA
    # profile's SGLANG_DSA_HOST_DEDUP_FANOUT cannot change an MLA baseline.
    # The C2g tuning words (SGLANG_DSA_HOST_DEDUP_PEER_*, LAYER_PARTS, ...)
    # apply to both.
    SGLANG_MLA_HOST_DEDUP_FANOUT = EnvStr("nccl")
    # Shared pool only: before every load, check (one small gloo all-reduce)
    # that all TP ranks load the same host rows. Followers read the source's
    # pages by their own host indices, so a divergence would be a silent
    # wrong read; this fails the load instead.
    SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY = EnvBool(True)
    # Verify every N load ops via a running digest (1 = an all-reduce per op).
    SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY_EVERY = EnvInt(1)
    # Shared pool: one 2D copy per page across layers (needs the layer-strided
    # device pools that shared mode allocates); 0 = per-(page, layer) copies.
    SGLANG_DSA_HOST_DEDUP_SHARED_2D = EnvBool(True)
    # Shared pool: submit loads on a per-rank loader thread (v0.2); 0 = on the
    # scheduler thread. Copies go in chunks of N pages, at most 2 in flight.
    SGLANG_DSA_HOST_DEDUP_SHARED_LOADER = EnvBool(True)
    SGLANG_DSA_HOST_DEDUP_LOADER_CHUNK_PAGES = EnvInt(256)
    # Draft-layer (height-1) copies per cudaMemcpyBatchAsync call; 0 = height-1
    # 2D copies (shortest driver calls; best NCCL-launch p99, more GPU time).
    SGLANG_DSA_HOST_DEDUP_DRAFT_BATCH = EnvInt(64)
    # C2g (FANOUT=ce_peer): a follower fails its load if the source publishes
    # no progress for this long.
    SGLANG_DSA_HOST_DEDUP_PEER_TIMEOUT_S = EnvFloat(120.0)
    # Overlap load-back M0: GPU timings of the shared loader's start/fence
    # waits and of the forward stream's load waits (event pairs resolved
    # lazily, no sync), plus forward-in-flight-at-issue counts. Logging only.
    SGLANG_DSA_HOST_DEDUP_LOAD_TIMING = EnvBool(False)
    # Overlap load-back M1: layer-group
    # consumption of shared / C2g admission loads. LAYER_PARTS = number of
    # layer parts per op (1 = whole op, today); the consuming forward waits
    # only for the part of each layer it reads. LAYER_PART_BOUNDS overrides
    # the uniform split ("0,20,39,58,78"). LAYERWISE_EAGER: run consuming
    # batches without the prefill CUDA graph (fallback only). RACE_AMPLIFIER
    # (debug, correctness runs only): GB/s cap on non-follower H2D plus
    # destination poison before every part.
    SGLANG_DSA_HOST_DEDUP_LAYER_PARTS = EnvInt(1)
    SGLANG_DSA_HOST_DEDUP_LAYER_PART_BOUNDS = EnvStr("")
    # Copy shapes of layer-wise ops: "page" = one
    # copy per page and layer run (today); "hybrid" = hybrid page/layer-major
    # copies on the dedup source, run-coalesced copies on C2g followers,
    # which then wait for the source's whole part. Whole ops stay "page".
    SGLANG_DSA_HOST_DEDUP_PART_PLAN = EnvStr("page")
    SGLANG_DSA_HOST_DEDUP_LAYERWISE_EAGER = EnvBool(False)
    SGLANG_DSA_HOST_DEDUP_RACE_AMPLIFIER = EnvFloat(0.0)
    # I9: redraw the shared loader's copy streams at startup while they alias
    # forward_stream or fail the overlap self-test; raise if they still fail.
    SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW = EnvBool(False)
    # I9 / Q4: keep a preset CUDA_DEVICE_MAX_CONNECTIONS (1-32)
    # instead of the engine's 8. Only for a deployment whose loader streams
    # still fail I9 after redraws; then every configuration compared there sets it.
    SGLANG_HONOR_PRESET_CUDA_DEVICE_MAX_CONNECTIONS = EnvBool(False)
    # C2g: who pulls from whom (star: all from the source; chain: rank i from
    # i-1; tree: rank i from (i-1)//2), and a per-rank cap on the issue rate
    # of peer copies in GB/s (0 = uncapped).
    SGLANG_DSA_HOST_DEDUP_PEER_TOPOLOGY = EnvStr("star")
    SGLANG_DSA_HOST_DEDUP_PEER_RATE_GBPS = EnvFloat(0.0)
    # C2g: the same cap for bootstrap-window prefetch ops only (they overlap
    # forwards; nothing waits for them until admission). Lifted mid-op when an
    # admission load queues behind one. 0 = uncapped.
    SGLANG_DSA_HOST_DEDUP_BWP_PEER_RATE_GBPS = EnvFloat(50.0)
    # C2g, bootstrap-window prefetch ops only (M5): apply
    # the cap above only while a forward is in flight on the rank AND the op's
    # remaining bytes fit the cap before its estimated window ends (minus the
    # margin below); otherwise the op runs uncapped. The pacer restarts its
    # token bucket whenever the effective rate changes. Default off: the cap
    # applies for the whole op until lifted, as before.
    SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP = EnvBool(False)
    SGLANG_DSA_HOST_DEDUP_BWP_ADAPTIVE_CAP_MARGIN_MS = EnvFloat(100.0)
    # Bootstrap-window prefetch (M5): settle finished ops before admission
    # matches (one MIN all-reduce, moved, not added), so a landed op is used
    # instead of deferring its request one pass.
    SGLANG_BWP_POLL_FIRST = EnvBool(False)
    # Bootstrap-window prefetch (M5): admission consumes an in-flight prefetch
    # op (the forward waits for the rest of the copy) instead of deferring the
    # request until the op's ack.
    SGLANG_BWP_CONSUME_INFLIGHT = EnvBool(False)
    # Bootstrap-window prefetch (M5 3.3; needs SGLANG_BWP_CONSUME_INFLIGHT and
    # layer-wise loads, SGLANG_DSA_HOST_DEDUP_LAYER_PARTS > 1): prefetch ops are
    # copied in the same layer parts as admission loads, and a forward that
    # adopted one in flight waits per layer for the op's part of that layer
    # instead of for the whole op at launch.
    SGLANG_BWP_CONSUME_PER_PART = EnvBool(False)
    # Bootstrap-window prefetch, chunked (parts B and B1;
    # needs concurrent submission and depth >= 2): the oldest candidate whose
    # host prefix does not fit the pin budget is loaded in prefix order in
    # chunks of SGLANG_BWP_CHUNK_TOKENS (one op per chunk), up to a separate
    # head budget of SGLANG_BWP_CHUNKED_HEAD_FRACTION x the device pool (and the
    # headroom), with the not-yet-loaded rest host-locked. B1: a candidate
    # skipped for the pin budget is re-examined as soon as the budget frees
    # (not after 64 passes). Pressure release also runs on busy passes.
    # Default off: the prefetcher runs as before.
    SGLANG_BWP_CHUNKED = EnvBool(False)
    SGLANG_BWP_CHUNK_TOKENS = EnvInt(131072)
    SGLANG_BWP_CHUNKED_HEAD_FRACTION = EnvFloat(0.4)
    # With the IPC fan-out: the source loads whole host pages with copy
    # engines into a device staging buffer, then scatters per layer.
    SGLANG_DSA_HOST_DEDUP_CE_LOAD = EnvBool(False)
    SGLANG_DSA_HOST_DEDUP_CE_STAGING_MB = EnvInt(256)
    # IPC fan-out only, experimental: let admission loads overlap the forward
    # layer by layer instead of making the forward wait for the whole op.
    # Not safe yet: it hung under load on GB300 1P1D and the
    # cross-rank cycle is not established. Keep off until it has a hang
    # regression test.
    SGLANG_DSA_HOST_DEDUP_IPC_LAYERWISE = EnvBool(False)
    # DSA: device-load the bounded serving-path Triton kernels (indexer K/S
    # gather, paged allocator, get_last_loc) at backend init and the decode-
    # graph metadata kernels at capture, instead of at first use mid-serving.
    SGLANG_DSA_PRELOAD_SERVING_KERNELS = EnvBool(True)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)

    # ===================================================================
    # Radix and sparse KV caches
    # ===================================================================
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)
    # Kill-switch for the shared-index (IndexShare) swap-in prefetch
    # (auto-enabled for GLM-5.2-style DSA); set True to A/B synchronous swap-in.
    SGLANG_DISABLE_HISPARSE_PREFETCH = EnvBool(False)
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(True)
    # KV events of bigram (EAGLE) radix keys: publish each page's own tokens and
    # a block hash chained over them (not over the (t_j, t_j+1) pairs), so the
    # block hash is a function of what a unigram consumer hashes. Removals are
    # published only when no bigram variant of that page remains on the tier.
    SGLANG_KV_EVENTS_UNIGRAM_BLOCK_HASH = EnvBool(False)
    # KV events of a host->device load-back (6f): the
    # load-back commit enqueues one O(1) entry per loaded node (token-id array
    # reference, hashes, parent hash captured at commit); take() builds the
    # per-page BlockStored events in queue order, off the admission path. The
    # published stream is byte-identical to the eager one.
    SGLANG_KV_EVENTS_LAZY_LOAD_BACK = EnvBool(False)
    # Debug for the above (correctness runs): also build the eager events at
    # commit and compare them with the deferred ones at take(); publishes the
    # commit-time events and logs every mismatch.
    SGLANG_KV_EVENTS_LAZY_LOAD_BACK_CHECK = EnvBool(False)
    # With SGLANG_KV_EVENTS_LAZY_LOAD_BACK: take() builds each lazy entry as
    # the events its per-page events coalesce into (one for the full pages,
    # one for a short last page), with batched slices instead of one object
    # per page and one Python tuple per bigram token. Same events, byte for
    # byte. In the ltval2 profile take() was 82 % of report_prefill_stats on
    # the P result path. With SGLANG_KV_EVENTS_LAZY_LOAD_BACK_CHECK on, the
    # commit-time events are still published, and the merged build is also
    # compared with the per-page build (fast_build counters in the 6f line).
    SGLANG_KV_EVENTS_LAZY_FAST_BUILD = EnvBool(False)
    # Batched KV-event publication, one flush per scheduler pass. Each
    # publish_kv_events() call of the result path (report_prefill_stats,
    # report_decode_stats) only closes the queued segment (O(1)); the loop
    # takes the closed segments once per pass, right after the result
    # processing (in the overlap loops also after the next launch) and ahead
    # of the routing sample and the prefill state push, and sends them in as
    # few ZMQ messages as the caps below allow. The events are the ones one
    # take() per call gave, byte for byte, in the same order; the idle and
    # pause calls still publish at once. Needs a loop that calls the hook (PD
    # prefill, PD decode, and the normal/overlap loops, symmetric included);
    # other loops keep immediate publication.
    SGLANG_KV_EVENTS_BATCH_PUBLISH = EnvBool(False)
    # With SGLANG_KV_EVENTS_BATCH_PUBLISH: the largest encoded event message
    # (bytes; 0 = no cap, values below 4096 are raised to 4096). A batch above
    # it goes out as several messages, split between events; an event above it
    # is split at block boundaries into events of the same kind (a BlockStored
    # chain keeps its parent links), the form the queue has before it
    # coalesces pages. A >1 MiB message once froze the router's indexer bridge.
    SGLANG_KV_EVENTS_BATCH_MAX_BYTES = EnvInt(512 * 1024)
    # With SGLANG_KV_EVENTS_BATCH_PUBLISH: the most events in one message
    # (0 = no limit).
    SGLANG_KV_EVENTS_BATCH_MAX_EVENTS = EnvInt(0)
    # KV-event publication counters on the publishing rank, one INFO line per
    # ~60 s ("KV events publish over N s"): publish calls, deferred calls,
    # flushes, events, lazy entries built, take (build) and publish ms, the
    # longest deferral, and from the ZMQ thread: messages, bytes, the largest
    # message, split batches/events, encode+send ms. Counting only; usable in
    # every configuration of an A/B.
    SGLANG_KV_EVENTS_PUBLISH_STATS = EnvBool(False)
    # Decode batches between SWA out-of-window evictions.
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    # Deprecated: the unified radix tree is the default tree cache now, so the
    # registry no longer reads this. Kept because a few model/arch call sites
    # still assert on it; do not use in new code.
    SGLANG_ENABLE_UNIFIED_RADIX_TREE = EnvBool(False)
    # Registered TreeCore backend serving the unified radix cache.
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("python")
    # TODO(DSV4): @ispobock this has bug on main branch when retract
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
    # Default off; DSV4 window SWA residency turns it on unless it is set
    # (with a position-based condition, ScheduleBatch._maybe_spill_swa_window).
    SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)
    # DSV4 window SWA residency (--dsv4-uniform-swa-residency window): pages
    # per request-SWA spill, the host headroom target as a fraction of host
    # pages, and a per-step debug check that every running request's SWA
    # window is still mapped on the device.
    SGLANG_DSV4_SWA_SPILL_INTERVAL_PAGES = EnvInt(1)
    # Window residency default (unset): 0.01 (mem_cache/dsv4_window_defaults.py).
    SGLANG_DSV4_SWA_HOST_LOW_WATERMARK = EnvFloat(0.03)
    SGLANG_DSV4_SWA_WINDOW_CHECK = EnvBool(False)
    # Debug (window residency, off by default, read once at cache build):
    # compare the device and host rows of SWA and its SWA-anchored state pools
    # bit for bit at every spill ack (before the detached slots are freed) and
    # at every load-back completion. Host syncs; the counters appear under
    # /server_info dsv4_window.copy_verify and mismatches are logged.
    SGLANG_DSV4_SWA_COPY_VERIFY = EnvBool(False)
    # Window residency, prefill, symmetric or non-disaggregated worker:
    # the SWA host pool stores only the tail of each SWA page, the bytes of
    # its last SWA_WINDOW tokens plus the page's scale block. A host hit ends
    # on a page boundary b and attention reads SWA only in [b - window, b),
    # so the page head is never read after a load. The tree, the host token
    # index space and the page counts are unchanged; only the bytes per SWA
    # host page shrink, so the same host bytes (--dsv4-hicache-host-gb) hold
    # more pages, while --dsv4-hicache-host-tokens stays a literal token
    # count that pins fewer bytes. With host dedup it is TP0's one host copy
    # (loads broadcast whole device pages). Refused with a decode role, the
    # peer-KV servicer or GPU export, a HiCache storage backend, or a
    # non-default HiCache layout / io backend (hybrid_pool_assembler). Unset
    # under window residency it defaults on for prefill, symmetric and
    # non-disaggregated workers and falls back to whole pages (one warning)
    # where refused; an explicit 0 or 1 wins (mem_cache/dsv4_window_defaults.py).
    SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY = EnvBool(False)
    # With SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY: the c4 / c4-indexer compress
    # state host pools keep only the last 8 rows of each page's ring (16 rows
    # with MTP): an extend or decode from a page boundary b reads the state
    # of positions [b - 8, b) only (create_paged_compress_data_kernel).
    # Unset, it follows the effective SGLANG_DSV4_WINDOW_HOST_TAIL_ONLY.
    SGLANG_DSV4_WINDOW_HOST_STATE_TAIL_ONLY = EnvBool(False)
    # Debug for the above (works with them on or off): after every
    # host->device SWA page load, overwrite the bytes of the page's head
    # tokens (those before the tail: their data and their scales) with 0xFF,
    # a NaN in fp8 e4m3fn, bf16 and ue8m0, and after every c4 state load the
    # ring rows before the last 8 (a NaN in fp32), so any read of them shows
    # up as NaN. The load-phase SGLANG_DSV4_SWA_COPY_VERIFY skips those bytes.
    SGLANG_DSV4_WINDOW_HOST_TAIL_POISON_HEAD = EnvBool(False)
    # Window residency, symmetric owner: spill an export's SWA below the export
    # floor after its FINAL is published, not before. The spill's D2H copy
    # holds the GPU's D2H copy engine and FINAL waits behind it in its
    # page-index readbacks. 0 restores the spill-then-FINAL order (A/B only).
    SGLANG_DSV4_EXPORT_SPILL_AFTER_FINAL = EnvBool(True)
    # Window residency, symmetric owner, overlap loop: run the chunk-boundary
    # export spill (symmetric_stash_chunked_request) after the next batch is
    # launched, before its WAR barrier, instead of before the launch. The
    # owner's loop is serialized behind its per-chunk publication readback,
    # so every pre-launch ms is GPU idle; after the launch the spill (detach
    # check + D2H submit, ~1 ms per chunk) overlaps the forward. Its
    # range only covers KV of chunks whose results were processed, below what
    # the chunks in flight read. Default off (A/B); window residency default
    # (unset) on symmetric workers: on (mem_cache/dsv4_window_defaults.py).
    SGLANG_DSV4_EXPORT_SPILL_AFTER_LAUNCH = EnvBool(False)
    # Window residency, overlap loops (std prefill, symmetric, colocated): run
    # the chunk-boundary write-through of the previous stash's chain
    # (UnifiedRadixCache._backup_unbacked_chain, the deferred-by-one-stash
    # backup) after the next batch is launched, before its WAR barrier,
    # instead of inside the stash before the launch. Those loops wait for the
    # forward just launched in their result processing, so every pre-launch
    # ms is GPU idle (standard prefill at concurrency 32: inter-chunk gap
    # 5.6-7.5 ms with the work before the launch vs 2.1 ms after it);
    # after the launch the backup (host alloc / eviction + write submission)
    # overlaps the forward. Same chain and final bytes; no device frees move.
    # Default off (A/B); window residency default (unset) on prefill workers:
    # on (mem_cache/dsv4_window_defaults.py).
    SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH = EnvBool(False)
    # Symmetric owner, overlap loop (any residency): read a published chunk's
    # page indices and generations (CHUNK_READY / FINAL) on a side stream that
    # is ordered after the scheduler's work up to the last launch, instead of
    # on the schedule stream that the WAR barrier just fenced behind that
    # forward. Those readbacks then no longer wait for the forward in flight:
    # they held every chunk until the next forward ended (~58 ms spin,
    # then 4.5-6 ms of host work with the GPU idle) and every FINAL by ~25 ms.
    # They only read launched, export-held state (req_to_token rows, page
    # generations, full->SWA mapping), so overlapping the forward's reads is
    # safe. Default off (A/B, both arms).
    SGLANG_SYMMETRIC_PUBLISH_UNFENCED = EnvBool(False)
    # Symmetric owner, overlap loop: read a published chunk's page ids and page
    # generations with copy-engine DMAs into pinned host memory, on a side
    # stream (checked at loop start not to share a hardware queue with the
    # forward stream) ordered after the same pre-WAR marks; a row written after
    # the mark takes the old read (symmetric/publish_readback.py). The
    # publication no longer waits for the forward in flight, nor for SMs
    # (Kimi-K2.6 computer-use workload: 1,110 ms per pass, GPU idle 6.7 % of P
    # busy). Covers plain paged pools (MLA, DSA); other shapes keep the
    # flag-off path. Default off (A/B, both arms).
    SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK = EnvBool(False)
    # Smoke only: also run the old read (it waits for the forward) and compare.
    SGLANG_SYMMETRIC_PUBLISH_ASYNC_READBACK_CHECK = EnvBool(False)
    # Opt-in ordinary symmetric imports: reuse existing L2 only when a complete
    # packed restore can start now. Busy or unavailable restores retain READ.
    SGLANG_SYMMETRIC_IMPORT_L2_RESTORE = EnvBool(False)
    # Skip only the attempt-bound, all-TP common prefix advertised by admitted D.
    SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION = EnvBool(False)
    # Exporting (P-role) symmetric worker: its release-time insert of the
    # computed prompt leaves the node shape a standard P leaves, one node per
    # prefill chunk (cut at the chunk ends this prompt ran), not one node for
    # the whole suffix. Host eviction frees whole nodes, so one big node loses
    # the whole context where a standard P trims only tail chunks. FULL-only
    # unified trees. Default off (A/B).
    SGLANG_SYMMETRIC_RELEASE_INSERT_CHUNKED = EnvBool(False)
    # Symmetric worker with --kv-events-config (unified radix tree): publish
    # no KV events for radix nodes that D-role work created: the prompt an
    # import inserts, a D-leg's decode and finish inserts, a D-leg's local
    # recompute, and later the host copies and removals of those nodes. In
    # force-split P/D the router reads only P-role events (Dynamo indexer of
    # the P-role workers, gateway cache-aware tree); a standard D's tree is
    # read by no router either. P-role nodes publish as before. When P-role
    # work inserts a path through or below such nodes, they are published
    # first (one BlockStored per tier they hold), so a consumer never gets a
    # child without its parent or a removal without its store. Counters in
    # the SGLANG_KV_EVENTS_PUBLISH_STATS line. Default off (A/B).
    SGLANG_SYMMETRIC_SKIP_D_ROLE_KV_EVENTS = EnvBool(False)
    # D admission (standard and symmetric loops): the PREBUILT batch's host
    # data (req_pool_indices, seq_lens, orig_seq_lens, last tokens, the EAGLE
    # draft seed) goes to the GPU from pinned host memory with non_blocking
    # copies on the same (schedule) stream, instead of blocking copies. A
    # blocking copy synchronizes that stream, which waits behind the forward
    # in flight (WAR barrier), so the radix insert and admit_pending after it
    # ran with the GPU idle (measured on GB300: prebuilt_build 12.4 ms per
    # import). GPU order is unchanged; only the host wait moves to the first
    # remaining real sync (the DSA seed's .item(), after the insert). The
    # caching host allocator keeps each pinned block until its copy is done.
    # Default off (A/B).
    SGLANG_PREBUILT_NONBLOCKING_H2D = EnvBool(False)
    # Compute-first (late D) owner: publish computed KV right after each
    # processed result, not only in the next symmetric_pre_schedule. With the
    # gate-pending reorder a result processed before a launch otherwise waits
    # for that whole launch (GB300 3P6D symmetric at concurrency 90: 33 % of
    # requests, mean 0.69 s Pdone -> first publication). The call shares the
    # pass's catch-up budget and runs at the same point of every TP rank's
    # loop. Default off (A/B).
    SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT = EnvBool(False)
    # Compute-first (late D) owner: charge the per-pass publication budget
    # (SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS) only to the KV that an export
    # had computed before its reader became ready (fixed at the bind). KV
    # computed after the bind, and the FINAL, are published without the
    # budget once that KV is out. This is not the standard P/D rule: it does
    # not meter the result of a forward in flight at the bind (a multi-chunk
    # export that binds during its first forward publishes its whole first
    # result at once) or results that arrive during the catch-up, and it
    # meters a parked export and the catch-up left at the last result, which
    # standard P/D sends at once (ExportTable._publish_computed_pre_bind).
    # Off, one budget per pass covers every publication in transfer-id order
    # and stops when it is spent (GB300 3P6D symmetric at concurrency 90,
    # HiCache ratio 4.0: 1.38 s of the 1.69 s mean KV handoff tail).
    # Default off (A/B).
    SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY = EnvBool(False)
    # Compute-first (late D) owner: charge the per-pass publication budget
    # (SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS) only to what standard P/D
    # meters for a late-D request (prefill.py late_d_catchup): the catch-up
    # of an export that was launched before its bind, from the bind until
    # its published KV reaches its computed KV (that end moves with each
    # result). Other KV goes out at its result, and FINAL goes out at once
    # after the last result (a parked export: at the bind), with all KV
    # left. Differences from standard P/D: catch-ups are served in
    # transfer-id order, and the device prefix at the bind is part of the
    # catch-up (ExportTable._publish_computed_standard_pd). Exclusive with
    # SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY. Default off (A/B).
    SGLANG_SYMMETRIC_BUDGET_STANDARD_PD = EnvBool(False)
    # Compute-first owner: order of publish_computed within the per-pass
    # catch-up budget. 0 (default): transfer-id (FIFO) order. N > 0: rooms
    # that have had something to publish for >= N passes first, in FIFO
    # order; then the rest by smallest remaining publication. FIFO lets
    # small publications wait behind large cached-prompt ones (GB300 3P6D at
    # concurrency 90: every launch window a waiting request skipped was at
    # the cap; one 576-token request waited 15 s behind 250k-380k-token
    # rooms). Pass counts, not clocks, so every TP rank orders alike.
    # Exclusive with SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY and
    # SGLANG_SYMMETRIC_BUDGET_STANDARD_PD: their paths keep transfer-id
    # order, so ExportTable refuses N > 0 with either of them.
    SGLANG_SYMMETRIC_PUBLISH_SMALL_FIRST_AGE_PASSES = EnvInt(0)
    # Symmetric worker: run the bootstrap prefetcher (--disaggregation-
    # bootstrap-prefetch, with SGLANG_BWP_CHUNKED) on the waiting
    # queue through the lookahead plan; a symmetric worker has no bootstrap
    # queue. Pins also leave room for running decodes and bound imports.
    # Off (default): the prefetch flags stay inactive on symmetric workers,
    # so shared P/D launch configs keep today's behavior. Needs the overlap
    # loop. Default off (A/B).
    SGLANG_SYMMETRIC_BWP = EnvBool(False)
    # BWP v2: umbrella flag. Each part
    # below runs only with SGLANG_BWP_V2=1 AND its own flag. All default off;
    # with them off the prefetcher takes today's decisions (golden traces,
    # test_bwp_v2_golden.py).
    SGLANG_BWP_V2 = EnvBool(False)
    # C3: on a symmetric worker the prefetcher owns its lookahead plan and
    # installs no scheduler.late_d_lookahead (so SGLANG_SYMMETRIC_BWP does not
    # change the late-D bound reserve input); the routing snapshot reads it
    # through plan_view().
    SGLANG_BWP_SYM_PRIVATE_PLAN = EnvBool(False)
    # C4 (waiting source): admission-first memory. Pins leave room for every
    # row the next formation admits per the real plan and the first row by
    # due pass, summed so that any formation order fits: what PrefillAdder
    # takes for the row (host load, page-rounded extend, clipped max_new, a
    # page; at least its planned need + 1 and its late-D park-gate terms)
    # plus the unlocked device prefix its admission lock takes (shared nodes
    # once); the candidate's own park-gate terms; the bound reserve once; at
    # least today's reserve; plus the symmetric reserve. Pressure release
    # also runs before formation (after the MIN poll); it frees latest-due
    # pins first, never an in-flight pin or a pin of a row admitted next, the
    # first-due row's pin last, and beyond today's release target only if
    # that makes the next formation fit (reason admission_first).
    SGLANG_BWP_ADMISSION_SAFE = EnvBool(False)
    # C5 (waiting source, concurrent submission): loader backlog guard. A
    # new op is submitted only if the host tokens this pass's formation queued
    # for admission loads, plus the in-flight ops' tokens, plus the op fit in
    # one nominal forward at the nominal load rate (this rate x
    # SGLANG_JIT_PULL_NOMINAL_PASS_S, 1.0 M tokens/s x 0.38 s), so an
    # admission load never queues behind more prefetch than a forward hides.
    # Constants, not measured rates: every TP rank decides alike.
    SGLANG_BWP_FIFO_GUARD = EnvBool(False)
    SGLANG_BWP_FIFO_GUARD_LOAD_TOKENS_PER_S = EnvInt(1_000_000)
    # C6 (waiting source; symmetric and standard P/D alike): gate-free due
    # pass, the rule "the late-D park gate never stops or delays a host
    # prefetch". Prefetch selection order, due pass, op size, cap lifts,
    # uncapping and the adaptive-cap deadline come from the plan rerun with
    # every late-D blocker cleared (park gate, hold / intent skips), so they
    # read no late-D gate input. Memory (the C4 reserve and release) still
    # keeps room for the row the real plan admits next. Needs
    # SGLANG_BWP_ADMISSION_SAFE (and SGLANG_BWP_FIFO_GUARD with concurrent
    # submission): the guards land before the part that adds candidates.
    # Two known exceptions, neither in the symmetric P/D deployment
    # (ce_peer fan-out, so concurrent submission; no JIT pull):
    # - F6, serial submission (NCCL / CUDA IPC host-dedup fan-out): the
    #   GPU-idle test still reads which waiting rows the real plan admits, so
    #   a closed gate can make a pass idle and add a prefetch. It never
    #   removes or delays one (a gate-free idle test would stop the prefetch
    #   of gated rows instead).
    # - F7, outside BWP code (standard P/D with JIT pull): a deferred peer
    #   pull is issued at its row's real start (gated rows: the gate-open
    #   start; hold / intent rows have none), and BWP skips a row while its
    #   pull is pending, so a late-D hold or intent still delays that row's
    #   local host prefetch. Pull timing is out of scope here.
    SGLANG_BWP_GATE_FREE = EnvBool(False)
    # C7 (waiting source): growth re-check. A row examined for good (host
    # part below the minimum, or pinned and later released for TTL /
    # pressure / admission-first) is examined again once its plan host part
    # grew by at least max(min tokens, one page) over the size seen then
    # (its device copy was evicted to the host while it waited).
    SGLANG_BWP_GROWTH_RECHECK = EnvBool(False)
    # C8 (waiting source): fresh-row step plan. In a pass whose first
    # SGLANG_LATE_D_PLAN_MAX_REQS waiting rows hold 1 to
    # SGLANG_BWP_STEP_PLAN_MAX_FRESH rows the post-result plan has not seen
    # (they arrived this pass and the formation left them), the step plans
    # once more after the formation: the post-result candidates still
    # waiting (cached peeks), the fresh rows (read-only peeks, not stored)
    # and the running chunked request; start 1 is the next formation. The
    # step plan is private to that step (never installed). Count limits
    # only (no clock), so every TP rank plans alike. Needs the C4/C5 guards.
    SGLANG_BWP_STEP_PLAN = EnvBool(False)
    SGLANG_BWP_STEP_PLAN_MAX_FRESH = EnvInt(8)
    # Symmetric worker: the overlap loop launches each plain extend pass that
    # has a previous result queued on one worker thread, and processes that
    # result on the scheduler thread meanwhile (it no longer waits behind
    # the launch, which blocks on a full CUDA launch queue for about one
    # forward). Decode, mixed and the first pass after idle launch inline as
    # before. The PD loop's SGLANG_PREFILL_LAUNCH_THREAD has no effect on a
    # symmetric worker. Needs the overlap loop. Every TP rank logs
    # "SYM_LAUNCH_THREAD ..." (disaggregation/symmetric/launch_thread.py).
    # Default off (A/B).
    SGLANG_SYMMETRIC_LAUNCH_THREAD = EnvBool(False)
    # Symmetric worker: the import walk (ImportQueue.admit_pending: the D-leg
    # prealloc, the late-D walk and BEGIN) runs right after the launch in the
    # overlap loop instead of inside symmetric_pre_schedule before it. Its
    # first blocking device read (the prealloc's length tensors) waits,
    # through the last WAR barrier, for the forward in flight; before the
    # launch that wait delays the launch by most of a forward (GLM-5.2 on
    # GB300: allocation steps 54 ms launch to launch vs 26 ms). After
    # the launch it overlaps that forward, whose result the loop waits for
    # next anyway. A pass that may process the previous result before its
    # launch (idle, disable-overlap, symmetric launch thread, pending
    # shared-loader gate) walks before the launch, after batch formation.
    # Needs the overlap loop (refused otherwise). Every TP rank logs
    # "SYM_ADMIT_AFTER_LAUNCH ..." (disaggregation/symmetric/
    # admit_after_launch.py). Default off (A/B).
    # Startup refuses it without late D, with M4, with the bootstrap-window
    # prefetch or SGLANG_HICACHE_PREFETCH_NEXT_BATCH, and without
    # SGLANG_HICACHE_DECODE_CLEAR_CONSUMER (admit_after_launch.startup_refusal).
    SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH = EnvBool(False)
    # A decode pass drops the HiCache load consumers that its batch object
    # still carries from the prefill pass that consumed them: an extend batch
    # adopted as the running batch keeps hicache_consumer_index (and
    # hicache_extra_waits), so each later decode launch calls
    # set_consumer(stale slot) and waits on whatever load reuses that slot.
    # The extend forward already waited for its loads on the forward stream,
    # so no decode pass needs the wait. Every TP rank logs
    # "DECODE_CLEAR_CONSUMER ..." (managers/scheduler_components/
    # decode_load_consumer.py). Default off: the off path is unchanged.
    SGLANG_HICACHE_DECODE_CLEAR_CONSUMER = EnvBool(False)
    # Window residency: copy request-SWA spills to the host with SM kernels
    # instead of the staged copy-engine path. A copy-engine D2H blocks every
    # later device->host readback on any stream (.item(), .cpu(), batch
    # result copies) until it is done; an SM copy runs beside them (H100: a
    # 165 MB spill delays a 1 KB readback by 2.0 ms staged, 0 ms SM), but
    # takes SMs from the forward. Unset: chosen per spill role
    # (unified_radix_cache.swa_spill_sm_copy): the SM copy for prefill-side
    # spills (symmetric owner exports; owner chunk gap 9.8 ms staged vs
    # 4.0 ms SM), the staged copy for the decode-batch spills of a DECODE
    # engine or a SYMMETRIC D-leg (decode step: 0.8 ms staged vs 2.2 ms
    # SM). 1 / 0 force the SM / staged copy for every spill; the declared
    # default is not read.
    SGLANG_DSV4_SWA_SPILL_SM_COPY = EnvBool(True)
    # Opt-in, DSv4 host pools: the same SM copy for every HiCache write-through
    # (default off: those copies keep the staged copy-engine path).
    SGLANG_HICACHE_WRITE_BACK_SM_COPY = EnvBool(False)
    # Opt-in, --hicache-io-backend direct with --hicache-mem-layout
    # page_first_direct (MLA KV and DSA indexer host pools): HiCache
    # write-throughs use the kernel io backend's staged copy (SM relayout
    # kernel into a device staging buffer, then one copy-engine entry per
    # host page) instead of one cudaMemcpyBatchAsync entry per page x layer,
    # and the device indices stay on the GPU (no blocking .cpu() on the
    # scheduler thread). Host bytes are the same; loads keep the direct path.
    # Default off: the direct path, unchanged.
    SGLANG_HICACHE_DIRECT_LAYOUT_KERNEL_BACKUP = EnvBool(False)
    # Window residency with a peer KV servicer: publish each request's spilled
    # SWA pages to the servicer (SWA-only shadow entries, keyed like a
    # reader's backfill lookup) as each spill acks, instead of only once the
    # finished prompt is inserted and its FULL write-through acks. A sym
    # reader's backfill can then be served right after the owner's FINAL
    # (at concurrency 32: first hit 0.4-0.7 s after the first lookup, 23-26 % of
    # backfills never served). Default off (A/B).
    SGLANG_DSV4_PEER_PUBLISH_SPILL_SWA = EnvBool(False)
    # Window residency, SWA backfill: nice value of the backfill IO thread.
    # Its RDMA read (Mooncake batch_transfer_sync_read) busy-polls a CPU for
    # the whole transfer, on every TP rank at once (the reads are lockstep).
    # A positive value lets the scheduler and gloo threads preempt it (0:
    # unchanged; 19 recommended with SGLANG_MOONCAKE_WORKER_NICE). Window
    # residency default (unset): 19.
    SGLANG_DSV4_SWA_BACKFILL_IO_NICE = EnvInt(0)
    # Set by the launcher: NAME=VALUE of each env it set as a window residency
    # default (mem_cache/dsv4_window_defaults.py; read by the
    # DSV4_WINDOW_CONFIG line). Not a setting.
    SGLANG_DSV4_WINDOW_DEFAULTED = EnvStr("")

    # ===================================================================
    # PD disaggregation runtime
    # ===================================================================
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT = EnvInt(1)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    # --disaggregation-late-d: the prefill bounds a late room by an admission
    # timer (sender creation -> first ADMIT; unset: max(1800,
    # SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT)) and then by the bootstrap
    # timeout from the first ADMIT. The decode aborts a request DEFERRED for
    # longer than the TTL (unset: admission timeout + 120 s). Seconds.
    SGLANG_LATE_D_ADMISSION_TIMEOUT = EnvInt(None)
    SGLANG_LATE_D_DEFERRED_TTL = EnvInt(None)
    # Late D in symmetric mode (symmetric P/D), seconds unless noted. Owner rank 0
    # bind timer, from the first ADMIT to the reader's BEGIN (unset: max(900,
    # bootstrap timeout, waiting timeout)). Reader rank 0: an ADMITTED or HELD
    # entry not allocated within the HELD TTL ends with HTTP 503 (unset: bind
    # timer + 120 s); a control whose request has not arrived yet waits at
    # most the early-control TTL.
    SGLANG_LATE_D_BIND_TIMEOUT = EnvInt(None)
    SGLANG_LATE_D_HELD_TTL = EnvInt(None)
    SGLANG_LATE_D_EARLY_CONTROL_TTL = EnvInt(60)
    # Symmetric reader: while the oldest held import (the claim) is blocked by
    # memory, new prefills of requests without an export leave its pages
    # free; after CLAIM_ESCALATE_PASSES such walk passes (unset: 2 x
    # --disaggregation-late-d-claim-iters; 0: never) new P-leg starts do too,
    # capped at the import budget left. 0 / false only for ablations.
    SGLANG_LATE_D_CLAIM_HOLDS_LOCAL = EnvBool(True)
    SGLANG_LATE_D_CLAIM_ESCALATE_PASSES = EnvInt(None)
    # Symmetric owner: a bound late P-leg enters the prefill queue at its
    # arrival position, and only earlier requests count as ahead of a held one
    # (false only for an ablation).
    SGLANG_LATE_D_ARRIVAL_ORDER = EnvBool(True)
    # Debug: every N late walk passes all TP ranks compare a digest of the walk
    # and raise on a mismatch (0: off).
    SGLANG_LATE_D_WALK_DIGEST_EVERY = EnvInt(0)
    # Compute before binding to a decode worker. These values are read only
    # when --disaggregation-late-d-lead-mode=need is selected.
    SGLANG_LATE_D_CATCHUP_TOKENS_PER_PASS = EnvInt(78000)
    SGLANG_LATE_D_PARK_CAP = EnvFloat(0.35)
    SGLANG_LATE_D_PARK_EXIT = EnvStr("abort")
    SGLANG_LATE_D_PARK_TIMEOUT_S = EnvFloat(300.0)
    # Park-gate aging (symmetric compute-first workers; 0 = off, the default).
    # A waiting request first refused this many seconds ago (by the park gate,
    # or by an aged head's hold-back; owner rank 0's clock, agreed in the export
    # table's MIN vote) is aged; the aged request that arrived first is the
    # worker's aged head: it is scanned first, its park cap is
    # SGLANG_LATE_D_PARK_AGED_CAP, and while it is refused no other unbound
    # request starts, for at most SGLANG_LATE_D_PARK_AGED_BLOCK_S seconds from
    # its first hold-back. 1e9 = log-only (stock admission, aging diagnostics).
    SGLANG_LATE_D_PARK_AGE_S = EnvFloat(0.0)
    # The aged head's park cap (fraction of the pool); 0 = SGLANG_LATE_D_PARK_CAP.
    SGLANG_LATE_D_PARK_AGED_CAP = EnvFloat(0.0)
    # How long one aged head may hold the other unbound requests back.
    SGLANG_LATE_D_PARK_AGED_BLOCK_S = EnvFloat(10.0)
    SGLANG_LATE_D_WARM_HOLD_MIN_TOKENS = EnvInt(0)
    SGLANG_LATE_D_BIND_MARGIN = EnvInt(1)
    SGLANG_LATE_D_PLAN_MAX_REQS = EnvInt(32)
    SGLANG_LATE_D_PLAN_REMATCH_PASSES = EnvInt(4)
    # CPU restore source queue: "waiting" or "bootstrap". Empty (default)
    # selects "waiting" in need lead mode and "bootstrap" otherwise.
    SGLANG_BWP_SOURCE = EnvStr("")
    SGLANG_BWP_LEAD_TOKENS_PER_PASS = EnvInt(131072)
    SGLANG_BWP_LEAD_MARGIN = EnvInt(1)
    SGLANG_BWP_DUE_OP_TOKENS = EnvInt(262144)
    # Test only: delay ADMIT to exercise computation without a destination.
    SGLANG_TEST_LATE_D_DELAY_ADMIT_S = EnvFloat(0.0)
    SGLANG_TEST_LATE_D_DELAY_ADMIT_RID_PREFIX = EnvStr("")
    SGLANG_TEST_LATE_D_FORCE_BOUND_RID_PREFIX = EnvStr("")
    SGLANG_TEST_LATE_D_WITHHOLD_ADMIT_RID_PREFIX = EnvStr("")
    # Late-D overhead readout: the scheduler leader times its request broadcast
    # to the other ranks (sglang:late_d_recv_broadcast_* by payload; metrics
    # on). Always on for a decode with late D; 1 turns it on for any other
    # server too (the late-D-off baseline of an A/B, a prefill).
    SGLANG_RECV_BROADCAST_READOUT = EnvBool(False)
    # symmetric P/D symmetric mode (disaggregation/symmetric, disaggregation/pull):
    # control-plane poll/flush interval of the owner and reader IO threads in
    # ms. Every FINAL -> last READ -> DONE hop waits up to one interval, so the
    # old 50 ms default added up to ~150 ms per request to TTFT.
    SGLANG_SYMMETRIC_PD_PLANE_POLL_MS = EnvInt(5)
    # Reader-side prefix reuse: match the D-leg prompt against the local radix
    # tree and import only the missing suffix (upstream PD's decode_prefix_len).
    SGLANG_SYMMETRIC_PD_IMPORT_PREFIX_REUSE = EnvBool(True)
    # D-role SWA-tail import: on a symmetric engine whose SWA pool is smaller
    # than its full pool (no --dsv4-uniform-cache-lifecycle, i.e. the upstream
    # decode shape --swa-full-tokens-ratio 0.1), the reader allocates and READs
    # SWA pages only for the sliding-window tail of the imported prompt, like
    # upstream PD decode (`DecodePreallocQueue._uses_swa_tail_prealloc`).
    # Auto-detected from the allocator / tree; this only turns it off.
    SGLANG_SYMMETRIC_PD_IMPORT_SWA_TAIL = EnvBool(True)
    # Symmetric import-admission liveness (both default off; set them on every
    # symmetric worker). An import preallocates its whole prompt and then
    # waits for the peer's P-leg; when both workers fill their pools with such
    # imports, neither can run the other's P-leg and both wedge. F-A: import
    # admission leaves the KV the local in-flight chunked prefill still has to
    # allocate, so a started P-leg always finishes.
    SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE = EnvBool(False)
    # F-B: active imports (whole prompt + decode headroom each) may hold at most
    # this fraction of max_total_num_tokens; an import above it is admitted only
    # when no other import is active. 0 = no cap. The liveness argument needs
    # a P-leg's prompt + max_new + one page to fit in the remaining (1 - f) of
    # the pool; 0.5 suits GLM-5.2 on GB300.
    SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION = EnvFloat(0.0)
    # F-C, reader-first exports: a P-leg enters the prefill queue only once its
    # reader admitted the import (wire BEGIN, agreed by every TP rank). A
    # finished export then always waits on a reader that already holds its own
    # pages; without it, finished exports can fill the owner's pool while their
    # readers' budgets are full of imports waiting for the P-legs those pages
    # would run. Readers send BEGIN only with this set. A held export whose
    # reader has not begun within SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER_TIMEOUT
    # seconds (0: SGLANG_DISAGGREGATION_WAITING_TIMEOUT) is queued anyway, as
    # without this flag.
    SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER = EnvBool(False)
    SGLANG_SYMMETRIC_PD_EXPORT_AWAIT_READER_TIMEOUT = EnvFloat(0.0)
    # /v1/loads on a symmetric worker: count the held P-legs (prompt x recent
    # cache-miss rate) in num_waiting_uncached_tokens. Unset: on with
    # --disaggregation-late-d, else off; set 1 for F-C without late D.
    SGLANG_SYMMETRIC_PD_LOADS_COUNT_HELD = EnvBool(None)
    # Symmetric mode: the streamed-response background abort
    # (TokenizerManager.create_abort_task, 2 s after the stream ends) reaches
    # the scheduler only while the request is still live, as upstream (a no-op
    # for finished requests). Off restores the earlier behavior, where
    # that abort also revokes a completed P response's export, so a reader
    # that has not finished within ~2 s (D admission waiting for pages) falls
    # back to recomputing the prompt locally. Explicit /abort_request calls
    # revoke completed exports either way.
    SGLANG_SYMMETRIC_POST_STREAM_ABORT_LIVE_ONLY = EnvBool(True)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS = EnvInt(16384)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS = EnvInt(0)
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)
    # PD decode with decode radix + HiCache: TP0 logs a start and a done line
    # per host->device restore op (tokens, pages, fan-out, copy / wall /
    # stream-wait ms; disaggregation/decode_hicache_restore_log.py).
    SGLANG_DECODE_HICACHE_RESTORE_LOG = EnvBool(True)
    # Deferred decode-side KV release: on abort, hold an in-flight request's KV
    # pages/slot until the prefill acks the transfer drained, or the timeout
    # below fires. Off by default (no behavior/perf impact when disabled).
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE = EnvBool(False)
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT = EnvFloat(30.0)
    # Mooncake, NIXL: when a prefill room fails before any KV chunk was sent
    # (bootstrap timeout, abort), tell the decode side so it drops the request and
    # frees its KV now, including a request still in its preallocation queue,
    # instead of holding it until SGLANG_DISAGGREGATION_WAITING_TIMEOUT. Decode
    # metadata that arrives later for such a room is answered with Failed, and
    # decode remembers a notice that beats its own request for 60 s. A prefill
    # admission abort (prompt larger than the prefill pool) takes this path.
    SGLANG_DISAGGREGATION_NOTIFY_DECODE_ON_PREFILL_FAILURE = EnvBool(True)

    # ===================================================================
    # Distributed and model-parallel runtime
    # ===================================================================
    SGLANG_ENABLE_CP_V2 = EnvBool(False)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Comma-separated bundle indices for Ray Custom PG mode (e.g., "0,1,2,7").
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Replicate the input embedding across TP ranks instead of sharding it
    # along the vocab dimension (saves an all-reduce/all-gather in the embed
    # lookup at the cost of replicated embedding weights). Drives both the
    # target and every draft that shares its embedding (see
    # get_embedding_tp_kwargs); they must stay in lock-step. Currently only
    # applies to the Deepseek-V2 family (Deepseek V3.1, Kimi K2.5) + drafts.
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)

    # ===================================================================
    # Tool calling and native web search
    # ===================================================================
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)
    # Native web search (Exa). EXA_API_KEY is the vendor BYOK credential
    # (kept as-is, not renamed to SGLANG_*); the SGLANG_EXA_* knobs tune the
    # request defaults for the built-in GPT-OSS web_search tool.
    EXA_API_KEY = EnvStr(None, secret=True)
    SGLANG_EXA_NUM_RESULTS = EnvInt(10)
    SGLANG_EXA_SEARCH_TYPE = EnvStr("auto")
    SGLANG_EXA_INCLUDE_HIGHLIGHTS = EnvBool(True)
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # ===================================================================
    # HiCache storage backends and mmap allocation
    # ===================================================================
    # Per-call cudaHostRegister limit in GB.
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB = EnvInt(256)
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    # File-backend LRU eviction (opt-in; sizes accept SI/IEC suffixes, "0" disables).
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    # Enable client-side metadata caching to optimize filesystem checks (e.g. for Lustre/NFS/FUSE)
    SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE = EnvBool(False)
    # Positive cache TTL for filesystem metadata lookups (-1 disables positive expiration)
    SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL = EnvFloat(5.0)
    # Buffer mode: pin a staged prefetch's device anchor from IO commit to
    # consumption so eviction cannot waste the fetch; cap = fraction of pool.
    SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK = EnvBool(False)
    SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP = EnvFloat(0.5)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Enable O_DIRECT when opening NIXL POSIX backend files (bypasses OS page cache).
    # Disable with SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0 or via the
    # "use_direct_io": false key in --hicache-storage-backend-extra-config.
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    # Hint-only storage backends (peer_kv): skip the storage prefetch and the
    # storage hit query of a request that carries no kv_pull_hint, which can
    # never hit. 0 restores the old always-prefetch behavior (A/B only).
    SGLANG_HICACHE_SKIP_HINTLESS_PREFETCH = EnvBool(True)
    # Storage backends whose writes are acknowledged no-ops (peer_kv): issue
    # no storage backup ops for them. 0 restores the old no-op backups (A/B).
    SGLANG_HICACHE_SKIP_NOOP_STORAGE_BACKUP = EnvBool(True)
    # --enable-kv-session-retention host eviction: within a retention class,
    # nodes covered by a registration with a queued or running request on this
    # worker are evicted last (soft pin, no recency change). 0 = ablation: the
    # key without this level. The counters and their KV_SESSION_STATS fields
    # are kept either way.
    SGLANG_KV_SESSION_ACTIVE_SOFT_PIN = EnvBool(True)
    # --enable-kv-session-retention host eviction, the order inside class 1
    # (nodes of live registrations; mem_cache/unified_cache/retention_order.py).
    # predicted_return_fixed (default): the node whose agent is predicted to
    # return last goes first, by the expected remaining idle time m(idle age)
    # from a fixed table (SGLANG_KV_SESSION_RETENTION_MTAU_JSON, else the
    # built-in table) on a TP-lockstep clock. lru: CPU access tick.
    # predicted_return (L1, online): the same key with m learned from this
    # worker's return gaps (the LRU key until 200 gaps). Class 0, the soft
    # pin and the kind level are unchanged. Every value keeps the same
    # KV_SESSION_STATS counters. Another value refuses to start.
    SGLANG_KV_SESSION_RETENTION_ORDER = EnvStr("predicted_return_fixed")
    # predicted_return_fixed: m(tau) in seconds for 1-s idle-age bins
    # tau = 0, 1, 2, ... s (the last bin also serves every longer tau), as a
    # JSON list or {"m_s": [...], "dead_after_s": optional, "label":
    # optional}, inline or in a file (path). Unset: the built-in table
    # (retention_order.DEFAULT_FIXED_TABLE). A bad table refuses to start.
    SGLANG_KV_SESSION_RETENTION_MTAU_JSON = EnvStr(None)
    # Symmetric late D: the owner adds its P leg's
    # kv_session tag to the ADMIT it sends the chosen reader, and the reader
    # registers its D leg with that tag when the ADMIT applies, so only the
    # admitted candidate registers. Off: ADMIT frames are unchanged.
    SGLANG_KV_SESSION_ADMIT_TAG = EnvBool(False)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")
    # Bind host-dedup source-pool stripes to the owner's and the other NUMA node
    # (mbind before first touch); see pool_host/numa_split.py. 1 = on (refuse to
    # start if a node cannot hold its share), 0 = off, unset = auto: on for GB300
    # hosts, and for DeepSeek-V4 host dedup on H100 hosts (there with the private
    # mapping), falling back to the default allocation if it cannot be placed
    # (pool_host/numa_split_policy.py).
    SGLANG_DEDUP_POOL_NUMA_SPLIT = EnvBool(None)
    # Owner (GPU-local) node share of each dedup pool; unset = the auto
    # platform's fraction (0.5 on GB300), or with =1 as much as fits.
    SGLANG_DEDUP_POOL_LOCAL_FRACTION = EnvFloat(None)
    # Available GB (MemFree + file LRU) each node must keep after placement.
    SGLANG_DEDUP_POOL_NUMA_RESERVE_GB = EnvFloat(64.0)
    # Opt-in (needs SGLANG_DEDUP_POOL_NUMA_SPLIT): "private" maps the dedup pool
    # as private anonymous memory with MADV_DONTFORK instead of shared anonymous
    # (shmem). Pinned private anon pages are skipped by compaction, so the GPU's
    # ATS translations of the pool are not invalidated by futile migrations.
    # Unset: shared, except under the H100 DeepSeek-V4 auto split (private); an
    # explicit "shared" turns that auto split off.
    SGLANG_DEDUP_POOL_MAP = EnvStr("shared")
    # Opt-in (needs SGLANG_DEDUP_POOL_MAP=private): MADV_HUGEPAGE on the pool, so
    # it is backed by PMD-size THPs where free blocks allow (fewer ATS entries).
    SGLANG_DEDUP_POOL_THP = EnvBool(False)

    # ===================================================================
    # KV-transfer staging and Mooncake transport
    # ===================================================================
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)

    # ===================================================================
    # Mooncake store
    # ===================================================================
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    # Nice value for the threads a Mooncake TransferEngine starts in
    # initialize (RDMA workers, monitor, handshake): the workers spin for
    # 100 ms after their last transfer (worker_pool.cpp kWaitPeriodInNano),
    # 16 per engine with 8 HCAs. When every TP rank transfers at once they
    # outnumber the CPUs and starve the scheduler and its gloo collectives;
    # a positive value makes them yield to those threads (0: unchanged).
    # Window residency default (unset): 19.
    SGLANG_MOONCAKE_WORKER_NICE = EnvInt(0)
    # Put the same threads (and the SWA backfill IO thread) under SCHED_IDLE:
    # a waking normal thread then always preempts them at once, which a nice
    # value alone does not guarantee. Unprivileged; one-way for the process.
    SGLANG_BUSY_POLL_SCHED_IDLE = EnvBool(False)
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)
    MOONCAKE_TENANT_ID = EnvStr("default")

    # ===================================================================
    # MoRI transport and expert dispatch
    # ===================================================================
    SGLANG_DEEPEP_V2_FORCE_MAX_LEN = EnvBool(False)
    # Send CPU-resident AUX data via RDMA instead of ZMQ TCP (default: TCP).
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    # Number of RDMA Queue Pairs (QPs) used per transfer operation. Higher
    # values can increase parallelism and bandwidth utilization.
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    # Number of RDMA work requests posted in a single batch to each QP. Larger
    # batch sizes reduce per-operation overhead and improve throughput at the
    # cost of higher latency. -1 selects automatic sizing based on the number
    # of merged work requests and available endpoints.
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    # Number of worker threads in the RDMA executor thread pool. More workers
    # can improve parallelism for large batch transfers across multiple QPs,
    # but excessive threads may cause contention.
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    # Number of sharded synchronous worker threads that drain KV transfers.
    # Also the bound on outstanding (posted-but-not-completed) transfers, so it
    # is the primary throttle keeping the RDMA send queue from overflowing.
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    # Poll cadence (ms) at which a transfer worker wakes to check the SLA while
    # waiting for completion; real completion still wakes it immediately.
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    # Per-transfer SLA (ms) before a KV transfer is failed; 0 disables the SLA
    # and relies on the RDMA retry-exceeded timeout only.
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)

    # ===================================================================
    # AMD, ROCm, and AITER
    # ===================================================================
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_USE_AITER_AG = EnvBool(True)
    # Use reduce_scatter (instead of all_reduce + dp_scatter) for the equal-chunk
    # MAX_LEN DP-MoE combine. Default ON for ROCm/HIP (uses the aiter custom
    # symmetric-memory kernel), OFF elsewhere (would fall back to RCCL); override
    # explicitly to force on/off on any platform.
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    # Quantize the variable-length DP-MoE gather payload (SGLANG_DP_USE_GATHERV
    # path, prefill/extend only) to fp8-e4m3 with per-token-group-128 scales:
    # halves the gathered hidden-state bytes over NCCL; the combine
    # (reduce_scatterv) leg stays bf16 (NCCL SUM cannot run on fp8).  Lossy on
    # the wire — same group quantization the MoE expert GEMMs apply to their
    # input anyway, but router/shared-expert reads see rounded values, so this
    # stays accuracy-gated and default OFF.
    SGLANG_ENABLE_DP_GATHER_FP8 = EnvBool(False)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    # Select the gate/up tile layout for AITER MoE: True -> interleave
    # (matches FlyDSL `gate_mode="interleave"` kernels), False -> separated
    # (matches `gate_mode="separated"`, the layout used by gptoss_fp4 tuned
    # configs and by Mxfp4MoEMethod's post-fix weight shuffle).
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    # Fold `silu(gate) * up` into the triton MoE up-GEMM epilogue. W13 rows are
    # permuted in place at load so gate/up land in adjacent columns of the same
    # output tile, which removes intermediate_cache1 and the standalone
    # activation launch per MoE layer. Opt-in because the in-place permute is
    # not compatible with runtime weight updates or EPLB expert rearrangement,
    # both of which assume the checkpoint's halves layout.
    SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED = EnvBool(False)
    # Fuse the `residual_add + RMSNorm + zero-pad` triplet that appears
    # before the MoE block for models whose MoE input hidden_size must be
    # padded up to a stride (e.g. GPT-OSS MXFP4 needs pad to multiple of
    # 256). When False (default) the pad runs as a separate
    # torch.nn.functional.pad call inside the MoE method. When True, the
    # aiter Triton kernel `fused_add_rmsnorm_pad` produces a padded
    # post-attention layernorm output in one launch and the MoE method
    # skips the explicit pad. Currently only takes effect on the
    # post_attention_layernorm path with aiter backend and TP=1.
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    # Physical layout for MHA KV cache. "nhd" (default) keeps the existing
    # (size, head_num, head_dim) per-token storage that
    # `aiter.mha.mha_batch_prefill_func`/`unified_attention` consume directly.
    # "vectorized_5d" allocates K as (num_blocks, H_kv, head_dim/x, page_size, x)
    # and V as (num_blocks, H_kv, page_size/x, head_dim, x) (x = 16 / dtype_size),
    # matching the SHUFFLE layout that aiter's CK FmhaBatchPrefill kernel and
    # `aiter.ops.triton.gluon.pa_decode_gluon` both consume natively. This is
    # the SHUFFLE KV layout that enables pa_decode_gluon for full-attn
    # decode without runtime permutes.
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    USE_ROCM_AITER_ROPE_BACKEND = EnvStr("0")
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("tilelang")
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)

    # DSV4 Aiter flags
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    # Unified KV wired the fused qk-norm-rope kernel to decode only, so MTP
    # target-verify kept running the norm+RoPE as separate kernels. Set to 0 to
    # go back to the unfused chain on the verify path.
    SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY = EnvBool(True)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)

    # ===================================================================
    # Apple Silicon and MLX
    # ===================================================================
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    # Number of decode steps between periodic mx.clear_cache() calls.
    # Set to 0 to disable cache clearing entirely.
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)
    # MLX buffer-cache cap in GB.
    SGLANG_MLX_CACHE_LIMIT_GB = EnvFloat(None)

    # ===================================================================
    # Ascend NPU
    # ===================================================================
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Enable int4x2 weights loading
    SGLANG_NPU_W4A4_NEW_PACKING = EnvBool(False)
    # Use the graph-safe Triton-Ascend kernel for masked speculative KV commits.
    SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NPU_USE_TRITON_KV_CACHE_STORE"
    )
    # Quantize x to int8 in the dispatch operator (vendor alias consumed by the
    # Ascend DeepEP library; the MTP draft-build scopes override it to False).
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False)
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")

    # ===================================================================
    # MUSA
    # ===================================================================
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # ===================================================================
    # Quantization
    # ===================================================================
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")
    # On by default; set SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0 as a kill switch.
    # Consults the tuned per-(N, K, M) Triton tile config table in
    # apply_fp8_linear. When a tuned config exists for this GPU / weight shape /
    # token count, run the Triton w8a8 FP8 GEMM with it; otherwise keep the
    # default CUTLASS path. Only takes effect on a GPU with a matching
    # dtype=fp8_w8a8_channelwise config JSON under
    # kernels/ops/quantization/configs/ (currently L40S), so it is a no-op on
    # any other GPU / untuned shape even when enabled.
    SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE = EnvBool(True)

    # ===================================================================
    # Humming quantization
    # ===================================================================
    SGLANG_HUMMING_ONLINE_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_INPUT_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_USE_F16_ACCUM = EnvBool(False)
    SGLANG_HUMMING_MOE_GEMM_TYPE = EnvStr("")

    # ===================================================================
    # FlashInfer, FlashMLA, and TRT-LLM
    # ===================================================================
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Per-rank dispatch capacity of the FlashInfer MoE A2A dispatcher. Unset
    # means each call site keeps its own default.
    SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(None)
    # Enable per-token FP32 activation scaling for serialized ModelOpt FP4 with
    # FlashInfer TRT-LLM or CuTe DSL v2 MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # Launch the TRT-LLM MoE grouped GEMMs with PDL only at or below this
    # token count.
    SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS = EnvInt(8192)
    # Use FlashInfer's fused atomic CUTLASS/CuTe DSL MoE finalize.
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE = EnvBool(True)
    # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
    # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
    FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
    FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # Split TRTLLM-GEN decode attention into sorted, equal-size request groups.
    # One preserves the default single-call path; values above one are useful
    # for batches whose KV sequence lengths have a large spread.
    SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS = EnvInt(1)
    # SM120 FlashMLA decode backend: "flashinfer" (default), "triton", or "torch".
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    # Also autotune one EXTEND-shaped dummy at max_prefill_tokens during
    # warmup. Opt-in: the extra forward needs transient activation headroom
    # that small-VRAM or tightly-packed configs may not have.
    SGLANG_FLASHINFER_AUTOTUNE_EXTEND = EnvBool(False)

    # ===================================================================
    # Triton and Torch compilation
    # ===================================================================
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)
    # A-B kill-switch for Work-Centric (Lean) Attention. When True, forces the
    # standard Triton decode kernel even if --enable-lean-attention or the auto-gate
    # would select Lean. Used to isolate the Lean kernel in benchmarks.
    SGLANG_DISABLE_LEAN_ATTENTION = EnvBool(False)
    # Persistent-grid size multiplier for the Lean decode kernel:
    # total_programs = round(device_CU_count * this). Default 1.0 (one CTA per CU), which
    # maximizes KV work-tiles per CTA and minimizes the cross-CTA combine/atomic reduction.
    # Kernel + E2E A/B sweeps found 1.0 beats 2.0 across uniform and ragged configs on both
    # MI300X (gfx942) and MI355X (gfx950) — 2.0 oversubscribed the CUs and regressed high-batch
    # decode. Exposed as a knob (e.g. set 2.0) for grid A/B tuning without a rebuild.
    SGLANG_FORCE_LEAN_GRID_CU_MULT = EnvFloat(1.0)

    # Torch Compile
    # Compact extend-attention query-tile grid: AMD/HIP-only optimization
    # (parity with flash-attn's ragged-aware launch). The feature checks _is_hip
    # explicitly in code; this env var allows override (0=force off, 1=force on).
    SGLANG_TRITON_COMPACT_EXTEND_ATTENTION = EnvBool(True)
    # Raise if Triton loads a kernel after the engine starts serving. This
    # verifies that startup warmup covers every kernel specialization used at
    # serving time.
    SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY = EnvBool(False)
    SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS = EnvFloat(1.0)
    SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB = EnvFloat(1.0)
    # gfx950 MLA decode stage-1: pick the launch geometry and split count per batch.
    # Reorders the fp32 accumulation, so off by default.
    SGLANG_MLA_DECODE_TUNE = EnvBool(False)
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # ===================================================================
    # Expert parallel load balancing
    # ===================================================================
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    # Chunk size for the rebalance expert-weight P2P exchange; set
    # >= num_physical_experts to submit a single batch_isend_irecv.
    SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE = EnvIntWithAlias(
        32, deprecated_name="SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE"
    )

    # ===================================================================
    # DeepGEMM
    # ===================================================================
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    # Enable the allowlisted low-M BF16 Split-K GEMM path on Blackwell. Shapes
    # outside the measured allowlist continue to use CuTe DSL/cuBLAS.
    SGLANG_ENABLE_BF16_SPLITK_GEMM = EnvBool(True)
    SGLANG_DEEPGEMM_STANDARD_LAYOUT = EnvStr("auto")
    SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION = EnvFloat(0.25)
    # Cap the DeepGEMM masked grouped-GEMM per-expert padded capacity at
    # round_up(max(masked_m), 256) instead of round_up(rank_tokens, 256):
    # shrinks the [num_local_experts, m, *] MoE intermediates ~4x under
    # load imbalance (they otherwise OOM saturated --moe-runner-backend
    # deep_gemm serving).  Costs one D2H sync per MoE layer.
    SGLANG_OPT_DG_MASKED_M_CAP = EnvBool(False)
    # Wide-DP eager prefill uses compact routing storage; masked storage scales
    # with num_local_experts and can OOM on skewed batches.
    SGLANG_OPT_DG_COMPACT_EAGER = EnvBool(False)
    # Drop dp-attention MAX_LEN pad rows from MoE dispatch (StandardDispatcher
    # post-translation topk_ids -> -1): pad rows otherwise run the router on
    # stale hidden values and burn expert compute whose outputs are discarded;
    # colliding pad top-ks also inflate the DeepGEMM masked-GEMM workspace to
    # OOM at saturation.  Capture-safe (reads only global_num_tokens_gpu).
    SGLANG_OPT_MASK_DP_PAD_MOE = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    # Resolved lazily so it tracks SGLANG_CACHE_DIR, which is defined below.
    SGLANG_DG_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)

    # ===================================================================
    # Cache directories
    # ===================================================================
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    # JIT kernel build cache. None = unset, resolving to ~/.cache/sglang/jit;
    # point it at a persistent mount to share builds across CI jobs.
    SGLANG_JIT_CACHE_DIR = EnvStr(None)
    # Log, at INFO, which dependency changed whenever a module is rebuilt.
    SGLANG_JIT_CACHE_DEBUG = EnvBool(False)
    # How many builds to keep per module variant. None = unset = keep all, which
    # is what makes reverting an edit an instant hit instead of a rebuild; set
    # it to trade that away for disk (1 keeps only the most recent build).
    SGLANG_JIT_CACHE_KEEP = EnvInt(None)

    # ===================================================================
    # Expert-parallel dispatch and MoE execution
    # ===================================================================
    # Deprecated in favor of '--deepep-dispatcher-output-dtype bf16' but still
    # read by several call sites; do not use in new code.
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    # Per-rank buffer capacity, not a model token limit.
    SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    # 0 lets ElasticBuffer select its theoretical communication SM/QP counts.
    SGLANG_DEEPEP_V2_NUM_SMS = EnvInt(0)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP = EnvBool(True)
    # Force dynamic Waterfill with runtime EP all-reduce instead of the default
    # static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(True)
    # DeepSeek/GLM MoE (deepseek_v2.py): quantize the (dp-gathered) MoE input
    # to per-token-group-128 fp8 ONCE and feed both the fused shared-expert
    # GEMM (cutlass w8a8 linear) and the routed experts' triton fused runner,
    # instead of quantizing the same [T, hidden] tensor twice with different
    # scale layouts. Only engages on CUDA with fp8 block-128 weights, the
    # standard dispatcher, and the triton MoE runner; falls back silently
    # otherwise.
    SGLANG_OPT_MOE_QUANT_ONCE = EnvBool(False)

    # ===================================================================
    # DeepGEMM Mega MoE
    # ===================================================================
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(8192)
    # Blackwell MegaMoE uses a whole-grid software barrier. Keep a small
    # residency margin so every cluster can launch beside other streams.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS = EnvInt(2)

    # ===================================================================
    # Top-k kernels
    # ===================================================================
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    # Opt-in: route DeepSeek-V3 grouped topk through the unified Triton router
    # instead of the flashinfer/AOT grouped kernels. Off by default (flashinfer is
    # the tuned production path); the Triton path is bit-exact on DeepSeek-V3.2 e2e
    # and benchmarks at parity, so this is a consolidation escape hatch, not a perf flip.
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK = EnvBool(False)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)

    # ===================================================================
    # Kernel selection and fused backends
    # ===================================================================
    # MiniCPM sparse attention developer switches
    SGLANG_MINICPM_FUSE_TOPK = EnvBool(False)
    SGLANG_MINICPM_DENSE_AS_SPARSE = EnvBool(False)
    SGLANG_MINICPM_FORCE_DENSE = EnvBool(False)

    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)
    # Force every decoupled_engine.kernels BaseFusedOp onto one backend (a KernelBackend
    # value, e.g. "torch" / "torch_compile" / "triton" / "aot"); unset =
    # auto-select by priority. "torch" flips all fused ops to their pure-torch
    # reference implementations for numerical-bug bisection.
    SGLANG_FORCE_FUSED_OP_BACKEND = EnvStr(None)
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)

    # ===================================================================
    # Logits and log-probability processing
    # ===================================================================
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    # Sanitize NaN logits before sampling kernels and log a throttled warning
    # (see sanitize_nan_logits).
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)
    SGLANG_ENABLE_LOGPROB_CHUNK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK"
    )
    SGLANG_LOGPROB_CHUNK_SIZE = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"
    )
    # Compute input logprobs from logits via per-row logsumexp instead of
    # materializing the full-vocab log-softmax. Escape hatch only; the two
    # paths are mathematically identical.
    SGLANG_ENABLE_FAST_INPUT_LOGPROBS = EnvBool(True)

    # ===================================================================
    # Deterministic inference and all-reduce
    # ===================================================================
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    # NCCL channel count pinned on CUDA so the all-reduce reduces a token the
    # same way whatever else shares its batch. Raise it to buy back bandwidth
    # on links that can drive more channels.
    SGLANG_DETERMINISTIC_NCCL_NCHANNELS = EnvInt(8)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    # Default per-direction workspace cap for CustomAllReduceV2; explicit
    # constructor sizes take precedence over this.
    SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = EnvInt(16 * 1024)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = EnvInt(None)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = EnvInt(None)

    # ===================================================================
    # RoPE cache
    # ===================================================================
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_FP32 = EnvBool(False)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # ===================================================================
    # Speculative decoding
    # ===================================================================
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)
    # Capture the per-replay attention-metadata prep (init_forward_metadata_out_graph)
    # into a small CUDA graph, collapsing its host dispatch cost to one launch.
    # Experimental; auto-falls back to eager if the backend's prep is not capturable.
    SGLANG_ENABLE_METADATA_GLUE_GRAPH = EnvBool(False)
    SGLANG_OPT_FUSED_KDA_VERIFY = EnvBool(False)
    # A/B: keep the DFLASH draft greedy head eager (not folded in-graph).
    SGLANG_DFLASH_EAGER_DRAFT_SAMPLER = EnvBool(False)
    SGLANG_RAGGED_VERIFY_MODE = EnvStr("static")
    SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE = EnvBool(False)
    # Skip draft_extend while adaptive spec is at steps=0 (drafting disabled).
    # Saves the per-step draft forward, but the draft KV goes stale: an upshift
    # back to steps>0 starts from a cold draft state (low accept until it recovers).
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    # Which speculative decisions rank 0 broadcasts to its TP group; narrowing
    # it under live traffic isolates where ranks actually diverge. Comma
    # separated presets ("all", "rng", "init", "off"), or SpecTpSyncSite slugs
    # and numbers, each negatable with a leading "-": "all,-dspark-plan,-6".
    SGLANG_SPEC_TP_SYNC = EnvStr("all")
    # Kill-switch for the draft-extend cuda graph. Draft extend then always runs
    # eager. Escape hatch for setups where the capture's memory pool costs more
    # than the graph saves (e.g. DeepEP MoE workspace captured at full dispatch
    # capacity).
    SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)
    # Use the split-KV (flash-decode) kernel for EAGLE target-verify on the
    # Triton backend (ROCm). Only active at speculative topk == 1; falls back to
    # extend_attention_fwd for unsupported cases or when set false (e.g. for
    # debugging). Correctness is unaffected; this only changes performance.
    SGLANG_ENABLE_SPLITKV_VERIFY = EnvBool(True)
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # ===================================================================
    # Multimodal processing
    # ===================================================================
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    # Use the fully-vectorized ViT position-embedding interpolation (no per-image
    # Python loop / CPU<->GPU sync). Bit-exact with the legacy implementation;
    # set False to fall back to the per-image loop.
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)
    # For pre-tokenized (list[int]) multimodal prompts,
    # preserve the user's original tokens to avoid retokenization drift.
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)

    # ===================================================================
    # Multimodal CUDA IPC transport
    # ===================================================================
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    # Reuse the mapping for the already-allocated bounded CUDA IPC pool. This
    # has no effect unless CUDA IPC feature transport is explicitly selected.
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(True)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # ===================================================================
    # Mamba state and cache
    # ===================================================================
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)
    # Kill-switch for the fused per-slot conv clear/copy kernel (MambaPool);
    # falls back to the per-conv-type Python loop.
    SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS = EnvBool(False)
    # Opt-in: on the unified radix tree, leave the matched-prefix mamba evictable
    # during decode (it is already COW'd to the request's own slot) and shrink the
    # mamba pool ratio accordingly. Frees one resident slot per running request,
    # raising max_running_requests. Off = original locking + ratio (escape hatch).
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK = EnvBool(False)

    # ===================================================================
    # CUDA graphs and execution buffers
    # ===================================================================
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)
    # Guards CUDA graph executable dedup via cudaGraphExecUpdate.
    SGLANG_ENABLE_CUDA_GRAPH_DEDUP = EnvBool(False)
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)
    # Reuse wholly-free graph-pool segments for step-local eager allocations.
    SGLANG_ENABLE_GRAPH_POOL_BORROW = EnvBool(False)
    # Eager forward wraps the ForwardBatch's own tensors instead of copying them
    # into the CUDA graph buffer registry (no per-iter device-to-device copy).
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)

    # ===================================================================
    # Tokenizer, request state, embeddings, and reasoning controls
    # ===================================================================
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)
    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)
    SGLANG_PATCH_TOKENIZER = EnvBool(True)
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)
    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ===================================================================
    # Encoder pipeline and disaggregation
    # ===================================================================
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    SGLANG_ENCODER_MM_LOAD_WORKERS = EnvInt(4)
    # EncoderBootstrapServer health-check tuning.  Interval == 0 disables it.
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    # Seconds before permanently dropping an unhealthy encoder (0 = keep probing).
    SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL = EnvFloat(600.0)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)

    # ===================================================================
    # Native gRPC server
    # ===================================================================
    # Native gRPC server. SGLANG_GRPC_PORT is the env fallback for the
    # --grpc-port CLI flag; setting either enables the native server alongside
    # HTTP. The worker-threads knob stays env-only (internal tuning, no CLI
    # surface).
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_GRPC_WORKER_THREADS = EnvInt(4)

    # ===================================================================
    # NUMA and CPU affinity
    # ===================================================================
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(True)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)

    # ===================================================================
    # DeepSeek V4
    # ===================================================================

    # Model and Quantization
    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    # Set True to dequantize the FP4 experts to FP8 at runtime
    SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
    # Flash-0731 also accepts "low"; the active profile is checkpoint-resolved.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")
    # Quantize the SWA fp8 KV cache from bf16-rounded values (matches
    # trainer-side QAT and the DSA-CP path) instead of fp32 registers.
    SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE = EnvBool(False)

    # Kernels and indexer
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_FLASHINFER_MHC = EnvBool(False)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER = EnvBool(True)
    # Per-rank local query rows (after DP-attention sharding when enabled),
    # not request ISL.
    SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    # Upper bound (bytes) on the fp32 [query_rows, max_c4_seq_len] logits the
    # DSV4 C4 indexer materializes at once in eager (non-graph) forwards; above
    # it the query rows are processed in slices with exactly the same per-row
    # results. 0 disables row chunking.
    SGLANG_DSV4_INDEXER_LOGITS_MAX_BYTES = EnvInt(512 * 1024 * 1024)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)
    # Rows of the DSV4 sparse-prefill KV workspace per attention call (bf16:
    # 1 KiB per row). 0 = auto (one request at the context limit: c4 rows +
    # SWA rows + 1). Larger chunks are split into request groups with the same
    # per-query results. < 0 = one group per chunk and a workspace that grows
    # without bound, but still the packed per-request layout. No value gives
    # the old padded layout (num_reqs * max_seq_len // 4 rows); an A/B against
    # that layout needs an engine build without this change.
    SGLANG_DSV4_SPARSE_PREFILL_WORKSPACE_ROWS = EnvInt(0)

    # cache, GEMM, and distributed
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    # Route the decode wo_a bf16 batched matmul off rocBLAS/Tensile onto aiter's
    # tuned batched_gemm_bf16 (gfx95). Off by default; see deepseek_v4.py
    # _apply_wo_a_bf16_matmul.
    SGLANG_OPT_USE_AITER_BATCHED_GEMM = EnvBool(False)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # ===================================================================
    # Inkling
    # ===================================================================
    SGLANG_OPT_USE_FUSED_GATE_TOPK = EnvBool(True)
    # Inside the fused gate: use the CUDA JIT top-k+renorm kernel (v2) instead
    # of the triton kernel when the production Inkling shape applies.
    SGLANG_OPT_USE_GATE_TOPK_JIT = EnvBool(True)
    # Inside the fused gate: replace the cublas gate linear with the
    # expert-per-block GEMV JIT kernel at small token counts (GateGemvMode).
    SGLANG_OPT_GATE_GEMV_MODE = EnvInt(GateGemvMode.PAIR)
    # Capture all multi-layer EAGLE draft-extend steps and the in-graph chain
    # rotation into ONE CUDA graph instead of one captured graph per step.
    SGLANG_ENABLE_SINGLE_CG_DRAFT = EnvBool(True)
    # Draft sampler uses the Gumbel-max trick (argmax(probs / Exp(1))) instead of
    # torch.multinomial, whose device-side validity assert breaks draft-graph replay.
    SGLANG_OPT_USE_GUMBEL_SAMPLE = EnvBool(True)
    # Multi-layer chain-MTP boundary-KV fix: widen the draft-extend window to
    # rewrite rejected-draft KV rows before reuse (acc_len repair; on by default).
    SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX = EnvBool(True)
    SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP = EnvBool(True)
    SGLANG_OPT_USE_INKLING_SHEARED_BIAS = EnvBool(True)
    # Use feature-stacked GEMMs for the no-LoRA BF16 shared sink. Eligible LoRA
    # serving enables this layout independently of the flag.
    SGLANG_OPT_LINEARIZED_SHARED_SINK = EnvBool(True)
    # Use the autotuned JIT all-reduce, falling back to torch multimem for
    # shapes where it wins.
    SGLANG_OPT_USE_INKLING_CUSTOM_AR = EnvBool(True)
    # Fuse small-batch decode all-reduce, MLP convolution, and attention norm.
    # Requires the custom all-reduce; other shapes use the unfused path.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM = EnvBool(True)
    # Fuse eligible extend all-reduce, convolution, and cache updates.
    # Supports scattered or full-width state and requires the custom all-reduce.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV = EnvBool(True)
    # Fuse eligible convolution, QK norm, window, and KV-store prologue work.
    # Non-BF16 caches retain the backend KV store.
    SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE = EnvBool(True)
    # Override shared-expert selection: true uses grouped GEMM, false uses BMM.
    # When unset, selection follows model, quantization, and LoRA requirements.
    SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE = EnvBool(True)
    # Fold the conditional long-context log-scaling tau into its producers
    # instead of separate output-sized scale kernels: the fused attn
    # prologue's q path (bit-exact, before MXFP8 quantization there) and the
    # rel_logits projection's r OPERAND (the diagonal scale commutes through
    # the einsum, shrinking the pass by rel_extent/d_rel = 64x; rounding moves
    # before the GEMM). Flag-off keeps the standalone apply_log_scaling_tau
    # on the outputs.
    # Fold the MoE shared-expert partials into the custom AR kernels instead
    # of a separate {routed + shared} torch.add per MoE layer; some buckets
    # keep a pre-add during the AR stage-in. torch.add numerics
    # (bit-identical). Requires SGLANG_OPT_USE_INKLING_CUSTOM_AR.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SHARED = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU = EnvBool(True)
    # Dispatch the rel_logits projection around einsum's hidden compaction
    # copy of the strided r operand (a view into the packed qkvr output):
    # zero-copy strided-batched matmul at small t, JIT row-compact + einsum
    # above the band, single-launch tau-folded kernel in the small-t tau
    # band. Bit-identical to the plain einsum; flag-off restores it.
    SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH = EnvBool(True)
    # Quantize and store MXFP8 K/V data and scales in one fused kernel.
    SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE = EnvBool(True)
    # Default reasoning effort in [0.0, 0.99] when omitted by a request.
    # An empty string falls back to the protocol default (0.9); the effort
    # directive is always emitted.
    SGLANG_INKLING_DEFAULT_REASONING_EFFORT = EnvStr("0.9")
    SGLANG_INKLING_RS_MM_PREPROCESS = EnvBool(True)

    # ===================================================================
    # DSA backend (GLM 5 and DeepSeek V3.2)
    # ===================================================================
    SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_NSA_FUSE_TOPK"
    )
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"
    )
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NSA_HIP_DISABLE_PRESHUFFLE"
    )
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)
    SGLANG_DISABLE_DSA_INDEXER_FUSION = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: fuse the
    # absorbed q bmm with the nope/rope concat + fp8 cast so q is written
    # directly in fp8 ("born fp8") and the standalone concat-cast kernel
    # disappears.  Not bit-exact vs the default path (same rounding stages,
    # different GEMM accumulation order), hence default OFF until accuracy-
    # gated (oracle + full-set gsm8k).
    SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: pass a
    # per-row valid-topk count (derived from the trailing -1 pad run of the
    # topk indices) so the kernel skips whole pad-only topk blocks instead of
    # computing masked zero contributions.  Bit-exact by construction: skipped
    # blocks contain only -1 pads, and -1 entries inside the consumed range
    # still take the in-kernel clamp+mask path.
    SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH = EnvBool(False)
    # Opt-in: run the born-fp8 q-prep (absorbed bmm + concat + fp8 cast,
    # ~173us/layer-call) on alt_stream underneath the DSA indexer — the two
    # chains fork independently from the q_a_layernorm output.  Requires
    # SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q; eager-prefill-only via the born
    # predicate.  Coarse per-layer join keeps the single-slot born-q buffer
    # WAR-safe.
    SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = EnvBool(False)
    # Opt-in: fuse the Q8KV8 non-prefix KV prep — cast-concat k/k_rope
    # directly into the persistent fp8 kv buffer and zero the pad band in one
    # Triton kernel (replaces bf16 _cat + copy_ cast + zero_ tail).
    SGLANG_ENABLE_DSA_Q8KV8_KV_CAT_FUSION = EnvBool(False)
    # Q8KV8 born-fp8 q-prep codegen: "auto" = per-K Triton dispatch (default);
    # "cuda" = the hand-written SM90 WGMMA kernel (bitwise identical to the
    # Triton two_dot variant, 1.16-1.38x faster across GLM/DS shapes).
    SGLANG_OPT_Q8KV8_QPREP_VARIANT = EnvStr("auto")

    # ===================================================================
    # MiniMax M3
    # ===================================================================
    SGLANG_OPT_USE_BF16_ROUTER_GEMM = EnvBool(True)
    SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE = EnvBool(False)
    SGLANG_DISABLE_MSA = EnvBool(False)
    SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH = EnvBool(False)
    # Kill switch for the derived fp8 attention-GEMM mode (m3_fp8_attn_gemm_enabled):
    # forces the pre-fp8 behavior (bf16 indexer + widening sparse path, bf16 q)
    # even when kv_cache_dtype fp8_e4m3 + trtllm_mha + SM100 would activate it.
    SGLANG_DISABLE_M3_FP8_ATTN_GEMM = EnvBool(False)
    # MiniMax-M3 sparse decode indexer: single JIT radix-select kernel replaces the 2-stage split-K Triton topk.
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(True)
    # Fused JIT store (minimax_store_kv_index) of main+index K/V instead of separate
    # set_*_buffer copies; falls back when main/index dtypes differ or non-CUDA.
    SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE = EnvBool(True)
    # MiniMax-M3 MXFP8 MoE experimental fusion toggles (default off; A/B only).
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)
    # MiniMax M3 NPU prefill MAIN-attention: route the sparse main attention through
    # the native Ascend FA op `torch.ops.npu.npu_fused_infer_attention_score` (FIA)
    # with a per-query CUSTOM block_table
    SGLANG_MINIMAX_NPU_PREFILL_FIA = EnvBool(True)
    # MiniMax-M3 NPU sparse INDEXER (decode + verify topk block selection): route
    # through the native AscendC packed indexer op instead of the Triton indexer.
    SGLANG_MINIMAX_NPU_NATIVE_INDEXER = EnvBool(False)
    # MiniMax-M3 NPU sparse MAIN-attention (decode-main + verify-main): route the
    # sparse main attention through the native AscendC sparse-attention op with the
    # cached block_table override.
    SGLANG_MINIMAX_NPU_NATIVE_ATTN = EnvBool(False)
    # MiniMax-M3 on ROCm force-disables custom all-reduce in its model override
    # (arg_groups/overrides.py) when aiter all-reduce fusion is off. Set this to
    # opt back in and keep custom/quick all-reduce enabled -- e.g. to run the
    # INT4 quick-reduce path via ROCM_QUICK_REDUCE_QUANTIZATION={INT4,INT6,INT8}.
    SGLANG_M3_ALLOW_CUSTOM_AR = EnvBool(False)

    # ===================================================================
    # Kimi K3
    # ===================================================================
    # MNNVL fused all-reduce (bf16, TP8): zero-copy 1shot multicast-push for
    # small messages and in-place NVLS 2shot on symmetric-memory tensors for
    # large ones, with an optional fused residual add. Covers the KDA o_proj
    # output and the latent|shared MoE reduce; everything else falls back to
    # the regular all-reduce path. Auto-enabled on SM100/SM103 when
    # CustomAllReduceV2 with multicast is available; set 0/1 to override in
    # either direction. See srt/layers/k3_ar_fusion.py.
    SGLANG_K3_AR_FUSION = EnvBool(False)
    # K3 SP-MoE fused residual + reduce-scatter and matching all-gather over
    # CustomAllReduceV2's MNNVL push workspace. Auto-probed for the validated
    # TP8 GB300 configuration; set 0/1 to override. See
    # srt/layers/k3_sp_collective.py.
    SGLANG_K3_SP_COLLECTIVE = EnvBool(False)
    # Keep K3's post-MoE residual stream token-sharded between consecutive
    # SP-MoE layers. The next attention-residual aggregation and snapshot
    # bank write run on the local shard, then only the normalized attention
    # input is all-gathered. Requires SGLANG_K3_SP_COLLECTIVE.
    SGLANG_K3_SP_ATTN_RES = EnvBool(False)
    # Fused o_proj GEMM + all-reduce (bf16, TP 2..8, SM100+): one
    # kernel computes the TP-local o_proj partial and the cross-rank sum over
    # a P2P comm region, replacing the GEMM + NCCL AR pair at M <= 512.
    SGLANG_K3_GEMM_AR = EnvBool(False)
    # Merge the router gate and routed_expert_down_proj weights so the K3 MoE
    # front reads hidden_states once, and run the top-k plus the bf16 cast in one
    # epilogue kernel. See kernels/ops/moe/moe_front.py. Default on.
    SGLANG_K3_FUSED_FRONT = EnvBool(True)
    # Use the ROCm radix-4 router for covered K3 top-k workloads.
    SGLANG_K3_RADIX4_TOPK = EnvBool(False)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_CACHE_CAPACITY = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MIN_HITS = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MAX_SEQLEN = EnvInt(6144)

    # ===================================================================
    # Symmetric memory
    # ===================================================================
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Qwen3.5 and GDN
    SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV = EnvBool(True)
    SGLANG_TRACE_QWEN35_FINAL_NORM = EnvBool(False)
    SGLANG_QWEN35_NATIVE_FINAL_NORM = EnvBool(False)
    # One switch enables deferred MoE finalize and AR + residual + RMSNorm.
    SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION = EnvBool(False)
    # Distinct workspace configurations allowed in one process. Production
    # uses one model/configuration per rank, so fail closed on accidental reuse.
    SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES = EnvInt(1)

    # ===================================================================
    # Plugin system
    # ===================================================================
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")

    # ===================================================================
    # KV-Canary and Token-Oracle (testing only)
    # ===================================================================
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)

    # ===================================================================
    # Rust server
    # ===================================================================
    SGLANG_RUST_SERVER = EnvBool(False)
    # Build a missing Rust extension from source (auto), require a bundled or
    # cached extension (never), or rebuild the local cache entry (force).
    SGLANG_RUST_BUILD_MODE = EnvStr("auto")
    # Most batched requests one /generate HTTP call may expand into.
    SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ = EnvInt(4096)

    # ===================================================================
    # Weight Cache Daemon
    # ===================================================================
    # Paths the daemon and the engine ranks it serves must agree on. Both are
    # format templates and must keep the {global_rank} placeholder: each rank
    # talks to the daemon on its own GPU, so a rank-independent path would point
    # every rank at one daemon and map another rank's shard.
    SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_rank{global_rank}.sock"
    )
    SGLANG_WEIGHT_CACHE_READY_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_rank{global_rank}.ready"
    )


envs = Envs()
EnvField._allow_set_name = False


def exportable_env_vars() -> dict[str, str]:
    return {
        field.name: _exportable_value(os.environ[field.name])
        for field in sorted(
            (value for value in vars(Envs).values() if isinstance(value, EnvField)),
            key=lambda field: field.name,
        )
        if not field.secret and field.name in os.environ
    }


def _exportable_value(value: str) -> str:
    try:
        value.encode()
    except UnicodeEncodeError:
        return (
            _NON_UTF8_PREFIX
            + base64.b64encode(value.encode(errors="surrogateescape")).decode()
        )
    return value


class _DeprecatedEnv:
    """One deprecated env var: warn if it is set, and optionally forward its
    (possibly transformed) value to a replacement env var."""

    def __init__(
        self,
        replacement: Optional[str] = None,
        transform: Optional[Callable[[str], str]] = None,
        note: Optional[str] = None,
    ):
        self.replacement = replacement
        self.transform = transform
        self.note = note

    def apply(self, old_name: str):
        if old_name not in os.environ:
            return
        message = f"Environment variable {old_name} is deprecated."
        if self.replacement is not None:
            message += f" Please use {self.replacement} instead."
        if self.note is not None:
            message += f" {self.note}"
        warnings.warn(message)
        if self.replacement is not None:
            value = os.environ[old_name]
            if self.transform is not None:
                value = self.transform(value)
            os.environ[self.replacement] = value


def _ms_to_s(value: str) -> str:
    return str(float(value) / 1000.0)


def _invert_bool(value: str) -> str:
    return "0" if value.lower() in ("true", "1", "yes", "y") else "1"


# The single registry for deprecated environment variables, processed once at
# import by _handle_deprecated_envs(). Add new deprecations here instead of
# ad-hoc warnings. For a rename where the old name must keep working through a
# descriptor, use EnvBoolWithAlias / EnvIntWithAlias instead.
_DEPRECATED_ENVS: Dict[str, _DeprecatedEnv] = {
    # Renamed: the value is forwarded to the replacement.
    "SGLANG_GC_LOG": _DeprecatedEnv(replacement="SGLANG_LOG_GC"),
    "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH": _DeprecatedEnv(
        replacement="SGLANG_MOE_NVFP4_DISPATCH"
    ),
    "SGLANG_ENABLE_THINKING": _DeprecatedEnv(replacement="SGLANG_DEFAULT_THINKING"),
    "SGLANG_REASONING_EFFORT": _DeprecatedEnv(
        replacement="SGLANG_DSV4_REASONING_EFFORT"
    ),
    "SGLANG_USE_JIT_ALL_REDUCE": _DeprecatedEnv(
        replacement="SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
    ),
    # The legacy DISABLE flags have the opposite polarity of their replacement.
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": _DeprecatedEnv(
        replacement="SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", transform=_invert_bool
    ),
    # Renamed with a unit change.
    "SGLANG_QUEUED_TIMEOUT_MS": _DeprecatedEnv(
        replacement="SGLANG_REQ_WAITING_TIMEOUT",
        transform=_ms_to_s,
        note="Note the unit change: milliseconds -> seconds.",
    ),
    "SGLANG_FORWARD_TIMEOUT_MS": _DeprecatedEnv(
        replacement="SGLANG_REQ_RUNNING_TIMEOUT",
        transform=_ms_to_s,
        note="Note the unit change: milliseconds -> seconds.",
    ),
    # Removed without replacement.
    "SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2": _DeprecatedEnv(),
    # Superseded by the unified JIT per_token_group_quant, the default CUDA path.
    "SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT": _DeprecatedEnv(),
    "SGLANG_MASKED_GEMM_FAST_ACT": _DeprecatedEnv(),
    "SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN": _DeprecatedEnv(),
    # sconv-family kernels always use the CUDA-JIT ports when supported; no toggle.
    "SGLANG_OPT_USE_CUDA_SCONV": _DeprecatedEnv(),
    # The direct dense BF16 GEMM source is vendored in-tree.
    "SGLANG_FLASHINFER_PR4266_SOURCE": _DeprecatedEnv(),
    # DSV4 compressor V2 is always used.
    "SGLANG_OPT_USE_COMPRESSOR_V2": _DeprecatedEnv(),
    # Replaced by CLI flags.
    "SGLANG_ENABLE_GRPC": _DeprecatedEnv(
        note="Please use '--grpc-port' to enable the native gRPC server."
    ),
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": _DeprecatedEnv(
        note="Please use '--enable-prefill-delayer' instead."
    ),
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": _DeprecatedEnv(
        note="Please use '--prefill-delayer-max-delay-passes' instead."
    ),
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": _DeprecatedEnv(
        note="Please use '--prefill-delayer-token-usage-low-watermark' instead."
    ),
    "SGLANG_CUTLASS_MOE": _DeprecatedEnv(
        note="Please use '--moe-runner-backend=cutlass' and/or "
        "'--speculative-moe-runner-backend=cutlass' instead."
    ),
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": _DeprecatedEnv(
        note="Please use '--enable-w4a4-mxfp4-megamoe' instead."
    ),
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": _DeprecatedEnv(
        note="Please use '--enable-w4a4-mxfp4-megamoe' instead."
    ),
    "SGLANG_DFLASH_PREFILL_REFILL_TARGET": _DeprecatedEnv(
        note="DFlash now auto-enables the min-free-slots delay; unset this env. "
        "To override the threshold, use '--min-free-slots-delay'."
    ),
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": _DeprecatedEnv(
        note="The unified radix tree is the default tree cache now; unset this "
        "env. The field is still defined for legacy call sites."
    ),
}


def _handle_deprecated_envs():
    for old_name, deprecation in _DEPRECATED_ENVS.items():
        deprecation.apply(old_name)

    # Rewrite the legacy SGL_ prefix to SGLANG_ (names not covered above).
    for key, value in list(os.environ.items()):
        if key.startswith("SGL_") and key not in _DEPRECATED_ENVS:
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


def third_party_cache_defaults() -> Dict[str, str]:
    base = os.path.expanduser(envs.SGLANG_CACHE_DIR.get())
    return {
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),
        "CUDA_CACHE_PATH": os.path.join(base, "nv"),
        # FlashInfer appends ".cache/flashinfer" to this base itself, so this
        # is the base dir rather than the final cache dir.
        "FLASHINFER_WORKSPACE_BASE": base,
    }


def redirect_third_party_caches():
    """Point third-party JIT caches at SGLANG_CACHE_DIR, so a run's compiled
    kernels can be cleaned, warmed or volume-mounted as one directory.

    Must be called early. The redirect silently does nothing if either of
    these has already happened:

    - FlashInfer was imported. It resolves its workspace at import time.
    - Inductor made its first ``cache_dir()`` call. That call setdefaults
      TORCHINDUCTOR_CACHE_DIR itself.
    """
    for key, value in third_party_cache_defaults().items():
        os.environ.setdefault(key, value)


_handle_deprecated_envs()

# Trigger auto-injection of CUDA coredump env vars when SGLANG_CUDA_COREDUMP=1.
# Best-effort; for strict guarantees, set CUDA_* env vars in the shell before
# launching Python. Imported conditionally to keep the default import of this
# module free of non-stdlib side effects.
if envs.SGLANG_CUDA_COREDUMP.get():
    import decoupled_engine.srt.debug_utils.cuda_coredump  # noqa: F401, E402  # isort: skip

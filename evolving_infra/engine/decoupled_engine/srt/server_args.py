# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from SGLang python/sglang/srt/server_args.py (commit 297e54fd10), modified for evolving-infra.
"""Server argument construction, resolution, CLI registration, and network ports."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import importlib
import logging
import sys
import tempfile
import uuid
from typing import Any, Literal, NoReturn, Optional

import msgspec

from decoupled_engine.srt.arg_groups.arg_utils import (
    add_cli_args_from_dataclass,
    is_record,
    record_fields,
)
from decoupled_engine.srt.arg_groups.argparse_actions import (
    DeprecatedStoreConstAction,
    DeprecatedStoreTrueAction,
    ExplicitLateDAction,
)
from decoupled_engine.srt.arg_groups.model_override_base import ep_joiner_of, ep_scale_joiner_of
from decoupled_engine.srt.arg_groups.overrides import (
    model_config_of,
    remote_instance_transfer_engine_of,
    resolution_result,
    resolving_view,
)
from decoupled_engine.srt.arg_groups.validation_hook import (
    validate_mps_model_config,
    validate_standard_mps_server_args,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.hardware_backend.mlx.runtime import use_mlx
from decoupled_engine.srt.hardware_backend.mps.runtime import validate_mps_runtime
from decoupled_engine.srt.runtime_context import (
    attn_dp_enabled_of,
    get_platform,
    num_dp_ranks_of,
    publish,
)
from decoupled_engine.srt.speculative.decoupled_spec_io import DecoupledSpecIpcConfig
from decoupled_engine.srt.utils.network import (
    NetworkAddress,
    get_free_port_below_ephemeral,
    wait_port_available,
)

logger = logging.getLogger(__name__)


def _reasoning_parser_choices():
    # Importing the registry here costs seconds in every process that parses
    # arguments; a plugin that registered a parser has already imported it.
    module = sys.modules.get("decoupled_engine.srt.parser.reasoning_parser")
    if module is not None:
        return list(module.ReasoningParser.DetectorMap)
    from decoupled_engine.srt.parser.reasoning_parser_names import REASONING_PARSER_NAMES

    return list(REASONING_PARSER_NAMES)


def _tool_call_parser_choices():
    module = sys.modules.get("decoupled_engine.srt.function_call.function_call_parser")
    if module is not None:
        return list(module.FunctionCallParser.ToolCallParserEnum)
    from decoupled_engine.srt.function_call.parser_names import TOOL_CALL_PARSER_NAMES

    return list(TOOL_CALL_PARSER_NAMES)


def _real_kv_hash_modes():
    # Lazy: this pulls the whole decoupled_engine.kernels package (~2 s) into every
    # process that imports server_args, most of which never use it.
    from decoupled_engine.kernels.ops.kv_canary.consts import RealKvHashMode

    return list(RealKvHashMode)


# Compatibility re-exports for callers importing through server_args.
from decoupled_engine.srt.arg_groups.arg_utils import NS, A, Arg  # noqa: F401
from decoupled_engine.srt.arg_groups.argparse_actions import LoRAPathAction  # noqa: F401

# Public choice lists and plugin registration helpers.
from decoupled_engine.srt.arg_groups.choices import (  # noqa: F401
    ATTENTION_BACKEND_CHOICES,
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    DETERMINISTIC_ATTENTION_BACKEND_CHOICES,
    DISAGG_TRANSFER_BACKEND_CHOICES,
    DRAFT_ATTENTION_BACKEND_CHOICES,
    FP4_GEMM_RUNNER_BACKEND_CHOICES,
    FP8_GEMM_RUNNER_BACKEND_CHOICES,
    GRAMMAR_BACKEND_CHOICES,
    LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
    LOAD_FORMAT_CHOICES,
    MOE_RUNNER_BACKEND_CHOICES,
    MXFP8_MOE_RUNNER_BACKEND_CHOICES,
    QUANTIZATION_CHOICES,
    RADIX_EVICTION_POLICY_CHOICES,
    RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND,
    RL_ON_POLICY_TARGET_CHOICES,
    SAMPLING_BACKEND_CHOICES,
    add_attention_backend_choices,
    add_chunked_prefix_cache_attention_backend,
    add_deterministic_attention_backend_choices,
    add_disagg_transfer_backend_choices,
    add_draft_attention_backend_choices,
    add_fp4_gemm_runner_backend_choices,
    add_fp8_gemm_runner_backend_choices,
    add_grammar_backend_choices,
    add_linear_attn_kernel_backend_choices,
    add_load_format_choices,
    add_moe_runner_backend_choices,
    add_mxfp8_moe_runner_backend_choices,
    add_quantization_method_choices,
    add_radix_eviction_policy_choices,
    add_radix_supported_deterministic_attention_backend_choices,
    add_rl_on_policy_target_choices,
)
from decoupled_engine.srt.arg_groups.fields import collect_input_fields
from decoupled_engine.srt.arg_groups.fields.device import (
    Device,
)
from decoupled_engine.srt.arg_groups.fields.disagg import (
    Disagg,
)
from decoupled_engine.srt.arg_groups.fields.exec_ import (
    ExecComm,
    ExecDeterministic,
    ExecDllm,
    ExecFeatures,
    ExecGraph,
    ExecKernel,
    ExecMamba,
    ExecMoe,
    ExecOffload,
    ExecOverlap,
)
from decoupled_engine.srt.arg_groups.fields.lora import (
    Lora,
)
from decoupled_engine.srt.arg_groups.fields.memory import (
    Memory,
)
from decoupled_engine.srt.arg_groups.fields.mm import (
    Mm,
)
from decoupled_engine.srt.arg_groups.fields.model import (
    Model,
)
from decoupled_engine.srt.arg_groups.fields.observability import (
    Observability,
)
from decoupled_engine.srt.arg_groups.fields.parallel import (
    Parallel,
)
from decoupled_engine.srt.arg_groups.fields.schedule import (
    Schedule,
)
from decoupled_engine.srt.arg_groups.fields.serving import (
    Serving,
)
from decoupled_engine.srt.arg_groups.fields.spec import (
    Spec,
)
from decoupled_engine.srt.lora.lora_registry import LoRARef  # noqa: F401
from decoupled_engine.srt.model_executor.cuda_graph_config import (  # noqa: F401
    CudaGraphConfig,
    parse_cuda_graph_config_arg,
)
from decoupled_engine.srt.utils.common import (  # noqa: F401
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
    human_readable_int,
    json_list_type,
    nullable_str,
)

# Re-exported like the imports above, but resolved on first use: importing them
# eagerly is what the choices helpers avoid, and most processes never read them.
_LAZY_REEXPORTS = {
    "FunctionCallParser": "decoupled_engine.srt.function_call.function_call_parser",
    "ReasoningParser": "decoupled_engine.srt.parser.reasoning_parser",
    "RealKvHashMode": "decoupled_engine.kernels.ops.kv_canary.consts",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_REEXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def _plain(value: Any) -> Any:
    """Convert nested Structs and dataclasses to dicts, copying all values."""
    if not isinstance(value, type) and is_record(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in record_fields(type(value))
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(_plain(item) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    if isinstance(value, dict):
        return type(value)((_plain(k), _plain(v)) for k, v in value.items())
    return copy.deepcopy(value)


class ServerArgs:
    """Raw server configuration, sealed when resolution starts.

    Add fields to the matching namespace in ``arg_groups/fields/`` using
    ``A[T, help]`` or ``A[T, Arg(...)]``; see ``arg_utils.Arg`` for CLI metadata.
    Only dynamic choices, deprecated flags, and ``--config`` need manual
    registration in ``add_cli_args``. ``POSITIONAL_FIELD_ORDER`` preserves
    the positional constructor signature when these namespaces are assembled.
    """

    def __post_init__(self):
        """Leave construction unresolved; launchers and publishers call ``resolve_once``."""

    def resolve_once(self) -> None:
        """Resolve once, preserving declarations across pickling to child processes.

        Handlers are not idempotent over their own output. A failed resolution
        cannot be retried on the same record.
        """
        if getattr(self, "_resolution_finished", False):
            return
        if getattr(self, "_resolution_failed", False):
            raise RuntimeError(
                "resolution already failed on this ServerArgs; the handlers that "
                "ran left their writes on the record, and a second pass would "
                "read that partial output as fresh input. Build a new record "
                "from the corrected arguments."
            )
        from decoupled_engine.srt.arg_groups.pipeline import run_resolution_pipeline

        self._input_frozen = True

        try:
            # Validate Torch MPS before model resolution; MLX has its own gate.
            # Platform detection covers the usual omitted --device on macOS.
            if not use_mlx():
                requested_device = getattr(self, "device", None)
                explicitly_mps = (
                    requested_device is not None
                    and str(requested_device).split(":", 1)[0] == "mps"
                )
                if explicitly_mps or (
                    requested_device is None and get_platform().is_mps
                ):
                    validate_mps_runtime()

            run_resolution_pipeline(self)

            # Validate resolved arguments and checkpoint-derived constraints.
            cfg = resolving_view(self)
            if (
                cfg.device == "mps"
                and not use_mlx()
                and str(cfg.model_path).lower() not in ("none", "dummy")
            ):
                validate_standard_mps_server_args(self)
                lora_enabled = bool(cfg.enable_lora) or (
                    cfg.enable_lora is None and bool(cfg.lora_paths)
                )
                validate_mps_model_config(
                    model_config_of(self),
                    lora_enabled=lora_enabled,
                )
        except BaseException:
            self._resolution_failed = True
            raise
        finally:
            self._input_frozen = False
        # Also mark the dummy/absent-model path, which returns early from the pipeline.
        self._resolution_finished = True

    @property
    def launch_command(self) -> str | None:
        """Original CLI arguments or Engine call, with secrets redacted."""
        return getattr(self, "_launch_command", None)

    def __repr__(self) -> str:
        hidden = {"watermark_key", "watermark_key_b", "watermark_config"}
        values = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}"
            for field in record_fields(type(self))
            if field.name not in hidden
        )
        return f"ServerArgs({values})"

    def resolved_dict(self) -> dict[str, Any]:
        """Serialize resolved field values, expanding nested records and excluding bookkeeping.

        One exception: the deprecated `enable_dp_attention` reports whether an
        attention-DP width was configured, for clients that still read it from
        `/server_info`. Resolution leaves the field false. It reports the width
        rather than `attn_dp_enabled`, whose elastic scale-joiner arm is true at
        width one: a dump carrying that would fold the joiner's replicas into
        attention-DP groups when it is read back.
        """

        resolved = {
            field.name: _plain(resolution_result(self, field.name))
            for field in record_fields(type(self))
        }
        # TODO: drop together with `--enable-dp-attention` after 2026-12-31.
        resolved["enable_dp_attention"] = resolving_view(self).attn_dp_size > 1
        return resolved

    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
        "DeepseekV4ForCausalLM",
        "MuseGlimmerForConditionalGeneration",
        "Cosmos3ForConditionalGeneration",
        "Cosmos3EdgeForConditionalGeneration",
    )

    # The attention-backend allow-list is enforced via
    # --enable-page-major-kv-layout (implied by the unified pool in
    # _handle_page_major_kv_layout); the model-family gate is enforced at pool
    # construction in model_runner_kv_cache_mixin._init_pools.

    def _unified_memory_pd_transfer_backends(self) -> set[str]:
        return {"mooncake"}

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):

        add_cli_args_from_dataclass(parser, ServerArgs)

        # --- Fields with dynamic choices (computed at add_cli_args time) ---
        sampling_backend_choices = set(SAMPLING_BACKEND_CHOICES)
        if envs.SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.get():
            sampling_backend_choices.add("token_oracle")
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=sampling_backend_choices,
            default=_declared_default("sampling_backend"),
            help="Choose the kernels for sampling layers.",
        )

        reasoning_parser_choices = _reasoning_parser_choices()
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=["auto"] + reasoning_parser_choices,
            default=_declared_default("reasoning_parser"),
            help=f"Specify the parser for reasoning models. "
            f"Use 'auto' to detect from chat template or checkpoint metadata. "
            f"Options include: {reasoning_parser_choices}.",
        )
        tool_call_parser_choices = _tool_call_parser_choices()
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=["auto"] + tool_call_parser_choices,
            default=_declared_default("tool_call_parser"),
            help=f"Specify the parser for handling tool-call interactions. "
            f"Use 'auto' to detect from chat template or checkpoint metadata. "
            f"Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--kv-canary-real-data",
            type=str,
            default=_declared_default("kv_canary_real_data"),
            choices=[m.name.lower() for m in _real_kv_hash_modes()],
            help=(
                "Check the real KV-cache in the canary. "
                "'none' (default) disables the feature. "
                "'partial' checks the first 16 bytes of each real-KV slot. "
                "'all' checks the full real-KV slot."
            ),
        )

        # --- Configuration file support ---
        parser.add_argument(
            "--config",
            type=str,
            help="Read CLI options from a config file. Must be a YAML file with configuration options.",
        )

        # --- Deprecated argument registrations ---
        # `disable_cuda_graph` is `no_cli=True`, so this deprecated spelling is
        # its only command-line entry point.
        parser.add_argument(
            "--disable-cuda-graph",
            action=DeprecatedStoreTrueAction,
            new_flag="--cuda-graph-backend-{decode,prefill}=disabled",
            help="Deprecated. Use --cuda-graph-backend-{decode,prefill}=disabled instead.",
        )
        # `enable_dp_attention` is `no_cli=True` too; resolution turns it into
        # `attn_dp_size`. TODO: remove the flag and the field after 2026-12-31.
        parser.add_argument(
            "--enable-dp-attention",
            action=DeprecatedStoreTrueAction,
            new_flag="--attn-dp-size <dp-size>",
            help="Deprecated. Use --attn-dp-size <dp-size> instead of "
            "--dp-size <dp-size> --enable-dp-attention.",
        )
        # amd was the only monokernel backend the flag could name.
        parser.add_argument(
            "--monokernel-backend",
            action=DeprecatedStoreConstAction,
            dest="enable_amd_monokernel",
            nargs=None,
            choices=["amd"],
            const_value=True,
            default=False,
            new_flag="--enable-amd-monokernel",
            help="Deprecated. Use --enable-amd-monokernel instead of "
            "--monokernel-backend amd.",
        )
        parser.add_argument(
            "--enable-flashinfer-allreduce-fusion",
            action="store_true",
            help="(Deprecated: use --flashinfer-allreduce-fusion-backend=auto) "
            "Enable FlashInfer allreduce fusion with Residual RMSNorm.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Window residency sizes a smaller SWA pool from the ratio, so only
        # the ``full`` residency rejects an explicit one. The ratio defaults
        # to None, so any value here was given explicitly.
        if (
            getattr(args, "dsv4_uniform_cache_lifecycle", False)
            and getattr(args, "dsv4_uniform_swa_residency", "full") == "full"
            and getattr(args, "swa_full_tokens_ratio", None) is not None
            and args.swa_full_tokens_ratio != 1.0
        ):
            raise ValueError(
                "Uniform DSv4 lifecycle conflicts with an explicit SWA/FULL ratio other than 1.0."
            )
        # Some dataclass fields (e.g. stat_loggers) intentionally have no CLI
        # surface and won't appear on the argparse Namespace. Skip them so the
        # dataclass default applies.
        attrs = [attr.name for attr in record_fields(cls) if hasattr(args, attr.name)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def get_tokenizer_worker_class(self):
        from decoupled_engine.srt.managers.multi_tokenizer_mixin import TokenizerWorker

        return TokenizerWorker

    def url(self, port: int | None = None):
        scheme = "https" if self.ssl_certfile else "http"
        # When binding to all interfaces, use loopback for internal requests.
        host = self.host
        if not host or host == "0.0.0.0":
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        return NetworkAddress(host, port if port is not None else self.port).to_url(
            scheme
        )

    @property
    def engine_info_bootstrap_url(self):
        return self.url(port=self.engine_info_bootstrap_port)

    def __setattr__(self, name, value):
        # Seal configuration fields, including underscore-prefixed ones, but allow bookkeeping.
        if not name.startswith("_") or name in _underscore_field_names():
            if getattr(self, "_input_frozen", False):
                raise AttributeError(
                    f"server_args.{name} assigned during resolution; the record "
                    "is the operator's input and resolution does not write it -- "
                    "declare the decision with declare_resolution(server_args, "
                    "source, **fields) so it carries a source and leaves the "
                    "input intact."
                )
            if getattr(self, "_resolution_finished", False):
                raise AttributeError(
                    f"server_args.{name} assigned after resolution; server_args is "
                    "read-only -- use get_context().override(source, ...) to change "
                    "resolved config; a value one runner owns travels as a "
                    "constructor argument."
                )
        # The Struct's own setter, spelled explicitly: this method is copied
        # into the class `defstruct` builds, so a zero-argument `super()` would
        # still close over the class it was written in. `object.__setattr__`
        # does not reach a Struct's fields at all.
        msgspec.Struct.__setattr__(self, name, value)

    def __reduce__(self):
        """Preserve resolution bookkeeping as well as Struct fields when pickling.

        Restore fields before bookkeeping so the write seal is re-armed last.
        """
        return (
            _rebuild_server_args,
            (type(self), msgspec.structs.asdict(self), dict(self.__dict__)),
        )

    def check_server_args(self):
        from decoupled_engine.srt.arg_groups.validation_hook import check_server_args

        check_server_args(self)

    def remote_instance_weight_loader_use_transfer_engine(self, load_format=None):
        """``load_format`` overrides the seed's: a draft runner loading under
        ``--speculative-draft-load-format`` needs its own transfer engine."""
        return remote_instance_transfer_engine_of(resolving_view(self), load_format)


# --------------------------------------------------------------------------
# EI input fields.
# --------------------------------------------------------------------------
# EI's server arguments: symmetric P/D, late decode binding, bootstrap
# prefetch, the peer-KV servicer, the DSv4/DSA/MLA HiCache host tier, KV-session
# retention, scheduler traces and strict replay. Upstream declares its fields
# in arg_groups/fields/; EI's live here, one class per namespace, with the
# ``_NS_PATH`` of the upstream namespace they belong to, so they project into
# the same config bags (get_memory(), get_disagg(), ...). Their validation is
# in the arg_groups hooks (kv_cache_hook, serving_hook, pd_disaggregation_hook,
# hicache_hook, speculative_hook) and in overrides.py.


class EiMemory(msgspec.Struct):
    """EI fields of the ``memory`` namespace."""

    _NS_PATH = "memory"

    dsv4_uniform_cache_lifecycle: A[
        bool,
        "Retain complete DSv4 SWA/CSA/HCA prefixes with joint cache lifecycle (experimental, CUDA HiCache prefill/standalone only).",
    ] = False
    dsv4_uniform_swa_residency: A[
        Literal["full", "window"],
        "Device SWA residency under --dsv4-uniform-cache-lifecycle. 'full' keeps every cached SWA page on the GPU and forces --swa-full-tokens-ratio 1.0. 'window' keeps only the sliding window on the GPU and every SWA page on the host; it defaults --swa-full-tokens-ratio to 0.1 and requires --dsv4-hicache-host-tokens or --dsv4-hicache-host-gb.",
    ] = "full"
    dsv4_hicache_host_tokens: A[
        Optional[int],
        "Host tier capacity in tokens for the DSv4 KV anchor and SWA pools (and their derived state pools) under --dsv4-uniform-swa-residency window, where --hicache-ratio is ignored. Must be at least 1.5x the device FULL tokens. Taken literally: its bytes follow the host layout (the default tail-only SWA and c4 state layouts take about 0.55x the bytes per token of whole pages). Alternative to --dsv4-hicache-host-gb.",
    ] = None
    dsv4_hicache_host_gb: A[
        Optional[float],
        "Host tier budget in GB (1e9 bytes) per byte-holding rank for every DSv4 host pool together under --dsv4-uniform-swa-residency window, where --hicache-ratio is ignored. Converted at cache build into whole pages at the active host layout's bytes per page, so the same budget holds more tokens with the tail-only layouts. The resulting tokens must be at least 1.5x the device FULL tokens. Alternative to --dsv4-hicache-host-tokens.",
    ] = None
    dsv4_swa_backfill: A[
        Literal["auto", "off"],
        "Host->host SWA backfill of received tail-only prefixes under --dsv4-uniform-swa-residency window. 'auto' enables it when a peer-KV storage backend is configured and resolves to 'off' without a HiCache storage backend; 'off' truncates such inserts at their prefix.",
    ] = "auto"
    dsv4_swa_backfill_timeout_s: A[
        float,
        "Per-request deadline in seconds for a window-residency SWA backfill, including the wait for the owner to publish and retries. It may exceed the owner's lease TTL: every retry takes a fresh grant, a grant older than half its TTL is dropped before host slots are allocated, and a read is never started (or accepted) near or past its lease expiry.",
    ] = 60.0
    dsv4_swa_backfill_max_inflight_gb: A[
        float,
        "Rate limit (GB in flight) for window-residency SWA backfill reads.",
    ] = 16.0
    enable_kv_session_retention: A[
        bool,
        "Track versioned trajectory references for CPU KV eviction without enabling native session-cache behavior. Requires the unified FULL-only write-through HiCache.",
    ] = False
    enable_dsv4_hicache_host_dedup: A[
        bool,
        "Keep one DeepSeek-V4 host KV payload per TP group and broadcast loaded "
        "pages to its GPUs. Supports single-node TP with local or peer_kv HiCache.",
    ] = False
    enable_dsa_hicache_host_dedup: A[
        bool,
        "Keep DSA KV and indexer payloads on one CPU rank and broadcast loaded "
        "data to the TP GPUs. Requires single-node TP, kernel IO and page_first.",
    ] = False
    enable_mla_hicache_host_dedup: A[
        bool,
        "Keep one CPU copy of replicated MLA target, packed MTP and draft KV and "
        "broadcast restored rows to the TP GPUs. Requires single-node TP, "
        "kernel IO and page_first layout.",
    ] = False
    hicache_storage_repull_on_admission: A[
        bool,
        "Prefill only, with a hint-only HiCache storage backend (peer_kv): when "
        "admission reaches a request whose device+host prefix is short of its "
        "kv_pull_hint expected_hit_tokens by at least "
        "--hicache-storage-repull-min-tokens (and the hint's "
        "min_local_deficit_tokens), pull the prefix again and admit the "
        "requests behind it until the pull lands. Inactive on decode workers.",
    ] = False
    hicache_storage_repull_min_tokens: A[
        int,
        "Smallest prefix deficit (tokens) that triggers an admission re-pull.",
    ] = 8192
    hicache_storage_repull_max_attempts: A[
        int,
        "Maximum admission re-pulls per request.",
    ] = 1
    hicache_storage_jit_pull: A[
        str,
        Arg(
            help=(
                "Prefill only, with a hint-only HiCache storage backend (peer_kv): "
                "'queue' defers a hinted request's peer pull while the queues "
                "ahead of it are long (--hicache-storage-jit-pull-now-*) and "
                "issues it once the request is near the head of the waiting "
                "queue (--hicache-storage-jit-pull-lead-*). Inactive on decode "
                "workers."
            ),
            choices=["off", "queue"],
        ),
    ] = "off"
    hicache_storage_jit_pull_now_reqs: A[
        int,
        "Pull at arrival when at most this many requests are queued ahead.",
    ] = 8
    hicache_storage_jit_pull_now_tokens: A[
        int,
        "Pull at arrival when at most this many input tokens are queued ahead.",
    ] = 1100000
    hicache_storage_jit_pull_lead_reqs: A[
        int,
        "Issue a deferred pull when at most this many requests are ahead of it "
        "in the waiting queue.",
    ] = 4
    hicache_storage_jit_pull_lead_tokens: A[
        int,
        "Issue a deferred pull when at most this many input tokens are ahead "
        "of it in the waiting queue.",
    ] = 131072
    hicache_queued_prefix_protect: A[
        str,
        Arg(
            help=(
                "Prefill only (PD prefill, unified HiCache tree, write-through): "
                "host locks on the cached prefixes of queued requests until "
                "admission. 'pull' locks peer-pulled pages when they land; 'all' "
                "also locks the local prefix at arrival. Bounded by "
                "--hicache-queued-prefix-max-tokens and released under host "
                "pressure. Inactive on decode workers."
            ),
            choices=["off", "pull", "all"],
        ),
    ] = "off"
    hicache_queued_prefix_max_tokens: A[
        int,
        "Queued-prefix lock budget in host tokens (sum of locked path depths).",
    ] = 3000000
    hicache_queued_prefix_max_fraction: A[
        float,
        "Queued-prefix lock budget cap, as a fraction of the host pool's usable "
        "tokens.",
    ] = 0.30
    hicache_queued_prefix_min_tokens: A[
        int,
        "No queued-prefix lock for a cached prefix shorter than this.",
    ] = 8192
    hicache_queued_prefix_ttl: A[
        float,
        "Queued-prefix lock age limit in seconds (a leak bound); 0 uses "
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT.",
    ] = 0.0
    enable_peer_kv_servicer: A[
        bool,
        "Enable the kv-p2p owner-side PeerKVServicer: a per-TP-rank thread that "
        "leases this engine's HiCache pinned host pages to peer engines for "
        "one-sided RDMA reads. Requires --enable-hierarchical-cache and the "
        "unified radix cache path with peer-enabled host pools (DeepSeek-V4 or DSA).",
    ] = False
    peer_kv_servicer_port: A[
        int,
        "Base TCP port for the PeerKVServicer's ZMQ REP listener; each TP rank "
        "binds base_port + attn_tp_rank.",
    ] = 18100
    enable_peer_kv_gpu_export: A[
        bool,
        "Prepare GPU-only prefixes on demand for peer reads through HiCache. "
        "Requires uniform DSv4 cache lifecycle, cache-mode write-through and "
        "single-node TP with CP=DP=PP=1; host dedup is not yet supported.",
    ] = False
    peer_kv_gpu_export_max_tokens: A[
        int,
        "Maximum complete source prefix protected by one GPU export. "
        "The prefix must also fit the ordinary host pools and peer lease budgets.",
    ] = 262144
    peer_kv_gpu_export_timeout_s: A[
        float,
        "Bounded admission/preparation lifetime for a GPU-only export ticket.",
    ] = 30.0
    peer_kv_servicer_max_leased_gb: A[
        float,
        "Total bytes (GB) of host pages that may be under peer lease at once; "
        "LOOKUPs beyond the budget are refused (fail-open to recompute).",
    ] = 16.0
    peer_kv_servicer_trace_dir: A[
        Optional[str],
        "Directory for the PeerKVServicer's per-LOOKUP JSONL trace "
        "(servicer_trace.*.jsonl). Disabled when unset.",
    ] = None
    peer_kv_servicer_reserve_ratio_kv: A[
        float,
        "Fraction of the KV-anchor host pages withheld from the free list on "
        "every TP rank as the kv-p2p pin reserve (pages a peer may still be "
        "RDMA-reading after eviction park here instead of being reused). Bounds "
        "the number of concurrently leased KV pages; rank-uniform by construction.",
    ] = 0.10
    peer_kv_servicer_reserve_ratio_swa: A[
        float,
        Arg(
            help=(
                "Same as --peer-kv-servicer-reserve-ratio-kv for the SWA host pool. "
                "Under --dsv4-uniform-swa-residency window the default follows "
                "--peer-kv-servicer-reserve-ratio-kv (0.10): full-SWA-prefix leases "
                "pin as many SWA pages as KV pages. An explicit value equal to the "
                "default (0.02) cannot be told apart from an unset flag and is "
                "treated as unset (it follows the KV ratio in window mode); pass a "
                "slightly different value (e.g. 0.021) to keep a reserve near 0.02."
            ),
            resolvable=True,
        ),
    ] = 0.02
    peer_kv_servicer_expiry_cooling_s: A[
        float,
        "Seconds a page stays out of the free list after its lease EXPIRED (not "
        "released): covers an RDMA read that started just before the TTL ran out. "
        "Should exceed the transfer engine's slice timeout (MC_SLICE_TIMEOUT).",
    ] = 12.0
    peer_kv_servicer_verify_checksums: A[
        bool,
        "Debug: the servicer returns a crc32 per (pool, page) with each LOOKUP "
        "and readers verify pulled pages against it (fails the pull on mismatch).",
    ] = False
    peer_kv_demote_source: A[
        bool,
        "After a peer has read every page it leased from this worker (complete "
        "RELEASE), host eviction takes this worker's copy of the pulled path "
        "first: the deepest leased node and its ancestors up to the first branch "
        "point or the first local access since the lookup. A later local access "
        "clears the flag. Requires --enable-peer-kv-servicer.",
    ] = False


class EiObservability(msgspec.Struct):
    """EI fields of the ``observability`` namespace."""

    _NS_PATH = "observability"

    enable_prefill_work_snapshot: A[
        bool,
        "Publish per-request remaining prefill compute for PD prefill routing. Missing request IDs never imply completion.",
    ] = False
    sched_trace_dir: A[
        Optional[str],
        "Directory for scheduler JSONL trace streams: per-iteration "
        "(iter_trace.*.jsonl) and per-finished-request (req_trace.*.jsonl) "
        "records, written by the attn_tp_rank-0 scheduler. If not specified, "
        "scheduler tracing is disabled.",
    ] = None


class EiSpec(msgspec.Struct):
    """EI fields of the ``spec`` namespace."""

    _NS_PATH = "spec"

    enable_eagle_strict_replay: A[
        bool,
        "Enable forced_output_token_ids strict replay through the EAGLE verify "
        "path. Off by default so ordinary EAGLE execution is unchanged.",
    ] = False
    synthetic_oracle_mtp: A[
        bool,
        "EAGLE strict-replay benchmarking mode. Replace eligible linear-chain "
        "draft candidates with recorded future tokens before target verification. "
        "This measures controlled acceptance, not natural MTP quality.",
    ] = False
    synthetic_oracle_mtp_accept_rate: A[
        Optional[float],
        "Target aggregate EAGLE draft accept rate in (0, 1] for "
        "--synthetic-oracle-mtp. Omit for full tail-adjusted oracle acceptance.",
    ] = None
    synthetic_oracle_dspark_accept_rate: A[
        Optional[float],
        "DSPARK strict-replay only. Target draft accept rate in (0, 1] for "
        "forced_output_token_ids requests: each verify round injects a seeded "
        "truncated-geometric number of replay tokens into the drafted block and "
        "corrupts the next draft position, so the trace-authoritative verify "
        "accepts exactly the sampled count (before the ragged-verify cap). The "
        "confidence head and verify-length schedule keep acting on the real "
        "draft. Rows without forced output are unaffected. Synthetic "
        "performance-control mode, not natural DSpark acceptance.",
    ] = None


class EiDisagg(msgspec.Struct):
    """EI fields of the ``disagg`` namespace."""

    _NS_PATH = "disagg"

    disaggregation_decode_radix_cache_mode: A[
        Literal["full_state", "content_only", "window_host"],
        "Select what a decode-side radix prefix hit may reuse. 'full_state' preserves upstream behavior. 'content_only' reuses only stable FULL content KV while transferring the live SWA/request state fresh; it is restricted to DeepSeek-V4 with DSPARK, Mooncake or NIXL, and CUDA's non-unified physical KV layout. 'window_host' is the decode mode of --dsv4-uniform-swa-residency window: content_only's FULL-only reuse and fresh SWA tail, plus a host-complete finish insert (the SWA below the tail is backfilled host->host from the prefill's PeerKVServicer, later SWA is spilled to this worker's host); it needs --enable-hierarchical-cache and, for the backfill, --enable-peer-kv-servicer and a peer_kv storage backend.",
    ] = "full_state"
    disaggregation_bootstrap_prefetch: A[
        str,
        Arg(
            help="Prefill only (HiCache): while a request waits in the bootstrap queue "
            "for decode memory, load its host-cached prefix to the device on idle "
            "scheduler passes and pin it until admission. 'local' covers requests "
            "without a router KV pull hint, 'remote' covers hinted requests after "
            "their peer pull landed, 'both' covers all. Data movement only.",
            choices=["off", "local", "remote", "both"],
        ),
    ] = "off"
    disaggregation_bootstrap_prefetch_max_tokens: A[
        int,
        "Device tokens that bootstrap-prefetch pins may protect in total.",
    ] = 524288
    disaggregation_bootstrap_prefetch_max_fraction: A[
        float,
        "Also cap bootstrap-prefetch pins at this fraction of the device KV pool "
        "(the budget is the smaller of the two; <= 0 disables the fraction cap).",
    ] = 0.25
    disaggregation_bootstrap_prefetch_ttl: A[
        float,
        "Seconds a bootstrap-prefetch pin may wait for admission (<= 0: no TTL).",
    ] = 60.0
    disaggregation_bootstrap_prefetch_min_tokens: A[
        int,
        "Skip bootstrap prefetch when the host part is shorter than this.",
    ] = 2048
    disaggregation_bootstrap_prefetch_depth: A[
        int,
        "Bootstrap-prefetch ops in flight at once (concurrent submission only). "
        "1 = one merged op at a time. >1: one op per request, oldest first, at "
        "most one op per pass and only once every in-flight op has started on "
        "every rank; the queue head's op runs uncapped. The pin budget stays "
        "--disaggregation-bootstrap-prefetch-max-tokens / -max-fraction.",
    ] = 1
    disaggregation_late_d: A[
        Literal["off", "alloc", "choose"],
        Arg(
            help=(
                "Defer decode memory allocation. 'alloc': the router selects the decode "
                "worker at arrival. That worker holds the request without KV memory, a "
                "request slot or a metadata buffer until prefill permits allocation. "
                "'choose': select the decode worker later through the router control "
                "link, then release the other candidates with HTTP 409. Requests without "
                "the router's late_d field use alloc behavior. Set the same mode on "
                "prefill and decode workers. Separate prefill/decode requires Mooncake; "
                "DP attention and PP > 1 are unsupported. Symmetric mode supports alloc "
                "and choose with count mode. To compute before reader allocation in "
                "symmetric mode, use alloc or remote choose + need + first_chunk, with "
                "router LOCAL off, overlap scheduling and "
                "context parallel size 1. Symmetric mode also requires "
                "SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE=1 and "
                "SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION > 0."
            ),
            action=ExplicitLateDAction,
            action_kwargs={"choices": ["off", "alloc", "choose"]},
        ),
    ] = "off"
    # Internal provenance: --disaggregation-late-d was given on the command line,
    # so the symmetric `paper` preset keeps its value (an explicit `off` included).
    _disaggregation_late_d_explicit: A[Optional[bool], Arg(no_cli=True)] = None
    symmetric_preset: A[
        Literal["paper", "none"],
        "--disaggregation-mode symmetric only. 'paper' (default): turn on the "
        "features of the paper's configuration that the other flags leave off: "
        "the host KV tier, the prefill work snapshot, late decode choice "
        "(need, first_chunk), the peer KV servicer and pulls when "
        "--disaggregation-ib-device is set, KV session retention on FULL-only "
        "models, and host dedup for the model's attention kind on single-node "
        "TP. 'none': the flags as given.",
    ] = "paper"
    disaggregation_late_d_lead_min_reqs: A[
        int,
        "Late D: minimum lead window (bound but not started requests per prefill). Below it the token cap does not apply.",
    ] = 2
    disaggregation_late_d_lead_max_reqs: A[
        int, "Late D: maximum lead window (requests)."
    ] = 8
    disaggregation_late_d_lead_max_tokens: A[
        Optional[int],
        "Late D: input-token cap of the lead window above the minimum. Default: 4 x chunked_prefill_size. Lead mode 'work': no default; if set, it caps the prompt tokens bound ahead (D KV held early).",
    ] = None
    disaggregation_late_d_lead_mode: A[
        Literal["count", "work", "need"],
        "Control when prefill prepares decode allocation. 'count' (default): "
        "limit requests prepared before their first prefill chunk, using the "
        "launch rate and the configured request and token limits. 'work': use "
        "measured allocation and prefill iteration times to prepare enough "
        "uncached tokens to cover the allocation delay, plus the configured "
        "margin. Work mode requires separate prefill/decode workers. 'need': "
        "allow prefill computation before decode allocation, with a retained-KV "
        "budget and a wait timeout. In symmetric mode, need requires alloc "
        "or remote choose with the first_chunk allocation trigger; router LOCAL "
        "must be off for choose + need.",
    ] = "count"
    disaggregation_late_d_allocation_trigger: A[
        Literal["lead", "first_chunk"],
        "'lead': use the configured allocation preparation window. 'first_chunk': "
        "send allocation permission, or request a decode worker choice, at the "
        "first prefill CPU forward-launch boundary. This is not a GPU kernel "
        "start marker. It requires need mode and lets prefill compute without "
        "waiting for decode allocation. Symmetric mode supports this trigger "
        "only with alloc.",
    ] = "lead"
    disaggregation_late_d_lead_margin_iters: A[
        int,
        "Late D, lead mode 'work': prefill iterations added to ceil(T_bind / T_iter).",
    ] = 1
    disaggregation_late_d_max_outstanding_per_d: A[
        int,
        "Late D: maximum ADMITs a prefill keeps outstanding to one decode worker.",
    ] = 4
    disaggregation_late_d_hold_detect_ms: A[
        int,
        "Late D: ADMIT age after which the prefill treats the request as held on its decode and binds later requests.",
    ] = 200
    disaggregation_late_d_claim_iters: A[
        int,
        "Late D: decode preallocation passes before the oldest held request becomes the reservation claim.",
    ] = 20
    disaggregation_late_d_control_port: A[
        Optional[int],
        "Late D stage 2 ('choose'): the TCP port of the prefill's control link to the router (a ZMQ ROUTER socket on attn-TP rank 0). Required on a prefill with 'choose'.",
    ] = None
    disaggregation_late_d_choice_timeout_ms: A[
        int,
        "Late D stage 2: how long the prefill waits for the router's decision before it commits the default candidate.",
    ] = 20
    disaggregation_late_d_export_aux_reserve: A[
        Optional[int],
        "Late D, symmetric mode: metadata (aux) slots that imports never take, so a P-leg whose reader began always gets one at its bind. Default: --disaggregation-late-d-lead-max-reqs + 2.",
    ] = None


# Collect input fields only; field_order.py preserves positional argument order.


_INPUT_NAMESPACES = [
    Model,
    ExecDeterministic,
    ExecDllm,
    ExecOffload,
    ExecOverlap,
    Lora,
    Disagg,
    Spec,
    ExecMoe,
    ExecComm,
    ExecGraph,
    ExecMamba,
    ExecKernel,
    Observability,
    Device,
    Parallel,
    Memory,
    Schedule,
    ExecFeatures,
    Mm,
    Serving,
    # EI namespaces (declared above), appended after upstream's fields.
    EiMemory,
    EiObservability,
    EiSpec,
    EiDisagg,
]

_annotations, _defaults, _namespaces = collect_input_fields(_INPUT_NAMESPACES)
ServerArgs.__annotations__ = {**_annotations, **ServerArgs.__annotations__}
ServerArgs._NS_BY_FIELD = _namespaces
ServerArgs._NAMESPACES = _INPUT_NAMESPACES
# dict=True retains resolution bookkeeping and memoized values outside the fields.
ServerArgs = msgspec.defstruct(
    "ServerArgs",
    [
        (_name, _ann, _defaults[_name]) if _name in _defaults else (_name, _ann)
        for _name, _ann in ServerArgs.__annotations__.items()
    ],
    namespace={
        _k: _v
        for _k, _v in vars(ServerArgs).items()
        if _k not in ("__dict__", "__weakref__", "__annotations__")
    },
    dict=True,
)


# --------------------------------------------------------------------------
# Module-level ServerArgs helpers and runtime shims.
# --------------------------------------------------------------------------


def resolve_encoder_transfer_backend(
    backend: str, model_arch: str, tp_size: int
) -> str:
    if backend != "auto":
        return backend
    if model_arch == "KimiK3ForConditionalGeneration" and tp_size > 1:
        return "zmq_to_tokenizer"
    return "zmq_to_scheduler"


def compute_world_size(*, dp_size: int, tp_size: int, pp_size: int) -> int:
    """Total GPU count across all data-parallel replicas.

    Takes the values rather than a config object: the sizes are the widths
    the launch asked for, which the Ray driver needs before any process group
    exists, and passing a context would hand it the live groups instead.
    Attention DP runs inside the TP group, so it adds no GPUs.
    """
    return dp_size * tp_size * pp_size


def m3_fp8_attn_gemm_enabled(args) -> bool:
    """Whether MiniMax-M3 attention GEMMs run in fp8 (no opt-in flag; active
    whenever possible): fp8_e4m3 main + index KV caches, fp8-cast q, fp8
    sparse/MSA kernels, with dense layers on trtllm_mha's fp8-q path. Needs
    kv_cache_dtype fp8_e4m3 (e5m2 would silently mis-dispatch fmha_sm100's
    e4m3 kernel), the trtllm_mha backend (the only dense backend with fp8-q
    GEMMs), and SM100 (MSA fp8 variants and trtllm-gen fp8 dense kernels are
    sm100-only). SGLANG_DISABLE_M3_FP8_ATTN_GEMM=1 is the kill switch:
    it forces bf16 q with the widening sparse/indexer path without having to
    move off trtllm_mha.
    """
    from decoupled_engine.srt.environ import envs

    return (
        args.kv_cache_dtype == "fp8_e4m3"
        and args.attention_backend == "trtllm_mha"
        and get_platform().is_sm100
        and not envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.get()
    )


@functools.lru_cache(maxsize=1)
def _underscore_field_names() -> frozenset:
    """Configuration fields that must stay sealed despite their underscore prefix."""
    return frozenset(
        field.name for field in record_fields(ServerArgs) if field.name.startswith("_")
    )


# NOTE: The process-wide ServerArgs is owned by the runtime context
# (decoupled_engine.srt.runtime_context). The two publish functions below are LEGACY
# shims kept for the existing call-sites; they hand over the same live object
# by reference. Do not add new call-sites. The third function is retired and
# only raises.
# Imports are in-function so the two modules stay cycle-free at import time.
def set_global_server_args_for_scheduler(server_args: ServerArgs):
    """Legacy publish shim (role=scheduler) — prefer
    ``runtime_context.publish(server_args, role=...)`` in new code."""

    publish(server_args, role="scheduler")


def set_global_server_args_for_tokenizer(server_args: ServerArgs):
    """Legacy publish shim (role=tokenizer). Not aliased to the scheduler shim:
    the process role differs."""

    publish(server_args, role="tokenizer")


def get_global_server_args() -> NoReturn:
    """Retired accessor retained to raise a migration error for existing imports."""
    raise RuntimeError(
        "get_global_server_args() is retired. Read the value that is in effect "
        "from its namespace bag -- `get_exec().kernel.attention_backend`, "
        "`get_schedule().max_running_requests`, and so on "
        "(decoupled_engine.srt.runtime_context). For the operator's raw input, which is a "
        "different question, `get_server_args()` still answers it."
    )


def _rebuild_server_args(cls, fields, bookkeeping):
    """Rebuild a pickled record: fields through the constructor, the rest after."""
    record = cls(**fields)
    record.__dict__.update(bookkeeping)
    return record


def _declared_default(name: str):
    """Return a field default; Struct class attributes are slot descriptors."""
    return next(
        field.default
        for field in msgspec.structs.fields(ServerArgs)
        if field.name == name
    )


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(parser)

    # Check for config file and merge arguments if present
    if "--config" in argv:
        # Import here to avoid circular imports
        from decoupled_engine.srt.utils.server_args_config_parser import ConfigArgumentMerger

        # Extract boolean actions from the parser to handle them correctly
        config_merger = ConfigArgumentMerger(parser)
        argv = config_merger.merge_config_with_args(argv)

    radix_eviction_policy_explicitly_set = any(
        arg == "--radix-eviction-policy" or arg.startswith("--radix-eviction-policy=")
        for arg in argv
    )

    raw_args = parser.parse_args(argv)
    raw_args._radix_eviction_policy_explicitly_set = (
        radix_eviction_policy_explicitly_set
    )

    # Set up basic logging before ServerArgs.__post_init__ so that
    # logger.info / logger.warning calls there are properly formatted.
    logging.basicConfig(
        level=getattr(logging, raw_args.log_level.upper()),
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    server_args = ServerArgs.from_cli_args(raw_args)
    # Not a field: the record's fields are the configuration, and this is how
    # the configuration was asked for. It rides along on the record so a
    # subprocess copy can answer the same question the launcher can.
    # lazy: watermark imports torch, outside server_args' top-level boundary.
    from decoupled_engine.srt.sampling.watermarking.core import redact_watermark_command_line

    server_args._launch_command = redact_watermark_command_line(
        argv, raw_args.watermark_config
    )
    return server_args


# --------------------------------------------------------------------------
# Networking constants and PortArgs.
# --------------------------------------------------------------------------


ZMQ_TCP_PORT_DELTA = 233
DP_ATTENTION_HANDSHAKE_PORT_DELTA = 13


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # The ipc filename for MultiTokenizerRouter to receive inputs from TokenizerWorker processes (zmq)
    tokenizer_worker_ipc_name: str | None

    # The ipc endpoints between verifier scheduler and drafter scheduler
    decoupled_spec_ipc_config: DecoupledSpecIpcConfig | None

    # zmq address for load snapshot PUSH/PULL (dp-attention TCP mode only;
    # empty when IPC mode derives the address from instance_id).
    load_collector_ipc_name: str = ""

    # Stable token shared by all processes in one server instance, used to
    # derive the /dev/shm path for load snapshots.
    instance_id: str = ""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        dp_rank: int | None = None,
        worker_ports: list[int] | None = None,
    ) -> PortArgs:
        cfg = resolving_view(server_args)
        if server_args.nccl_port is None:
            # The scheduler child binds this later; keep it out of the ephemeral range.
            nccl_port = get_free_port_below_ephemeral()
        else:
            nccl_port = server_args.nccl_port

        if server_args.tokenizer_worker_num == 1:
            tokenizer_worker_ipc_name = None
        else:
            tokenizer_worker_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )

        instance_id = uuid.uuid4().hex[:12]

        decoupled_spec_ipc_config = None
        if server_args.decoupled_spec_role != "null":
            if (
                server_args.decoupled_spec_bind_endpoint is None
                or server_args.decoupled_spec_connect_endpoints is None
                or server_args.decoupled_spec_rank is None
            ):
                raise ValueError(
                    "--decoupled-spec-bind-endpoint, "
                    "--decoupled-spec-connect-endpoints, and "
                    "--decoupled-spec-rank are required for decoupled speculative decoding."
                )
            decoupled_spec_ipc_config = DecoupledSpecIpcConfig(
                bind_endpoint=server_args.decoupled_spec_bind_endpoint,
                connect_endpoints=tuple(server_args.decoupled_spec_connect_endpoints),
                rank=int(server_args.decoupled_spec_rank),
            )

        if not attn_dp_enabled_of(cfg):
            # Normal case, use IPC within a single node
            return PortArgs(
                tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                nccl_port=nccl_port,
                rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
                decoupled_spec_ipc_config=decoupled_spec_ipc_config,
                instance_id=instance_id,
            )
        else:
            # DP attention. Use TCP + port to handle both single-node and multi-node.
            if server_args.nnodes == 1 and server_args.dist_init_addr is None:
                derived_port = server_args.port + ZMQ_TCP_PORT_DELTA
                if derived_port > 65535:
                    derived_port = server_args.port - ZMQ_TCP_PORT_DELTA
                na = NetworkAddress("127.0.0.1", derived_port)
            else:
                na = NetworkAddress.parse(server_args.dist_init_addr)

            dist_init_host = na.host
            dist_init_port = na.port

            # Reserve port_base+0..NUM_DERIVED_PORTS-1 (6 fixed ports + one
            # rust-path slot per DP rank); derive from server_args only (never dp_rank) so
            # every init_new call agrees, decrementing below dist_init_port on
            # overflow.
            is_rust_server = envs.SGLANG_RUST_SERVER.get()
            NUM_DERIVED_PORTS = 6 if not is_rust_server else 6 + num_dp_ranks_of(cfg)
            if ep_scale_joiner_of(resolving_view(server_args)):
                port_base = server_args.port + ZMQ_TCP_PORT_DELTA
                if port_base + NUM_DERIVED_PORTS > 65535:
                    port_base = server_args.port - ZMQ_TCP_PORT_DELTA
            elif dist_init_port + NUM_DERIVED_PORTS > 65535:
                port_base = dist_init_port - NUM_DERIVED_PORTS - 1
            else:
                port_base = dist_init_port + 1

            detokenizer_port = port_base + 1
            rpc_port = port_base + 2
            metrics_port = port_base + 3
            load_collector_port = port_base + 5
            if dp_rank is None:
                # TokenizerManager to DataParallelController
                scheduler_input_port = port_base + 4
            elif is_rust_server:
                # Rust server path (SGLANG_RUST_SERVER + dp attention): there is no
                # DataParallelController allocating worker ports.
                scheduler_input_port = port_base + 6 + dp_rank
            else:
                assert worker_ports is not None
                scheduler_input_port = worker_ports[dp_rank]

            is_joiner = ep_joiner_of(resolving_view(server_args))
            # Under SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE, SGLang never binds
            # dist_init_port / nccl_port (rendezvous uses the externally-managed
            # store; see distributed/bootstrap.py:_resolve_dist_init_method), so
            # their prechecks could only false-positive and are skipped.
            dist_init_overridden = bool(
                envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
            )
            try:
                if dp_rank is None:
                    if not (is_joiner or dist_init_overridden):
                        wait_port_available(dist_init_port, "dist_init_port")
                    wait_port_available(port_base, "port_base")
                    wait_port_available(detokenizer_port, "detokenizer_port")
                    if not dist_init_overridden:
                        wait_port_available(nccl_port, "nccl_port")
                    wait_port_available(rpc_port, "rpc_port")
                    wait_port_available(metrics_port, "metrics_port")
                    if server_args.nnodes > 1:
                        wait_port_available(load_collector_port, "load_collector_port")
                # Check scheduler_input_port only for dp.
                # Skip check when using worker_ports since the port is already bound by our ZMQ socket
                if dp_rank is None or worker_ports is None:
                    wait_port_available(scheduler_input_port, "scheduler_input_port")
            except ValueError:
                logger.exception(
                    f"Port is already in use. {dist_init_port=} {port_base=} {detokenizer_port=} {nccl_port=} {scheduler_input_port=}"
                )
                raise

            return PortArgs(
                tokenizer_ipc_name=NetworkAddress(dist_init_host, port_base).to_tcp(),
                scheduler_input_ipc_name=NetworkAddress(
                    dist_init_host, scheduler_input_port
                ).to_tcp(),
                detokenizer_ipc_name=NetworkAddress(
                    dist_init_host, detokenizer_port
                ).to_tcp(),
                nccl_port=nccl_port,
                rpc_ipc_name=NetworkAddress(dist_init_host, rpc_port).to_tcp(),
                metrics_ipc_name=NetworkAddress(dist_init_host, metrics_port).to_tcp(),
                tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
                decoupled_spec_ipc_config=decoupled_spec_ipc_config,
                load_collector_ipc_name=NetworkAddress(
                    dist_init_host, load_collector_port
                ).to_tcp(),
                instance_id=instance_id,
            )

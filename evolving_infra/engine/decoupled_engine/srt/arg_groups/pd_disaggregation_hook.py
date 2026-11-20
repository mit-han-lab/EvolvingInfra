# Adapted from SGLang python/sglang/srt/arg_groups/pd_disaggregation_hook.py (commit 297e54fd10), modified for evolving-infra.
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from decoupled_engine.srt.arg_groups.arg_utils import record_fields
from decoupled_engine.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.runtime_context import attn_dp_enabled_of, num_dp_ranks_of

if TYPE_CHECKING:
    from decoupled_engine.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: ServerArgs) -> None:
    """Validate and normalize PD-disaggregation server args."""
    cfg = resolving_view(server_args)

    # "mooncake_tcp" is mooncake with the TCP transport forced: set MC_FORCE_TCP
    # so mooncake installs TcpTransport instead of RDMA, rewrite the backend to
    # mooncake, and skip RDMA HCA selection. Must run before backend-name checks.
    if cfg.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_transfer_backend="mooncake",
        )
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_ib_device=None,
        )
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    decode_radix_cache_mode = cfg.disaggregation_decode_radix_cache_mode
    if decode_radix_cache_mode not in ("full_state", "content_only", "window_host"):
        raise ValueError(
            "--disaggregation-decode-radix-cache-mode must be one of "
            "'full_state', 'content_only' or 'window_host', got "
            f"{decode_radix_cache_mode!r}."
        )
    content_only_decode_radix = decode_radix_cache_mode == "content_only"
    # window_host: the decode side of DSv4 window SWA residency. The residency
    # itself is checked in handle_cache_compatibility.
    window_host_decode_radix = decode_radix_cache_mode == "window_host"
    if (content_only_decode_radix or window_host_decode_radix) and not (
        cfg.disaggregation_mode == "decode"
        and cfg.disaggregation_decode_enable_radix_cache
    ):
        raise ValueError(
            f"--disaggregation-decode-radix-cache-mode {decode_radix_cache_mode} "
            "requires --disaggregation-mode decode and "
            "--disaggregation-decode-enable-radix-cache."
        )

    if cfg.disaggregation_decode_allocation_policy == "prefill_complete":
        _validate_prefill_complete(server_args)
        if (
            cfg.disaggregation_mode == "prefill"
            and cfg.optimistic_prefill_attempts == 0
        ):
            declare_resolution(
                server_args, "handle_pd_disaggregation", optimistic_prefill_attempts=1
            )

    if cfg.disaggregation_mode == "prefill" and cfg.dcp_size > 1:
        logger.warning(
            "DCP on a PD prefill server is supported when prefill and decode "
            "use the same DCP layout, but it usually adds communication "
            "overhead without improving prefill performance."
        )

    if not 0 <= cfg.disaggregation_decode_host_receive_threshold <= 1:
        raise ValueError(
            "--disaggregation-decode-host-receive-threshold must be between 0 and 1"
        )
    if cfg.disaggregation_decode_host_receive_threshold < 1:
        if cfg.enable_hisparse or cfg.enable_pd_role_switch:
            raise ValueError(
                "Decode host receive does not yet support HiSparse or role switching"
            )

        if cfg.disaggregation_decode_retraction_backup == "cpu_tensor":
            raise ValueError("Decode host KV buffering requires host_pool retraction")
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_decode_retraction_backup="host_pool",
        )

    if cfg.disaggregation_mode == "decode" and cfg.dcp_size > 1:
        # Fake transfer moves no KV and is only used for synthetic decode
        # benchmarks, so it does not need the DCP relayout from Mooncake/NIXL.
        if cfg.disaggregation_transfer_backend not in (
            "mooncake",
            "nixl",
            "mori",
            "ascend",
            "fake",
        ):
            raise ValueError(
                "PD decode DCP requires --disaggregation-transfer-backend "
                "mooncake, nixl, mori, ascend, or fake for synthetic benchmarking, got "
                f"{cfg.disaggregation_transfer_backend!r}."
            )

    if cfg.disaggregation_mode == "decode":
        if cfg.disaggregation_decode_enable_radix_cache:
            if cfg.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if cfg.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            # Upstream validates the speculative algorithm itself
            # (speculative_hook: decode radix cache takes None, DSPARK, EAGLE
            # or EAGLE3). EI's decode radix modes add their own constraints.
            speculative_algorithm = (cfg.speculative_algorithm or "").upper()
            attn_dp_enabled = attn_dp_enabled_of(resolved_view(server_args))
            if content_only_decode_radix:
                if speculative_algorithm != "DSPARK":
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode content_only "
                        "currently requires --speculative-algorithm DSPARK."
                    )
                if cfg.disaggregation_transfer_backend not in (
                    "mooncake",
                    "nixl",
                ):
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode content_only "
                        "currently requires --disaggregation-transfer-backend "
                        "mooncake or nixl."
                    )
                if cfg.hicache_storage_backend is not None:
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode content_only "
                        "supports the HiCache host tier only; an L3 storage "
                        "backend is not supported."
                    )
                if cfg.enable_session_radix_cache:
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode content_only "
                        "is incompatible with --enable-session-radix-cache."
                    )
            elif window_host_decode_radix:
                # Unlike content_only, an L3/peer-KV storage backend is allowed:
                # it serves the host->host SWA backfill of the received prefix.
                if speculative_algorithm != "DSPARK":
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode window_host "
                        "currently requires --speculative-algorithm DSPARK."
                    )
                if cfg.disaggregation_transfer_backend not in (
                    "mooncake",
                    "nixl",
                ):
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode window_host "
                        "currently requires --disaggregation-transfer-backend "
                        "mooncake or nixl."
                    )
                if (
                    getattr(cfg, "dsv4_swa_backfill", "auto") != "off"
                    and cfg.hicache_storage_backend is None
                ):
                    logger.warning(
                        "--disaggregation-decode-radix-cache-mode window_host "
                        "without --hicache-storage-backend: the SWA backfill "
                        "from the prefill servicer is off, so a finished "
                        "request's insert stops at the prefix this worker "
                        "already held (its decode_prefix_len)."
                    )
            elif cfg.enable_hierarchical_cache:
                # With a HiCache host tier, full_state limits EAGLE and EAGLE3
                # as below; without one, they take only upstream's checks.
                # TODO: restore EAGLE and EAGLE3 drafts from the host tier
                # without these limits.
                # GLM EAGLE can retain target and draft KV at their shared
                # physical indices. A HiCache host tier is allowed only with
                # the packed MTP host layout, whose one restore op brings back
                # the target and draft rows of the KV and DSA indexer pools
                # (checked once the cache is built: kv_cache_builder.
                # _validate_decode_eagle_hicache_draft). A target-only restore
                # would leave the draft prefix stale.
                if speculative_algorithm == "EAGLE":
                    if (
                        cfg.speculative_eagle_topk != 1
                        or cfg.enable_multi_layer_eagle
                        or cfg.enable_session_radix_cache
                        or cfg.pp_size != 1
                        or attn_dp_enabled
                        or cfg.disaggregation_transfer_backend
                        not in ("mooncake", "nixl")
                    ):
                        raise ValueError(
                            "PD decode EAGLE radix cache requires topk=1, PP=1, "
                            "no DP attention or multi-layer EAGLE, no session "
                            "cache, and mooncake or nixl."
                        )
                    if cfg.hicache_storage_backend is not None:
                        # The decode EAGLE radix cache restores from the host
                        # tier only, and PD decode never prefetches from L3.
                        # P and D may share one launch config (role switch),
                        # so the L3 backend is inactive here, not an error.
                        logger.info(
                            "--hicache-storage-backend %s is inactive on a PD "
                            "decode worker with the EAGLE radix cache (it "
                            "uses the HiCache host tier only).",
                            cfg.hicache_storage_backend,
                        )
                        declare_resolution(
                            server_args,
                            "handle_pd_disaggregation",
                            hicache_storage_backend=None,
                        )
                elif speculative_algorithm == "EAGLE3":
                    # Kimi's flat MLA target and external draft use the existing
                    # complete HiCache draft sidecar. Validate the live layout
                    # before the decode queue can accept a request.
                    if (
                        cfg.speculative_eagle_topk != 1
                        or cfg.enable_multi_layer_eagle
                        or cfg.enable_session_radix_cache
                        or cfg.pp_size != 1
                        or attn_dp_enabled
                        or cfg.disaggregation_transfer_backend
                        not in ("mooncake", "nixl")
                    ):
                        raise ValueError(
                            "Native EAGLE3 radix requires topk1, PP1, no DP/multi-layer/session cache"
                        )

            if attn_dp_enabled:
                if content_only_decode_radix or window_host_decode_radix:
                    raise ValueError(
                        "--disaggregation-decode-radix-cache-mode "
                        f"{decode_radix_cache_mode} currently does not support "
                        "DP attention."
                    )
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=False,
            )
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=True,
            )
            logger.warning("Radix cache is disabled for decode server")

        _default_decode_extra_slots(server_args)

    elif cfg.disaggregation_mode == "symmetric":
        # symmetric P/D: the aggregated scheduler also holds
        # in-flight imports outside the batch, so size the extra req_to_token
        # slots exactly like a decode server. Chunked-prefill / activation
        # sizing keep the null-mode path (validation_hook, memory_hook).
        _default_decode_extra_slots(server_args)
        if cfg.disaggregation_decode_enable_radix_cache:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is a PD decode "
                "flag; symmetric mode uses the regular radix cache."
            )

    elif cfg.disaggregation_mode == "prefill":
        assert cfg.disaggregation_transfer_backend != "fake", (
            "Prefill server does not support 'fake' as the transfer backend"
        )

        if envs.SGLANG_RUST_SERVER.get():
            _alias_bootstrap_port_to_api_port(server_args)

    if cfg.disaggregation_mode in ("prefill", "decode"):
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and cfg.disaggregation_transfer_backend not in ("mooncake", "nixl")
        ):
            raise ValueError(
                f"SGLANG_DISAGG_STAGING_BUFFER requires "
                f"disaggregation_transfer_backend='mooncake' or 'nixl', "
                f"got '{cfg.disaggregation_transfer_backend}'."
            )

        # Reject features whose role-specific state is not rebuilt on a flip.
        if cfg.enable_pd_role_switch:
            view = resolved_view(server_args)
            unsupported = []
            if view.ep_size > 1:
                unsupported.append(f"expert parallelism (--ep-size {view.ep_size})")
            if view.moe_a2a_backend != "none":
                unsupported.append(
                    f"MoE all-to-all (--moe-a2a-backend {view.moe_a2a_backend})"
                )
            if view.pp_size > 1:
                unsupported.append(f"pipeline parallelism (--pp-size {view.pp_size})")
            if (
                attn_dp_enabled_of(view)
                and view.enable_dp_attention_local_control_broadcast
            ):
                unsupported.append("DP attention local control broadcast")
            if view.dcp_size > 1:
                unsupported.append(
                    f"decode context parallelism (--dcp-size {view.dcp_size})"
                )
            if view.speculative_algorithm is not None:
                unsupported.append(
                    "speculative decoding "
                    f"(--speculative-algorithm {view.speculative_algorithm})"
                )
            if unsupported:
                raise ValueError(
                    "--enable-pd-role-switch does not rebuild role-specific "
                    "state for the following features: "
                    + ", ".join(unsupported)
                    + ". Remove these options or drop --enable-pd-role-switch."
                )


def _validate_prefill_complete(server_args: ServerArgs) -> None:
    cfg = resolving_view(server_args)
    unsupported = []
    if cfg.disaggregation_mode not in ("prefill", "decode"):
        unsupported.append("aggregated serving")
    if cfg.disaggregation_transfer_backend != "mooncake":
        unsupported.append("transfer backends other than Mooncake")
    if cfg.pp_size != 1 or cfg.attn_cp_size != 1 or cfg.dcp_size != 1:
        unsupported.append("pipeline or attention context parallelism")
    attn_dp_size = cfg.dp_size if cfg.enable_dp_attention else 1
    if cfg.tp_size != attn_dp_size:
        unsupported.append("attention tensor parallelism greater than one")
    if cfg.enable_pdmux or cfg.decoupled_spec_role != "null":
        unsupported.append("PD multiplexing or decoupled speculation")
    if cfg.enable_pd_role_switch:
        unsupported.append("runtime PD role switching")
    if cfg.encoder_only or cfg.language_only:
        unsupported.append("encoder disaggregation")
    if cfg.enable_hisparse:
        unsupported.append("HiSparse")
    if cfg.disaggregation_decode_host_receive_threshold < 1:
        unsupported.append("decode host KV buffering")
    if cfg.disaggregation_mode == "decode" and (
        cfg.disaggregation_decode_enable_radix_cache or cfg.enable_hierarchical_cache
    ):
        unsupported.append("decode radix cache or decode HiCache")
    if (
        cfg.disaggregation_mode == "prefill"
        and cfg.enable_hierarchical_cache
        and (
            cfg.hicache_storage_backend is not None
            or cfg.hicache_write_policy != "write_back"
        )
    ):
        unsupported.append("prefill HiCache other than L2 write_back")
    if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
        unsupported.append("staging buffers")
    if envs.SGLANG_RUST_SERVER.get():
        unsupported.append("the Rust bootstrap registry")
    if unsupported:
        raise ValueError(
            "--disaggregation-decode-allocation-policy prefill_complete does not "
            "yet support " + ", ".join(unsupported)
        )


def _default_decode_extra_slots(server_args: ServerArgs) -> None:
    """Default the number of *extra* decode req_to_token slots reserved for
    in-transfer (being-received-from-prefill / imported) requests, on top of
    the max_running_requests-derived pool. Large batches get none; small
    per-worker batches reserve 2x the batch as cheap overlap headroom."""
    cfg = resolving_view(server_args)
    if cfg.disaggregation_decode_extra_slots is None:
        extra_slots = 0
        if cfg.max_running_requests is not None:
            per_worker = max(1, cfg.max_running_requests // num_dp_ranks_of(cfg))
            if per_worker <= 32:
                extra_slots = per_worker * 2
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_decode_extra_slots=extra_slots,
        )


def _alias_bootstrap_port_to_api_port(server_args: ServerArgs) -> None:
    """Rust-server prefill serves the KV bootstrap registry on the api listener
    itself, so the resolved bootstrap port must BE the api port — every internal
    consumer (KVManager registration, PrefillBootstrapQueue) reads the resolved
    field and agrees automatically. Decode is untouched: there the field names
    the PREFILL side's bootstrap port and must stay as the operator set it.
    A transfer backend that brings its own bootstrap server keeps the port as
    configured: that server, not the Rust registry, listens there.
    """
    # Lazy: disagg_service pulls in the disaggregation package. Its helper calls
    # disagg_service.get_kv_class, which a transfer backend registered at
    # runtime may rebind.
    from decoupled_engine.srt.managers.disagg_service import rust_server_serves_kv_bootstrap

    cfg = resolving_view(server_args)
    if not rust_server_serves_kv_bootstrap(cfg.disaggregation_transfer_backend):
        logger.info(
            "SGLANG_RUST_SERVER: transfer backend %s runs its own KV bootstrap "
            "server; keeping disaggregation_bootstrap_port %d",
            cfg.disaggregation_transfer_backend,
            cfg.disaggregation_bootstrap_port,
        )
        return
    default_port = next(
        f.default
        for f in record_fields(type(server_args))
        if f.name == "disaggregation_bootstrap_port"
    )
    if cfg.disaggregation_bootstrap_port not in (
        default_port,
        cfg.port,
    ):
        raise ValueError(
            "SGLANG_RUST_SERVER serves the PD KV bootstrap registry on the api "
            "port itself; --disaggregation-bootstrap-port "
            f"{cfg.disaggregation_bootstrap_port} conflicts with --port "
            f"{cfg.port}. Drop --disaggregation-bootstrap-port (decode "
            "nodes and the PD router must then target the prefill api port)."
        )
    if cfg.disaggregation_bootstrap_port != cfg.port:
        logger.info(
            "SGLANG_RUST_SERVER: KV bootstrap registry is served on the api "
            "port; disaggregation_bootstrap_port %d -> %d",
            cfg.disaggregation_bootstrap_port,
            cfg.port,
        )
        declare_resolution(
            server_args,
            "_alias_bootstrap_port_to_api_port",
            disaggregation_bootstrap_port=cfg.port,
        )


def handle_encoder_disaggregation(server_args: Any):
    from decoupled_engine.srt.arg_groups.model_hook import handle_language_model_only
    from decoupled_engine.srt.arg_groups.validation_hook import validate_ib_devices
    from decoupled_engine.srt.server_args import resolve_encoder_transfer_backend

    cfg = resolving_view(server_args)
    handle_language_model_only(server_args)
    if cfg.enable_prefix_mm_cache and not cfg.encoder_only:
        raise ValueError(
            "--enable-prefix-mm-cache requires --encoder-only to be enabled"
        )
    if cfg.encoder_only and cfg.language_only:
        raise ValueError("Cannot set --encoder-only and --language-only together")
    if cfg.encoder_only and not cfg.disaggregation_mode == "null":
        raise ValueError(
            "Cannot set --encoder-only and --disaggregation-mode prefill/decode together"
        )

    if cfg.language_only and len(cfg.encoder_urls) == 0:
        logger.info(
            "--language-only is set without --encoder-urls. Encoders are "
            "expected to register dynamically via the "
            "EncoderBootstrapServer."
        )

    # Validate IB devices when mooncake backend is used
    if (
        cfg.disaggregation_transfer_backend == "mooncake"
        and cfg.disaggregation_mode in ("prefill", "decode")
    ) or cfg.encoder_transfer_backend == "mooncake":
        declare_resolution(
            server_args,
            "_handle_encoder_disaggregation",
            disaggregation_ib_device=validate_ib_devices(cfg.disaggregation_ib_device),
        )

    # Validate model type for encoder disaggregation
    hf_config = model_config_of(server_args).hf_config
    model_arch = hf_config.architectures[0]
    if cfg.encoder_transfer_backend == "auto":
        declare_resolution(
            server_args,
            "_handle_encoder_disaggregation",
            encoder_transfer_backend=resolve_encoder_transfer_backend(
                cfg.encoder_transfer_backend, model_arch, cfg.tp_size
            ),
        )
        if cfg.encoder_only or cfg.language_only:
            logger.info(
                "Encoder transfer backend auto-resolved to %s for %s at TP%d.",
                cfg.encoder_transfer_backend,
                model_arch,
                cfg.tp_size,
            )
    if (cfg.encoder_only or cfg.language_only) and model_arch not in [
        "Qwen2VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen2AudioForConditionalGeneration",
        "Qwen2_5OmniForConditionalGeneration",
        "Dots3NoteForCausalLM",
        "KimiVLForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "MiMoV2ForCausalLM",
        "Glm5NextForConditionalGeneration",
        "MiniMaxM3SparseForConditionalGeneration",
        "MuseSparkForConditionalGeneration",
    ]:
        raise ValueError(
            f"Model type {model_arch} is not supported for encoder disaggregation. "
            f"Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, "
            f"Qwen2Audio, Qwen2.5Omni, Dots3-Note, Kimi, MiMoV2, GLM5Next, MiniMax-M3, MuseSpark."
        )

    if (
        model_arch == "MuseSparkForConditionalGeneration"
        and cfg.language_only
        and cfg.enable_adaptive_dispatch_to_encoder
    ):
        raise ValueError(
            "Muse language-only servers do not load local encoders; "
            "--enable-adaptive-dispatch-to-encoder is not supported."
        )

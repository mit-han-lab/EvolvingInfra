# SPDX-License-Identifier: Apache-2.0
"""Defaults that follow from the symmetric P/D server arguments.

With ``--disaggregation-mode symmetric`` and ``--symmetric-preset paper`` (the
default), resolution first turns on the features of the paper's configuration
that the other flags leave off (:func:`plan_features`): the host KV tier, the
prefill work snapshot, late decode choice computing before the decode is
bound, the peer KV servicer and pulls (when the RDMA devices are known), KV
session retention (FULL-only models) and host dedup for the model's attention
kind (single-node TP). ``--symmetric-preset none`` leaves the flags as given.

Each switch below is an environment variable that is off on its own. When the
server arguments select the feature it belongs to, resolution turns it on,
unless the variable is already set (``=0`` keeps a switch off):

- ``--disaggregation-mode symmetric`` with ``--disaggregation-late-d alloc``
  or ``choose``: the import admission rules that late D requires.
- symmetric with ``--disaggregation-late-d choose``: the KV hand-off and loop
  switches (L2 restore of imports, skip of the prefix the reader holds,
  publication after each result, the standard P/D catch-up budget, the launch
  thread, the import walk after the launch), and with
  ``--enable-kv-session-retention`` the branch tag on ADMIT.
- symmetric: non-blocking host-to-GPU copies at D admission; the unserialized
  prefill loop (overlap schedule); 1500 s bootstrap and waiting timeouts; a
  20 GB cap on retract backups in host memory; with the peer KV servicer,
  range pushes to its index; with HiCache on plain TP, the pipelined and
  idle-skipping per-step HiCache sync; with ``--kv-events-config``, lazy,
  merged and batched KV event publication.
- ``--enable-prefill-work-snapshot``: the prefill routing snapshot that the
  router's cost function reads, sized for long queues.
- symmetric with ``--enable-dsa-hicache-host-dedup`` or
  ``--enable-mla-hicache-host-dedup``: the copy-engine fan-out (``ce_peer``)
  and the direct host layout it needs. ``SGLANG_DSA_HOST_DEDUP_FANOUT=nccl``
  (or the MLA word) keeps the NCCL broadcast and the kernel layout. With DSA
  dedup, the per-load cross-rank check of the shared pool is off.

A switch that resolution turned on and that a later startup check rejects for
this model or loop is turned off again with a log line (:func:`refuse`). A
switch that the user set still fails startup.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Names that resolution set, comma-separated. Scheduler processes inherit it
# and consult it in refuse().
APPLIED_ENV = "SGLANG_AUTO_DEFAULTS_APPLIED"

# Host layouts from which the dedup default may move to the direct layout:
# the defaults, or the direct layout itself.
_DEDUP_LAYOUTS = {("kernel", "page_first"), ("direct", "page_first_direct")}
_DEDUP_FANOUTS = (
    ("enable_dsa_hicache_host_dedup", "SGLANG_DSA_HOST_DEDUP_FANOUT"),
    ("enable_mla_hicache_host_dedup", "SGLANG_MLA_HOST_DEDUP_FANOUT"),
)


def auto_applied(name: str, environ: Mapping[str, str] = os.environ) -> bool:
    """Whether resolution, not the user, set ``name``."""
    return name in environ.get(APPLIED_ENV, "").split(",")


def _user_set(name: str, environ: Mapping[str, str]) -> bool:
    return name in environ and not auto_applied(name, environ)


def _truthy(environ: Mapping[str, str], name: str) -> bool:
    return environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def _positive_int(environ: Mapping[str, str], name: str) -> bool:
    try:
        return int(environ.get(name, "0")) > 0
    except ValueError:
        return False


def plan(cfg: Any, environ: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """The variables to set and the server-argument fields to declare."""
    env: Dict[str, str] = {}
    fields: Dict[str, Any] = {}

    def default(name: str, value: str) -> None:
        if not _user_set(name, environ):
            env[name] = value

    symmetric = cfg.disaggregation_mode == "symmetric"
    late_d = getattr(cfg, "disaggregation_late_d", "off")
    overlap = not getattr(cfg, "disable_overlap_schedule", False)

    if symmetric and late_d in ("alloc", "choose"):
        default("SGLANG_SYMMETRIC_PD_IMPORT_LOCAL_RESERVE", "1")
        default("SGLANG_SYMMETRIC_PD_IMPORT_BUDGET_FRACTION", "1.0")

    if symmetric and late_d == "choose":
        if cfg.enable_hierarchical_cache:
            default("SGLANG_SYMMETRIC_IMPORT_L2_RESTORE", "1")
        default("SGLANG_SYMMETRIC_SKIP_REUSED_PREFIX_PUBLICATION", "1")
        default("SGLANG_SYMMETRIC_PUBLISH_AFTER_RESULT", "1")
        # Exclusive with the other budget rules and the small-first order.
        if not (
            _truthy(environ, "SGLANG_SYMMETRIC_BUDGET_PRE_BIND_ONLY")
            or _positive_int(environ, "SGLANG_SYMMETRIC_PUBLISH_SMALL_FIRST_AGE_PASSES")
        ):
            default("SGLANG_SYMMETRIC_BUDGET_STANDARD_PD", "1")
        if overlap:
            default("SGLANG_SYMMETRIC_LAUNCH_THREAD", "1")
            # The walk after the launch excludes both host prefetches.
            if getattr(cfg, "disaggregation_bootstrap_prefetch", "off") == "off" and not _truthy(
                environ, "SGLANG_HICACHE_PREFETCH_NEXT_BATCH"
            ):
                default("SGLANG_SYMMETRIC_ADMIT_AFTER_LAUNCH", "1")
                default("SGLANG_HICACHE_DECODE_CLEAR_CONSUMER", "1")
        if getattr(cfg, "enable_kv_session_retention", False):
            default("SGLANG_KV_SESSION_ADMIT_TAG", "1")
        # Decode capacity for the router's fit_req rule.
        default("SGLANG_DECODE_SELECTION_METRICS", "1")

    if symmetric:
        # Host-tier and transfer paths: the same work in fewer blocking syncs.
        default("SGLANG_PREBUILT_NONBLOCKING_H2D", "1")
        if overlap:
            default("SGLANG_PREFILL_UNSERIALIZE_LOOP", "1")
        # Long agent prompts: a prefill and its transfer can take minutes.
        default("SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", "1500")
        default("SGLANG_DISAGGREGATION_WAITING_TIMEOUT", "1500")
        default("SGLANG_RETRACT_BACKUP_MAX_GB", "20")
        if getattr(cfg, "enable_peer_kv_servicer", False):
            default("SGLANG_PEER_KV_PUSH_FAST", "1")
        plain_tp = getattr(cfg, "pp_size", 1) == 1 and not getattr(
            cfg, "enable_dp_attention", False
        )
        if cfg.enable_hierarchical_cache and plain_tp:
            default("SGLANG_HICACHE_PIPELINED_SYNC", "1")
            # The idle-step skip needs the TP digest off.
            if environ.get("SGLANG_HICACHE_TP_DIGEST_MODE", "").strip().lower() in ("", "off"):
                default("SGLANG_HICACHE_SKIP_IDLE_SYNC", "1")

    if getattr(cfg, "enable_prefill_work_snapshot", False):
        default("SGLANG_PREFILL_ROUTING_SNAPSHOT", "1")
        if symmetric and overlap:
            default("SGLANG_SYMMETRIC_ROUTING_SNAPSHOT", "1")
        # Room for a long queue: records, bytes, and the load-snapshot slot
        # that carries them.
        default("SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_RECORDS", "256")
        default("SGLANG_PREFILL_ROUTING_SNAPSHOT_MAX_BYTES", "98304")
        default("SGLANG_LOAD_SNAPSHOT_SLOT_BYTES", "131072")

    if symmetric and getattr(cfg, "kv_events_config", None):
        # Build the load-back events off the admission path, merged, and
        # publish them once per loop pass. The stream is byte-identical.
        default("SGLANG_KV_EVENTS_LAZY_LOAD_BACK", "1")
        default("SGLANG_KV_EVENTS_LAZY_FAST_BUILD", "1")
        default("SGLANG_KV_EVENTS_BATCH_PUBLISH", "1")

    if symmetric:
        layout = (cfg.hicache_io_backend, cfg.hicache_mem_layout)
        for flag, name in _DEDUP_FANOUTS:
            if getattr(cfg, flag, False) and not _user_set(name, environ) and layout in _DEDUP_LAYOUTS:
                env[name] = "ce_peer"
                fields.update(hicache_io_backend="direct", hicache_mem_layout="page_first_direct")
        if getattr(cfg, "enable_dsa_hicache_host_dedup", False):
            # The per-load cross-rank check of the shared pool (a gloo all-reduce).
            default("SGLANG_DSA_HOST_DEDUP_SHARED_VERIFY", "0")
    return env, fields


class _Overlay:
    """``cfg`` with ``fields`` in place of its values (read only)."""

    def __init__(self, cfg: Any, fields: Mapping[str, Any]):
        self._cfg, self._fields = cfg, fields

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        return getattr(self._cfg, name)


_PEER_KV_BACKEND = {
    "backend_name": "peer_kv",
    "module_path": "decoupled_engine.srt.mem_cache.storage.peer_kv.peer_kv_backend",
    "class_name": "PeerKVBackend",
    "interface_v1": 1,
}
_DEDUP_FLAGS = (
    "enable_dsa_hicache_host_dedup",
    "enable_mla_hicache_host_dedup",
    "enable_dsv4_hicache_host_dedup",
)


def ib_devices(value: Optional[str]) -> str:
    """Every RDMA device that ``--disaggregation-ib-device`` names, comma-separated."""
    if not value:
        return ""
    text = value.strip()
    if not text.startswith("{") and os.path.isfile(text):
        with open(text) as f:
            text = f.read()
    if text.lstrip().startswith("{"):
        parts = []
        for item in json.loads(text).values():
            parts.extend(str(item).split(","))
    else:
        parts = text.split(",")
    seen = []
    for part in (p.strip() for p in parts):
        if part and part not in seen:
            seen.append(part)
    return ",".join(seen)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def plan_features(
    cfg: Any,
    model: Optional[Mapping[str, Any]] = None,
    free_port: Callable[[], int] = _free_port,
    ratio_unset: bool = False,
) -> Dict[str, Any]:
    """The server-argument fields of the paper's configuration still off.

    ``model``: ``{"dedup": <dedup flag name or None>, "full_only": bool}``
    from the model configuration; None skips dedup and retention.
    ``ratio_unset``: the user gave no ``--hicache-ratio``; with host dedup it
    becomes the paper's 4 (4.445 with the peer KV servicer, which withholds
    10 % of the host tier)."""
    fields: Dict[str, Any] = {}
    if cfg.disaggregation_mode != "symmetric" or getattr(cfg, "symmetric_preset", "paper") != "paper":
        return fields
    single_engine = (
        getattr(cfg, "pp_size", 1) == 1
        and getattr(cfg, "dp_size", 1) == 1
        and getattr(cfg, "attn_cp_size", 1) == 1
        and not getattr(cfg, "enable_dp_attention", False)
    )
    if not cfg.enable_hierarchical_cache:
        fields["enable_hierarchical_cache"] = True
    if not getattr(cfg, "enable_prefill_work_snapshot", False):
        fields["enable_prefill_work_snapshot"] = True
    if (
        getattr(cfg, "disaggregation_late_d", "off") == "off"
        and single_engine
        and not getattr(cfg, "disable_overlap_schedule", False)
    ):
        fields["disaggregation_late_d"] = "choose"
        if getattr(cfg, "disaggregation_late_d_control_port", None) is None:
            fields["disaggregation_late_d_control_port"] = free_port()
        if getattr(cfg, "disaggregation_late_d_lead_mode", "count") == "count":
            fields["disaggregation_late_d_lead_mode"] = "need"
        if getattr(cfg, "disaggregation_late_d_allocation_trigger", "lead") == "lead":
            fields["disaggregation_late_d_allocation_trigger"] = "first_chunk"
    devices = ib_devices(getattr(cfg, "disaggregation_ib_device", None))
    if (
        devices
        and not getattr(cfg, "enable_peer_kv_servicer", False)
        and getattr(cfg, "hicache_storage_backend", None) is None
    ):
        fields["enable_peer_kv_servicer"] = True
        fields["hicache_storage_backend"] = "dynamic"
        fields["hicache_storage_backend_extra_config"] = json.dumps(
            {**_PEER_KV_BACKEND, "ib_device": devices}
        )
        if getattr(cfg, "peer_kv_servicer_max_leased_gb", 16.0) == 16.0:
            fields["peer_kv_servicer_max_leased_gb"] = 64.0
    if model is not None:
        dedup = model.get("dedup")
        if (
            dedup
            and single_engine
            and getattr(cfg, "nnodes", 1) == 1
            and not any(getattr(cfg, flag, False) for flag in _DEDUP_FLAGS)
        ):
            fields[dedup] = True
        dedup_on = any(fields.get(f, getattr(cfg, f, False)) for f in _DEDUP_FLAGS)
        if dedup_on and ratio_unset:
            servicer = fields.get("enable_peer_kv_servicer", getattr(cfg, "enable_peer_kv_servicer", False))
            fields["hicache_ratio"] = 4.445 if servicer else 4.0
        if (
            model.get("full_only")
            and not getattr(cfg, "enable_kv_session_retention", False)
            and not getattr(cfg, "disable_radix_cache", False)
            and getattr(cfg, "radix_eviction_policy", "lru") == "lru"
        ):
            fields["enable_kv_session_retention"] = True
    return fields


def model_info(server_args: Any) -> Optional[Dict[str, Any]]:
    """The host dedup flag and FULL-only tree of the configured model, or None."""
    try:
        from decoupled_engine.srt.arg_groups.overrides import model_config_of
        from decoupled_engine.srt.configs.model_config import (
            AttentionArch,
            is_deepseek_dsa,
            is_deepseek_v4,
        )

        config = model_config_of(server_args)
        hf = config.hf_config
        v4 = is_deepseek_v4(hf)
        if v4:
            dedup = "enable_dsv4_hicache_host_dedup"
        elif is_deepseek_dsa(hf):
            dedup = "enable_dsa_hicache_host_dedup"
        elif getattr(config, "attention_arch", None) == AttentionArch.MLA:
            dedup = "enable_mla_hicache_host_dedup"
        else:
            dedup = None
        from decoupled_engine.srt.mem_cache.kv_cache_builder import uses_ssm_state

        full_only = not (v4 or getattr(config, "is_hybrid_swa", False) or uses_ssm_state(config))
        return {"dedup": dedup, "full_only": full_only}
    except Exception as e:  # no model config at resolution (tests, dummy paths)
        logger.info("Symmetric defaults: model not inspected (%s)", e)
        return None


def apply_symmetric_defaults(server_args: Any) -> None:
    """Set the defaults that ``plan_features`` and ``plan`` return."""
    from decoupled_engine.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    view = resolving_view(server_args)
    features: Dict[str, Any] = {}
    if view.disaggregation_mode == "symmetric" and getattr(view, "symmetric_preset", "paper") == "paper":
        features = plan_features(
            view,
            model_info(server_args),
            ratio_unset=getattr(server_args, "hicache_ratio", None) is None,
        )
    if features:
        declare_resolution(server_args, "_apply_symmetric_features", **features)
        logger.info("Symmetric features: %s", features)
    env, fields = plan(_Overlay(view, features), os.environ)
    if fields:
        declare_resolution(server_args, "_apply_symmetric_defaults", **fields)
    if not env:
        return
    applied = [n for n in os.environ.get(APPLIED_ENV, "").split(",") if n]
    for name, value in env.items():
        os.environ[name] = value
        if name not in applied:
            applied.append(name)
    os.environ[APPLIED_ENV] = ",".join(applied)
    logger.info(
        "Symmetric defaults: %s%s",
        " ".join(f"{k}={v}" for k, v in sorted(env.items())),
        f"; {fields}" if fields else "",
    )


def refuse(name: str, message: str) -> None:
    """Fail startup for a switch the user set; turn off one resolution set."""
    if not auto_applied(name):
        raise ValueError(message)
    os.environ[name] = "0"
    logger.info("%s off: %s", name, message)


# Host dedup fan-out tuning of the paper's GB300 runs, applied with the
# ce_peer fan-out that resolution chose (see :func:`apply_gb300_dedup_tuning`).
GB300_DEDUP_TUNING = {
    "SGLANG_DSA_HOST_DEDUP_LAYER_PARTS": "8",
    "SGLANG_DSA_HOST_DEDUP_PART_PLAN": "hybrid",
    "SGLANG_DSA_HOST_DEDUP_STREAM_REDRAW": "1",
}


def apply_gb300_dedup_tuning(fanout_env: str) -> None:
    """On GB300, with the ce_peer fan-out that resolution set in
    ``fanout_env``, set the tuning variables the user left unset. Called by
    each rank when it builds its host dedup, so every rank decides alike."""
    if not auto_applied(fanout_env):
        return
    try:
        from decoupled_engine.srt.mem_cache.pool_host.numa_split_policy import (
            _is_gb300,
            probe_platform,
        )

        if not _is_gb300(probe_platform()):
            return
    except Exception as e:
        logger.debug("GB300 dedup tuning skipped: %s", e)
        return
    set_now = {k: v for k, v in GB300_DEDUP_TUNING.items() if k not in os.environ}
    os.environ.update(set_now)
    if set_now:
        logger.info("GB300 host dedup tuning: %s", set_now)

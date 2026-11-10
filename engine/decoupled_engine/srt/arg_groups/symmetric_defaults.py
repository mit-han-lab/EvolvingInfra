# SPDX-License-Identifier: Apache-2.0
"""Environment defaults that follow from the symmetric P/D server arguments.

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
- symmetric: non-blocking host-to-GPU copies at D admission; with the peer
  KV servicer, range pushes to its index; with HiCache on plain TP, the
  pipelined and idle-skipping per-step HiCache sync.
- ``--enable-prefill-work-snapshot``: the prefill routing snapshot that the
  router's cost function reads, sized for long queues.
- symmetric with ``--enable-dsa-hicache-host-dedup`` or
  ``--enable-mla-hicache-host-dedup``: the copy-engine fan-out (``ce_peer``)
  and the direct host layout it needs. ``SGLANG_DSA_HOST_DEDUP_FANOUT=nccl``
  (or the MLA word) keeps the NCCL broadcast and the kernel layout.

A switch that resolution turned on and that a later startup check rejects for
this model or loop is turned off again with a log line (:func:`refuse`). A
switch that the user set still fails startup.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Tuple

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

    if symmetric:
        # Host-tier and transfer paths: the same work in fewer blocking syncs.
        default("SGLANG_PREBUILT_NONBLOCKING_H2D", "1")
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

    if symmetric:
        layout = (cfg.hicache_io_backend, cfg.hicache_mem_layout)
        for flag, name in _DEDUP_FANOUTS:
            if getattr(cfg, flag, False) and not _user_set(name, environ) and layout in _DEDUP_LAYOUTS:
                env[name] = "ce_peer"
                fields.update(hicache_io_backend="direct", hicache_mem_layout="page_first_direct")
    return env, fields


def apply_symmetric_defaults(server_args: Any) -> None:
    """Set the defaults that ``plan`` returns for these server arguments."""
    from decoupled_engine.srt.arg_groups.overrides import (
        declare_resolution,
        resolving_view,
    )

    env, fields = plan(resolving_view(server_args), os.environ)
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

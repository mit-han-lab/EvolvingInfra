"""Validated defaults of DSv4 window SWA residency.

Under ``--dsv4-uniform-cache-lifecycle --dsv4-uniform-swa-residency window``
the engine defaults to the configuration measured on 1P1D 8xH100 (standard
and symmetric P/D, DeepSeek-V4-Flash, concurrency 16 and 32). Other
residencies (full, N, F) keep every stock default.

Each default holds only when its env is unset (an explicit value, including
"0", always wins) and only for the roles it was validated on:

* env knobs, applied once in the launcher process before the schedulers
  start (``apply_window_env_defaults``): the HiCache sync pipelining and
  idle-sync skip, the Mooncake worker and SWA backfill IO nice values, the
  SWA host low watermark, chunk-boundary backups after the launch (prefill)
  and export spills after the launch (symmetric);
* the tail-only SWA and c4-state host layouts, decided at cache build where
  the worker's storage backend, servicer and pool shapes are known
  (hybrid_pool_assembler._dsv4_window_host_tail);
* ``--dsv4-swa-backfill auto`` resolving to off without a HiCache storage
  backend (kv_cache_hook), where no backfill can run.

The SWA spill copy path follows the spill role when
SGLANG_DSV4_SWA_SPILL_SM_COPY is unset (unified_radix_cache.swa_spill_sm_copy),
which also matches the measured configuration. The host tier is sized by
--dsv4-hicache-host-tokens (literal) or --dsv4-hicache-host-gb (bytes,
converted at the active layout's bytes per page).

``format_window_config_line`` renders the effective configuration as one
``DSV4_WINDOW_CONFIG`` line, logged once by TP rank 0 at cache build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

# Comma-separated NAME=VALUE of the envs the launcher set as window defaults
# (read back by the startup config line to tell defaults from explicit values).
DEFAULTED_ENV = "SGLANG_DSV4_WINDOW_DEFAULTED"


@dataclass(frozen=True)
class WindowEnvDefault:
    name: str
    value: str
    # Disaggregation modes it applies to; None: every mode.
    roles: Optional[tuple] = None


WINDOW_ENV_DEFAULTS = (
    WindowEnvDefault("SGLANG_HICACHE_PIPELINED_SYNC", "1"),
    WindowEnvDefault("SGLANG_HICACHE_SKIP_IDLE_SYNC", "1"),
    WindowEnvDefault("SGLANG_MOONCAKE_WORKER_NICE", "19"),
    WindowEnvDefault("SGLANG_DSV4_SWA_BACKFILL_IO_NICE", "19"),
    WindowEnvDefault("SGLANG_DSV4_SWA_HOST_LOW_WATERMARK", "0.01"),
    # Validated on the standard prefill loop only.
    WindowEnvDefault("SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH", "1", ("prefill",)),
    # Export spills exist on symmetric owners only.
    WindowEnvDefault("SGLANG_DSV4_EXPORT_SPILL_AFTER_LAUNCH", "1", ("symmetric",)),
)

# Effective values shown on the config line, in order.
_CONFIG_LINE_ENVS = (
    ("pipelined_sync", "SGLANG_HICACHE_PIPELINED_SYNC"),
    ("skip_idle_sync", "SGLANG_HICACHE_SKIP_IDLE_SYNC"),
    ("mooncake_nice", "SGLANG_MOONCAKE_WORKER_NICE"),
    ("backfill_io_nice", "SGLANG_DSV4_SWA_BACKFILL_IO_NICE"),
    ("low_watermark", "SGLANG_DSV4_SWA_HOST_LOW_WATERMARK"),
    ("chunk_work_after_launch", "SGLANG_DSV4_CHUNK_WINDOW_WORK_AFTER_LAUNCH"),
    ("export_spill_after_launch", "SGLANG_DSV4_EXPORT_SPILL_AFTER_LAUNCH"),
    ("publish_spill_swa", "SGLANG_DSV4_PEER_PUBLISH_SPILL_SWA"),
)


def is_window_mode(cfg: Any) -> bool:
    return bool(getattr(cfg, "dsv4_uniform_cache_lifecycle", False)) and (
        getattr(cfg, "dsv4_uniform_swa_residency", "full") == "window"
    )


def apply_window_env_defaults(
    cfg: Any, environ: Optional[MutableMapping[str, str]] = None
) -> list:
    """Set the window-mode env defaults that are unset; return their names.

    Called by the launcher (entrypoints/engine.py _set_envs_and_config)
    before the schedulers start, so every process reads them. Outside window
    mode it sets nothing."""
    environ = os.environ if environ is None else environ
    # Undo an earlier launch's defaults in this process (where the value is
    # still the one it set), so they neither pass for explicit settings nor
    # leak into a later engine that is not in window mode.
    for name, value in _defaulted(environ).items():
        if environ.get(name) == value:
            del environ[name]
    environ.pop(DEFAULTED_ENV, None)
    if not is_window_mode(cfg):
        return []
    mode = getattr(cfg, "disaggregation_mode", None) or "null"
    mode = getattr(mode, "value", mode)
    digest = environ.get("SGLANG_HICACHE_TP_DIGEST_MODE", "").strip().lower()
    applied = []
    for default in WINDOW_ENV_DEFAULTS:
        if default.roles is not None and mode not in default.roles:
            continue
        if default.name in environ:
            continue
        if default.name == "SGLANG_HICACHE_SKIP_IDLE_SYNC" and digest not in (
            "",
            "off",
        ):
            # SKIP_IDLE_SYNC requires the TP digest off.
            continue
        environ[default.name] = default.value
        applied.append(default.name)
    environ[DEFAULTED_ENV] = ",".join(f"{name}={environ[name]}" for name in applied)
    return applied


def _defaulted(environ: MutableMapping[str, str]) -> dict:
    """The NAME -> VALUE pairs DEFAULTED_ENV records."""
    pairs = {}
    for item in filter(None, environ.get(DEFAULTED_ENV, "").split(",")):
        name, _, value = item.partition("=")
        pairs[name] = value
    return pairs


def _provenance(name: str, environ: MutableMapping[str, str]) -> str:
    if name in _defaulted(environ):
        return "default"
    return "env" if name in environ else "stock"


def format_window_config_line(
    *,
    mode: str,
    host_tail: Any,
    host_sizing: str,
    host_tokens: int,
    num_host_pages: int,
    bytes_per_page: int,
    swa_backfill: str,
    decode_spill_sm_copy: bool,
    export_spill_sm_copy: bool,
    environ: Optional[MutableMapping[str, str]] = None,
) -> str:
    """One grep-able ``DSV4_WINDOW_CONFIG`` line with the effective window
    configuration: each value and where it came from (default = a window
    default, env = set explicitly, stock = the code default, role = chosen
    by the engine's disaggregation role, fallback = a default this worker
    cannot use). ``host_sizing``: "tokens" or "gb", the host size flag."""
    from decoupled_engine.srt.environ import envs

    environ = os.environ if environ is None else environ
    sources = getattr(host_tail, "sources", None) or {}
    tail = getattr(host_tail, "swa_tail", None) is not None
    state = bool(getattr(host_tail, "state_tail", False))
    spill_source = "env" if envs.SGLANG_DSV4_SWA_SPILL_SM_COPY.is_set() else "role"
    parts = [
        f"mode={mode}",
        f"tail={int(tail)}({sources.get('tail', 'off')})",
        f"state_tail={int(state)}({sources.get('state_tail', 'off')})",
        f"host_sizing={host_sizing}",
        f"host_tokens={host_tokens}",
        f"host_pages={num_host_pages}",
        f"bytes_per_page={bytes_per_page}",
        f"host_gb_per_rank={num_host_pages * bytes_per_page / 1e9:.2f}",
        f"swa_backfill={swa_backfill}",
        f"decode_spill_sm_copy={int(decode_spill_sm_copy)}({spill_source})",
        f"export_spill_sm_copy={int(export_spill_sm_copy)}({spill_source})",
    ]
    for label, name in _CONFIG_LINE_ENVS:
        value = getattr(envs, name).get()
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{label}={value}({_provenance(name, environ)})")
    return "DSV4_WINDOW_CONFIG " + " ".join(parts)

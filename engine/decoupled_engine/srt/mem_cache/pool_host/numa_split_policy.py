"""Whether the host-dedup source pool uses NUMA-split placement (numa_split.py).

``SGLANG_DEDUP_POOL_NUMA_SPLIT``:

- ``1``: on. A NUMA node that cannot hold its share refuses to start.
  ``SGLANG_DEDUP_POOL_LOCAL_FRACTION`` unset
  means max-local. The mapping follows ``SGLANG_DEDUP_POOL_MAP`` (default
  shared) on every platform.
- ``0``: off.
- unset: auto. On only for a host-dedup source pool on a platform listed in
  ``AUTO_PLATFORMS`` whose storage backend uses the default host allocator
  (not mooncake, mori or shm), and only for the host-dedup kinds that platform
  lists. The owner node share is ``SGLANG_DEDUP_POOL_LOCAL_FRACTION`` or, when
  unset, the platform's validated fraction. If the split cannot be placed (a
  node cannot hold its share, no 2 memory nodes, a failed pre-fault, ...) the
  pool falls back to the default allocation with one warning that names the
  node and the shortfall, instead of refusing to start. Auto never refuses to
  start for a placement reason; only an explicit ``1`` does.

``AUTO_PLATFORMS``:

- GB300, any host-dedup kind: owner share 0.5; the mapping follows
  ``SGLANG_DEDUP_POOL_MAP`` (default shared).
- H100, DeepSeek-V4 host dedup only: owner share 0.5 and a private anonymous
  mapping (what ``SGLANG_DEDUP_POOL_MAP=private`` selects, with
  ``MADV_DONTFORK``); THP stays off unless ``SGLANG_DEDUP_POOL_THP=1``.
  Compaction tries to migrate pinned shmem pages, and each futile migration
  unmaps and remaps the page in TP0's page tables; compaction skips pinned
  private anonymous pages (numa_split.py). An explicit
  ``SGLANG_DEDUP_POOL_MAP=shared`` or ``SGLANG_DEDUP_POOL_NUMA_SPLIT=0``
  turns the H100 policy off: the previous H100 path (the default shmem
  allocation under the process NUMA policy).

GB300 detection needs all three: an aarch64 host, an NVIDIA Grace CPU (MIDR
implementer 0x41 / part 0xd4f, Arm Neoverse V2, from /proc/cpuinfo), and a
Blackwell Ultra GPU (device name containing "GB300", or compute capability
10.3). H100 detection needs an x86_64 host and a GPU whose device name
contains "H100" (H200, H800 and GH200 do not match). Other hosts keep the
default (off). To cover another platform, add an ``AutoPlatform`` to
``AUTO_PLATFORMS``.
"""

from __future__ import annotations

import dataclasses
import logging
import platform
from typing import Callable, Optional, Tuple

from decoupled_engine.srt.environ import envs

logger = logging.getLogger(__name__)

# Storage backends with their own host allocator, which the split cannot
# replace; every other backend (e.g. peer_kv, file) uses the default allocator.
UNSUPPORTED_ALLOCATOR_TYPES = ("mooncake", "mori", "shm")
_GRACE_MIDR = ("0x41", "0xd4f")  # Arm Ltd, Neoverse V2 (NVIDIA Grace)


@dataclasses.dataclass(frozen=True)
class PlatformProbe:
    arch: str
    cpu_midrs: Tuple[Tuple[str, str], ...]  # distinct (implementer, part)
    gpu_name: str
    gpu_capability: Optional[Tuple[int, int]]

    @property
    def grace_cpu(self) -> bool:
        return bool(self.cpu_midrs) and all(m == _GRACE_MIDR for m in self.cpu_midrs)

    def describe(self) -> str:
        cc = (
            f"{self.gpu_capability[0]}.{self.gpu_capability[1]}"
            if self.gpu_capability
            else "none"
        )
        cpu = "Grace" if self.grace_cpu else (
            ",".join(map(":".join, self.cpu_midrs)) or "unknown"
        )
        return f"arch={self.arch} cpu={cpu} gpu={self.gpu_name or 'none'} cc={cc}"


@dataclasses.dataclass(frozen=True)
class AutoPlatform:
    name: str
    matches: Callable[[PlatformProbe], bool]
    # Owner-node share when SGLANG_DEDUP_POOL_LOCAL_FRACTION is unset.
    local_fraction: float
    # Host-dedup kinds (the dedup object's ``dedup_kind``: "dsv4", "dsa") that
    # the auto policy covers; None = every kind.
    dedup_kinds: Optional[Tuple[str, ...]] = None
    # Pool mapping when SGLANG_DEDUP_POOL_MAP is unset; None = that variable's
    # own default (shared). With a value, an explicit different
    # SGLANG_DEDUP_POOL_MAP turns the auto policy off on this platform.
    map_mode: Optional[str] = None


def _is_gb300(p: PlatformProbe) -> bool:
    blackwell_ultra = "GB300" in p.gpu_name.upper() or p.gpu_capability == (10, 3)
    return p.arch == "aarch64" and p.grace_cpu and blackwell_ultra


def _is_h100(p: PlatformProbe) -> bool:
    return p.arch == "x86_64" and "H100" in p.gpu_name.upper()


AUTO_PLATFORMS: Tuple[AutoPlatform, ...] = (
    # Validated on GB300 host-dedup runs at fraction 0.5.
    AutoPlatform("GB300", _is_gb300, local_fraction=0.5),
    # H100 DeepSeek-V4 dedup: the slow-P mode of TP0's pinned shmem pool
    # (futile compaction migrations). Settings validated on H100: split 0.5,
    # private mapping, THP off.
    AutoPlatform(
        "H100",
        _is_h100,
        local_fraction=0.5,
        dedup_kinds=("dsv4",),
        map_mode="private",
    ),
)


@dataclasses.dataclass(frozen=True)
class NumaSplitDecision:
    enabled: bool
    source: str  # "env" or "auto"
    reason: str
    # Refuse to start when the split cannot be placed (explicit =1); auto falls
    # back to the default allocation instead.
    strict: bool = True
    # Owner-node share when SGLANG_DEDUP_POOL_LOCAL_FRACTION is unset; None =
    # max-local.
    default_local_fraction: Optional[float] = None
    # Pool mapping when SGLANG_DEDUP_POOL_MAP is unset; None = shared.
    map_mode: Optional[str] = None


def _cpu_midrs() -> Tuple[Tuple[str, str], ...]:
    implementer, midrs = None, set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip().lower()
                if key == "CPU implementer":
                    implementer = value
                elif key == "CPU part" and implementer is not None:
                    midrs.add((implementer, value))
    except OSError:
        pass
    return tuple(sorted(midrs))


def probe_platform() -> PlatformProbe:
    gpu_name, capability = "", None
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device) or ""
            capability = tuple(torch.cuda.get_device_capability(device))
    except Exception as e:  # no driver, or CUDA not initialised
        logger.debug("Dedup pool NUMA split: GPU probe failed: %s", e)
    return PlatformProbe(platform.machine(), _cpu_midrs(), gpu_name, capability)


def resolve_numa_split(
    env_value: Optional[bool],
    allocator_type: str,
    hugepage_size: str,
    probe: Optional[Callable[[], PlatformProbe]] = None,
    *,
    dedup_kind: Optional[str] = None,
    map_env: Optional[str] = None,
) -> NumaSplitDecision:
    """Pure resolution for a host-dedup source pool; see the module docstring.

    ``dedup_kind``: the source dedup object's ``dedup_kind`` (None if it has
    none). ``map_env``: SGLANG_DEDUP_POOL_MAP when set and not empty
    (stripped, lower case), else None.
    """
    if env_value is not None:
        return NumaSplitDecision(
            enabled=env_value,
            source="env",
            reason=f"SGLANG_DEDUP_POOL_NUMA_SPLIT={int(env_value)}",
        )
    if allocator_type in UNSUPPORTED_ALLOCATOR_TYPES:
        return NumaSplitDecision(False, "auto", f"host allocator {allocator_type!r}")
    if (hugepage_size or "").strip():
        return NumaSplitDecision(False, "auto", "SGLANG_HUGEPAGE_SIZE is set")
    p = (probe or probe_platform)()
    for auto in AUTO_PLATFORMS:
        if not auto.matches(p):
            continue
        if auto.dedup_kinds is not None and dedup_kind not in auto.dedup_kinds:
            return NumaSplitDecision(
                False,
                "auto",
                f"{auto.name} auto covers {'/'.join(auto.dedup_kinds)} host dedup "
                f"only, this pool's is {dedup_kind or 'unknown'} ({p.describe()})",
            )
        if auto.map_mode is not None and map_env and map_env != auto.map_mode:
            return NumaSplitDecision(
                False,
                "auto",
                f"{auto.name} auto needs map {auto.map_mode}, "
                f"SGLANG_DEDUP_POOL_MAP={map_env} keeps the default allocation "
                f"({p.describe()})",
            )
        return NumaSplitDecision(
            enabled=True,
            source="auto",
            reason=f"{auto.name} ({p.describe()})",
            strict=False,
            default_local_fraction=auto.local_fraction,
            map_mode=auto.map_mode,
        )
    return NumaSplitDecision(
        False, "auto", f"no auto platform matches ({p.describe()})"
    )


def explicit_pool_map() -> Optional[str]:
    """SGLANG_DEDUP_POOL_MAP when set and not empty (stripped, lower case)."""
    if not envs.SGLANG_DEDUP_POOL_MAP.is_set():
        return None
    return (envs.SGLANG_DEDUP_POOL_MAP.get() or "").strip().lower() or None


def resolve_dedup_pool_numa_split(
    allocator_type: str, dedup_kind: Optional[str] = None
) -> NumaSplitDecision:
    """Resolve from the environment for a host-dedup source pool and log it."""
    map_env = explicit_pool_map()
    decision = resolve_numa_split(
        envs.SGLANG_DEDUP_POOL_NUMA_SPLIT.get(),
        allocator_type,
        envs.SGLANG_HUGEPAGE_SIZE.get(),
        dedup_kind=dedup_kind,
        map_env=map_env,
    )
    if decision.enabled:
        fraction = envs.SGLANG_DEDUP_POOL_LOCAL_FRACTION.get()
        if fraction is None:
            fraction = decision.default_local_fraction
        mapping = ""
        if decision.map_mode is not None:
            source = "SGLANG_DEDUP_POOL_MAP" if map_env else "auto"
            mapping = f"; pool mapping {map_env or decision.map_mode} ({source})"
        logger.info(
            "Dedup pool NUMA split: on (source=%s: %s); owner share %s; %s%s",
            decision.source,
            decision.reason,
            "max-local" if fraction is None else fraction,
            "refuse to start if it cannot be placed"
            if decision.strict
            else "fall back to the default allocation if it cannot be placed",
            mapping,
        )
    else:
        logger.info(
            "Dedup pool NUMA split: off (source=%s: %s)",
            decision.source,
            decision.reason,
        )
    return decision

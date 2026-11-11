"""NUMA placement of the host-dedup source pool (SGLANG_DEDUP_POOL_NUMA_SPLIT).

Whether it is used is decided in numa_split_policy.py: on with =1, off with =0,
and by default on GB300-class hosts and for DeepSeek-V4 host dedup on H100
hosts, where a split that cannot be placed falls back to the default
allocation instead of refusing to start.

With host dedup, TP0 owns one host pool for the whole node and is the only
rank that reads or writes it (followers are allocator-only). Without explicit
placement its pages follow the process policy (``numactl --interleave=all``),
which silently falls back to the other node when one node is short of free
pages, so the split varies per start.

This allocator maps the pool without populating it, binds fixed-size stripes
to the owner (GPU-local) node and the other node with ``mbind(MPOL_BIND)``,
then pre-faults it. A bound shared-anonymous (shmem) range cannot fall back:
if a node cannot hold its part, an explicit =1 fails startup here instead of
skewing. The owner-node share is ``SGLANG_DEDUP_POOL_LOCAL_FRACTION``; when
unset, the auto default's platform fraction (0.5 on GB300 and H100), or with =1 as
much as fits while each node keeps ``SGLANG_DEDUP_POOL_NUMA_RESERVE_GB``
available. Stripes alternate in that ratio, so any subset of host pages (the
slot allocator hands out low page indices first) has the same split.

Mapping kind (default shared; private is opt-in, and the H100 auto default):

* ``SGLANG_DEDUP_POOL_MAP=shared`` (default): shared anonymous (shmem).
* ``SGLANG_DEDUP_POOL_MAP=private``: private anonymous + ``MADV_DONTFORK``.
  With the variable unset, the H100 auto policy for DeepSeek-V4 host dedup
  selects it too (numa_split_policy.py).
  The pool is long-term pinned by the transfer engine's memory registration.
  Compaction does not isolate pinned anonymous pages, but it does try to
  migrate pinned shmem pages; each such futile migration unmaps and remaps the
  page, which invalidates the GPU's ATS translations (GB300 slow load-back)
  and, on H100, costs TP0 CPU time in its page tables (slow-P mode).
  ``MADV_DONTFORK`` keeps forked children from inheriting the range (a fork
  would otherwise have to copy the pinned private pages).
* ``SGLANG_DEDUP_POOL_THP=1`` (needs ``private``): ``MADV_HUGEPAGE`` before
  pre-faulting, so PMD-size THPs back the pool where free blocks allow. The
  mapping is rounded up to a PMD multiple so the kernel aligns it.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import math
import mmap
import os
import platform
import time

import torch

from decoupled_engine.srt.environ import envs
from decoupled_engine.srt.mem_cache.pool_host.common import HostTensorAllocator
from decoupled_engine.srt.mem_cache.storage.mmap import mmap_allocator as _mm

logger = logging.getLogger(__name__)

STRIPE_BYTES = 1 << 30
_GiB = float(1 << 30)
_MPOL_BIND = 2
_MPOL_F_ADDR = 2
_MPOL_NAMES = {
    0: "default",
    1: "preferred",
    2: "bind",
    3: "interleave",
    4: "local",
    5: "preferred_many",
    6: "weighted_interleave",
}
# (mbind, get_mempolicy) syscall numbers; glibc has no mbind wrapper.
_SYSCALLS = {"x86_64": (237, 239), "aarch64": (235, 236)}
_NODE_SYS = "/sys/devices/system/node"
_page_cache_dropped = False
_THP_PMD_SIZE = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size"


def pool_map_mode(default: str = "shared") -> tuple[str, bool]:
    """(SGLANG_DEDUP_POOL_MAP, SGLANG_DEDUP_POOL_THP), validated. ``default`` is
    the mapping when SGLANG_DEDUP_POOL_MAP is unset or empty: shared, or the
    auto platform's mapping (private for H100 DeepSeek-V4 host dedup)."""
    raw = (
        envs.SGLANG_DEDUP_POOL_MAP.get()
        if envs.SGLANG_DEDUP_POOL_MAP.is_set()
        else None
    )
    mode = ((raw or "").strip() or default).strip().lower()
    thp = bool(envs.SGLANG_DEDUP_POOL_THP.get())
    if mode not in ("shared", "private"):
        raise ValueError(
            f"SGLANG_DEDUP_POOL_MAP={mode!r}: expected 'shared' or 'private'"
        )
    if thp and mode != "private":
        raise ValueError("SGLANG_DEDUP_POOL_THP=1 needs SGLANG_DEDUP_POOL_MAP=private")
    return mode, thp


def pmd_size() -> int:
    with open(_THP_PMD_SIZE) as f:
        return int(f.read())


def parse_smaps_range(text: str, start: int, end: int) -> dict[str, int]:
    """Summed smaps fields (bytes) of VMAs starting in [start, end), plus 'vmas'."""
    out, keep = {"vmas": 0}, False
    for line in text.splitlines():
        head = line.split(None, 1)[0] if line.strip() else ""
        if "-" in head and not head.endswith(":"):
            lo = int(head.split("-", 1)[0], 16)
            keep = start <= lo < end
            out["vmas"] += keep
        elif keep and head.endswith(":") and line.rstrip().endswith(" kB"):
            out[head[:-1]] = out.get(head[:-1], 0) + int(line.split()[1]) * 1024
    return out


def parse_nodelist(text: str) -> list[int]:
    nodes = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            nodes.extend(range(int(lo), int(hi or lo) + 1))
    return nodes


def node_meminfo() -> dict[int, dict[str, int]]:
    """Per-node meminfo in bytes, e.g. info[0]["MemFree"]."""
    info = {}
    for path in glob.glob(f"{_NODE_SYS}/node[0-9]*/meminfo"):
        node = int(path.split("/")[-2][4:])
        fields = {}
        with open(path) as f:
            for line in f:
                # "Node 0 MemFree:   123456 kB"
                parts = line.split()
                if len(parts) >= 4:
                    fields[parts[2].rstrip(":")] = int(parts[3]) * 1024
        info[node] = fields
    return info


def available_bytes(fields: dict[str, int]) -> int:
    """Free plus file-backed LRU pages, which a bound fault may reclaim."""
    return (
        fields.get("MemFree", 0)
        + fields.get("Active(file)", 0)
        + fields.get("Inactive(file)", 0)
    )


def _fmt_nodes(values: dict[int, int]) -> str:
    return (
        "{" + ", ".join(f"{n}: {v / _GiB:.1f}" for n, v in sorted(values.items())) + "}"
    )


def _read_nodelist(name: str) -> list[int]:
    with open(f"{_NODE_SYS}/{name}") as f:
        return parse_nodelist(f.read())


def memory_nodes() -> list[int]:
    """CPU NUMA nodes with memory that this process may allocate from."""

    nodes = set(_read_nodelist("has_memory")) & set(_read_nodelist("has_cpu"))
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Mems_allowed_list:"):
                nodes &= set(parse_nodelist(line.split(":", 1)[1]))
    return sorted(nodes)


def owner_node(nodes: list[int]) -> int:
    """NUMA node local to this rank's GPU, else to its CPU affinity."""
    try:
        from decoupled_engine.srt.utils.numa_utils import _query_numa_node_for_gpu

        gpu_nodes = [
            n
            for n in _query_numa_node_for_gpu(torch.cuda.current_device())
            if n in nodes
        ]
        if len(gpu_nodes) == 1:
            return gpu_nodes[0]
    except Exception as e:
        logger.debug("NVML NUMA query failed: %s", e)
    cpus = os.sched_getaffinity(0)
    cpu_nodes = [n for n in nodes if cpus & set(_read_nodelist(f"node{n}/cpulist"))]
    if len(cpu_nodes) != 1:
        raise RuntimeError(
            "SGLANG_DEDUP_POOL_NUMA_SPLIT: cannot determine the dedup owner's NUMA "
            f"node (GPU affinity unknown, CPU affinity spans nodes {cpu_nodes})."
        )
    return cpu_nodes[0]


def drop_checkpoint_page_cache() -> None:
    """Best effort: evict the model checkpoint's clean page cache, once."""
    global _page_cache_dropped
    if _page_cache_dropped:
        return
    _page_cache_dropped = True
    try:
        from decoupled_engine.srt.runtime_context import get_server_args

        args = get_server_args()
    except Exception as e:
        logger.warning("Dedup pool NUMA split: no server args, page cache kept: %s", e)
        return
    files, start = [], time.monotonic()
    for path in (args.model_path, getattr(args, "speculative_draft_model_path", None)):
        if not path:
            continue
        if not os.path.isdir(path):
            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(path, local_files_only=True)
            except Exception:
                continue
        for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
            files.extend(glob.glob(os.path.join(path, pattern)))
    before = node_meminfo()
    total = failed = 0
    for name in files:
        try:
            fd = os.open(name, os.O_RDONLY)
            try:
                total += os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            failed += 1
    after = node_meminfo()
    logger.info(
        "Dedup pool NUMA split: fadvise(DONTNEED) on %d checkpoint files "
        "(%.1f GiB, %d failed) in %.1f s; node MemFree GiB before %s after %s",
        len(files),
        total / _GiB,
        failed,
        time.monotonic() - start,
        _fmt_nodes({n: v.get("MemFree", 0) for n, v in before.items()}),
        _fmt_nodes({n: v.get("MemFree", 0) for n, v in after.items()}),
    )


def plan_stripes(
    nbytes: int, local_stripes: int, local: int, remote: int, stripe_bytes: int
) -> list[tuple[int, int, int]]:
    """(offset, length, node) runs; local stripes spread evenly (Bresenham)."""
    n = math.ceil(nbytes / stripe_bytes)
    runs = []
    for i in range(n):
        node = (
            local
            if ((i + 1) * local_stripes) // n > (i * local_stripes) // n
            else remote
        )
        offset = i * stripe_bytes
        length = min(stripe_bytes, nbytes - offset)
        if runs and runs[-1][2] == node:
            runs[-1] = (runs[-1][0], runs[-1][1] + length, node)
        else:
            runs.append((offset, length, node))
    return runs


def _syscall(index: int, *args) -> None:
    numbers = _SYSCALLS.get(platform.machine())
    if numbers is None or _mm._libc is None:
        raise RuntimeError(f"NUMA syscalls unsupported on {platform.machine()}")
    _mm._libc.syscall.restype = ctypes.c_long
    rc = _mm._libc.syscall(ctypes.c_long(numbers[index]), *args)
    if rc != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def mbind(addr: int, length: int, node: int) -> None:
    mask = ctypes.c_ulong(1 << node)
    _syscall(
        0,
        ctypes.c_void_p(addr),
        ctypes.c_ulong(length),
        ctypes.c_long(_MPOL_BIND),
        ctypes.byref(mask),
        ctypes.c_ulong(ctypes.sizeof(mask) * 8 + 1),
        ctypes.c_ulong(0),
    )


def mempolicy_at(addr: int) -> str:
    """get_mempolicy(MPOL_F_ADDR) as e.g. 'bind:0' (no page fault)."""
    mode, mask = ctypes.c_int(0), (ctypes.c_ulong * 16)()
    try:
        _syscall(
            1,
            ctypes.byref(mode),
            mask,
            ctypes.c_ulong(ctypes.sizeof(mask) * 8),
            ctypes.c_void_p(addr),
            ctypes.c_ulong(_MPOL_F_ADDR),
        )
    except (OSError, RuntimeError) as e:
        return f"error:{e}"
    nodes = [i for i in range(len(mask) * 64) if mask[i // 64] >> (i % 64) & 1]
    return f"{_MPOL_NAMES.get(mode.value, mode.value)}:{','.join(map(str, nodes))}"


def parse_numa_maps(text: str, start: int, end: int):
    """Resident bytes per node and VMA policies of mappings starting in [start, end)."""
    per_node, policies, vmas = {}, [], 0
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or not start <= int(tokens[0], 16) < end:
            continue
        vmas += 1
        policy, rest = [], []
        for tok in tokens[1:]:
            (rest if rest or "=" in tok else policy).append(tok)
        policies.append(" ".join(policy))
        kv = dict(t.split("=", 1) for t in rest if "=" in t)
        page = int(kv.get("kernelpagesize_kB", "4")) * 1024
        for key, val in kv.items():
            if key[0] == "N" and key[1:].isdigit():
                node = int(key[1:])
                per_node[node] = per_node.get(node, 0) + int(val) * page
    return per_node, sorted(set(policies)), vmas


class _CannotPlace(Exception):
    """The split cannot be placed: strict mode raises ``error``, the auto
    default logs ``reason`` and falls back."""

    def __init__(self, error: Exception, reason: str | None = None):
        super().__init__(reason or str(error))
        self.error = error


class NumaSplitHostTensorAllocator(HostTensorAllocator):
    """Allocator for the dedup source pool; see the module docstring.

    ``strict`` (explicit =1): refuse to start when the split cannot be placed.
    Otherwise (auto default) log one warning and allocate the old way.
    ``default_local_fraction`` is the owner-node share when
    SGLANG_DEDUP_POOL_LOCAL_FRACTION is unset; None means max-local.
    ``default_map_mode`` is the mapping when SGLANG_DEDUP_POOL_MAP is unset
    (shared; private under the H100 auto policy).
    """

    stripe_bytes = STRIPE_BYTES

    def __init__(
        self,
        strict: bool = True,
        default_local_fraction: float | None = None,
        default_map_mode: str = "shared",
    ):
        super().__init__()
        self.strict = strict
        self.default_local_fraction = default_local_fraction
        self.default_map_mode = default_map_mode
        self.fallback_reason: str | None = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", f"got device={device!r}"
        self.dtype, self.dims = dtype, dims
        self.fallback_reason = None
        try:
            return self._allocate_split(dims, dtype)
        except _CannotPlace as e:
            if self.strict:
                raise e.error from e.__cause__
            self.fallback_reason = str(e)
            self.plan = {}
            logger.warning(
                "Dedup pool NUMA split (source=fallback): %s; allocating %.1f GiB "
                "the default way (no NUMA binding). Set "
                "SGLANG_DEDUP_POOL_NUMA_SPLIT=1 to refuse to start instead, or =0 "
                "to skip the split.",
                e,
                math.prod(dims) * dtype.itemsize / _GiB,
            )
            return super().allocate(dims, dtype, device)

    def _allocate_split(self, dims: tuple, dtype: torch.dtype) -> torch.Tensor:
        count, nbytes, runs, want = self._plan_runs(dims, dtype)
        map_mode, thp = pool_map_mode(self.default_map_mode)
        map_bytes = nbytes
        if map_mode == "shared":
            flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS
        else:
            flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            if thp:
                pmd = pmd_size()
                map_bytes = math.ceil(nbytes / pmd) * pmd
        if (
            map_mode == "shared"
            and type(self)._map is not NumaSplitHostTensorAllocator._map
        ):
            mm = self._map(nbytes)  # subclass mapping (shared memfd pool)
        else:
            mm = mmap.mmap(-1, map_bytes, flags=flags, prot=_mm._PROT_RW)
        tensor = torch.frombuffer(mm, dtype=dtype, count=count).reshape(dims)
        base = tensor.data_ptr()
        if map_mode == "private":
            mm.madvise(mmap.MADV_DONTFORK)
            if thp:
                mm.madvise(mmap.MADV_HUGEPAGE)
            logger.info(
                "Dedup pool NUMA split: map=private (MADV_DONTFORK)%s; base=%#x mapped %.1f GiB",
                (
                    f" + MADV_HUGEPAGE, pmd {pmd >> 20} MiB, base % pmd = {base % pmd}"
                    if thp
                    else ""
                ),
                base,
                map_bytes / _GiB,
            )
        for offset, length, node in runs:
            mbind(base + offset, length, node)
        start = time.monotonic()
        try:
            if map_bytes == nbytes:
                mm.madvise(_mm._MADV_POPULATE_WRITE)
            else:  # leave the PMD round-up tail untouched (not part of the tensor)
                mm.madvise(_mm._MADV_POPULATE_WRITE, 0, nbytes)
        except OSError as e:
            error = RuntimeError(
                "SGLANG_DEDUP_POOL_NUMA_SPLIT: pre-faulting the bound dedup pool "
                f"failed ({e}); node available GiB now "
                f"{_fmt_nodes({k: available_bytes(v) for k, v in node_meminfo().items()})}"
            )
            # Release the partly faulted mapping before any fallback.
            del tensor
            try:
                mm.close()
            except BufferError:
                pass  # still referenced; freed with the last reference
            raise _CannotPlace(error) from e
        self.plan = want
        self.map_mode, self.thp = map_mode, thp
        logger.info(
            "Dedup pool NUMA split: populated %.1f GiB in %.1f s; policy@base=%s",
            nbytes / _GiB,
            time.monotonic() - start,
            mempolicy_at(base),
        )
        if thp:
            self.log_thp(base, nbytes)
        return tensor  # noqa: F821 (the except branch above re-raises after its del)

    def plan_runs(self, dims: tuple, dtype: torch.dtype):
        """(count, nbytes, [(offset, length, node)], {node: bytes}) for the pool;
        raises if a node cannot hold its share (no memory is touched). Strict, as
        an explicit =1: the shared pool's VMM backend plans with it."""
        try:
            return self._plan_runs(dims, dtype)
        except _CannotPlace as e:
            raise e.error from e.__cause__

    def _plan_runs(self, dims: tuple, dtype: torch.dtype):
        """plan_runs; a split that cannot be placed raises ``_CannotPlace``."""
        self.dtype, self.dims = dtype, dims
        if (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip():
            raise _CannotPlace(
                ValueError(
                    "SGLANG_DEDUP_POOL_NUMA_SPLIT does not support SGLANG_HUGEPAGE_SIZE"
                )
            )
        if not _mm._has_madv_populate_write():
            raise _CannotPlace(
                RuntimeError("SGLANG_DEDUP_POOL_NUMA_SPLIT needs MADV_POPULATE_WRITE")
            )
        map_mode, thp = pool_map_mode(self.default_map_mode)
        try:
            nodes = memory_nodes()
        except OSError as e:
            raise _CannotPlace(e) from None
        if len(nodes) != 2:
            raise _CannotPlace(
                RuntimeError(
                    f"SGLANG_DEDUP_POOL_NUMA_SPLIT expects 2 CPU memory nodes, found {nodes}"
                )
            )
        drop_checkpoint_page_cache()
        try:
            local = owner_node(nodes)
        except RuntimeError as e:
            raise _CannotPlace(e) from None
        remote = nodes[1 - nodes.index(local)]

        count = math.prod(dims)
        nbytes = math.ceil(count * dtype.itemsize / mmap.PAGESIZE) * mmap.PAGESIZE
        stripe = self.stripe_bytes
        n = math.ceil(nbytes / stripe)
        reserve = int(envs.SGLANG_DEDUP_POOL_NUMA_RESERVE_GB.get() * _GiB)
        info = node_meminfo()
        room = {node: available_bytes(info[node]) - reserve for node in nodes}
        fraction = envs.SGLANG_DEDUP_POOL_LOCAL_FRACTION.get()
        if fraction is None:
            fraction = self.default_local_fraction
        if fraction is None:
            mode = "max-local"
            k = n if room[local] >= nbytes else max(room[local], 0) // stripe
        else:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"SGLANG_DEDUP_POOL_LOCAL_FRACTION={fraction} not in [0, 1]"
                )
            mode = f"fraction={fraction}"
            k = round(fraction * n)
        runs = plan_stripes(nbytes, k, local, remote, stripe)
        want = {node: 0 for node in nodes}
        for _, length, node in runs:
            want[node] += length
        logger.info(
            "Dedup pool NUMA split: plan %.1f GiB dims=%s mode=%s owner_node=%d "
            "-> node %d %.1f GiB (%.1f%%), node %d %.1f GiB; %d stripes of %d MiB "
            "in %d runs; available GiB %s, reserve %.1f GiB/node",
            nbytes / _GiB,
            tuple(dims),
            mode,
            local,
            local,
            want[local] / _GiB,
            100.0 * want[local] / nbytes,
            remote,
            want[remote] / _GiB,
            n,
            stripe >> 20,
            len(runs),
            _fmt_nodes({node: available_bytes(info[node]) for node in nodes}),
            reserve / _GiB,
        )
        for node in nodes:
            if want[node] > room[node]:
                short = (want[node] - room[node]) / _GiB
                raise _CannotPlace(
                    RuntimeError(
                        f"SGLANG_DEDUP_POOL_NUMA_SPLIT: NUMA node {node} cannot hold its "
                        f"{want[node] / _GiB:.1f} GiB of the dedup host pool (short by "
                        f"{short:.1f} GiB): available "
                        f"(MemFree + file LRU) {available_bytes(info[node]) / _GiB:.1f} GiB, "
                        f"reserve {reserve / _GiB:.1f} GiB. Refusing to start instead of "
                        "skewing; adjust SGLANG_DEDUP_POOL_LOCAL_FRACTION / "
                        "SGLANG_DEDUP_POOL_NUMA_RESERVE_GB, lower --hicache-ratio, or "
                        "set SGLANG_DEDUP_POOL_NUMA_SPLIT=0."
                    ),
                    reason=(
                        f"NUMA node {node} cannot hold its {want[node] / _GiB:.1f} GiB "
                        f"share, short by {short:.1f} GiB (available "
                        f"{available_bytes(info[node]) / _GiB:.1f} GiB, "
                        f"reserve {reserve / _GiB:.1f} GiB)"
                    ),
                )
        return count, nbytes, runs, want

    def _map(self, nbytes: int):
        """The mapping the stripe plan is applied to (shared memfd: shared_dedup)."""
        return mmap.mmap(
            -1, nbytes, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, prot=_mm._PROT_RW
        )

    def log_thp(self, base: int, nbytes: int) -> None:
        """THP coverage of the pool from /proc/self/smaps (THP mode only)."""
        start = time.monotonic()
        try:
            with open("/proc/self/smaps") as f:
                info = parse_smaps_range(f.read(), base, base + nbytes)
        except OSError as e:
            logger.warning("Dedup pool NUMA split: smaps unreadable: %s", e)
            return
        rss = max(info.get("Rss", 0), 1)
        logger.info(
            "Dedup pool NUMA split: THP coverage AnonHugePages %.1f GiB of Rss %.1f GiB (%.1f%%), vmas=%d; smaps %.1f s",
            info.get("AnonHugePages", 0) / _GiB,
            rss / _GiB,
            100.0 * info.get("AnonHugePages", 0) / rss,
            info["vmas"],
            time.monotonic() - start,
        )

    def log_placement(self, buffer: torch.Tensor) -> None:
        """Log measured per-node bytes after pinning (called by alloc_with_host_register)."""
        start = time.monotonic()
        base = buffer.data_ptr()
        end = base + buffer.numel() * buffer.element_size()
        with open("/proc/self/numa_maps") as f:
            per_node, policies, vmas = parse_numa_maps(f.read(), base, end)
        total = max(sum(per_node.values()), 1)
        logger.info(
            "Dedup pool NUMA split%s: placement after pinning: %s "
            "(planned %s); vmas=%d policies=%s policy@base=%s; numa_maps %.1f s",
            " (fallback, not bound)" if self.fallback_reason else "",
            ", ".join(
                f"node {n} {b / _GiB:.1f} GiB ({100.0 * b / total:.1f}%)"
                for n, b in sorted(per_node.items())
            ),
            _fmt_nodes(getattr(self, "plan", {})),
            vmas,
            policies,
            mempolicy_at(base),
            time.monotonic() - start,
        )

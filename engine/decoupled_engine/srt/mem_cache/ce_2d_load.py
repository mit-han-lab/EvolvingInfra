"""Shared host-dedup loads with one 2D copy per page across layers (v0.1).

The direct page_first_direct path issues one copy descriptor per (page, layer):
~247k per 100k tokens for GLM-5.2, and CPU submission (~300 ns each in the
driver) then sets the load time (~77 ms / 100k). Here every page is one
``cudaMemcpy2DAsync`` per *run* of layers whose host offsets and device
addresses both advance by a constant step:

- host page_first_direct keeps a page's layers back to back (row pitch = one
  page-layer chunk);
- in shared mode the device KV and indexer pools are one layer-strided
  allocation (``layer_strided_buffers``), so the device pitch is constant.

Runs that would be a single layer are plain 1D copies. Registration ranges
of the host pool must start on page boundaries (a copy must not span two
``cudaHostRegister`` ranges); otherwise, or when the device strides are not
constant, ``plan`` returns None and the caller keeps the per-layer path.

Copy engines only: no kernel, no collective. Still submitted on the
scheduler thread in v0.1; the loader thread is v0.2.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from decoupled_engine.srt.mem_cache.staged_ce_load import registration_boundaries, token_pages

logger = logging.getLogger(__name__)

_H2D = 1  # cudaMemcpyHostToDevice
MAX_RUNS = 8
_LIB = None


def _cudart():
    """The libcudart this process already uses (torch's), via ctypes."""
    global _LIB
    if _LIB is not None:
        return _LIB
    names = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart.so" in line:
                    names.append(line.split()[-1])
                    break
    except OSError:
        pass
    names += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    err = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as e:
            err = e
    else:
        raise OSError(f"libcudart not found: {err}")
    fn = lib.cudaMemcpy2DAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    _LIB = lib
    return lib


def _memcpy2d_cuda(dst, dpitch, src, spitch, width, height, stream_handle):
    rc = _cudart().cudaMemcpy2DAsync(
        dst, dpitch, src, spitch, width, height, _H2D, stream_handle
    )
    if rc != 0:
        raise RuntimeError(
            f"cudaMemcpy2DAsync failed rc={rc} (dst={dst:#x} dpitch={dpitch} "
            f"src={src:#x} spitch={spitch} width={width} height={height})"
        )


# Tests replace this with a host memmove implementation.
memcpy2d: Callable = _memcpy2d_cuda


def _memset2d_cuda(dst, pitch, value, width, height, stream_handle):
    lib = _cudart()
    fn = lib.cudaMemset2DAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    rc = fn(dst, pitch, value, width, height, stream_handle)
    if rc != 0:
        raise RuntimeError(f"cudaMemset2DAsync failed rc={rc} (dst={dst:#x})")


# Debug only (layer-wise race amplifier); tests replace it like memcpy2d.
memset2d: Callable = _memset2d_cuda


def poison(arrs: dict, stream_handle, value: int = 0xFF) -> int:
    """Debug (M1 race amplifier): overwrite every destination row of ``arrs``
    with ``value`` on the loader stream, before the real copies. A consumer
    that reads a part's rows before waiting on its event then sees poison
    instead of plausible stale rows. Memset operations, no kernels."""
    n = 0
    for j in range(int(arrs["dst"].size)):
        memset2d(
            int(arrs["dst"][j]),
            int(arrs["dpitch"][j]),
            value,
            int(arrs["width"][j]),
            int(arrs["height"][j]),
            stream_handle,
        )
        n += 1
    for j in range(int(arrs["d1_dst"].size) if "d1_dst" in arrs else 0):
        w = int(arrs["d1_size"][j])
        memset2d(int(arrs["d1_dst"][j]), w, value, w, 1, stream_handle)
        n += 1
    return n


def has_rows(transfer, layer_ids: Sequence[int]) -> bool:
    """False if every layer in ``layer_ids`` is a 0-row device placeholder
    (skip-topk indexer layers): a layer part may legitimately have nothing
    to copy for a transfer. Unknown pool types count as having rows."""
    from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost

    if isinstance(transfer.host_pool, DSAIndexerPoolHost):
        dev_layers = getattr(transfer.device_pool, "index_k_with_scale_buffer", None)
    else:
        dev_layers = getattr(transfer.device_pool, "kv_buffer", None)
    if dev_layers is None:
        return True
    for layer_id in layer_ids:
        dev = dev_layers[0 if transfer.is_draft else layer_id]
        if dev.numel() != 0:
            return True
    return False


@dataclass(frozen=True)
class Run:
    host_off: int  # bytes from the start of a host page to the first layer
    dev_ptr: int  # device address of the first layer's page 0
    height: int  # layers in the run
    spitch: int  # host bytes between consecutive layers of the run
    dpitch: int  # device bytes between consecutive layers of the run


def group_runs(layers: Sequence[Tuple[int, int]], width: int) -> List[Run]:
    """Greedy runs over (host_off, dev_ptr) pairs with constant positive steps.

    Pitches must be >= width (rows must not overlap); a 1-layer run uses
    pitch = width.
    """
    runs: List[Run] = []
    i = 0
    while i < len(layers):
        h0, d0 = layers[i]
        j = i + 1
        sp = dp = None
        while j < len(layers):
            hs = layers[j][0] - layers[j - 1][0]
            ds = layers[j][1] - layers[j - 1][1]
            if sp is None:
                if hs < width or ds < width:
                    break
                sp, dp = hs, ds
            elif hs != sp or ds != dp:
                break
            j += 1
        n = j - i
        runs.append(Run(h0, d0, n, sp if n > 1 else width, dp if n > 1 else width))
        i = j
    return runs


@dataclass
class TransferPlan:
    host_base: int
    host_page_bytes: int
    dev_page_bytes: int
    width: int
    runs: List[Run]
    host_pages: torch.Tensor  # int64 CPU
    dev_pages: torch.Tensor  # int64 CPU
    # Host page indices where a new cudaHostRegister range starts (a copy
    # must not span two ranges); None for peer plans (device memory).
    reg_bounds: Optional[Tuple[int, ...]] = None


def kv_page_bytes(host_pool) -> int:
    """Bytes of one page of one layer in an MLA host pool.

    Computed from fields every MLA host pool sets in ``__init__`` (via
    ``get_size_per_token``), including a host-dedup follower's allocator-only
    pool. ``token_stride_size`` is set only in ``init_kv_buffer``, which an
    allocator-only follower never runs (an AttributeError on the first C2g
    follower load)."""
    return host_pool.page_size * host_pool.kv_cache_dim * host_pool.dtype.itemsize


def _pool_layout(host_pool, device_pool, layer_ids, is_draft):
    """(host tensor, [(host layer, device layer tensor)], width, dev unit) or None."""
    from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
    from decoupled_engine.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    if getattr(host_pool, "layout", None) != "page_first_direct":
        return None
    if host_pool._is_device_layer_sharded(device_pool):
        return None
    if getattr(host_pool, "dcp_size", 1) != 1:
        return None
    pairs = []
    if isinstance(host_pool, DSAIndexerPoolHost):
        host = host_pool.index_k_with_scale_buffer
        dev_layers = device_pool.index_k_with_scale_buffer
        width = host_pool.indexer_page_stride_size
        tokens_per_unit = host_pool.page_size  # device rows are pages
    elif isinstance(host_pool, MLATokenToKVPoolHost):
        host = host_pool.kv_buffer
        dev_layers = device_pool.kv_buffer
        width = kv_page_bytes(host_pool)
        tokens_per_unit = 1  # device rows are tokens
    else:
        return None
    for layer_id in layer_ids:
        host_layer = layer_id if is_draft else host_pool._host_layer_index(layer_id)
        dev = dev_layers[0 if is_draft else layer_id]
        if dev.numel() == 0:
            continue  # skip-topk placeholder: the per-layer path copies nothing
        pairs.append((host_layer, dev))
    return host, pairs, width, tokens_per_unit


def plan(transfer, layer_ids: Sequence[int]) -> Optional[TransferPlan]:
    """2D plan for one L2Transfer over its local layer ids, or None (fallback)."""
    host_pool, device_pool = transfer.host_pool, transfer.device_pool
    layout = _pool_layout(host_pool, device_pool, layer_ids, transfer.is_draft)
    if layout is None:
        return None
    host, pairs, width, tokens_per_unit = layout
    page_size = host_pool.page_size
    esz = host.element_size()
    host_page_bytes = host.stride(0) * esz
    host_layer_bytes = host.stride(1) * esz
    if host_layer_bytes != width:
        return None
    reg_bounds = registration_boundaries(host, host_page_bytes)
    if reg_bounds is None:
        return None
    host_idx = transfer.host_indices
    dev_idx = transfer.device_indices
    if host_idx is None or dev_idx is None or host_idx.is_cuda or dev_idx.is_cuda:
        return None
    host_pages = token_pages(host_idx, page_size)
    dev_pages = token_pages(dev_idx, page_size)
    if (
        host_pages is None
        or dev_pages is None
        or host_pages.numel() != dev_pages.numel()
    ):
        return None
    layers = []
    dev_page_bytes = None
    for host_layer, dev in pairs:
        unit_bytes = dev.stride(0) * dev.element_size()
        pb = unit_bytes * (page_size // tokens_per_unit)
        if dev_page_bytes is None:
            dev_page_bytes = pb
        elif pb != dev_page_bytes:
            return None
        layers.append((host_layer * host_layer_bytes, dev.data_ptr()))
    if not layers:
        return None
    if dev_page_bytes < width:
        return None
    runs = group_runs(layers, width)
    if len(runs) > max(MAX_RUNS, len(layers) // 8):
        # Device layers are not one strided allocation: 2D would not cut calls.
        return None
    return TransferPlan(
        host_base=host.data_ptr(),
        host_page_bytes=host_page_bytes,
        dev_page_bytes=dev_page_bytes,
        width=width,
        runs=runs,
        host_pages=host_pages,
        dev_pages=dev_pages,
        reg_bounds=tuple(reg_bounds),
    )


def submit(p: TransferPlan, stream_handle) -> int:
    """Issue the plan's copies on ``stream_handle``; returns the call count."""
    hp = (p.host_pages * p.host_page_bytes + p.host_base).tolist()
    dp = (p.dev_pages * p.dev_page_bytes).tolist()
    width = p.width
    if memcpy2d is _memcpy2d_cuda:
        fn = _cudart().cudaMemcpy2DAsync
        for run in p.runs:
            dev, off, dpitch, spitch, height = (
                run.dev_ptr,
                run.host_off,
                run.dpitch,
                run.spitch,
                run.height,
            )
            for h, d in zip(hp, dp):
                rc = fn(
                    dev + d, dpitch, h + off, spitch, width, height, _H2D, stream_handle
                )
                if rc:
                    _memcpy2d_cuda(
                        dev + d, dpitch, h + off, spitch, width, height, stream_handle
                    )
    else:
        for run in p.runs:
            for h, d in zip(hp, dp):
                memcpy2d(
                    run.dev_ptr + d,
                    run.dpitch,
                    h + run.host_off,
                    run.spitch,
                    width,
                    run.height,
                    stream_handle,
                )
    return len(p.runs) * len(hp)


# ---- whole-op submission (v0.2): one GIL-free native call per op ----------

_OP_FIELDS = ("dst", "dpitch", "src", "spitch", "width", "height")


def op_arrays(plans: Sequence[TransferPlan], batch_1d: bool = True) -> dict:
    """Every copy of the op as uint64 arrays (vectorized; no per-page Python).

    Runs of height > 1 become 2D copies (``_OP_FIELDS``); height-1 runs (the
    MTP draft layer, isolated layers) become 1D copies (``d1_*``), which the
    native helper sends as cudaMemcpyBatchAsync batches, or, with
    ``batch_1d=False``, height-1 2D copies (short driver calls: a batch call
    holds the driver ~19-75 us and delays other threads' NCCL launches).
    """
    import numpy as np

    cols = {k: [] for k in _OP_FIELDS}
    d1 = {"d1_dst": [], "d1_src": [], "d1_size": []}
    for p in plans:
        hp = p.host_pages.numpy().astype(np.uint64) * np.uint64(
            p.host_page_bytes
        ) + np.uint64(p.host_base)
        dp = p.dev_pages.numpy().astype(np.uint64) * np.uint64(p.dev_page_bytes)
        n = hp.size
        for run in p.runs:
            if run.height == 1 and batch_1d:
                d1["d1_dst"].append(dp + np.uint64(run.dev_ptr))
                d1["d1_src"].append(hp + np.uint64(run.host_off))
                d1["d1_size"].append(np.full(n, p.width, np.uint64))
                continue
            cols["dst"].append(dp + np.uint64(run.dev_ptr))
            cols["src"].append(hp + np.uint64(run.host_off))
            cols["dpitch"].append(np.full(n, run.dpitch, np.uint64))
            cols["spitch"].append(np.full(n, run.spitch, np.uint64))
            cols["width"].append(np.full(n, p.width, np.uint64))
            cols["height"].append(np.full(n, run.height, np.uint64))
    out = {}
    for k, v in list(cols.items()) + list(d1.items()):
        out[k] = (
            np.ascontiguousarray(np.concatenate(v)) if v else np.zeros(0, np.uint64)
        )
    return out


# ---- layer-part copy shapes (layer-wise loads with PART_PLAN=hybrid) -------
#
# A microbenchmark picked n = 8 layer parts with these shapes: page-major parts
# of 8 cost C(8)/C(1) = 2.48 (one copy per page and part), hybrid /
# run-coalesced 1.095. Same rows as op_arrays, fewer and larger copies. All
# copies are 2D (no 1D batches).


def _page_runs(hp, dp=None):
    """(starts, lengths) of the runs where ``hp`` (and ``dp``) advance by 1."""
    import numpy as np

    n = int(hp.size)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    brk = hp[1:] != hp[:-1] + 1
    if dp is not None:
        brk = brk | (dp[1:] != dp[:-1] + 1)
    starts = np.concatenate(
        [np.zeros(1, np.int64), np.nonzero(brk)[0].astype(np.int64) + 1]
    )
    lens = np.diff(np.concatenate([starts, np.array([n], np.int64)]))
    return starts, lens


def _split_at_bounds(h0, d0, ln, bounds):
    """Split runs (host page h0, device page d0, ln pages) where a host
    registration range starts (``bounds``: sorted host page indices).
    Returns (run index, h, d, n) per piece, in run order."""
    import numpy as np

    k = int(h0.size)
    if not bounds or k == 0:
        return np.arange(k), h0, d0, ln
    b = np.asarray(bounds, dtype=np.int64)
    lo = np.searchsorted(b, h0, side="right")  # first boundary > h0
    hi = np.searchsorted(b, h0 + ln, side="left")  # first boundary >= h0 + ln
    npc = hi - lo + 1
    run = np.repeat(np.arange(k), npc)
    q = np.arange(int(npc.sum())) - np.repeat(np.cumsum(npc) - npc, npc)
    last = b.size - 1
    start = np.where(q == 0, h0[run], b[np.clip(lo[run] + q - 1, 0, last)])
    end = np.where(
        q == npc[run] - 1, h0[run] + ln[run], b[np.clip(lo[run] + q, 0, last)]
    )
    return run, start, d0[run] + (start - h0[run]), end - start


def _rows_dict(parts) -> dict:
    """op_arrays-style dict from [(dst, dpitch, src, spitch, width, height)]
    column tuples (all 2D, no 1D batches)."""
    import numpy as np

    out = {}
    for i, k in enumerate(_OP_FIELDS):
        cols = [np.asarray(p[i], dtype=np.uint64) for p in parts]
        out[k] = (
            np.ascontiguousarray(np.concatenate(cols))
            if cols
            else np.zeros(0, np.uint64)
        )
    for k in ("d1_dst", "d1_src", "d1_size"):
        out[k] = np.zeros(0, np.uint64)
    return out


def _page_major_cols(p: TransferPlan, run: Run, hp, dp):
    import numpy as np

    n = int(hp.size)
    return (
        dp * p.dev_page_bytes + run.dev_ptr,
        np.full(n, run.dpitch if run.height > 1 else p.width, np.int64),
        hp * p.host_page_bytes + (p.host_base + run.host_off),
        np.full(n, run.spitch if run.height > 1 else p.width, np.int64),
        np.full(n, p.width, np.int64),
        np.full(n, run.height, np.int64),
    )


def hybrid_arrays(plans: Sequence[TransferPlan]) -> dict:
    """Source (H2D) copies of one layer part, hybrid shape (MB1 "hy").

    Per run of r pages contiguous on host and device, and per layer run of G
    layers: r <= G -> r page-major copies (one per page, height G, as
    op_arrays); r > G -> G layer-major copies (one per layer: width = row
    bytes, height = pages, pitches = host / device page bytes), split where a
    host registration range starts. A height-1 run (the MTP draft) becomes one
    copy per run. Copies follow the page runs' order, layer by layer inside a
    layer-major run."""
    import numpy as np

    parts = []
    for p in plans:
        hp = p.host_pages.numpy().astype(np.int64)
        dp = p.dev_pages.numpy().astype(np.int64)
        starts, lens = _page_runs(hp, dp)
        if lens.size == 0:
            continue
        if p.host_page_bytes < p.width or p.dev_page_bytes < p.width:
            # Layer-major rows need page pitches >= the row width (plan()
            # guarantees it; keep one copy per page and layer run otherwise).
            parts += [_page_major_cols(p, run, hp, dp) for run in p.runs]
            continue
        page_run = np.repeat(np.arange(lens.size), lens)
        for run in p.runs:
            g = int(run.height)
            short = lens <= g
            sel = short[page_run]
            pm = _page_major_cols(p, run, hp[sel], dp[sel])
            pm_key = page_run[sel]
            long_idx = np.nonzero(~short)[0]
            r_of, ph, pd, pn = _split_at_bounds(
                hp[starts[long_idx]], dp[starts[long_idx]], lens[long_idx], p.reg_bounds
            )
            counts = np.bincount(r_of, minlength=long_idx.size).astype(np.int64)
            tot = counts * g
            crun = np.repeat(np.arange(long_idx.size), tot)
            w = np.arange(int(tot.sum())) - np.repeat(np.cumsum(tot) - tot, tot)
            layer = w // counts[crun]
            piece = (np.cumsum(counts) - counts)[crun] + w % counts[crun]
            nl = int(crun.size)
            lm = (
                pd[piece] * p.dev_page_bytes + layer * run.dpitch + run.dev_ptr,
                np.full(nl, p.dev_page_bytes, np.int64),
                ph[piece] * p.host_page_bytes
                + layer * run.spitch
                + (p.host_base + run.host_off),
                np.full(nl, p.host_page_bytes, np.int64),
                np.full(nl, p.width, np.int64),
                pn[piece],
            )
            order = np.argsort(np.concatenate([pm_key, long_idx[crun]]), kind="stable")
            parts.append(tuple(np.concatenate([a, b])[order] for a, b in zip(pm, lm)))
    return _rows_dict(parts)


def coalesced_arrays(plans: Sequence[TransferPlan]) -> dict:
    """C2g follower copies of one layer part, run-coalesced (MB1 "rc").

    One 2D copy per run of pages contiguous on both sides (on a peer plan the
    source rows are the source rank's same device pages) and per layer run:
    width = run pages x row bytes, height = the run's layers, pitches = the
    layer strides (a height-1 run, e.g. the MTP draft: one row per run).
    Coalescing needs pages back to back in a layer (page bytes == row
    width); a plan without that keeps one copy per page and layer run."""
    import numpy as np

    parts = []
    for p in plans:
        hp = p.host_pages.numpy().astype(np.int64)
        dp = p.dev_pages.numpy().astype(np.int64)
        if hp.size == 0:
            continue
        starts, lens = _page_runs(hp, dp)
        width = lens * p.width
        n = int(lens.size)
        for run in p.runs:
            if (
                p.dev_page_bytes != p.width
                or p.host_page_bytes != p.width
                or (run.height > 1 and int(width.max()) > min(run.dpitch, run.spitch))
            ):
                parts.append(_page_major_cols(p, run, hp, dp))
                continue
            if run.height > 1:
                dpitch = np.full(n, run.dpitch, np.int64)
                spitch = np.full(n, run.spitch, np.int64)
            else:
                dpitch = spitch = width
            parts.append(
                (
                    dp[starts] * p.dev_page_bytes + run.dev_ptr,
                    dpitch,
                    hp[starts] * p.host_page_bytes + (p.host_base + run.host_off),
                    spitch,
                    width,
                    np.full(n, run.height, np.int64),
                )
            )
    return _rows_dict(parts)


def arrays_bytes(arrs: dict) -> int:
    """Bytes an op's copy arrays move (2D rows plus 1D copies)."""
    import numpy as np

    n2d = (
        int(np.sum(arrs["width"].astype(np.int64) * arrs["height"].astype(np.int64)))
        if arrs["width"].size
        else 0
    )
    n1d = (
        int(np.sum(arrs["d1_size"].astype(np.int64)))
        if "d1_size" in arrs and arrs["d1_size"].size
        else 0
    )
    return n2d + n1d


CUDA_MEMCPY_H2D = 1
CUDA_MEMCPY_DEFAULT = 4
CE_GATE_ERROR = 9001
CE_GATE_TIMEOUT = 9002
SEQ_SHIFT = 24


def progress_page(addr: int):
    """The C2g progress words at ``addr``: [0] progress, [1] op start, [2] error."""
    return (ctypes.c_uint64 * 3).from_address(addr)


def wait_progress(addr: int, seq: int, timeout_s: float = 120.0) -> None:
    """Python fallback of the follower gate: wait until op ``seq`` is complete."""
    words = progress_page(addr)
    deadline = time.monotonic() + timeout_s
    while True:
        if words[2]:
            raise RuntimeError("C2g gate: the dedup source's loader reported an error")
        start = words[1]
        if start >> SEQ_SHIFT > seq:
            return
        if start >> SEQ_SHIFT == seq and words[0] >= (seq << SEQ_SHIFT) | (
            start & ((1 << SEQ_SHIFT) - 1)
        ):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"C2g gate: no progress from the dedup source for {timeout_s:.0f} s"
            )
        time.sleep(0.0002)


class _MemcpyAttributes(ctypes.Structure):
    """cudaMemcpyAttributes (24 B): srcAccessOrder, srcLocHint, dstLocHint, flags."""

    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocType", ctypes.c_int),
        ("srcLocId", ctypes.c_int),
        ("dstLocType", ctypes.c_int),
        ("dstLocId", ctypes.c_int),
        ("flags", ctypes.c_uint),
    ]


_NATIVE = None


def _native_build_dir() -> str:
    d = os.environ.get("SGLANG_CE_SUBMIT_BUILD_DIR") or os.path.join(
        tempfile.gettempdir(), f"sglang-ce-submit-{os.getuid()}"
    )
    os.makedirs(d, exist_ok=True)
    return d


def native():
    """ce_submit.c compiled once per node (node-local temp dir), or None."""
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE or None
    try:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ce_submit.c")
        with open(src, "rb") as f:
            tag = hashlib.sha256(f.read()).hexdigest()[:16]
        so = os.path.join(
            _native_build_dir(), f"ce_submit-{tag}-{platform.machine()}.so"
        )
        if not os.path.exists(so):
            cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
            if not cc:
                raise RuntimeError("no C compiler (cc/gcc) on PATH")
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.run(
                [cc, "-O2", "-shared", "-fPIC", "-o", tmp, src],
                check=True,
                capture_output=True,
                timeout=120,
            )
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        fn = lib.ce_submit_op
        fn.restype = ctypes.c_int
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        fn.argtypes = (
            [vp] * 4  # cp, rec, query, sync
            + [vp] * 6  # 2D arrays
            + [sz, sz, vp, ctypes.c_int]  # n2d, chunk2d, batch, batch_has_fail_idx
            + [vp] * 3  # 1D arrays
            + [sz, sz, vp, vp, ctypes.c_int, vp, ctypes.c_uint64, vp, vp, vp]
            + [
                ctypes.c_int,
                vp,
                vp,
                ctypes.c_uint64,
                ctypes.c_uint64,
                vp,
            ]  # ..., rate word
            + [ctypes.c_int]  # gate_whole
            + [vp, vp]  # adaptive cap: busy word, deadline word (NULL = off)
            + [ctypes.c_uint64]  # adaptive cap: bytes of later parts of the job
        )
        _NATIVE = lib
        logger.info("ce_submit native helper: %s", so)
    except Exception as e:  # noqa: BLE001
        logger.warning("ce_submit native helper unavailable: %s", e)
        _NATIVE = False
    return _NATIVE or None


class PacedSubmitter:
    """Whole-op submission on one stream: 2D copies, then 1D batches, in
    chunks with at most ``window`` chunks in flight, all in one native call
    that holds no GIL. The wait for a chunk slot is cudaEventSynchronize on a
    BlockingSync event (sleeps; no spinning, no copy call on a full queue)."""

    def __init__(self, chunk_ops: int = 256, window: int = 2, chunk_1d: int = 256):
        self.chunk_ops = max(1, int(chunk_ops))
        self.chunk_1d = max(1, int(chunk_1d))
        self.window = max(1, int(window))
        self._events = None
        self._state = (ctypes.c_int * 2)(0, 0)
        self._fns = None
        self._attr = None
        self._batch_has_fail_idx = 0

    def _cuda_setup(self):
        lib = _cudart()
        if self._fns is None:
            for name, argtypes in (
                ("cudaEventCreateWithFlags", [ctypes.c_void_p, ctypes.c_uint]),
                ("cudaRuntimeGetVersion", [ctypes.c_void_p]),
                ("cudaGetDevice", [ctypes.c_void_p]),
            ):
                f = getattr(lib, name)
                f.restype = ctypes.c_int
                f.argtypes = argtypes
            ver = ctypes.c_int(0)
            lib.cudaRuntimeGetVersion(ctypes.byref(ver))
            self._batch_has_fail_idx = 0 if ver.value >= 13000 else 1
            dev = ctypes.c_int(0)
            lib.cudaGetDevice(ctypes.byref(dev))
            self._attr = _MemcpyAttributes(
                1, 2, 0, 1, dev.value, 0
            )  # Stream order; host -> device
            self._fns = tuple(
                ctypes.cast(getattr(lib, n), ctypes.c_void_p).value
                for n in (
                    "cudaMemcpy2DAsync",
                    "cudaEventRecord",
                    "cudaEventQuery",
                    "cudaEventSynchronize",
                    "cudaMemcpyBatchAsync",
                )
            )
        if self._events is None:
            evs = (ctypes.c_void_p * self.window)()
            for i in range(self.window):
                h = ctypes.c_void_p()
                rc = lib.cudaEventCreateWithFlags(
                    ctypes.byref(h), 0x3
                )  # BlockingSync|DisableTiming
                if rc:
                    raise RuntimeError(f"cudaEventCreateWithFlags failed rc={rc}")
                evs[i] = h.value
            self._events = evs
        return self._fns

    def peer_attr(self, src_device: int):
        """1D batch attributes for device-to-device peer copies."""
        if memcpy2d is not _memcpy2d_cuda:
            return None  # tests: copies are faked, no CUDA runtime
        self._cuda_setup()
        a = _MemcpyAttributes(1, 1, int(src_device), 1, self._attr.dstLocId, 0)
        self._peer_attrs = getattr(self, "_peer_attrs", [])
        self._peer_attrs.append(a)  # keep alive
        return a

    def _rate_arg(self, rate_gbps: float, rate_word) -> Optional[int]:
        """Address of the rate word for the native call (None = uncapped)."""
        if rate_word is not None:
            return ctypes.addressof(rate_word)
        if rate_gbps <= 0:
            return None
        if getattr(self, "_static_rate", None) is None:
            self._static_rate = ctypes.c_uint64(0)
        self._static_rate.value = int(rate_gbps * 1e9)
        return ctypes.addressof(self._static_rate)

    def n_chunks(self, arrs: dict) -> int:
        n2d = int(arrs["dst"].size)
        n1d = int(arrs["d1_dst"].size) if "d1_dst" in arrs else 0
        return -(-n2d // self.chunk_ops) + -(-n1d // self.chunk_1d)

    def submit(
        self,
        arrs: dict,
        stream_handle,
        *,
        kind: int = CUDA_MEMCPY_H2D,
        attr=None,
        pub: int = 0,
        gate: int = 0,
        seq: int = 0,
        timeout_s: float = 120.0,
        rate_gbps: float = 0.0,
        rate_word: Optional[ctypes.c_uint64] = None,
        gate_whole: bool = False,
        busy_word: Optional[ctypes.c_uint64] = None,
        deadline_word: Optional[ctypes.c_uint64] = None,
        tail_bytes: int = 0,
    ) -> Tuple[int, float, int]:
        """Returns (copies, ms waiting for chunk events, chunks).

        ``pub`` / ``gate``: address of a C2g progress page (3 x uint64) to
        publish this op's chunk completions to (source) or to wait on before
        each chunk (follower), for op ``seq``. A follower gates per chunk when
        both sides have the same chunk count; ``gate_whole`` makes it wait for
        the source's whole op instead (the sides copy different shapes). ``kind`` is the 2D copy kind
        (peer copies: CUDA_MEMCPY_DEFAULT); ``attr`` the 1D batch attributes;
        ``rate_gbps`` > 0 paces the issue rate (GB/s, e.g. NVLink peer reads);
        ``rate_word`` instead is a live cap (bytes/s) the native helper re-reads
        while the op runs, so its owner can lift it (set 0) mid-op.
        ``busy_word`` / ``deadline_word`` (adaptive cap, both or neither): the
        cap applies only while the busy word is non-zero and the op's remaining
        bytes fit the cap before the deadline (CLOCK_MONOTONIC ns, 0 = none);
        ``tail_bytes``: the job's bytes in later calls (its later layer parts),
        counted as remaining by that check.
        """
        n2d = int(arrs["dst"].size)
        n1d = int(arrs["d1_dst"].size) if "d1_dst" in arrs else 0
        self.last_gate_ms = 0.0
        self.last_rate_ms = 0.0
        self.last_flips = 0
        if n2d + n1d == 0 and not pub:
            return 0, 0.0, 0
        lib = native() if memcpy2d is _memcpy2d_cuda else None
        if lib is None:
            # Tests (memcpy2d replaced) or no compiler: per-copy Python loop;
            # C2g pages are honored at whole-op granularity.
            total = self.n_chunks(arrs)
            if pub:
                progress_page(pub)[1] = (seq << SEQ_SHIFT) | total
            if gate:
                wait_progress(gate, seq, timeout_s)
            for j in range(n2d):
                memcpy2d(*(int(arrs[k][j]) for k in _OP_FIELDS), stream_handle)
            for j in range(n1d):
                w = int(arrs["d1_size"][j])
                memcpy2d(
                    int(arrs["d1_dst"][j]),
                    w,
                    int(arrs["d1_src"][j]),
                    w,
                    w,
                    1,
                    stream_handle,
                )
            if pub:
                progress_page(pub)[0] = (seq << SEQ_SHIFT) | total
            return n2d + n1d, 0.0, 1
        if not stream_handle:
            raise RuntimeError("PacedSubmitter needs a non-default stream")
        cp, rec, query, sync, batch = self._cuda_setup()
        fail = ctypes.c_size_t(0)
        stats = (ctypes.c_uint64 * 6)()
        adaptive = busy_word is not None or deadline_word is not None
        rc = lib.ce_submit_op(
            cp,
            rec,
            query,
            sync,
            *(arrs[k].ctypes.data for k in _OP_FIELDS),
            n2d,
            self.chunk_ops,
            batch,
            self._batch_has_fail_idx,
            arrs["d1_dst"].ctypes.data,
            arrs["d1_src"].ctypes.data,
            arrs["d1_size"].ctypes.data,
            n1d,
            self.chunk_1d,
            ctypes.addressof(attr if attr is not None else self._attr),
            self._events,
            self.window,
            stream_handle,
            20000,
            ctypes.byref(fail),
            stats,
            self._state,
            int(kind),
            pub or None,
            gate or None,
            int(seq),
            int(timeout_s * 1e9),
            self._rate_arg(rate_gbps, rate_word),
            1 if gate_whole else 0,
            ctypes.addressof(busy_word) if busy_word is not None else None,
            ctypes.addressof(deadline_word) if deadline_word is not None else None,
            int(tail_bytes),
        )
        self.last_gate_ms = stats[2] / 1e6
        self.last_rate_ms = stats[3] / 1e6
        if adaptive:
            self.last_flips = int(stats[4])
        if rc:
            j = fail.value
            if j < n2d:
                what = (
                    f"2D copy {j} (dst={int(arrs['dst'][j]):#x} src={int(arrs['src'][j]):#x} "
                    f"width={int(arrs['width'][j])} height={int(arrs['height'][j])})"
                )
            elif j < n2d + n1d:
                k = j - n2d
                what = (
                    f"1D copy {k} (dst={int(arrs['d1_dst'][k]):#x} "
                    f"src={int(arrs['d1_src'][k]):#x} size={int(arrs['d1_size'][k])})"
                )
            elif rc == CE_GATE_ERROR:
                what = "C2g gate: the dedup source's loader reported an error"
            elif rc == CE_GATE_TIMEOUT:
                what = (
                    f"C2g gate: no progress from the dedup source for {timeout_s:.0f} s"
                )
            else:
                what = "chunk event"
            raise RuntimeError(f"ce_submit_op failed rc={rc} at {what}")
        return n2d + n1d, stats[0] / 1e6, int(stats[1])


def layer_strided_buffers(
    num_layers: int, row_shape: Sequence[int], *, dtype, device, rows_of=None
):
    """Per-layer views of ONE allocation (constant layer stride).

    ``rows_of(i)`` gives layer i's row count; layers with 0 rows get their own
    empty placeholder and do not take space in the shared allocation.
    """
    rows = [
        rows_of(i) if rows_of is not None else row_shape[0] for i in range(num_layers)
    ]
    live = [i for i, r in enumerate(rows) if r]
    if not live:
        return [
            torch.zeros((0, *row_shape[1:]), dtype=dtype, device=device) for _ in rows
        ]
    n_rows = max(rows[i] for i in live)
    if any(rows[i] != n_rows for i in live):
        raise ValueError("layer_strided_buffers needs equal row counts per live layer")
    big = torch.zeros((len(live), n_rows, *row_shape[1:]), dtype=dtype, device=device)
    views = big.unbind(0)
    out, k = [], 0
    for r in rows:
        if r:
            out.append(views[k])
            k += 1
        else:
            out.append(torch.zeros((0, *row_shape[1:]), dtype=dtype, device=device))
    return out


def shared_dedup_requested(mla: bool = False) -> bool:
    """Shared pool or C2g: both need layer-strided device KV/indexer pools.

    ``mla``: a plain (non-DSA) MLA pool, whose host dedup selects its fan-out
    with SGLANG_MLA_HOST_DEDUP_FANOUT (C2g only); DSA pools keep the DSA word."""
    from decoupled_engine.srt.environ import envs

    if mla:
        return str(envs.SGLANG_MLA_HOST_DEDUP_FANOUT.get()).lower() == "ce_peer"
    return str(envs.SGLANG_DSA_HOST_DEDUP_FANOUT.get()).lower() in ("shared", "ce_peer")


def joint_runs(plans: Sequence[TransferPlan]) -> Tuple[float, float]:
    """(mean run length, fraction of pages in runs > 78) of pages contiguous
    on both sides (host and device), over the op's first plan (the anchor)."""
    if not plans:
        return 0.0, 0.0
    p = plans[0]
    hp, dp = p.host_pages, p.dev_pages
    n = int(hp.numel())
    if n == 0:
        return 0.0, 0.0
    brk = ((hp[1:] - hp[:-1]) != 1) | ((dp[1:] - dp[:-1]) != 1)
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.nonzero(brk).flatten() + 1]
    )
    lens = torch.diff(torch.cat([starts, torch.tensor([n])]))
    long_pages = int(lens[lens > 78].sum())
    return n / int(lens.numel()), long_pages / n


def peer_plan(
    transfer, layer_ids: Sequence[int], peer_ptrs: dict
) -> Optional[TransferPlan]:
    """C2g follower: 2D D2D plan pulling ``transfer``'s rows from the dedup
    source's IPC-mapped device buffers (``peer_ptrs[(kind, host_layer)]``)
    into this rank's own buffers; None if a buffer is missing or strided
    differently (the caller fails loudly: a follower has no other source).

    ``transfer.host_pool`` is the follower's allocator-only dedup pool: only
    fields set before its early return in ``__init__`` may be read here
    (page_size, kv_cache_dim, dtype, indexer_page_stride_size, layer ranges),
    never buffers or fields set by ``init_kv_buffer``."""
    from decoupled_engine.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
    from decoupled_engine.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    host_pool, device_pool = transfer.host_pool, transfer.device_pool
    if isinstance(host_pool, DSAIndexerPoolHost):
        kind = "idx"
        dev_layers = device_pool.index_k_with_scale_buffer
        width = host_pool.indexer_page_stride_size
        tokens_per_unit = host_pool.page_size
    elif isinstance(host_pool, MLATokenToKVPoolHost):
        # A separate HiCache draft pool (EAGLE3 sidecar) has its own host layer
        # ids starting at 0: its source buffers are keyed apart from the
        # target's (dedup_ipc_fanout.layer_buffers, SIDECAR_DRAFT_KIND).
        kind = getattr(host_pool, "ce_peer_kind", "kv")
        dev_layers = device_pool.kv_buffer
        width = kv_page_bytes(host_pool)  # follower pools are allocator-only
        tokens_per_unit = 1
    else:
        return None
    dev_idx = transfer.device_indices
    if dev_idx is None or dev_idx.is_cuda:
        return None
    page_size = host_pool.page_size
    dev_pages = token_pages(dev_idx, page_size)
    if dev_pages is None:
        return None
    layers, dev_page_bytes = [], None
    for layer_id in layer_ids:
        host_layer = (
            layer_id if transfer.is_draft else host_pool._host_layer_index(layer_id)
        )
        dev = dev_layers[0 if transfer.is_draft else layer_id]
        if dev.numel() == 0:
            continue
        src = peer_ptrs.get((kind, host_layer))
        if src is None:
            return None
        pb = dev.stride(0) * dev.element_size() * (page_size // tokens_per_unit)
        if dev_page_bytes is None:
            dev_page_bytes = pb
        elif pb != dev_page_bytes:
            return None
        layers.append((int(src), dev.data_ptr()))
    if not layers or dev_page_bytes < width:
        return None
    runs = group_runs(layers, width)
    if len(runs) > max(MAX_RUNS, len(layers) // 8):
        return None
    return TransferPlan(
        host_base=0,
        host_page_bytes=dev_page_bytes,
        dev_page_bytes=dev_page_bytes,
        width=width,
        runs=runs,
        host_pages=dev_pages,
        dev_pages=dev_pages,
    )


class CostLog:
    """Scheduler-thread costs of the shared-dedup load path, logged every 60 s."""

    INTERVAL_S = 60.0

    def __init__(self):
        self.enabled = False  # set on the dedup source rank
        self._reset(time.monotonic())

    def _reset(self, now):
        self.t0 = now
        self.stats = {}

    def add(self, name: str, ms: float, n: int = 1) -> None:
        s = self.stats.setdefault(name, [0, 0.0, 0.0])
        s[0] += n
        s[1] += ms
        s[2] = max(s[2], ms)
        now = time.monotonic()
        if self.enabled and now - self.t0 >= self.INTERVAL_S:
            parts = [
                f"{k} n={v[0]} mean_ms={v[1] / max(v[0], 1):.3f} max_ms={v[2]:.3f} sum_ms={v[1]:.1f}"
                for k, v in sorted(self.stats.items())
            ]
            logger.info(
                "Shared host-dedup costs over %.0f s: %s",
                now - self.t0,
                "; ".join(parts),
            )
            self._reset(now)


COST = CostLog()

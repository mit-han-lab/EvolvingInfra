# Adapted from SGLang python/sglang/srt/mem_cache/cpp_utils/native_hash.py (commit 9a489f8d2f), modified for evolving-infra.
import os
import platform
import sys
import threading
from array import array
from functools import lru_cache
from typing import Any, Optional

# The first use JIT-builds and imports the extension. The HiCache prefetch
# thread and the scheduler thread both hash, and their first calls can overlap
# (fresh TORCH_EXTENSIONS_DIR: the build takes seconds). lru_cache does not
# serialize concurrent misses, so two torch load()s of the same extension could
# run in one process, and a caller got a module without get_hash
# (AttributeError: module 'hicache_hash_cpp' has no attribute 'get_hash' on the
# first real request: fatal in a prefill scheduler thread, or a one-off
# prefetch-thread miss; later calls were fine).
# Only one thread may run the load, and a module without get_hash is rejected.
_NATIVE_HASH_LOAD_LOCK = threading.Lock()


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


def _load_native_hash_module() -> Any:
    with _NATIVE_HASH_LOAD_LOCK:
        return _load_native_hash_module_once()


@lru_cache(maxsize=1)
def _load_native_hash_module_once() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        raise RuntimeError(
            "HiCache native hash is only supported on little-endian Linux"
        )

    try:
        from torch.utils.cpp_extension import load

        abs_path = os.path.dirname(os.path.abspath(__file__))
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
        module = load(
            name="hicache_hash_cpp",
            sources=[f"{abs_path}/hash_binding.cpp"],
            extra_cflags=extra_cflags,
            extra_ldflags=["-lcrypto"],
            with_cuda=False,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to load HiCache native hash extension") from exc
    if not hasattr(module, "get_hash"):
        # Never cache a half-initialized or foreign module (the raise skips
        # lru_cache, so the next call retries the load).
        raise RuntimeError(
            f"HiCache native hash extension {getattr(module, '__file__', module)!r} "
            "has no get_hash"
        )
    return module


def _native_hash_input(token_ids: Any) -> tuple[array, int, int, bool]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )

    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    if isinstance(raw, array) and raw.typecode in ("I", "q", "Q", "L"):
        if is_bigram and logical_len > 0 and len(raw) < logical_len + 1:
            raise ValueError("bigram token buffer is shorter than logical length")
        return raw, logical_len, 2 if is_bigram else 1, is_bigram

    if is_bigram:
        return array("I", raw[: logical_len + 1]), logical_len, 2, is_bigram

    if logical_len == 0:
        return array("I"), logical_len, 1, is_bigram

    first_token = raw[0]
    if isinstance(first_token, tuple):
        unit_width = len(first_token)
        return (
            array("I", (elem for token in raw[:logical_len] for elem in token)),
            logical_len,
            unit_width,
            is_bigram,
        )

    return array("I", raw[:logical_len]), logical_len, 1, is_bigram


def get_native_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int] = None
) -> str | list[str]:
    raw, logical_len, unit_width, is_bigram = _native_hash_input(token_ids)
    return _load_native_hash_module().get_hash(
        raw, logical_len, unit_width, is_bigram, prior_digest, page_size
    )

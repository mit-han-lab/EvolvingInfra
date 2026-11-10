import hashlib
import struct


def common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def block_aligned_prefix_length(prefix_tokens: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")
    return prefix_tokens - (prefix_tokens % block_size)


class GlobalPrefixRegistry:
    """Block-aligned prefix registry modeling ONE engine with INFINITE cache.

    The registry answers, per request: "how many prompt tokens could a single
    engine with unbounded KV-cache capacity have served from cache?" — the
    workload-inherent upper bound (global across sessions, no eviction, no
    multi-worker split, cache-salt agnostic). Completed requests insert the
    cumulative block-hash chain of their full processed sequence
    (prompt + generated tokens); lookups walk a candidate prompt's chain and
    stop at the first unseen block, mirroring radix-tree contiguous-prefix
    semantics. Callers should look up at request-send time and insert at
    request-completion time so the bound reflects what a real single engine
    could have cached when the request was dispatched.

    Chains use deterministic 128-bit BLAKE2b digests, so a collision is
    negligible at corpus scale and results are reproducible across runs.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be greater than 0")
        self._block_size = block_size
        self._chains: set[bytes] = set()

    @staticmethod
    def _block_digest(previous: bytes, block: list[int]) -> bytes:
        hasher = hashlib.blake2b(previous, digest_size=16)
        # Preserve the signed little-endian 64-bit hash format while packing
        # the entire block in C instead of updating the hasher per token.
        hasher.update(struct.pack(f"<{len(block)}q", *block))
        return hasher.digest()

    def insert(self, token_ids: list[int]) -> None:
        digest = b""
        block_size = self._block_size
        for start in range(0, len(token_ids) - block_size + 1, block_size):
            digest = self._block_digest(digest, token_ids[start : start + block_size])
            self._chains.add(digest)

    def match_tokens(self, token_ids: list[int]) -> int:
        # The final prompt token is never reusable from cache (its logits must
        # be recomputed), matching the reusable-prefix clamp used elsewhere.
        usable = max(len(token_ids) - 1, 0)
        matched = 0
        digest = b""
        block_size = self._block_size
        for start in range(0, usable - block_size + 1, block_size):
            digest = self._block_digest(digest, token_ids[start : start + block_size])
            if digest not in self._chains:
                break
            matched = start + block_size
        return matched

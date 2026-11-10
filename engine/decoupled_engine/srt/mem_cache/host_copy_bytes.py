"""Host bytes that a chunked device-to-host copy allocates.

PD decode retraction with the cpu_tensor backend copies a request's KV to
host tensors, one per layer and chunk (``get_cpu_copy`` of the KV pool). The
retraction backup cap (``mem_cache/common.py``) projects that size from the
request's token count and the pool shapes before it copies anything.
"""


def host_alloc_nbytes(nbytes: int, pinned: bool) -> int:
    """Host bytes one copied tensor of ``nbytes`` takes.

    Pinned (a ``non_blocking`` copy): torch's caching host allocator rounds
    every block up to a power of two and keeps it cached after release.
    Pageable (a blocking copy): the exact size, freed on release.
    """
    n = int(nbytes)
    if n <= 0:
        return 0
    return 1 << (n - 1).bit_length() if pinned else n


def chunked_copy_nbytes(
    num_rows: int, chunk_rows: int, row_nbytes: int, pinned: bool
) -> int:
    """Host bytes of copying ``num_rows`` rows in chunks of ``chunk_rows`` rows,
    one tensor per chunk."""
    num_rows, row_nbytes = int(num_rows), int(row_nbytes)
    if num_rows <= 0 or row_nbytes <= 0:
        return 0
    chunk_rows = max(1, int(chunk_rows))
    full, tail = divmod(num_rows, chunk_rows)
    full_nbytes = full * host_alloc_nbytes(chunk_rows * row_nbytes, pinned)
    return full_nbytes + host_alloc_nbytes(tail * row_nbytes, pinned)

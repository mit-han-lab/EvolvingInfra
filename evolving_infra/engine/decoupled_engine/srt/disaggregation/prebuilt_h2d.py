"""Host-to-GPU copies of the D admission (PREBUILT) path without a host wait.

``SGLANG_PREBUILT_NONBLOCKING_H2D`` (default off). A blocking copy
(``torch.tensor(..., device=cuda)``, ``cpu_tensor.to(cuda)``) ends in a
``cudaStreamSynchronize`` of the current stream. On the overlap loops that
is the schedule stream, which already waits behind the forward in flight
(the WAR barrier after each launch, or ``wait_stream(forward_stream)``), so
the CPU stalled for the rest of that forward and the work after it (radix
insert, ``admit_pending``) ran with the GPU idle.

With the flag on, the data are built in pinned host memory and copied with
``non_blocking=True`` on the same current stream:

- GPU order is unchanged: the copy is enqueued where the blocking copy was,
  and every consumer runs on that stream or on one that waits for it.
- Pinned blocks come from PyTorch's caching host allocator, which records
  an event on the current stream at each non-blocking copy and does not
  hand the block out again until that event has completed, so a block is
  never rewritten while its copy is pending.
- The source host memory (lists, or the request's views into the transfer
  metadata buffers) is read synchronously when the pinned staging tensor is
  filled, i.e. at the same point as before.
- Every TP rank takes the same branch (environment flag, same tensors).

With the flag off every call site keeps its original code.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from decoupled_engine.srt.environ import envs


def prebuilt_nonblocking_h2d() -> bool:
    return envs.SGLANG_PREBUILT_NONBLOCKING_H2D.get()


def _pin_for(device) -> bool:
    from decoupled_engine.srt.utils.common import is_pin_memory_available

    return torch.device(device).type != "cpu" and is_pin_memory_available(device)


def list_to_device(values: Sequence, dtype: torch.dtype, device) -> torch.Tensor:
    """``torch.tensor(values, dtype=dtype, device=device)`` without a host
    wait: staged in pinned memory, copied non-blocking on the current stream."""
    host = torch.tensor(values, dtype=dtype, pin_memory=_pin_for(device))
    return host.to(device, non_blocking=True)


def _all_host_tensors(items: Sequence) -> bool:
    if not items or not isinstance(items[0], torch.Tensor):
        return False
    shape = items[0].shape
    # Same shape required: copy_ would broadcast where torch.stack raises.
    return all(
        isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.shape == shape
        for t in items
    )


def stack_to_device(
    items: Sequence, device, dtype: Optional[torch.dtype] = None
) -> Optional[torch.Tensor]:
    """``torch.stack(items).to(device)`` for host tensors without a host wait.

    The rows are copied (and cast to ``dtype`` if given, as ``as_tensor``
    would) into one pinned tensor, which is copied non-blocking on the
    current stream. Returns None if any item is not a CPU tensor or the
    shapes differ (the caller keeps its original path, which raises or, for
    a GPU source, copies on the device instead of adding a D2H sync)."""
    if not _all_host_tensors(items):
        return None
    first = items[0]
    host = torch.empty(
        (len(items),) + tuple(first.shape),
        dtype=first.dtype if dtype is None else dtype,
        pin_memory=_pin_for(device),
    )
    if dtype is None or all(t.dtype == dtype for t in items):
        torch.stack(list(items), dim=0, out=host)
    else:
        for i, t in enumerate(items):
            host[i].copy_(t)
    return host.to(device, non_blocking=True)

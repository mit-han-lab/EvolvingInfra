"""Collective-communication kernels (custom all-reduce, ...).

Reserved group in the ``decoupled_engine.kernels`` namespace (RFC #29630). No thin
wrappers are exposed here: the collective ops (custom all-reduce and friends)
are stateful — they manage workspaces / IPC handles and are driven through a
``CustomAllreduce``-style object and ``torch.ops.sgl_kernel.*`` bindings rather
than standalone callable kernels, so a thin ``decoupled_engine.kernels.ops`` forwarder
would be misleading. Import them from ``sgl_kernel`` / ``decoupled_engine.kernels.jit``
directly until a proper stateful-op interface is designed.
"""

__all__ = []

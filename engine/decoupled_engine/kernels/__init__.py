"""Unified public kernel namespace for SGLang (RFC #29630).

SGLang runtime code and tests should import callable kernels from
``decoupled_engine.kernels.ops.<group>``, e.g.::

    from decoupled_engine.kernels.ops.layernorm import rmsnorm
    from decoupled_engine.kernels.ops.activation import silu_and_mul
    from decoupled_engine.kernels.ops.kvcache import reshape_and_cache_flash

Implementations still live in ``decoupled_engine.kernels.jit`` (JIT CUDA), the
``sgl_kernel`` wheel (AOT CUDA/C++), Triton op modules, etc. The ``ops.*``
functions are thin wrappers that forward to a chosen backend; the
:data:`~decoupled_engine.kernels.registry.registry` provides an inventory of every
backend so alternatives can be compared and selected. Multi-backend operators
are structured as :class:`~decoupled_engine.kernels.fused_op.BaseFusedOp` subclasses —
one ``forward_<backend>`` method per backend behind a single ``forward()``,
with a required pure-``torch`` ``forward_native`` reference and a
``SGLANG_FORCE_FUSED_OP_BACKEND`` global switch.

Importing this package (and any ``ops.*`` group) does not import a kernel
backend (``sgl_kernel`` / ``decoupled_engine.kernels.jit``) or trigger JIT compilation:
registration is metadata-only and backends are imported lazily on first call.
This keeps the namespace usable for inventory tooling on a CPU-only box.
"""

from decoupled_engine.kernels.fused_op import (
    BaseFusedOp,
    FusedOpTraceRecord,
    clear_fused_op_trace,
    disable_fused_op_trace,
    enable_fused_op_trace,
    get_fused_op_backend,
    get_fused_op_trace,
    register_fused_op,
    set_fused_op_backend,
)
from decoupled_engine.kernels.registry import KernelRegistry, register_kernel, registry
from decoupled_engine.kernels.selector import get_kernel, select_kernel
from decoupled_engine.kernels.spec import (
    CapabilityRequirement,
    DeviceType,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
    capabilities_satisfied,
)

# Importing the operator groups populates the registry (metadata only). Kept
# after the core imports above (and guarded from isort reordering) so those
# modules are fully initialized first.
from decoupled_engine.kernels import ops  # noqa: E402  # isort: skip

__all__ = [
    "ops",
    "BaseFusedOp",
    "CapabilityRequirement",
    "DeviceType",
    "capabilities_satisfied",
    "FormatSignature",
    "FusedOpTraceRecord",
    "KernelBackend",
    "KernelRegistry",
    "KernelSpec",
    "PlatformInfo",
    "clear_fused_op_trace",
    "disable_fused_op_trace",
    "enable_fused_op_trace",
    "get_fused_op_backend",
    "get_fused_op_trace",
    "get_kernel",
    "register_fused_op",
    "register_kernel",
    "registry",
    "select_kernel",
]

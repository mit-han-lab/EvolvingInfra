# [Deprecated] Re-export shim for backward compatibility. Use dsa.triton_kernel instead.
import warnings

warnings.warn(
    "decoupled_engine.srt.layers.attention.nsa.triton_kernel is deprecated; "
    "use decoupled_engine.kernels.ops.attention.dsa.triton_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from decoupled_engine.kernels.ops.attention.dsa.triton_kernel import *  # noqa: F401, F403

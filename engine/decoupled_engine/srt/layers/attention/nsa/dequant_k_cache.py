# [Deprecated] Re-export shim for backward compatibility. Use dsa.dequant_k_cache instead.
import warnings

warnings.warn(
    "decoupled_engine.srt.layers.attention.nsa.dequant_k_cache is deprecated; "
    "use decoupled_engine.kernels.ops.attention.dsa.dequant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from decoupled_engine.kernels.ops.attention.dsa.dequant_k_cache import *  # noqa: F401, F403

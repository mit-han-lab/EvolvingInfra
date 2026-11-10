# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_backend_mtp_precompute instead.
import warnings

warnings.warn(
    "decoupled_engine.srt.layers.attention.nsa.nsa_backend_mtp_precompute is deprecated; "
    "use decoupled_engine.srt.layers.attention.dsa.dsa_backend_mtp_precompute instead.",
    DeprecationWarning,
    stacklevel=2,
)
from decoupled_engine.srt.layers.attention.dsa.dsa_backend_mtp_precompute import *  # noqa: F401, F403

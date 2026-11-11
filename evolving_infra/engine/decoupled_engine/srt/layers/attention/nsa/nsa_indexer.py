# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
import warnings

warnings.warn(
    "decoupled_engine.srt.layers.attention.nsa.nsa_indexer is deprecated; "
    "use decoupled_engine.srt.layers.attention.dsa.dsa_indexer instead.",
    DeprecationWarning,
    stacklevel=2,
)
from decoupled_engine.srt.layers.attention.dsa.dsa_indexer import *  # noqa: F401, F403

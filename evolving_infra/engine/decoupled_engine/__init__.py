# Adapted from SGLang python/sglang/__init__.py (commit 9a489f8d2f), modified for evolving-infra.
"""DecoupledEngine runtime API.

Derived from SGLang's ``python/sglang`` (see engine/README.md). The SGLang
frontend language (``sglang.lang``) is not included; only the runtime API is.
"""

import platform as _platform
import sys as _sys

# decoupled_engine.srt.environ must run before the rest of this file's imports
# (hf_transformers_patches, ...), which pull in torch and FlashInfer: those
# claim these cache dirs early, and the first value set is the one that sticks.
# Safe here -- environ has no heavy dependency (no torch).
from decoupled_engine.srt.environ import (
    redirect_third_party_caches as _redirect_third_party_caches,
)

_redirect_third_party_caches()

if _sys.platform == "darwin" and _platform.machine() == "arm64":
    from decoupled_engine._platform_stubs import (
        install_platform_stubs as _install_platform_stubs,
    )

    _install_platform_stubs()

from decoupled_engine.srt.utils.hf_transformers_patches import (
    apply_all as _apply_hf_patches,
)

_apply_hf_patches()

from decoupled_engine.utils import LazyImport
from decoupled_engine.version import __version__

# Runtime API
ServerArgs = LazyImport("decoupled_engine.srt.server_args", "ServerArgs")
Engine = LazyImport("decoupled_engine.srt.entrypoints.engine", "Engine")

__all__ = [
    "Engine",
    "ServerArgs",
    "__version__",
]

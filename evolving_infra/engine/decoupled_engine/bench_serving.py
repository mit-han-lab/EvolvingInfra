"""Deprecated import path for ``decoupled_engine.benchmark.serving``.

``python -m decoupled_engine.bench_serving`` and ``from decoupled_engine.bench_serving import ...``
still work, but the implementation now lives in ``decoupled_engine.benchmark.serving``.
Update references to the new path.
"""

import warnings

from decoupled_engine.benchmark.serving import *  # noqa: F401,F403
from decoupled_engine.benchmark.serving import cli_main

warnings.warn(
    "`decoupled_engine.bench_serving` is deprecated and will be removed in a future "
    "release; use `decoupled_engine.benchmark.serving` instead "
    "(e.g. `python -m decoupled_engine.benchmark.serving`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()

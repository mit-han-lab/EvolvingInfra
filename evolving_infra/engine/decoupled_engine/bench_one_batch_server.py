"""Deprecated import path for ``decoupled_engine.benchmark.one_batch_server``.

``python -m decoupled_engine.bench_one_batch_server`` and
``from decoupled_engine.bench_one_batch_server import ...`` still work, but the
implementation now lives in ``decoupled_engine.benchmark.one_batch_server``.
Update references to the new path.
"""

import warnings

from decoupled_engine.benchmark.one_batch_server import *  # noqa: F401,F403
from decoupled_engine.benchmark.one_batch_server import cli_main

warnings.warn(
    "`decoupled_engine.bench_one_batch_server` is deprecated and will be removed in a "
    "future release; use `decoupled_engine.benchmark.one_batch_server` instead "
    "(e.g. `python -m decoupled_engine.benchmark.one_batch_server`).",
    FutureWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    cli_main()

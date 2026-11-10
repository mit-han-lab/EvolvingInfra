# Adapted from SGLang python/sglang/launch_server.py (commit 9a489f8d2f), modified for evolving-infra.
"""Launch the inference server."""

import asyncio
import os
import sys

from decoupled_engine.srt.arg_groups.overrides import resolving_view
from decoupled_engine.srt.plugins import load_plugins
from decoupled_engine.srt.server_args import prepare_server_args
from decoupled_engine.srt.utils import kill_process_tree
from decoupled_engine.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    # The flags dispatched on below are decided by resolution (`--grpc-mode`
    # folds into `smg_grpc_mode`), and `prepare_server_args` returns raw input.
    server_args.resolve_once()
    cfg = resolving_view(server_args)

    if cfg.encoder_only:
        # For encoder disaggregation
        if cfg.smg_grpc_mode or cfg.grpc_mode:
            from decoupled_engine.srt.disaggregation.encoder.grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from decoupled_engine.srt.disaggregation.encoder.http_server import launch_server

            launch_server(server_args)
    elif cfg.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). The native Rust gRPC
        # server is a separate path, enabled by --grpc-port, that starts
        # alongside the default HTTP server below.
        from decoupled_engine.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif cfg.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from decoupled_engine.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'decoupled-engine[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from decoupled_engine.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


def main(argv=None):
    """Console script ``decoupled-engine``; same as ``python -m decoupled_engine.launch_server``."""
    load_plugins()

    server_args = prepare_server_args(sys.argv[1:] if argv is None else argv)

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()

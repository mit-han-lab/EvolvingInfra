# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Planner command line: ``python -m evolving_infra.planner <command> ...``.

Commands:
  control               run the planner (Router needs --symmetric-pd-external-role-control); every launch
                        parameter is a flag with a default, see ``control --help``
  fit                   learning-end fit; started by ``control`` as a subprocess with ``--config
                        <run_dir>/planner_args.json``, runnable alone with the same flags
  serve-traces          per worker host: serve scheduler trace files and the host clock to the planner
  prior REPORT          print a prefill shape prior rebuilt from a fit report
"""
import argparse
import json
from pathlib import Path
import sys
import threading

from evolving_infra.planner.planner_args import PlannerArgs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evolving_infra.planner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("control", "Run the planner until it is stopped."),
                       ("fit", "Run the learning-end fit and write <run_dir>/calibration/report.json.")):
        PlannerArgs.add_cli_args(sub.add_parser(name, help=text, description=text))
    p = sub.add_parser("serve-traces", help="Serve trace files and /time on one worker host.",
                       description="Read-only HTTP server: GET /time, /list?dir=, /read?path=&offset= under ROOT.")
    p.add_argument("--root", required=True, help="parent of the workers' --sched-trace-dir directories")
    p.add_argument("--port", type=int, default=None, help="default: the planner's --trace-port default")
    p.add_argument("--host", default="0.0.0.0")
    p = sub.add_parser("prior", help="Rebuild a prefill shape prior from a fit report.")
    p.add_argument("report", type=Path)
    return parser


def main(argv=None):
    namespace = build_parser().parse_args(argv)
    if namespace.command == "prior":
        from evolving_infra.planner.perf_model.priors import build_prior
        print(json.dumps(build_prior(json.loads(namespace.report.read_text()))))
    elif namespace.command == "serve-traces":
        from evolving_infra.planner.observability.trace_server import TRACE_PORT, serve_traces
        server = serve_traces(namespace.root, namespace.port or TRACE_PORT, namespace.host)
        print(f"serving {Path(namespace.root).resolve()} on {namespace.host}:{server.server_port}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
    else:
        args = PlannerArgs.from_cli_args(namespace).check()
        if namespace.command == "control":
            from evolving_infra.planner.managers.planner_manager import run_planner
            run_planner(args)
        else:
            from evolving_infra.planner.perf_model.calibration import calibrate
            calibrate(args)


if __name__ == "__main__":
    sys.exit(main())

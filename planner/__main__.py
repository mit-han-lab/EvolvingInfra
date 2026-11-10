"""Planner command line: ``python -m evolving_infra.planner <command> ...``.

Commands:
  control CONFIG        run the live P/D role controller (Router needs --symmetric-pd-external-role-control)
  fit CONFIG            learning-end fit; started by ``control`` as a subprocess, runnable alone
  serve-traces          per worker host: serve scheduler trace files and the host clock to the controller
  prior REPORT          print a prefill shape prior rebuilt from a fit report
  calibrate             one-window P/D ratio calibration from a trace manifest
  node-plan             fixed per-node P/D plan from a completed learning capture
"""
import argparse
import json
from pathlib import Path
import sys
import threading


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evolving_infra.planner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("control", "Run the live P/D role controller."),
                       ("fit", "Run the learning-end fit and write <run_dir>/calibration/report.json.")):
        p = sub.add_parser(name, help=text, description=text)
        p.add_argument("config", type=Path, help="controller JSON (see planner/README.md)")
    p = sub.add_parser("serve-traces", help="Serve trace files and /time on one worker host.",
                       description="Read-only HTTP server: GET /time, /list?dir=, /read?path=&offset= under ROOT.")
    p.add_argument("--root", required=True, help="parent of the workers' --sched-trace-dir directories")
    p.add_argument("--port", type=int, default=None, help="default: loop.TRACE_PORT")
    p.add_argument("--host", default="0.0.0.0")
    p = sub.add_parser("prior", help="Rebuild a prefill shape prior from a fit report.")
    p.add_argument("report", type=Path)
    from evolving_infra.planner.calibration import __main__ as calibrate
    from evolving_infra.planner import node_plan
    calibrate.add_arguments(sub.add_parser("calibrate", help="One-window P/D ratio calibration.",
                                           description=calibrate.__doc__))
    node_plan.add_arguments(sub.add_parser("node-plan", help="Fixed per-node P/D plan from a learning capture.",
                                           description=node_plan.__doc__))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        from evolving_infra.planner.calibration import __main__ as calibrate
        return calibrate.run(args)
    if args.command == "node-plan":
        from evolving_infra.planner import node_plan
        return node_plan.run(args)
    from evolving_infra.planner import loop
    if args.command == "prior":
        print(json.dumps(loop.build_prior(json.loads(args.report.read_text()))))
    elif args.command == "serve-traces":
        server = loop.serve_traces(args.root, args.port or loop.TRACE_PORT, args.host)
        print(f"serving {Path(args.root).resolve()} on {args.host}:{server.server_port}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
    else:
        cfg = loop.load_config(args.config)
        {"control": loop.control, "fit": loop.fit}[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())

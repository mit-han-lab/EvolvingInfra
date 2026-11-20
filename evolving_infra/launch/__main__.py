# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""``evolving-infra serve``: start the stack on one host, or this host's part of a multi-host stack."""
import argparse
from pathlib import Path
import socket
import sys

import yaml

from evolving_infra.launch import hardware
from evolving_infra.launch.serve import Stack, _argument_groups, merge_args


def parse(argv=None):
    parser = argparse.ArgumentParser(prog="evolving-infra")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the engines, KV directory, frontend, router and planner")
    serve.add_argument("--config", help="YAML file with the model and its settings (examples/)")
    serve.add_argument("--model-path", help="the model (overrides the config)")
    serve.add_argument("--hosts", help="comma-separated hosts, the first runs the router (default: this host)")
    serve.add_argument("--node-rank", type=int, help="this host's position in --hosts (default: found by name)")
    serve.add_argument("--gpus-per-worker", type=int,
                       help="GPUs of one engine (default: all GPUs of a host; half of them on one host)")
    serve.add_argument("--prefill-workers", type=int, help="workers that start as prefill (default: half)")
    serve.add_argument("--planner", action="store_true", help="change the P/D split while serving")
    serve.add_argument("--work-dir", default="evolving-infra-run", help="logs and generated configs")
    return parser.parse_args(argv)


def load(path: Path) -> dict:
    """A config file; ``base: other.yaml`` starts from that file. Engine and router options given here
    replace the base's options of the same name, together with their values, and keep the others;
    mappings (``engine_env``, ``frontend``, ``gateway``, ``planner``) are merged key by key, and each
    role of ``roles`` the same way; any other key replaces the base's."""
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" not in cfg:
        return cfg
    out = load(path.parent / cfg.pop("base"))
    for key in ("engine_args", "router_args"):
        mine = cfg.pop(key, [])
        out[key] = merge_args(out.get(key, []), mine)
    for key in ("engine_env", "frontend", "gateway", "planner"):
        if isinstance(cfg.get(key), dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **cfg.pop(key)}
    if isinstance(cfg.get("roles"), dict) and isinstance(out.get("roles"), dict):
        roles = dict(out["roles"])
        for role, mine in cfg.pop("roles").items():
            theirs = roles.get(role) or {}
            roles[role] = {"engine_args": merge_args(list(theirs.get("engine_args", [])),
                                                     list((mine or {}).get("engine_args", []))),
                           "engine_env": {**(theirs.get("engine_env") or {}), **((mine or {}).get("engine_env") or {})}}
        out["roles"] = roles
    out.update(cfg)
    return out


def _absolute_file(value: str) -> str:
    return str(Path(value).resolve()) if not value.startswith("/") and Path(value).is_file() else value


def absolute(flags: list) -> list:
    """Option values that name a file relative to the current directory, with that file's full
    path, in both `--name=value` and `--name value` form: the processes run in the work directory."""
    out = []
    for group in _argument_groups(flags):
        name, eq, value = group[0].partition("=")
        out.append(f"{name}={_absolute_file(value)}" if eq else group[0])
        out.extend(_absolute_file(value) for value in group[1:])
    return out


def config(args) -> dict:
    cfg = load(Path(args.config)) if args.config else {}
    for key in ("engine_args", "router_args"):
        if cfg.get(key):
            cfg[key] = absolute(cfg[key])
    roles = cfg.get("roles") or {}
    unknown = set(roles) - {"prefill", "decode"}
    if unknown:
        sys.exit(f"roles: only prefill and decode, not {', '.join(sorted(unknown))}")
    for role in roles.values():
        if role and role.get("engine_args"):
            role["engine_args"] = absolute(role["engine_args"])
    if roles and args.planner:
        # A worker keeps the settings of its starting role; the planner changes roles.
        sys.exit("roles: per-role engine settings need fixed roles; remove them or drop --planner")
    if args.model_path:
        cfg["model"] = args.model_path
    if not cfg.get("model"):
        sys.exit("give --model-path or a --config with model")
    cfg.setdefault("served_model_name", Path(cfg["model"]).name.lower())
    cfg.setdefault("page_size", 64)
    cfg.setdefault("engine_args", ["--trust-remote-code"])
    for key in ("gpus_per_worker", "prefill_workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    # The config's planner section holds the planner's arguments; --planner turns it on.
    planner = cfg.pop("planner", None) or {}
    if args.planner:
        cfg["planner"] = planner
    return cfg


def main(argv=None):
    args = parse(argv)
    hosts = args.hosts.split(",") if args.hosts else [socket.gethostname()]
    rank = args.node_rank if args.node_rank is not None else hardware.own_index(hosts)
    Stack(config(args), hosts, rank, Path(args.work_dir).resolve()).run()


if __name__ == "__main__":
    main()

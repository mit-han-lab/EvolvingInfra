#!/usr/bin/env python3
"""Run an unchanged conformance probe and retain native replies, including on failure."""
import argparse
import inspect
import json
from pathlib import Path
import runpy
import sys
import threading
import time


def capture(script, output, arguments):
    # A unique output is required: never overwrite an earlier attempt's evidence.
    with output.open("x") as journal:
        loaded = runpy.run_path(str(script), run_name="captured_conformance")
        namespace = loaded["main"].__globals__
        original = namespace["_native_generate"]
        signature = inspect.signature(original)
        lock = threading.Lock()

        def recorded(*args, **kwargs):
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            started = time.time_ns()
            reply = original(*args, **kwargs)
            event = {
                "started_ns": started, "finished_ns": time.time_ns(),
                "label": call.arguments.get("label"),
                "prompt_ids": call.arguments.get("prompt_ids"),
                "max_new_tokens": call.arguments.get("max_new_tokens"),
                "forced": call.arguments.get("forced"),
                "response": reply,
            }
            with lock:
                journal.write(json.dumps(event, ensure_ascii=False) + "\n")
                journal.flush()
            return reply

        namespace["_native_generate"] = recorded
        saved_argv = sys.argv
        sys.argv = [str(script), *arguments]
        try:
            return loaded["main"]()
        finally:
            sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("probe_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.probe_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return capture(args.script, args.event_log, arguments)


if __name__ == "__main__":
    raise SystemExit(main())

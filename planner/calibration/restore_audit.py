"""Observation-time uncertainty audit, independent of fitted model versions."""
from collections import defaultdict
import gzip
import json
import math


class CounterBracketError(ValueError):
    """A window boundary has no sufficiently narrow counter observations."""


def include_scrape_collection_time(windows, telemetry, replicas, clock_pad_s, *,
                                   exclude_unbracketed=False):
    """Conservative counter bounds, including complete HTTP observation time.

    Both telemetry timestamps are controller time. A measured bound on the
    absolute worker/controller offset widens both ends to the worker time axis.
    This does not correct step assignment to the global learning window.
    All observations are validated before changing any caller-owned windows.
    A live caller may exclude windows with unusable time brackets. Malformed
    observations, counter resets, and invalid window intervals still fail.
    """
    if type(replicas) is not int or replicas < 1:
        raise ValueError("Require audited restore-counter replica count")
    if (isinstance(clock_pad_s, bool) or not math.isfinite(clock_pad_s)
            or clock_pad_s < 0):
        raise ValueError("Require a finite nonnegative clock padding")
    if type(exclude_unbracketed) is not bool:
        raise ValueError("Require an explicit window exclusion policy")
    observations = defaultdict(list)
    with gzip.open(telemetry, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for node in record["nodes"]:
                if node["role"] != "prefill" or node.get("errors"):
                    continue
                metrics = node.get("metrics", "")
                if "# HELP sglang:hicache_backup_bytes_total " not in metrics:
                    continue
                values = [float(s.split()[-1])/replicas for s in metrics.splitlines()
                          if s.startswith("sglang:load_back_tokens_total{")
                          and 'pool="kv"' in s]
                a, b, value = node["at"], record["at"], sum(values)
                if (len(values) > 1 or not all(math.isfinite(x) for x in (a, b, value))
                        or b < a or value < 0):
                    raise ValueError("Invalid counter collection interval or value")
                entries = observations[f"node{node['node']}"]
                current = (a-clock_pad_s, b+clock_pad_s, value)
                if entries and (current[0] <= entries[-1][0]
                                or current[1] <= entries[-1][1]
                                or value < entries[-1][2]):
                    raise ValueError("Restore counter reset or unordered collection intervals")
                entries.append(current)

    changes, accepted, excluded = [], [], []
    for window in windows:
        if (not all(math.isfinite(window[k]) for k in ("start", "end"))
                or window["end"] <= window["start"]):
            raise ValueError("Invalid capacity window")
        entries = observations[window["node"]]

        def bracket(timestamp):
            before = [entry for entry in entries if entry[1] <= timestamp]
            after = [entry for entry in entries if entry[0] >= timestamp]
            if not before or not after:
                raise CounterBracketError("Missing full-collection counter time brackets")
            left, right = before[-1], after[0]
            # Include the entire possible observation horizon, not just the
            # unexpanded node.at difference used by the legacy point estimate.
            if right[1]-left[0] >= 10:
                raise CounterBracketError("Full-collection counter bracket gap >=10s")
            return left[2], right[2]

        try:
            alo, ahi = bracket(window["start"])
            blo, bhi = bracket(window["end"])
        except CounterBracketError as error:
            if not exclude_unbracketed:
                raise
            excluded.append(dict(node=window['node'], start=window['start'],
                                 end=window['end'], reason=str(error)))
            continue
        accepted.append(window)
        changes.append((window, dict(restore_lo=max(0, blo-ahi), restore_hi=bhi-alo)))
    for window, change in changes:
        window.update(change)
    if exclude_unbracketed:
        windows[:] = accepted
    return excluded


"""Restored-token counters of the P workers: scraped by the controller, read back by the fit.

Every P engine exports ``sglang:load_back_tokens_total{pool="kv"}``, the tokens it loaded back from host memory into
GPU memory. Every TP rank increments it, so values are divided by the TP size (``replicas``). ``Telemetry`` writes
one record per worker scrape with its own collection interval. The fit reads the records back as one value per
scrape (``read_restore_counters``, ``counter_bracket``) and as conservative bounds that include the whole HTTP
collection time (``include_scrape_collection_time``).
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import bisect
import gzip
import json
import math
from pathlib import Path
import threading
import time

from evolving_infra.planner import envs
from evolving_infra.planner.utils import open_url

BACKUP_HELP = "# HELP sglang:hicache_backup_bytes_total "
RESTORE_COUNTER = "sglang:load_back_tokens_total{"


class Telemetry(threading.Thread):
    """Restore counters of the current P workers every period (restore-observations.jsonl, one record per
    worker scrape with its own collection interval)."""

    KEEP = (BACKUP_HELP, RESTORE_COUNTER)

    def __init__(self, workers: list[dict], path: Path, period_s: float):
        super().__init__(daemon=True)
        self.workers, self.path, self.period = workers, path, period_s
        self.roles: dict = {}  # url -> role, refreshed by the control loop
        self.halt, self.error = threading.Event(), None

    def scrape(self, w):
        began = time.time()
        node = dict(node=w["idx"], role="prefill", at=began, errors={})
        try:
            with open_url(w["url"] + "/metrics", envs.EVOLVING_INFRA_PLANNER_METRICS_TIMEOUT_SECS) as r:
                text = r.read().decode(errors="replace")
            node["metrics"] = "\n".join(x for x in text.splitlines() if x.startswith(self.KEEP)) + "\n"
        except Exception as error:  # noqa: BLE001 - recorded; the fit excludes windows without brackets
            node.update(metrics="", errors=dict(metrics=repr(error)))
        return dict(at=time.time(), nodes=[node])

    def run(self):
        try:
            with self.path.open("a", buffering=1) as out, ThreadPoolExecutor(len(self.workers)) as pool:
                while not self.halt.is_set():
                    began = time.monotonic()
                    current = [w for w in self.workers if self.roles.get(w["url"]) == "prefill"]
                    for row in pool.map(self.scrape, current):
                        out.write(json.dumps(row) + "\n")
                    self.halt.wait(max(0., self.period - (time.monotonic() - began)))
        except Exception as error:  # noqa: BLE001 - raised by the control loop
            self.error = error


def counter_replicas(capacities):
    """The restore counter's replica count: the TP size, which every worker must share (worker-capacities.json)."""
    sizes = {c["tp_size"] for c in capacities.values()}
    if len(sizes) != 1:
        raise ValueError(f"Workers report different TP sizes: {sorted(sizes)}")
    return sizes.pop()


def restore_counter_rows(path, replicas):
    """(scrape, record, kv-pool restore counter values / replicas) per usable P scrape of a
    restore-observations .jsonl.gz file.

    Every TP rank exports the counter, so `replicas` must be audited, never inferred from values.
    Each caller reads the scrape's node and times and applies its own checks, in its own order.
    """
    if not isinstance(replicas, int) or replicas < 1:
        raise ValueError("Require audited restore-counter replica count")
    with gzip.open(path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            for node in record["nodes"]:
                if node["role"] != "prefill" or node.get("errors"):
                    continue
                metrics = node.get("metrics", "")
                # A labeled Counter child first appears when inc() is called.
                # Verify this collector through its already-active backup series.
                if BACKUP_HELP not in metrics:
                    continue
                values = [float(s.split()[-1])/replicas for s in metrics.splitlines()
                          if s.startswith(RESTORE_COUNTER) and 'pool="kv"' in s]
                yield node, record, values


def read_restore_counters(path, replicas):
    """(scrape start, restore counter) samples per P node of a restore-observations file."""
    meters = defaultdict(list)
    for node, _, values in restore_counter_rows(path, replicas):
        if len(values) > 1:
            raise ValueError("Ambiguous restore counter series")
        meters[f"node{node['node']}"].append((node["at"], sum(values)))
    for values in meters.values():
        if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(values, values[1:])):
            raise ValueError("Restore counter reset or unordered scrapes")
    return meters


def counter_bracket(samples, timestamp):
    index = bisect.bisect_right([s[0] for s in samples], timestamp)
    if not 0 < index < len(samples):
        raise ValueError("Missing restore-counter time brackets")
    (a, ca), (b, cb) = samples[index-1:index+1]
    if not 0 < b-a < 10 or cb < ca:
        raise ValueError("Restore counter reset or scrape gap >=10s")
    return ca+(cb-ca)*(timestamp-a)/(b-a), ca, cb, b-a


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
    for node, record, values in restore_counter_rows(telemetry, replicas):
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

"""Clock audit of the worker hosts against the controller, over the trace servers' ``GET /time``."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

from ..common import save
from .trace_server import probe


class Clock:
    """Clock audit over HTTP: a fixed offset per host measured at start; every later exchange must stay
    within clock_pad_s of it, and exchanges must bracket each captured interval."""

    def __init__(self, cfg: dict, directory: Path):
        self.hosts, self.port, self.pad = cfg["hosts"], cfg["trace_port"], cfg["clock_pad_s"]
        self.dir = directory / "diagnostics"
        self.reference = directory / "clock-reference.json"

    def observe(self) -> dict:
        with ThreadPoolExecutor(len(self.hosts)) as pool:
            nodes = list(pool.map(lambda h: probe(h, self.port), self.hosts))
        row = dict(at=time.time(), nodes=nodes)
        if not self.reference.exists():
            save(self.reference, dict(at=row["at"], offsets_s={str(n["node"]): n["worker_minus_controller_s"]
                                                              for n in nodes}))
        self.validate(row)
        save(self.dir / f"clock-offsets-{time.time_ns()}.json", row)
        return row

    def offsets(self) -> dict:
        return {int(k): v for k, v in json.loads(self.reference.read_text())["offsets_s"].items()}

    def validate(self, obs: dict):
        reference = self.offsets()
        if {n["node"] for n in obs["nodes"]} != set(reference):
            raise ValueError("Clock host coverage differs")
        for n in obs["nodes"]:
            lo, hi, shift = n["lower_s"], n["upper_s"], reference[n["node"]]
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi or max(abs(lo - shift), abs(hi - shift)) > self.pad:
                raise ValueError(f"Host {n['node']} clock residual exceeds {self.pad} s: {lo - shift:.4f}..{hi - shift:.4f}")

    def verify(self, start: float, end: float, max_gap_s: float = 360) -> dict:
        rows = [json.loads(p.read_text()) for p in sorted(self.dir.glob("clock-offsets-*.json"))]
        for row in rows:
            self.validate(row)
        gap_seen = 0.
        for host in self.hosts:
            ex = sorted((n for r in rows for n in r["nodes"] if n["node"] == host["index"]),
                        key=lambda n: n["controller_before_ns"])
            before = [n for n in ex if n["controller_after_ns"] / 1e9 <= start]
            after = [n for n in ex if n["controller_before_ns"] / 1e9 >= end]
            if not before or not after:
                raise ValueError("Per-host clock exchanges must bracket the measurement")
            part = ex[ex.index(before[-1]):ex.index(after[0]) + 1]
            gap = max((b["controller_after_ns"] - a["controller_before_ns"]) / 1e9 for a, b in zip(part, part[1:]))
            gap_seen = max(gap_seen, gap)
            if gap > max_gap_s:
                raise ValueError(f"Host {host['index']} clock observation gap {gap:.0f} s > {max_gap_s} s")
        return dict(passed=True, pad_s=self.pad, offsets_s=self.offsets(), observations=len(rows),
                    max_observation_gap_s=gap_seen)

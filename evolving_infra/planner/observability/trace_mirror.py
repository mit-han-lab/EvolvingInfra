# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The controller's copies of every worker's scheduler traces, read from the hosts' trace servers."""
from pathlib import Path
import re
import urllib.parse

from evolving_infra.planner import envs
from evolving_infra.planner.utils import http_json, open_url


class Mirror:
    """Append-only local copies of each worker's iter/req trace stream (one stream per worker: a second
    pid file means the engine restarted, which the fit must not mix). `cursors` keeps, per captured file,
    what earlier captures already read (trace_snapshot.records_before), so a capture reads only new bytes."""

    def __init__(self, root: Path, workers: list[dict]):
        self.root, self.workers = Path(root), {w["node"]: w for w in workers}
        self.cursors: dict = {}

    def sync(self, nodes) -> dict:
        out = {}
        for node in nodes:
            w = self.workers[node]
            names = http_json(f"{w['trace_url']}/list?dir={w['name']}",
                              timeout=envs.EVOLVING_INFRA_PLANNER_TRACE_LIST_TIMEOUT_SECS)
            item = {}
            for kind, field in (("iter", "steps"), ("req", "requests")):
                found = [n for n in names if re.fullmatch(rf"{kind}_trace\.dp0_pp0\.pid\d+\.jsonl", n)]
                if len(found) != 1:
                    raise ValueError(f"{node}: expected one {kind} trace stream, found {found}")
                local = self.root / node / found[0]
                local.parent.mkdir(parents=True, exist_ok=True)
                while True:
                    offset = local.stat().st_size if local.exists() else 0
                    query = urllib.parse.urlencode({"path": f"{w['name']}/{found[0]}", "offset": offset})
                    with open_url(f"{w['trace_url']}/read?{query}",
                                  envs.EVOLVING_INFRA_PLANNER_TRACE_READ_TIMEOUT_SECS) as r:
                        data = r.read()
                    with local.open("ab") as f:
                        f.write(data)
                    if len(data) < 64 << 20:
                        break
                item[field] = str(local)
            out[node] = item
        return out

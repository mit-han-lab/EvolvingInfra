"""The trace server on every worker host, and the controller's clients of it.

Every worker host runs ``python -m evolving_infra.planner serve-traces`` over the parent of the engines'
``--sched-trace-dir`` directories. The controller copies each worker's iteration/request traces from it over HTTP
(``Mirror``) and measures the host clock with ``GET /time`` (``probe``); no shared filesystem or SSH is needed.
"""
import http.client
import http.server
import json
from pathlib import Path
import re
import threading
import time
import urllib.parse

from ..common import http_json, open_url

TRACE_PORT = 18500  # per host: scheduler trace files + /time


def serve_traces(root: str, port: int = TRACE_PORT, host: str = "0.0.0.0"):
    """Every host: GET /time -> {"time_ns"}; GET /list?dir=<worker> -> file names; GET /read?path=&offset=
    -> the file's bytes from offset (incremental mirror). Read-only, inside `root` only."""
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, body: bytes, kind="application/octet-stream", code=200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            query = dict(urllib.parse.parse_qsl(url.query))
            if url.path == "/time":
                return self.reply(json.dumps({"time_ns": time.time_ns()}).encode(), "application/json")
            target = (base / query.get("dir", query.get("path", ""))).resolve()
            if base not in (target, *target.parents):
                return self.reply(b"outside root", code=403)
            if url.path == "/list":
                names = sorted(p.name for p in target.iterdir()) if target.is_dir() else []
                return self.reply(json.dumps(names).encode(), "application/json")
            if url.path == "/read" and target.is_file():
                with target.open("rb") as f:
                    f.seek(int(query.get("offset", 0)))
                    return self.reply(f.read(64 << 20))
            return self.reply(b"not found", code=404)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Mirror:
    """Append-only local copies of each worker's iter/req trace stream (one stream per worker: a second
    pid file means the engine restarted, which the fit must not mix)."""

    def __init__(self, root: Path, workers: list[dict]):
        self.root, self.workers = Path(root), {w["node"]: w for w in workers}

    def sync(self, nodes) -> dict:
        out = {}
        for node in nodes:
            w = self.workers[node]
            names = http_json(f"{w['trace_url']}/list?dir={w['name']}", timeout=30)
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
                    with open_url(f"{w['trace_url']}/read?{query}", 120) as r:
                        data = r.read()
                    with local.open("ab") as f:
                        f.write(data)
                    if len(data) < 64 << 20:
                        break
                item[field] = str(local)
            out[node] = item
        return out


def probe(host: dict, port: int, exchanges: int = 8) -> dict:
    """One host's clock against this controller (best of `exchanges` HTTP keep-alive round trips)."""
    conn = http.client.HTTPConnection(host["ip"], port, timeout=15)
    samples = []
    try:
        for _ in range(exchanges):
            before = time.time_ns()
            conn.request("GET", "/time")
            remote = json.loads(conn.getresponse().read())["time_ns"]
            after = time.time_ns()
            samples.append(dict(roundtrip_s=(after - before) / 1e9, controller_before_ns=before,
                                controller_after_ns=after,
                                worker_minus_controller_s=(remote - (before + after) / 2) / 1e9,
                                lower_s=(remote - after) / 1e9, upper_s=(remote - before) / 1e9))
    finally:
        conn.close()
    best = min(samples, key=lambda s: s["roundtrip_s"])
    return dict(node=host["index"], host=host["ip"], **best, samples=samples)

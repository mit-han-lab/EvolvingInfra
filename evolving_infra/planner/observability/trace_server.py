"""The trace server on every worker host.

Every worker host runs ``python -m evolving_infra.planner serve-traces`` over the parent of the engines'
``--sched-trace-dir`` directories. The controller copies each worker's iteration/request traces from it over HTTP
(``trace_mirror``) and measures the host clock with ``GET /time`` (``clock_sync``); no shared filesystem or SSH is
needed. It imports no numerical library.
"""
import http.server
import json
from pathlib import Path
import threading
import time
import urllib.parse

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

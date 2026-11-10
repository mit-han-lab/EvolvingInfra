"""The per-host trace server and the controller's incremental copies of its files."""
from pathlib import Path

import pytest

from evolving_infra.planner.collect.trace_server import Mirror, serve_traces
from evolving_infra.planner.common import http_json


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mirror_appends_only_new_bytes(tmp_path):
    root = tmp_path / "host"
    (root / "w00").mkdir(parents=True)
    trace = root / "w00" / "iter_trace.dp0_pp0.pid7.jsonl"
    req = root / "w00" / "req_trace.dp0_pp0.pid7.jsonl"
    trace.write_bytes(b'{"a":1}\n{"a":2')
    req.write_bytes(b"")
    port = free_port()
    server = serve_traces(str(root), port)
    try:
        worker = dict(node="node0", name="w00", trace_url=f"http://127.0.0.1:{port}")
        mirror = Mirror(tmp_path / "mirror", [worker])
        paths = mirror.sync(["node0"])["node0"]
        with trace.open("ab") as f:
            f.write(b'}\n{"a":3}\n')
        mirror.sync(["node0"])
        assert Path(paths["steps"]).read_bytes() == trace.read_bytes()
        (root / "w00" / "iter_trace.dp0_pp0.pid8.jsonl").write_bytes(b"")  # engine restarted
        with pytest.raises(ValueError, match="expected one iter"):
            mirror.sync(["node0"])
        assert http_json(f"http://127.0.0.1:{port}/time")["time_ns"] > 0
    finally:
        server.shutdown()

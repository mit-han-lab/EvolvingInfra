"""Restored-token counters: what the scraper records and how the fit reads them back."""
import http.server
import threading

import pytest

from evolving_infra.planner.observability.restore_metrics import Telemetry, counter_bracket, read_restore_counters

from planner_support import restore_record, write_telemetry


def write_scrapes(path, points):
    """(scrape time, counter) of P node 0; a zero counter is not exported yet."""
    write_telemetry(path, [restore_record(at, counter or None) for at, counter in points])


def test_actual_restore_counter_normalizes_rank_replicas_and_brackets(tmp_path):
    path = tmp_path / "telemetry.gz"
    write_scrapes(path, [(0, 0), (2, 128), (4, 256)])
    observed = read_restore_counters(path, 4)["node0"]
    assert observed == [(0, 0), (2, 32), (4, 64)]
    assert counter_bracket(observed, 1) == (16, 0, 32, 2)
    with pytest.raises(ValueError, match="Missing"):
        counter_bracket(observed, 5)
    with pytest.raises(ValueError, match="gap"):
        counter_bracket([(0, 0), (11, 100)], 5)
    write_scrapes(path, [(0, 128), (2, 0)])
    with pytest.raises(ValueError, match="reset"):
        read_restore_counters(path, 4)


def test_telemetry_rows_parse_as_restore_counters(tmp_path):
    body = ("# HELP sglang:hicache_backup_bytes_total Bytes.\n# TYPE x counter\n"
            'sglang:load_back_tokens_total{model_name="m",pool="kv"} 400.0\n'
            'sglang:load_back_tokens_total{model_name="m",pool="swa"} 8.0\nsglang:other 1\n').encode()

    class Metrics(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Metrics)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        t = Telemetry([dict(idx=3, url=f"http://127.0.0.1:{server.server_port}")], tmp_path / "r.jsonl", 3)
        rows = [t.scrape(t.workers[0]) for _ in range(2)]
    finally:
        server.shutdown()
    write_telemetry(tmp_path / "r.jsonl.gz", rows)
    meters = read_restore_counters(tmp_path / "r.jsonl.gz", 4)
    assert [v for _, v in meters["node3"]] == [100.0, 100.0]  # kv pool only, / 4 TP ranks

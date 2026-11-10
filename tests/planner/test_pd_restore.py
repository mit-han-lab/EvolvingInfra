import gzip
import json

import pytest

from evolving_infra.planner.calibration.restore import (
    build_restore_windows,
    counter_bracket,
    prediction_errors,
    read_restore_counters,
)


def write_scrapes(path, points):
    with gzip.open(path, "wt") as stream:
        for timestamp, counter in points:
            metrics = "# HELP sglang:hicache_backup_bytes_total Backup bytes\n"
            if counter:
                metrics += f'sglang:load_back_tokens_total{{cache_type="UnifiedRadixCache",pool="kv"}} {counter}\n'
            stream.write(json.dumps(dict(nodes=[dict(node=0, role="prefill", errors=[],
                                                    at=timestamp, metrics=metrics)]))+"\n")


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


def test_wall_windows_clip_busy_period_and_do_not_use_bad_cache_tiers(tmp_path):
    scheduler = tmp_path / "node0/scheduler"
    scheduler.mkdir(parents=True)
    request_path = scheduler / "requests.jsonl"
    rows, requests = [], []
    for i, launch in enumerate([10, 30, 50, 70, 90, 110, 140]):
        rows.append(dict(source=str(scheduler / "steps.jsonl"), mode="extend",
            forward_iter=i, launch_bs=1, new_prefill_tokens=100,
            prefill_attention_pairs=1000, after_idle=i in (0, 6),
            ts=launch+20, dur_ms=20000))
        requests.append(dict(prefill_iters=[i], cached_tokens=10, cached_tokens_host=999))
    request_path.write_text("".join(json.dumps(r)+"\n" for r in requests))
    telemetry = tmp_path / "telemetry.gz"
    write_scrapes(telemetry, [(t, t*80) for t in range(0, 186, 5)])
    samples, eligible = build_restore_windows(rows, [request_path], telemetry, 4, 0, 180)
    assert eligible == 7 and len(samples) == 1
    sample = samples[0]
    assert (sample["start"], sample["end"], sample["seconds"]) == (60, 120, 60)
    assert sample["features"] == [3, 300, 3000, 1200]
    assert sample["requests"] == 3
    assert (sample["restore_lo"], sample["restore_hi"]) == (1100, 1300)


def test_validation_propagates_counter_edge_uncertainty():
    samples = [dict(node="node0", features=[1, 100, 1000, 100],
                    seconds=1., restore_lo=0, restore_hi=200)] * 30
    # Center prediction is exact, but the unresolved transfer count adds ±0.5s.
    errors = prediction_errors(samples, [.5, 0, 0, .005])
    assert errors["predicted_total_s"] == errors["observed_total_s"]
    assert errors["p95_relative"] == pytest.approx(.5)
    assert errors["worker_total_errors"]["node0"] == pytest.approx(.5)

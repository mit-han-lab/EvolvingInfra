import json

from trie.reporting import build_json_summary
from trie.results import BenchmarkResult, ServerMetrics


def test_build_json_summary_is_serializable_and_keeps_samples() -> None:
    result = BenchmarkResult(
        wall_time=2.0,
        expected_requests=1,
        completed_requests=1,
        completed_model_requests=2,
        latencies=[1.5],
        client_prefix_tokens=[32],
        block_aligned_prefix_tokens=[32],
        server_cached_tokens=[24],
        server_metrics=[ServerMetrics(prompt_tokens=100, cached_tokens=40)],
    )
    result._append_point(
        timestamp=2.0,
        completion_tokens=20,
        new_prompt_tokens=60,
        cached_prompt_tokens=40,
    )

    summary = build_json_summary(result, num_gpus=2)

    assert summary["completed_requests"] == 1
    assert summary["completed_prompt_tokens"] == 100
    assert summary["throughput"]["overall"]["prompt_tok_s"] == 50.0
    assert summary["samples"]["server_cached_tokens"] == [24]
    assert summary["distribution_summaries"]["cache_hit_rate"]["mean"] == 0.4
    json.dumps(summary)

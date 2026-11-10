from trie.results import BenchmarkResult


def test_replay_throughput_uses_server_reported_cached_tokens() -> None:
    result = BenchmarkResult()

    result.record_replay_request(
        timestamp=1.0,
        prompt_tokens=100,
        completion_tokens=10,
        client_prefix_tokens=80,
        block_aligned_prefix_tokens=64,
        server_cached_tokens=32,
    )

    assert result.completed_prompt_tokens == 100
    assert result.cached_prompt_tokens == 32
    assert result.new_prompt_tokens == 68


def test_replay_cached_tokens_are_clamped_to_prompt_length() -> None:
    result = BenchmarkResult()

    result.record_replay_request(
        timestamp=1.0,
        prompt_tokens=10,
        completion_tokens=1,
        client_prefix_tokens=10,
        block_aligned_prefix_tokens=0,
        server_cached_tokens=20,
    )

    assert result.cached_prompt_tokens == 10
    assert result.new_prompt_tokens == 0

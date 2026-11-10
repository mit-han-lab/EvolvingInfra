import pytest
from trie.types import AppendMessage, Generate, ReplayTrace, Wait, token_ids_sha256


def test_replay_trace_round_trip() -> None:
    trace = ReplayTrace(
        trace_id="session-1",
        initial_messages=[{"role": "user", "content": "hello"}],
        events=[
            Generate(
                max_tokens=4,
                recorded_assistant={"role": "assistant", "content": "hi"},
                recorded_prompt_token_ids=[1, 2, 3],
                recorded_prompt_token_count=3,
                recorded_prompt_token_ids_sha256=token_ids_sha256([1, 2, 3]),
                recorded_output_token_ids=[11, 12, 13, 14],
                source_request_id="request-1",
            ),
            Wait(seconds=1.5, actor="user"),
            AppendMessage(message={"role": "user", "content": "again"}),
            Generate(max_tokens=2),
        ],
        metadata={"agent": "Claude Code"},
    )

    parsed = ReplayTrace.from_jsonl_row(trace.to_jsonl_row())

    assert parsed.trace_id == trace.trace_id
    assert parsed.initial_messages == trace.initial_messages
    assert parsed.events == trace.events
    assert parsed.metadata == trace.metadata


def test_recorded_output_token_ids_must_match_generation_length() -> None:
    with pytest.raises(ValueError, match="length must equal max_tokens"):
        Generate(max_tokens=2, recorded_output_token_ids=[1])


def test_recorded_output_token_ids_must_be_non_negative_integers() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        Generate(max_tokens=1, recorded_output_token_ids=[-1])
    with pytest.raises(ValueError, match="no greater than"):
        Generate(max_tokens=1, recorded_output_token_ids=[4294967296])


def test_generate_requires_a_strict_integer_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        Generate(max_tokens=True)
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        Generate(max_tokens=1.0)


def test_recorded_assistant_requires_explicit_assistant_role() -> None:
    with pytest.raises(ValueError, match="role must be assistant"):
        Generate(max_tokens=1, recorded_assistant={"content": "missing role"})


def test_recorded_prompt_token_ids_must_be_non_empty_non_negative_integers() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        Generate(max_tokens=1, recorded_prompt_token_ids=[])
    with pytest.raises(ValueError, match="non-negative integers"):
        Generate(max_tokens=1, recorded_prompt_token_ids=[-1])
    with pytest.raises(ValueError, match="no greater than"):
        Generate(max_tokens=1, recorded_prompt_token_ids=[4294967296])
    with pytest.raises(ValueError, match="non-negative integers"):
        Generate(max_tokens=1, recorded_prompt_token_ids=(1,))


def test_compact_recorded_prompt_contract_is_complete_and_exact() -> None:
    prompt_ids = [1, 2, 3]
    Generate(
        max_tokens=1,
        recorded_prompt_token_count=3,
        recorded_prompt_token_ids_sha256=token_ids_sha256(prompt_ids),
    )
    with pytest.raises(ValueError, match="count and SHA256"):
        Generate(max_tokens=1, recorded_prompt_token_count=3)
    with pytest.raises(ValueError, match="count and SHA256"):
        Generate(
            max_tokens=1,
            recorded_prompt_token_count=3,
            recorded_prompt_token_ids_sha256="A" * 64,
        )
    with pytest.raises(ValueError, match="differ from their count/SHA256"):
        Generate(
            max_tokens=1,
            recorded_prompt_token_ids=prompt_ids,
            recorded_prompt_token_count=2,
            recorded_prompt_token_ids_sha256=token_ids_sha256(prompt_ids),
        )


def test_recorded_output_token_ids_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        Generate(max_tokens=1, recorded_output_token_ids=(1,))


def test_replay_trace_requires_generate() -> None:
    with pytest.raises(ValueError, match="generate event"):
        ReplayTrace(
            trace_id="session-1",
            initial_messages=[],
            events=[AppendMessage(message={"role": "user", "content": "hello"})],
        )

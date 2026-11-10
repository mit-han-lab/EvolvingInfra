import json
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest
from trie.types import ReplayTrace

ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "schema" / "replay-trace-v2.schema.json"
FIXTURE = ROOT / "schema" / "fixtures" / "strict-replay-v2.json"


def test_strict_replay_fixture_matches_json_schema_and_runtime_types() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(fixture)
    trace = ReplayTrace.from_jsonl_row(fixture)

    assert trace.to_jsonl_row() == fixture
    assert trace.metadata["routing_key"] == "conformance-session-1"


def test_runtime_uses_packaged_schema_resource() -> None:
    packaged_schema = files("trie").joinpath(
        "schema",
        "replay-trace-v2.schema.json",
    )

    assert json.loads(packaged_schema.read_text(encoding="utf-8")) == json.loads(
        SCHEMA.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "event",
    [
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_ids": [1],
            "recorded_output_tokens_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": True,
            "recorded_prompt_token_ids": [1],
            "recorded_output_token_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": 1.0,
            "recorded_prompt_token_ids": [1],
            "recorded_output_token_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_ids": [1.0],
            "recorded_output_token_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_assistant": {"content": "missing role"},
            "recorded_prompt_token_ids": [1],
            "recorded_output_token_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_ids": [1],
            "recorded_output_token_ids": [],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_ids": [4294967296],
            "recorded_output_token_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_ids": [1],
            "recorded_output_token_ids": [4294967296],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_count": 1,
            "recorded_output_token_ids": [2],
        },
        {
            "type": "generate",
            "max_tokens": 1,
            "recorded_prompt_token_count": 1,
            "recorded_prompt_token_ids_sha256": "A" * 64,
            "recorded_output_token_ids": [2],
        },
    ],
)
def test_runtime_rejects_schema_invalid_replay_rows(event: dict[str, object]) -> None:
    row = {
        "schema_version": 2,
        "trace_id": "invalid",
        "events": [event],
    }

    with pytest.raises(ValueError, match="schema validation failed"):
        ReplayTrace.from_jsonl_row(row)


def test_runtime_rejects_unknown_top_level_fields() -> None:
    row = {
        "schema_version": 2,
        "trace_id": "invalid",
        "events": [{"type": "generate", "max_tokens": 1}],
        "unexpected": True,
    }

    with pytest.raises(ValueError, match="schema validation failed"):
        ReplayTrace.from_jsonl_row(row)


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_runtime_requires_integer_schema_version(schema_version: object) -> None:
    row = {
        "schema_version": schema_version,
        "trace_id": "invalid",
        "events": [{"type": "generate", "max_tokens": 1}],
    }

    with pytest.raises(ValueError, match="schema_version must be 2"):
        ReplayTrace.from_jsonl_row(row)


def test_runtime_enforces_output_length_after_schema_validation() -> None:
    row = {
        "schema_version": 2,
        "trace_id": "invalid",
        "events": [
            {
                "type": "generate",
                "max_tokens": 2,
                "recorded_output_token_ids": [1],
            }
        ],
    }

    with pytest.raises(ValueError, match="length must equal max_tokens"):
        ReplayTrace.from_jsonl_row(row)

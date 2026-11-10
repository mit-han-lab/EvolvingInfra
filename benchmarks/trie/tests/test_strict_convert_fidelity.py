"""Regressions for structured outputs disappearing behind valid token arrays.

Use a lossless character codec so the tests need no model download. The DSV4
serving encoder is real; full-corpus validation separately uses the real BPE.
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from strict_convert import rendering, validate
from strict_convert.builder import build_trace, new_stats
from strict_convert.canonical import CanonicalTrace, CanonicalTurn


class CharacterTokenizer:
    def __len__(self):
        return 0x110000


class CharacterManager:
    def __init__(self, path):
        self._tokenizer = CharacterTokenizer()

    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


@pytest.fixture
def renderer(tmp_path, monkeypatch):
    encoder = rendering.engine_files_module().dsv4_encoder_path()
    if not encoder.is_file():  # Trie installed without the engine tree
        pytest.skip(f"DSV4 encoder not found at {encoder}")
    monkeypatch.setattr(rendering, "tokenizer_manager_module", lambda: SimpleNamespace(TokenizerManager=CharacterManager))
    path = tmp_path / "deepseek-v4-test"
    path.mkdir()
    (path / "tokenizer.json").write_text("{}")
    return rendering.TargetRenderer(str(path))


def tool_call(name, value, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps({"x": value})}}


def make_row(renderer):
    calls = [tool_call("alpha", "first", "a"), tool_call("beta", "second", "b")]
    trace = CanonicalTrace(
        trace_id="regression",
        initial_messages=[{"role": "user", "content": "question"}],
        tools=[{"type": "function", "function": {"name": "alpha", "description": "SCHEMA_SENTINEL", "parameters": {"type": "object"}}}],
        turns=[
            CanonicalTurn(assistant={"role": "assistant", "content": "", "reasoning_content": "recorded reasoning", "tool_calls": calls}),
            CanonicalTurn(assistant={"role": "assistant", "content": "", "reasoning_content": "unfinished reasoning"},
                          context_ops=[("append", [{"role": "tool", "tool_call_id": "b", "content": "RESULT_B"}, {"role": "tool", "tool_call_id": "a", "content": "RESULT_A"}])], truncated=True),
        ],
    )
    row = build_trace(trace, renderer, "trie_text", new_stats())
    # Capture-shaped aliases; the tests below independently corrupt payloads
    # or count availability while preserving all structural schema checks.
    row["metadata"]["strict_conversion"]["original_counts"] = [
        {"event_index": i, "completion_tokens": len(e["recorded_output_token_ids"]), "prompt_tokens": len(e["recorded_prompt_token_ids"])}
        for i, e in enumerate(row["events"]) if e["type"] == "generate"
    ]
    return row


def test_native_dsv4_keeps_fields_modes_and_tool_result_order(renderer):
    row = make_row(renderer)
    events = [e for e in row["events"] if e["type"] == "generate"]
    prompt, output = [renderer.decode(events[0][f"recorded_{k}_token_ids"]) for k in ("prompt", "output")]
    assert "SCHEMA_SENTINEL" in prompt
    assert prompt.endswith("<｜Assistant｜><think>")
    assert output.startswith("recorded reasoning</think>\n\n<｜DSML｜tool_calls>")
    assert '<｜DSML｜invoke name="alpha">\n<｜DSML｜parameter name="x" string="true">first</｜DSML｜parameter>' in output
    assert output.endswith("</｜DSML｜tool_calls><｜end▁of▁sentence｜>")
    next_prompt = renderer.decode(events[1]["recorded_prompt_token_ids"])
    assert "recorded reasoning" not in next_prompt
    assert "<tool_result>RESULT_A</tool_result>\n\n<tool_result>RESULT_B</tool_result>" in next_prompt
    assert renderer.decode(events[1]["recorded_output_token_ids"]) == "unfinished reasoning"
    messages = [{"role": "developer", "content": "rules"}, {"role": "user", "content": "q"}]
    # The reference drops an earlier developer in thinking mode without
    # tools; the converter's existing role-remap policy must preserve it.
    remapped, counts = renderer.remap_roles(messages, 0)
    assert counts == {"developer->system": 1}
    assert "rules" in renderer.render(remapped, tools=None)
    assert renderer.render([messages[-1]], tools=None, add_generation_prompt=False) == "<｜begin▁of▁sentence｜><｜User｜>q"
    chat = rendering.TargetRenderer(renderer.model_path, thinking_mode="chat")
    assert "<｜User｜>rules<｜User｜>q" in chat.render(messages, tools=None)
    assert chat.render([messages[-1]], tools=None).endswith("<｜Assistant｜></think>")
    invalid = CanonicalTrace(trace_id="chat", initial_messages=[messages[-1]], turns=[CanonicalTurn(assistant={"role": "assistant", "reasoning_content": "must not disappear"})])
    assert build_trace(invalid, chat, "trie_text", new_stats()) is None


def test_validation_checks_payload_not_just_token_totals(renderer):
    row = make_row(renderer)
    validate._init_validator(renderer.model_path)
    assert validate._validate_row(row, 0, True, True)["errors"] == []
    damaged = copy.deepcopy(row)
    event = damaged["events"][0]
    text = renderer.decode(event["recorded_output_token_ids"])
    # Equal-length corruption passes all token-volume checks.
    text = text.replace("recorded reasoning", "x" * len("recorded reasoning"))
    event["recorded_output_token_ids"] = renderer.encode(text)
    assert any("loses reasoning_content" in e for e in validate._validate_row(damaged, 0, True, True)["errors"])
    damaged = copy.deepcopy(row)
    event = damaged["events"][0]
    text = renderer.decode(event["recorded_output_token_ids"]).replace('name="alpha"', 'name="TEMP"').replace('name="beta"', 'name="alpha"').replace('name="TEMP"', 'name="beta"')
    event["recorded_output_token_ids"] = renderer.encode(text)
    event["max_tokens"] = len(event["recorded_output_token_ids"])
    assert any("loses tool call" in e for e in validate._validate_row(damaged, 0, True, True)["errors"])


def test_putnam_aliases_and_missing_count_coverage(renderer, tmp_path, monkeypatch):
    row = make_row(renderer)
    path = tmp_path / "workload.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = validate.validate_workload(str(path), renderer.model_path, require_source_counts=True)
    assert result["verdict"] == "pass"
    assert result["source_fidelity"]["output_count_coverage"] == 1
    del row["metadata"]["strict_conversion"]["original_counts"][0]["completion_tokens"]
    path.write_text(json.dumps(row) + "\n")
    result = validate.validate_workload(str(path), renderer.model_path, require_source_counts=True)
    assert result["verdict"] == "fail"
    assert result["source_fidelity"]["output_count_coverage"] == .5
    assert result["source_fidelity"]["output_ratio"] == 1  # paired denominator
    row["metadata"]["strict_conversion"]["original_counts"] = []
    path.write_text(json.dumps(row) + "\n")
    result = validate.validate_workload(str(path), renderer.model_path, require_source_counts=True)
    assert result["verdict"] == "fail"
    assert result["source_fidelity"]["status"] == "unavailable"
    monkeypatch.setenv("TRIE_SKIP_ROW_VALIDATION", "1")
    with pytest.raises(ValueError, match="offline validation"):
        validate.validate_workload(str(path), renderer.model_path)

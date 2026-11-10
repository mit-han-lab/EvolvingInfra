"""Preserve completed and unfinished Kimi tool emissions across vocabularies."""

import importlib.util
import os
from pathlib import Path

import pytest

# The CUA converter needs these beyond the replay client's dependencies.
pytest.importorskip("tiktoken")
pytest.importorskip("zstandard")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cua_convert", ROOT / "scripts/convert_cua_deepseek_v4.py"
)
conversion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion)
from trie import engine_files

encoder_path = Path(os.environ.get("SGLANG_DSV4_ENCODER", engine_files.dsv4_encoder_path()))
if not encoder_path.is_file():  # Trie installed without the engine tree
    pytest.skip(f"DSV4 encoder not found at {encoder_path}", allow_module_level=True)
spec = importlib.util.spec_from_file_location("dsv4_encoding", encoder_path)
encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder)


def test_complete_tool_calls_preserve_names_arguments_and_end_token():
    raw = (
        "Find the button.</think><|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.computer_click:17<|tool_call_argument_begin|>{"x": 0.4, "y": 0.6}'
        "<|tool_call_end|><|tool_calls_section_end|><|im_end|>"
    )
    expected = encoder.render_message(
        1,
        [
            {"role": "user", "content": ""},
            {
                "role": "assistant",
                "reasoning_content": "Find the button.",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "computer_click",
                            "arguments": '{"x": 0.4, "y": 0.6}',
                        },
                    }
                ],
            },
        ],
        thinking_mode="thinking",
        drop_thinking=False,
    )
    assert conversion.translate_kimi_output(raw, encoder) == expected


@pytest.mark.parametrize(
    "raw,suffix",
    [
        ("Still reasoning", "Still reasoning"),
        (
            "Reason.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.computer_click:190",
            '<｜DSML｜invoke name="computer_click',
        ),
        (
            "Reason.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.computer_click:190"
            '<|tool_call_argument_begin|>{"x": 0.',
            '<｜DSML｜parameter name="arguments" string="true">{"x": 0.',
        ),
    ],
)
def test_truncated_output_has_no_invented_closing_tokens(raw, suffix):
    result = conversion.translate_kimi_output(raw, encoder)
    assert result.endswith(suffix)
    assert encoder.eos_token not in result
    assert "</｜DSML｜invoke>" not in result
    assert "</｜DSML｜tool_calls>" not in result
    assert "<|tool_" not in result


def test_invalid_complete_arguments_remain_in_the_official_fallback_parameter():
    raw = (
        "Reason.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.computer_click:2"
        '<|tool_call_argument_begin|>{"x":<|tool_call_end|><|tool_calls_section_end|><|im_end|>'
    )
    result = conversion.translate_kimi_output(raw, encoder)
    assert (
        '<｜DSML｜parameter name="arguments" string="true">{"x":</｜DSML｜parameter>'
        in result
    )
    assert result.endswith(encoder.eos_token)


def test_degenerate_closing_loops_remain_malformed_without_dropping_repetitions():
    raw = (
        "Reason.</think><|tool_calls_section_begin|><|tool_call_begin|>functions.computer_click:2"
        '<|tool_call_argument_begin|>{"x": 0.4}<|tool_call_end|><|tool_calls_section_end|>}'
    )
    raw += "<|tool_call_end|><|tool_calls_section_end|>}" * 20
    result = conversion.translate_kimi_output(raw, encoder)
    assert result.count("</｜DSML｜invoke>") == 21
    assert result.count("</｜DSML｜tool_calls>") == 21
    assert result.count("<｜DSML｜invoke name=") == 1
    assert result.count("}") == 21
    assert encoder.eos_token not in result
    assert "<|tool_" not in result


def test_orphan_closings_before_a_second_tool_section_are_retained():
    call = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.computer_click:2"
        '<|tool_call_argument_begin|>{"x": 0.4}<|tool_call_end|><|tool_calls_section_end|>'
    )
    raw = (
        "Reason.</think>"
        + call
        + "<|tool_call_end|><|tool_calls_section_end|>Again.</think>"
        + call
        + "<|im_end|>"
    )
    result = conversion.translate_kimi_output(raw, encoder)
    assert result.count("<｜DSML｜invoke name=") == 2
    assert result.count("</｜DSML｜invoke>") == 3
    assert result.count("</｜DSML｜tool_calls>") == 3
    assert "Again.</think>" in result
    assert result.endswith(encoder.eos_token)
    assert "<|tool_" not in result

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from trie.tokenizer_manager import TokenizerManager


class RecordingTokenizer:
    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


def _recording_manager() -> tuple[TokenizerManager, RecordingTokenizer]:
    manager = object.__new__(TokenizerManager)
    tokenizer = RecordingTokenizer()
    manager._tokenizer = tokenizer
    return manager, tokenizer


def test_render_chat_matches_sglang_tool_normalization() -> None:
    manager, tokenizer = _recording_manager()
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"needle"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [{"type": "text", "text": "one"}, "two"],
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search documents.",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
    ]

    assert manager.render_chat(messages, tools=tools) == "rendered"
    assert tokenizer.messages[0]["content"] == ""
    assert tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "query": "needle"
    }
    assert tokenizer.messages[1]["content"] == "one two"
    assert list(tokenizer.kwargs["tools"][0]["function"]) == [
        "description",
        "name",
        "parameters",
        "strict",
    ]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"query":"needle"}'
    )


def test_render_chat_rejects_invalid_tool_call_arguments() -> None:
    manager, _ = _recording_manager()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": "[]"},
                }
            ],
        }
    ]

    with pytest.raises(TypeError, match="must be a JSON object"):
        manager.render_chat(messages)


def test_render_chat_forwards_reasoning_effort() -> None:
    manager, tokenizer = _recording_manager()

    manager.render_chat(
        [{"role": "user", "content": "hello"}],
        reasoning_effort="high",
    )

    assert tokenizer.kwargs["reasoning_effort"] == "high"


def test_tokenizers_backend_fallback_loads_local_artifacts(
    monkeypatch, tmp_path
) -> None:
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "unk_token": "[UNK]",
                "model_max_length": 1024,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        encoding="utf-8",
    )

    def unsupported_backend(*args, **kwargs):
        raise ValueError(
            "Tokenizer class TokenizersBackend does not exist or is not "
            "currently imported."
        )

    monkeypatch.setattr(
        "trie.tokenizer_manager.AutoTokenizer.from_pretrained",
        unsupported_backend,
    )

    manager = TokenizerManager(str(tmp_path))

    assert manager.encode("hello world") == [1, 2]
    assert manager.render_chat([{"role": "user", "content": "hello"}]) == "hello"


def test_tokenizer_loader_does_not_hide_other_value_errors(monkeypatch) -> None:
    def invalid_tokenizer(*args, **kwargs):
        raise ValueError("different tokenizer error")

    monkeypatch.setattr(
        "trie.tokenizer_manager.AutoTokenizer.from_pretrained",
        invalid_tokenizer,
    )

    with pytest.raises(ValueError, match="different tokenizer error"):
        TokenizerManager("not-a-model")

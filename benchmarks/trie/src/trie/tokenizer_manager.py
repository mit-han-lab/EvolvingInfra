import copy
import json
import os
import secrets
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import structlog

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = structlog.get_logger(__name__)


class TokenizerManager:
    def __init__(self, model_name: str, seed: int | None = None) -> None:
        self._model_name = model_name
        self._rng = np.random.default_rng(seed)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except ValueError as exc:
            if "Tokenizer class TokenizersBackend" not in str(exc):
                raise
            self._tokenizer = self._load_tokenizers_backend_fallback(model_name)

    @staticmethod
    def _load_tokenizers_backend_fallback(
        model_name: str,
    ) -> PreTrainedTokenizerFast:
        """Load transformers-v5 TokenizersBackend artifacts on transformers v4.

        Some model artifacts declare ``TokenizersBackend`` in
        tokenizer_config.json. That generic class exists in transformers v5,
        while trie intentionally pins v4. Loading the same tokenizer.json and
        chat template directly avoids a model-specific tokenizer dependency and
        keeps replay tokenization equal to the server.
        """
        model_path = Path(model_name)
        tokenizer_path = model_path / "tokenizer.json"
        config_path = model_path / "tokenizer_config.json"
        template_path = model_path / "chat_template.jinja"
        if not tokenizer_path.is_file() or not config_path.is_file():
            raise ValueError(
                "TokenizersBackend fallback requires a local model directory "
                "containing tokenizer.json and tokenizer_config.json"
            )

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        kwargs: dict[str, Any] = {
            "tokenizer_file": str(tokenizer_path),
            "clean_up_tokenization_spaces": config.get(
                "clean_up_tokenization_spaces", False
            ),
        }
        for field in ("bos_token", "eos_token", "unk_token", "pad_token"):
            value = config.get(field)
            if isinstance(value, (str, dict)):
                kwargs[field] = value
        model_max_length = config.get("model_max_length")
        if isinstance(model_max_length, int):
            kwargs["model_max_length"] = model_max_length
        if template_path.is_file():
            kwargs["chat_template"] = template_path.read_text(encoding="utf-8")

        logger.info(
            "loading generic TokenizersBackend artifact with PreTrainedTokenizerFast",
            model=model_name,
        )
        return PreTrainedTokenizerFast(**kwargs)

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False).input_ids

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._decode(token_ids)

    def count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def eos_token_id(self) -> int:
        token_id = self._tokenizer.eos_token_id
        if not isinstance(token_id, int):
            raise TypeError("tokenizer must define one primary EOS token ID")
        return token_id

    @cached_property
    def regular_token_ids(self) -> tuple[int, ...]:
        """Actual vocabulary IDs, excluding control/special tokens and ID holes."""
        special = set(self._tokenizer.all_special_ids)
        pool = tuple(sorted(set(self._tokenizer.get_vocab().values()) - special))
        if len(pool) < 2:
            raise ValueError("trajectory prefixes require at least two ordinary tokens")
        return pool

    def random_prefix_tokens(self, count: int) -> tuple[int, ...]:
        # Independent of benchmark seed/shuffle and different across process restarts.
        return tuple(secrets.choice(self.regular_token_ids) for _ in range(count))

    def render_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        reasoning_effort: str | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        normalized_messages = self._normalize_messages_for_sglang(messages)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": False,
        }
        if tools is not None:
            kwargs["tools"] = self._normalize_tools_for_sglang(tools)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if template_kwargs:
            kwargs.update(template_kwargs)
        rendered = self._tokenizer.apply_chat_template(normalized_messages, **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("chat template did not return text")
        return rendered

    @staticmethod
    def _normalize_messages_for_sglang(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = copy.deepcopy(messages)
        for message in normalized:
            if message.get("content") is None:
                message["content"] = ""

            if message.get("role") == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    function = (
                        tool_call.get("function")
                        if isinstance(tool_call, dict)
                        else None
                    )
                    if not isinstance(function, dict) or not isinstance(
                        function.get("arguments"), str
                    ):
                        continue
                    try:
                        arguments = json.loads(function["arguments"])
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "assistant tool call function.arguments must be valid JSON"
                        ) from exc
                    if not isinstance(arguments, dict):
                        raise TypeError(
                            "assistant tool call function.arguments must be a JSON object"
                        )
                    function["arguments"] = arguments

            content = message.get("content")
            if (
                message.get("role") == "tool"
                and isinstance(content, list)
                and all(
                    isinstance(part, str)
                    or (isinstance(part, dict) and part.get("type") == "text")
                    for part in content
                )
            ):
                message["content"] = " ".join(
                    part if isinstance(part, str) else part.get("text", "")
                    for part in content
                )
        return normalized

    @staticmethod
    def _normalize_tools_for_sglang(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = []
        for tool in tools:
            function = tool.get("function")
            if not isinstance(function, dict):
                raise TypeError("tool.function must be an object")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("tool.function.name must be a non-empty string")

            normalized_function = {
                "description": function.get("description"),
                "name": name,
                "parameters": copy.deepcopy(function.get("parameters")),
                "strict": bool(function.get("strict", False)),
            }
            defer_loading = function.get("defer_loading")
            if defer_loading is None:
                defer_loading = tool.get("defer_loading")
            if defer_loading is not None:
                normalized_function["defer_loading"] = defer_loading
            normalized.append(
                {
                    "type": tool.get("type", "function"),
                    "function": normalized_function,
                }
            )
        return normalized

    def _sample_token_ids(self, prompt_length: int) -> list[int]:
        return self._rng.integers(
            self._tokenizer.vocab_size, size=prompt_length
        ).tolist()

    def get_prompt(
        self,
        prompt_length: int,
        *,
        max_rounds: int = 10,
    ) -> str:
        """Return text that round-trips to ``prompt_length`` tokens, repairing
        decode/encode mismatches for up to ``max_rounds`` iterations."""
        if prompt_length == 0:
            return ""

        token_ids = self._sample_token_ids(prompt_length)
        prompt = self._decode(token_ids)
        retokenized_ids = self._encode(prompt)

        rounds = 0
        while len(retokenized_ids) != prompt_length and rounds < max_rounds:
            if len(retokenized_ids) < prompt_length:
                num_extras = prompt_length - len(retokenized_ids)
                token_ids = retokenized_ids + self._sample_token_ids(num_extras)
            else:
                token_ids = retokenized_ids[:prompt_length]
            prompt = self._decode(token_ids)
            retokenized_ids = self._encode(prompt)
            rounds += 1

        if len(retokenized_ids) != prompt_length:
            logger.warning(
                "Failed to synthesize prompt for "
                f"{self._model_name!r}: target_length={prompt_length}, "
                f"final_length={len(retokenized_ids)}, repair_rounds={rounds}, "
                f"max_rounds={max_rounds}. Returning best effort result."
            )
            return prompt

        return prompt

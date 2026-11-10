"""Target-tokenizer rendering, encode/decode, and role-compatibility probes.

Wraps the vendored ``trie.tokenizer_manager.TokenizerManager`` so conversion
renders prompts exactly the way the replay client would (message
normalization included). Startup probes detect roles the target chat
template silently drops and drive a deterministic remap so no recorded
content disappears from rendered prompts.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from strict_convert._vendor import engine_files_module, tokenizer_manager_module

_PROBE_MARKER = "STRICT-CONVERT-ROLE-PROBE-7f3a"


class BoundaryError(RuntimeError):
    """Continuation token ids do not decode back to the continuation text."""


def _flatten_parts_content(message: dict[str, Any]) -> dict[str, Any]:
    """Collapse OpenAI parts-list content to text, as the dsv4 serving path does.

    ``serving_chat`` flattens parts-list content with
    ``process_content_for_template_format(msg, "string", ...)`` before calling
    the dsv4 encoder, which is text-only and would otherwise embed the list
    itself into a content block. Corpora recorded through the OpenAI schema
    carry ``content`` as ``[{"type": "text", ...}]``, so conversion has to
    apply the same flattening to render what serving would render.
    """
    content = message.get("content")
    if isinstance(content, list):
        texts = [
            chunk["text"]
            for chunk in content
            if isinstance(chunk, dict) and chunk.get("type") in ("text", "input_text")
        ]
        message = dict(message)
        message["content"] = " ".join(texts) if texts else ""
    elif content is None and "content" in message:
        message = dict(message)
        message["content"] = ""
    return message


class TargetRenderer:
    def __init__(self, model_path: str, chat_encoding: str = "auto", thinking_mode: str = "thinking",
                 history_reasoning: str = "native", template_kwargs: dict[str, Any] | None = None) -> None:
        if chat_encoding not in ("auto", "template", "dsv4"):
            raise ValueError(f"unsupported chat encoding: {chat_encoding}")
        if thinking_mode not in ("thinking", "chat"):
            raise ValueError(f"unsupported thinking mode: {thinking_mode}")
        self.chat_encoding = (
            "dsv4" if "deepseek-v4" in Path(model_path).name.lower() else "template"
        ) if chat_encoding == "auto" else chat_encoding
        if history_reasoning not in ("native", "inline"):
            raise ValueError(f"unsupported history reasoning policy: {history_reasoning}")
        self.thinking_mode = thinking_mode
        # 'native' renders history the way the target chat encoding does, so a
        # DSV4 target does not re-feed an earlier turn's reasoning. 'inline'
        # folds the recorded emission into the context message content instead,
        # reproducing a capture whose harness did re-feed it. The two are
        # different workloads, not one right answer, so the policy is part of
        # the rendering identity.
        self.history_reasoning = history_reasoning
        self.force_history_inline = history_reasoning == "inline"
        self.reference = None
        self.rendering = {"chat_encoding": self.chat_encoding, "history_reasoning": history_reasoning}
        # Extra chat-template variables, as serving's chat_template_kwargs (e.g. GLM's
        # clear_thinking=false keeps every earlier turn's reasoning). Recorded only when
        # set, so workloads converted without them keep their rendering identity.
        self.template_kwargs = dict(template_kwargs or {})
        if self.template_kwargs:
            if self.chat_encoding == "dsv4":
                raise ValueError("template kwargs apply to chat templates, not the DSV4 encoder")
            self.rendering["template_kwargs"] = self.template_kwargs
        if self.chat_encoding == "dsv4":
            # Load the exact serving encoder without importing sglang/torch.
            path = engine_files_module().dsv4_encoder_path()
            spec = importlib.util.spec_from_file_location("strict_convert_dsv4_encoding", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load DSV4 reference encoder: {path}")
            self.reference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.reference)
            self.rendering.update({
                "thinking_mode": thinking_mode,
                "reasoning_effort_profile": "official",
                "encoder_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        else:
            path = Path(model_path) / "chat_template.jinja"
            if path.is_file():
                self.rendering["template_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manager_cls = tokenizer_manager_module().TokenizerManager
        self.model_path = model_path
        self._manager = manager_cls(model_path)
        self.vocab_len = len(self._manager._tokenizer)
        self.generation_tail = self._probe_generation_tail()
        self.forced_suffix_len = self._probe_forced_suffix_len()
        self.developer_dropped = not self._probe_role_kept(
            [
                {"role": "developer", "content": _PROBE_MARKER},
                {"role": "user", "content": "x"},
            ]
        )
        self.nonfirst_system_dropped = not self._probe_role_kept(
            [
                {"role": "system", "content": "a"},
                {"role": "user", "content": "x"},
                {"role": "system", "content": _PROBE_MARKER},
            ]
        )
        # Some templates (DeepSeek-V4 append) carry assistant reasoning only
        # inline in content and silently drop the OpenAI-style
        # reasoning_content field: a diff-derived output then collapses to
        # the turn-glue tokens and the recorded emission loses its think
        # block. Probe it so the builder can re-inline verbatim.
        reasoning_probe = self.render(
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "y",
                    "reasoning_content": _PROBE_MARKER,
                },
            ],
            tools=None,
            add_generation_prompt=False,
        )
        self.reasoning_field_dropped = _PROBE_MARKER not in reasoning_probe
        # Same trap for structured tool calls: a content-only template
        # renders neither the calls nor any marker of them, so a tool-call
        # turn's diff-derived output collapses to turn glue. The reasoning
        # probe cannot catch it — templates often branch on tool_calls.
        tool_calls_probe = self.render(
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "y",
                    "tool_calls": [
                        {
                            "id": "call_probe0",
                            "type": "function",
                            "function": {
                                "name": _PROBE_MARKER,
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            ],
            tools=None,
            add_generation_prompt=False,
        )
        self.tool_calls_dropped = _PROBE_MARKER not in tool_calls_probe

    def _probe_role_kept(self, messages: list[dict[str, Any]]) -> bool:
        return _PROBE_MARKER in self.render(messages, tools=None)

    def _probe_generation_tail(self) -> str:
        probe = [{"role": "user", "content": "x"}]
        with_prompt = self.render(probe, tools=None, add_generation_prompt=True)
        without = self.render(probe, tools=None, add_generation_prompt=False)
        if not with_prompt.startswith(without):
            raise RuntimeError(
                "target template generation prompt is not a suffix extension"
            )
        return with_prompt[len(without) :]

    def _probe_forced_suffix_len(self) -> int:
        """Length of the generation tail the completed render never shares.

        The generation tail splits into the assistant turn header (which a
        completed assistant block also starts with, e.g. ``]~b]ai\\n``) and a
        forced continuation the template appends only when generating (e.g.
        ``<think>\\n``). The prefix-fallback split must never consume the
        forced part: characters there match assistant content only by
        coincidence (``<think>`` vs ``<table>``), and eating them corrupts
        the recorded output head.
        """
        user = [{"role": "user", "content": "x"}]
        base = self.render(user, tools=None, add_generation_prompt=False)
        completed = self.render(
            user + [{"role": "assistant", "content": "z7"}],
            tools=None,
            add_generation_prompt=False,
        )
        assistant_block = completed[len(base) :]
        shared = 0
        limit = min(len(self.generation_tail), len(assistant_block))
        while (
            shared < limit
            and self.generation_tail[shared] == assistant_block[shared]
        ):
            shared += 1
        return len(self.generation_tail) - shared

    @property
    def manager(self):
        """The vendored TokenizerManager (for tools sharing its interface)."""
        return self._manager

    def render(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool = True,
    ) -> str:
        if self.reference is not None:
            # Match serving_chat's tool injection. The reference owns DSML,
            # role transitions, thinking history, and tool-result ordering.
            normalized = [_flatten_parts_content(m) for m in copy.deepcopy(messages)]
            if not normalized or normalized[0].get("role") != "system":
                normalized.insert(0, {"role": "system", "content": ""})
            if tools:
                normalized[0]["tools"] = copy.deepcopy(tools)
            result = self.reference.encode_messages(
                normalized, thinking_mode=self.thinking_mode,
                reasoning_effort_profile="official",
            )
            # encode_messages targets inference and always opens a final user
            # turn. Expose the HF flag without stripping completed assistants.
            last_role = normalized[-1].get("role")
            suffix = self.reference.ASSISTANT_SP_TOKEN + (
                "<think>" if self.thinking_mode == "thinking" else "</think>"
            )
            if not add_generation_prompt and last_role in ("user", "developer", "tool") and result.endswith(suffix):
                result = result[:-len(suffix)]
            return result
        return self._manager.render_chat(
            messages, tools=tools, add_generation_prompt=add_generation_prompt,
            template_kwargs=self.template_kwargs,
        )

    def encode(self, text: str) -> list[int]:
        return self._manager.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._manager.decode(token_ids)

    def tokenizer_identity(self) -> dict[str, Any]:
        tokenizer_json = Path(self.model_path) / "tokenizer.json"
        sha = (
            hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
            if tokenizer_json.is_file()
            else None
        )
        return {
            "path": self.model_path,
            "tokenizer_json_sha256": sha,
            "vocab_len": self.vocab_len,
            "rendering": self.rendering,
        }

    def identity_summary(self) -> dict[str, Any]:
        """Compact identity embedded in-band in every emitted trace.

        Guards against replaying token ids under the wrong tokenizer. Foreign
        ids may pass the target's numeric range checks, so a range check alone
        cannot catch a corpus/tokenizer mix-up.
        """
        if not hasattr(self, "_identity_summary"):
            identity = self.tokenizer_identity()
            sha = identity["tokenizer_json_sha256"]
            self._identity_summary = {
                "name": Path(self.model_path).name,
                "tokenizer_json_sha256_prefix": sha[:16] if sha else None,
                "vocab_len": self.vocab_len,
            }
        return self._identity_summary

    def remap_roles(
        self, messages: list[dict[str, Any]], base_index: int
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Remap roles the target template would silently drop.

        ``base_index`` is the position of ``messages[0]`` in the full
        context. Rules (applied only when the startup probe proved the role
        is dropped): ``developer`` at index 0 becomes ``system`` (it is the
        de-facto system prompt in the pi corpora); ``developer`` elsewhere
        and non-first ``system`` become ``user`` so their content stays in
        the rendered prompt. Counts are returned for trace metadata.
        """
        remapped: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for offset, message in enumerate(messages):
            index = base_index + offset
            role = message.get("role")
            new_role = None
            if role == "developer" and self.developer_dropped:
                new_role = "system" if index == 0 else "user"
            elif role == "system" and index > 0 and self.nonfirst_system_dropped:
                new_role = "user"
            if new_role is None:
                remapped.append(message)
                continue
            clone = copy.deepcopy(message)
            clone["role"] = new_role
            remapped.append(clone)
            counts[f"{role}->{new_role}"] = counts.get(f"{role}->{new_role}", 0) + 1
        return remapped, counts

    def continuation_ids(
        self, prefix_ids: list[int], text: str, *, tail: int = 32
    ) -> list[int]:
        """Encode ``text`` as a continuation of the already-encoded prefix.

        The ids are defined as ``encode(text)`` appended after
        ``prefix_ids``; the encode is accepted only if it decodes back to
        ``text`` both standalone and when replayed after the prefix tail
        (bounded-cost version of the client's boundary check).
        """
        ids = self.encode(text)
        if self.decode(ids) != text:
            raise BoundaryError("continuation does not decode back to its text")
        tail_ids = prefix_ids[-tail:]
        if tail_ids and self.decode(tail_ids + ids) != self.decode(tail_ids) + text:
            raise BoundaryError("continuation decode changes across the boundary")
        return ids

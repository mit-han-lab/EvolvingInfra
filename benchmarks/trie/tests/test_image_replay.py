import asyncio
import base64
import sys
import types

import pytest

from trie.client import Client
from trie.types import Generate, ReplayTrace


def test_recorded_images_require_authoritative_prompt_ids():
    with pytest.raises(ValueError, match="authoritative prompt token IDs"):
        Generate(1, recorded_image_data=["a.png"])
    with pytest.raises(ValueError, match="non-empty list"):
        Generate(1, recorded_prompt_token_ids=[1], recorded_image_data=[])


def test_replay_sends_screenshots_as_data_uris_and_expands_the_prompt(tmp_path, monkeypatch):
    (tmp_path / "shot.png").write_bytes(b"png-bytes")
    seen = {}

    def prepare_image_layout(prompt, images, config):
        seen.update(prompt=prompt, images=images, config=config)
        return None, prompt + [9, 9], prompt + [7], None

    monkeypatch.setitem(sys.modules, "sglang_image_layout",
                        types.SimpleNamespace(prepare_image_layout=prepare_image_layout))
    event = Generate(1, recorded_prompt_token_ids=[1, 2],
                     recorded_image_data=["shot.png", "data:image/png;base64,QQ=="])
    trace = ReplayTrace("t", [event], metadata={"image_config": {"patch": 14},
                                                "image_root": str(tmp_path)})
    images, expanded, cached = asyncio.run(Client._prepare_replay_images(trace, event, [1, 2]))
    assert images == ["data:image/png;base64," + base64.b64encode(b"png-bytes").decode(),
                      "data:image/png;base64,QQ=="]
    assert (expanded, cached) == ([1, 2, 9, 9], [1, 2, 7])
    assert seen["config"] == {"patch": 14}


def test_text_events_skip_image_preparation():
    event = Generate(1, recorded_prompt_token_ids=[1])
    trace = ReplayTrace("t", [event])
    assert asyncio.run(Client._prepare_replay_images(trace, event, [1])) == (None, [1], [1])


def test_rows_with_screenshots_pass_the_trace_schema():
    row = {"schema_version": 2, "trace_id": "t",
           "metadata": {"image_config": {"patch_size": 14}, "image_root": "/screens"},
           "events": [{"type": "generate", "max_tokens": 1, "recorded_prompt_token_ids": [1, 2],
                       "recorded_image_data": ["blobs/a.png"]}]}
    trace = ReplayTrace.from_jsonl_row(row)
    assert trace.events[0].recorded_image_data == ["blobs/a.png"]


def test_warm_start_renders_a_message_row_with_the_chat_template(monkeypatch):
    client = Client(endpoint="http://127.0.0.1:9/v1", model="m", api_key="x", max_retries=0,
                    replay_output_mode="model")
    rendered = {}

    class Tokenizer:
        def render_chat(self, messages, **kwargs):
            rendered["messages"] = messages
            return "rendered"

        def encode(self, text):
            return [11, 12, 13]

    async def execute(prompt, max_tokens, **kwargs):
        return types.SimpleNamespace(prompt_token_ids=list(prompt), generated_token_ids=[7])

    monkeypatch.setattr(client, "_require_tokenizer_manager", lambda purpose: Tokenizer())
    monkeypatch.setattr(client, "_execute_with_migration", execute)
    event = Generate(1, recorded_assistant={"role": "assistant", "content": "a"})
    trace = ReplayTrace("putnam", [event])
    messages = [{"role": "user", "content": "q"}]
    actual = asyncio.run(client._send_warmup(
        trace, event, 0, identity_trace_id="putnam", task_identity=None, replay_cycle_id=0,
        request_semaphore=None, prefix_registry=None, stream=False, trace_start=0.0,
        cache_salt=None, routing_key=None, strict_replay=True, block_tokens=None,
        messages=messages))
    assert rendered["messages"] == messages
    assert actual == [11, 12, 13, 7]
    asyncio.run(client._client.close())

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("zstandard")
from trie import engine_files

try:
    image_config = engine_files.image_layout().image_config
except ModuleNotFoundError as exc:  # Trie installed without the engine tree
    pytest.skip(str(exc), allow_module_level=True)

PATH = Path(__file__).resolve().parents[1] / "scripts/bind_cua_kimi_images.py"
SPEC = importlib.util.spec_from_file_location("kimi_bind", PATH)
BIND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BIND)
CONFIG = image_config(
    {"model_type": "kimi_k25"},
    163605,
    {
        "patch_size": 14,
        "merge_kernel_size": 2,
        "in_patch_limit": 16384,
        "patch_limit_on_one_side": 512,
        "fixed_output_tokens": None,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
    },
)


def screenshot(sha):
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": {
                        "$blob": sha,
                        "encoding": "data_url",
                        "media_type": "image/png",
                    }
                },
            }
        ],
    }


def generate(images):
    return {
        "type": "generate",
        "source_model": BIND.MODEL,
        "recorded_prompt_token_ids": [11] + [163605] * images + [12],
        "recorded_output_token_ids": [101, 102],
        "max_tokens": 2,
    }


def test_preserves_source_ids_waits_and_image_history_compaction():
    a, b = "a" * 64, "b" * 64
    first = generate(1)
    first["recorded_assistant"] = {"role": "assistant", "content": "recorded"}
    row = {
        "schema_version": 2,
        "initial_messages": [screenshot(a)],
        "metadata": {
            "model": BIND.MODEL,
            "requests": [
                {"num_images": 1, "prompt_tokens": 2693},
                {"num_images": 2, "prompt_tokens": 5384},
                {"num_images": 1, "prompt_tokens": 2693},
            ],
        },
        "events": [
            first,
            {"type": "wait", "seconds": 95.9, "actor": "tool"},
            {"type": "append_message", "message": screenshot(b)},
            generate(2),
            {
                "type": "replace_context",
                "reason": "compaction",
                "messages": [screenshot(b)],
            },
            generate(1),
        ],
    }
    before = copy.deepcopy(row)
    target, counts = BIND.bind_row(
        row, CONFIG, "/screenshots", lambda sha: (1920, 1080)
    )
    assert row == before
    events = [e for e in target["events"] if e["type"] == "generate"]
    assert [e["recorded_image_data"] for e in events] == [
        [BIND.blob_path(a)],
        [BIND.blob_path(a), BIND.blob_path(b)],
        [BIND.blob_path(b)],
    ]
    assert target["events"][1] == before["events"][1]
    assert target["events"][4] == before["events"][4]
    for e, old in zip(events, [first, row["events"][3], row["events"][5]], strict=True):
        assert e["recorded_output_token_ids"] == old["recorded_output_token_ids"]
        assert e["recorded_prompt_token_ids"] == old["recorded_prompt_token_ids"]
        assert e["max_tokens"] == old["max_tokens"]
        assert (
            e["recorded_prompt_token_ids_sha256"]
            == hashlib.sha256(
                json.dumps(
                    old["recorded_prompt_token_ids"], separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
    assert counts["generate"] == 3


@pytest.mark.parametrize("bad", ["model", "placeholder", "expanded_count"])
def test_wrong_model_missing_image_or_wrong_processor_expansion_is_rejected(bad):
    row = {
        "schema_version": 2,
        "initial_messages": [screenshot("a" * 64)],
        "metadata": {
            "model": BIND.MODEL,
            "requests": [{"num_images": 1, "prompt_tokens": 2693}],
        },
        "events": [generate(1)],
    }
    if bad == "model":
        row["events"][0]["source_model"] = "other"
    if bad == "placeholder":
        row["events"][0]["recorded_prompt_token_ids"] = [11, 12]
    if bad == "expanded_count":
        row["metadata"]["requests"][0]["prompt_tokens"] = 2694
    with pytest.raises(ValueError):
        BIND.bind_row(row, CONFIG, "/screenshots", lambda sha: (1920, 1080))

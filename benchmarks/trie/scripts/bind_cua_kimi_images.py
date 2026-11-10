"""Bind original OSWorld Kimi token IDs to content-addressed screenshot files.

Keep source tokens, budgets, waits, assistant messages and context replacements.
Only add the image-upload contract consumed by the existing replay client.
"""

import argparse
import copy
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import zstandard
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strict_convert._vendor import engine_files_module  # noqa: E402

_layout = engine_files_module().image_layout()
image_config, kimi_resize_config = _layout.image_config, _layout.kimi_resize_config

MODEL = "nvidia/Kimi-K2.6-NVFP4"


def screenshot_refs(messages):
    result = []
    for message in messages:
        parts = message.get("content")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if part.get("type") != "image_url":
                continue
            ref = part.get("image_url", {}).get("url")
            if (
                not isinstance(ref, dict)
                or not isinstance(ref.get("$blob"), str)
                or re.fullmatch(r"[0-9a-f]{64}", ref["$blob"]) is None
                or ref.get("media_type") != "image/png"
            ):
                raise ValueError(
                    "Expected an original content-addressed PNG screenshot"
                )
            result.append(ref["$blob"])
    return result


def blob_path(sha):
    return f"blobs/sha256/{sha[:2]}/{sha[2:]}"


def bind_row(row, config, image_root, dimensions):
    if row.get("schema_version") != 2 or row.get("metadata", {}).get("model") != MODEL:
        raise ValueError("Expected the original Kimi NVFP4 schema-v2 trace")
    target = copy.deepcopy(row)
    messages = list(row["initial_messages"])
    requests = row["metadata"]["requests"]
    position, counts = 0, Counter()
    for original, event in zip(row["events"], target["events"], strict=True):
        kind = original["type"]
        counts[kind] += 1
        if kind == "append_message":
            messages.append(original["message"])
        elif kind == "replace_context":
            messages = list(original["messages"])
        elif kind == "generate":
            if original.get("source_model") != MODEL:
                raise ValueError("Source token IDs belong to a different model")
            images = screenshot_refs(messages)
            ids = original["recorded_prompt_token_ids"]
            if ids.count(config["image_token_id"]) != len(images):
                raise ValueError(
                    "Prompt placeholders do not match the current screenshot history"
                )
            info = requests[position]
            if len(images) != info["num_images"]:
                raise ValueError(
                    "Current screenshot history differs from recorded request metadata"
                )
            expanded = len(ids)
            for sha in images:
                width, height = dimensions(sha)
                expanded += kimi_resize_config(width, height, config)["num_tokens"] - 1
            if expanded != info["prompt_tokens"]:
                raise ValueError(
                    "Checkpoint image expansion differs from the source request"
                )
            event["recorded_image_data"] = [blob_path(sha) for sha in images]
            event["recorded_prompt_token_count"] = len(ids)
            event["recorded_prompt_token_ids_sha256"] = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            if original.get("recorded_assistant") is not None:
                messages.append(original["recorded_assistant"])
            # Tokens and output length are authoritative; never retokenize or
            # repair the source's failed/truncated generation events.
            assert (
                event["recorded_prompt_token_ids"]
                == original["recorded_prompt_token_ids"]
            )
            assert (
                event["recorded_output_token_ids"]
                == original["recorded_output_token_ids"]
            )
            assert event["max_tokens"] == original["max_tokens"]
            position += 1
    if position != len(requests):
        raise ValueError("Request metadata and generate events have different lengths")
    target["metadata"].update(image_config=config, image_root=str(image_root))
    return target, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((args.model_path / "config.json").read_text())
    processor = json.loads((args.model_path / "preprocessor_config.json").read_text())
    config = image_config(model, model["media_placeholder_token_id"], processor)
    verified_images = {}

    def dimensions(sha):
        if sha not in verified_images:
            from PIL import Image

            path = args.image_root / blob_path(sha)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError(
                    "Screenshot bytes differ from their original content address"
                )
            with Image.open(io.BytesIO(data)) as image:
                size = image.size
            verified_images[sha] = {
                "width": size[0],
                "height": size[1],
                "bytes": len(data),
            }
        value = verified_images[sha]
        return value["width"], value["height"]

    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(
        source
    ) as reader, args.output.open("wb") as destination, zstandard.ZstdCompressor(
        level=6
    ).stream_writer(
        destination
    ) as writer:
        for line in io.TextIOWrapper(reader):
            row, found = bind_row(json.loads(line), config, args.image_root, dimensions)
            from trie.types import ReplayTrace

            ReplayTrace.from_jsonl_row(row)
            writer.write(
                (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
            counts.update(found)
            counts["episodes"] += 1
    if counts["episodes"] != 98 or counts["generate"] != 14920:
        raise ValueError(f"Incomplete OSWorld input: {dict(counts)}")
    receipt = dict(
        model=MODEL,
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        counts=dict(counts),
        image_config=config,
        image_root=str(args.image_root),
        verified_images=verified_images,
        source_token_ids_preserved=True,
        output_budgets_and_waits_preserved=True,
    )
    args.output.with_suffix(args.output.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k != "verified_images"}
            | {"unique_verified_images": len(verified_images)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

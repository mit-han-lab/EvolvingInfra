"""Retokenize the exact Kimi CUA trace for DeepSeek-V4 Vision replay.

Keep every request, wait and context replacement. Prompts carry one image
placeholder per screenshot; the replay client loads the referenced PNG files
and uploads their contents. Failed/truncated outputs remain failed/truncated
DSML streams, including incomplete arguments; they are never dropped or closed
with invented termination tokens.
"""

import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import tiktoken
import zstandard
from strict_convert._vendor import engine_files_module, types_module
from tokenizers import Tokenizer

MODEL = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REVISION = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
SOURCE_CONTROLS = re.compile(r"<\|(?:tool_call[^|]*|im_end|im_start)\|>")


def translate_kimi_output(text, encoder):
    """Translate source control syntax while preserving unfinished emissions."""
    ended = text.endswith("<|im_end|>")
    if ended:
        text = text[: -len("<|im_end|>")]
    section = "<|tool_calls_section_begin|>"
    if section not in text:
        # Degenerate source outputs can repeat orphan closing markers. Retain
        # that malformed structure in DSML instead of repairing or dropping it.
        result = text.replace("<|tool_call_end|>", f"\n</{encoder.dsml_token}invoke>")
        result = result.replace(
            "<|tool_calls_section_end|>", f"\n</{encoder.dsml_token}tool_calls>"
        )
    else:
        prefix, body = text.split(section, 1)
        closed = "<|tool_calls_section_end|>" in body
        suffix = ""
        if closed:
            body, suffix = body.split("<|tool_calls_section_end|>", 1)
        calls = body.split("<|tool_call_begin|>")
        if calls[0].strip():
            raise ValueError("Unexpected content before a Kimi tool call")
        converted = []
        for call in calls[1:]:
            marker = "<|tool_call_argument_begin|>"
            header, separator, arguments = call.partition(marker)
            name = header.rsplit(":", 1)[0].removeprefix("functions.")
            if not separator:
                converted.append(f'<{encoder.dsml_token}invoke name="{name}')
                continue
            invocation = f'<{encoder.dsml_token}invoke name="{name}">\n'
            arguments, completed, trailing = arguments.partition("<|tool_call_end|>")
            if completed:
                if trailing.strip():
                    raise ValueError("Unexpected trailing Kimi tool-call content")
                invocation += encoder.encode_arguments_to_dsml({"arguments": arguments})
                invocation += f"\n</{encoder.dsml_token}invoke>"
            else:
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    # The released encoder preserves invalid complete arguments
                    # in an `arguments` parameter. Leave that parameter open for
                    # an incomplete source emission, preserving the raw suffix.
                    invocation += (
                        f'<{encoder.dsml_token}parameter name="arguments" string="true">'
                        + arguments
                    )
                else:
                    invocation += encoder.encode_arguments_to_dsml(
                        {"arguments": arguments}
                    )
            converted.append(invocation)
        result = (
            translate_kimi_output(prefix, encoder)
            + f"\n\n<{encoder.dsml_token}tool_calls>\n"
            + "\n".join(converted)
        )
        if closed:
            result += f"\n</{encoder.dsml_token}tool_calls>" + translate_kimi_output(
                suffix, encoder
            )
    if ended:
        result += encoder.eos_token
    if SOURCE_CONTROLS.search(result):
        raise ValueError("Source tool syntax survived DeepSeek conversion")
    return result


def normalize_images(value):
    if isinstance(value, dict):
        if "$blob" in value:
            return "blob://" + value["$blob"]
        return {key: normalize_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_images(item) for item in value]
    return value


def convert_row(row, tokenizer, source_tokenizer, encoder, config, image_root):
    """Return one complete target trace and its conversion counters."""
    target = copy.deepcopy(row)
    messages = list(row["initial_messages"])
    counts = Counter()
    infos = row["metadata"]["requests"]
    position = 0
    origin_ms = infos[0]["request_time_ms"]
    types = types_module()
    for original, event in zip(row["events"], target["events"], strict=True):
        kind = original["type"]
        counts[kind] += 1
        if kind == "append_message":
            messages.append(original["message"])
        elif kind == "replace_context":
            messages = list(original["messages"])
        elif kind == "generate":
            rendered_messages = normalize_images(messages)
            rendered_messages[0] = {**rendered_messages[0], "tools": row["tools"]}
            text, media = encoder.encode_messages(
                rendered_messages,
                thinking_mode="thinking",
                drop_thinking=False,
                return_multi_modal_data=True,
            )
            prompt = tokenizer.encode(text, add_special_tokens=False).ids
            images = []
            for image in media["images"]:
                url = image["url"]
                if (
                    not isinstance(url, str)
                    or re.fullmatch(r"blob://[0-9a-f]{64}", url) is None
                ):
                    raise ValueError(
                        f"Invalid content-addressed screenshot reference: {url!r}"
                    )
                images.append(url[len("blob://") :] + ".png")
            if prompt.count(config["image_token_id"]) != len(images):
                raise ValueError(
                    "Target prompt image placeholders differ from image references"
                )
            if "[Unsupported" in text:
                raise ValueError("Target encoder did not preserve source content")
            assistant = original.get("recorded_assistant")
            if assistant is not None:
                emission = encoder.render_message(
                    1,
                    [{"role": "user", "content": ""}, assistant],
                    thinking_mode="thinking",
                    drop_thinking=False,
                )
                counts["structured_outputs"] += 1
            else:
                try:
                    emission = translate_kimi_output(
                        source_tokenizer.decode(original["recorded_output_token_ids"]),
                        encoder,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Output conversion failed: trace={row['trace_id']} request={original['source_request_id']}: {error}"
                    ) from error
                counts["recovered_outputs"] += 1
            output = tokenizer.encode(emission, add_special_tokens=False).ids
            if not prompt or not output:
                raise ValueError("Conversion produced empty prompt or output IDs")
            event.update(
                recorded_prompt_token_ids=prompt,
                recorded_prompt_token_count=len(prompt),
                recorded_prompt_token_ids_sha256=types.token_ids_sha256(prompt),
                recorded_output_token_ids=output,
                max_tokens=len(output),
                source_model=MODEL,
                source_t_s=(infos[position]["request_time_ms"] - origin_ms) / 1000,
                source_api_time_s=infos[position]["client_latency_ms"] / 1000,
            )
            if images:
                event["recorded_image_data"] = images
            provenance = target["metadata"]["requests"][position]
            provenance.update(
                source_output_token_count=len(original["recorded_output_token_ids"]),
                source_output_token_ids_sha256=types.token_ids_sha256(
                    original["recorded_output_token_ids"]
                ),
                source_prompt_token_count=len(original["recorded_prompt_token_ids"]),
                source_prompt_token_ids_sha256=types.token_ids_sha256(
                    original["recorded_prompt_token_ids"]
                ),
                target_raw_prompt_tokens=len(prompt),
                target_output_tokens=len(output),
                recovered_output=assistant is None,
            )
            if assistant is not None:
                messages.append(assistant)
            position += 1
    if position != len(infos):
        raise ValueError("Source request metadata differs from generate events")
    target["metadata"].update(
        model=MODEL,
        image_config=config,
        image_root=image_root,
        strict_conversion={
            "target_model": MODEL,
            "target_revision": REVISION,
            "source_model": row["metadata"]["model"],
            "output_encoding": "DeepSeek DSML; incomplete source arguments remain incomplete",
            "prompt_encoding": "original text IDs with one placeholder per uploaded PNG",
        },
    )
    types.ReplayTrace.from_jsonl_row(target)
    return target, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--image-root", default="/inputs/cua/images")
    parser.add_argument(
        "--encoding-file",
        type=Path,
        default=engine_files_module().dsv4_encoder_path(),
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dsv4_encoding", args.encoding_file)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    image_config = engine_files_module().image_layout().image_config

    tokenizer = Tokenizer.from_file(str(args.model_path / "tokenizer.json"))
    config = image_config(
        json.loads((args.model_path / "config.json").read_text()),
        tokenizer.token_to_id(encoder.IMAGE_PLACEHOLDER),
    )
    source_config = json.loads(
        (args.source_tokenizer / "kimi-tokenizer-config.json").read_text()
    )
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (
            line.split()
            for line in (args.source_tokenizer / "kimi-tiktoken.model")
            .read_bytes()
            .splitlines()
        )
    }
    source_tokenizer = tiktoken.Encoding(
        name="kimi-cua-source",
        pat_str=r".+",
        mergeable_ranks=ranks,
        special_tokens={
            value["content"]: int(key)
            for key, value in source_config["added_tokens_decoder"].items()
        },
    )
    totals = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(
        source
    ) as reader, args.output.open("wb") as destination, zstandard.ZstdCompressor(
        level=6
    ).stream_writer(
        destination
    ) as writer:
        for line in io.TextIOWrapper(reader):
            row, counters = convert_row(
                json.loads(line),
                tokenizer,
                source_tokenizer,
                encoder,
                config,
                args.image_root,
            )
            writer.write(
                (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
            )
            totals.update(counters)
            totals["episodes"] += 1
            print(json.dumps(dict(totals)), flush=True)
    if totals["episodes"] != 98 or totals["generate"] != 14920:
        raise ValueError(f"Incomplete CUA conversion: {dict(totals)}")
    receipt = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "target_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(args.encoding_file.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (args.model_path / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "counts": dict(totals),
        "model": MODEL,
        "revision": REVISION,
        "image_config": config,
    }
    args.output.with_suffix(args.output.suffix + ".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

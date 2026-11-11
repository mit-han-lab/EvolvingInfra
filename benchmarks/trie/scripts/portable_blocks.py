"""Make a hash-block workload independent of the model: drop the embedded token pool.

    python scripts/portable_blocks.py cc.workload.jsonl cc.portable.jsonl

The replay client then draws each block's tokens from the regular vocabulary of the tokenizer
it replays with (`tokenizer_model=<model directory>`), which is the pool the converters embed
for that tokenizer. Block ids, prefix sharing, timing and output lengths are unchanged.
"""
import json
import sys


def strip(node):
    """Remove every block_tokens.pool and the tokenizer the conversion targeted."""
    if isinstance(node, dict):
        spec = node.get("block_tokens")
        if isinstance(spec, dict):
            spec.pop("pool", None)
        conversion = node.get("strict_conversion")
        if isinstance(conversion, dict):
            conversion.pop("target_tokenizer", None)
        for value in node.values():
            strip(value)
    elif isinstance(node, list):
        for value in node:
            strip(value)


def main(source, target):
    rows = 0
    with open(source) as src, open(target, "x") as out:
        for line in src:
            row = json.loads(line)
            strip(row)
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows += 1
    print(f"wrote {rows} rows to {target}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])

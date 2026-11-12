#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Write tokenizer.json for a model that ships a tiktoken vocabulary (Kimi-K2.6).

    python export_tokenizer.py MODEL_DIR [OUTPUT]

The router reads a `tokenizers` JSON file. This converts the model's tiktoken vocabulary and
special tokens, checks every token id and a few encodings against the model's own tokenizer,
and writes OUTPUT (default MODEL_DIR/tokenizer.json). Needs `transformers` and `tiktoken`.
"""
from pathlib import Path
import sys

TEXTS = ["Hello world!", "中文与English mixed 123456", "\n\n  foo\tbar\n", "We're testing Ünicode naïve café 🙂",
         "<think>reason</think><|media_pad|>[BOS][EOS]"]


def main(model: str, output: str = "") -> int:
    from transformers import AutoTokenizer
    from transformers.convert_slow_tokenizer import TikTokenConverter

    slow = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    special = [token for token, _ in sorted(slow.special_tokens.items(), key=lambda item: item[1])]
    fast = TikTokenConverter(vocab_file=str(Path(model) / "tiktoken.model"), pattern=slow.pat_str,
                             additional_special_tokens=special).converted()
    if fast.get_vocab_size() != slow.vocab_size:
        raise SystemExit(f"vocabulary size differs: {fast.get_vocab_size()} and {slow.vocab_size}")
    for index in range(slow.vocab_size):
        if fast.id_to_token(index) != slow._convert_id_to_token(index):
            raise SystemExit(f"token {index} differs")
    for text in TEXTS:
        if fast.encode(text, add_special_tokens=False).ids != slow.encode(text):
            raise SystemExit(f"encoding differs for {text!r}")
    path = Path(output or Path(model) / "tokenizer.json")
    fast.save(str(path))
    print(f"wrote {path}: {slow.vocab_size} tokens, {len(special)} special")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) not in (2, 3) or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))

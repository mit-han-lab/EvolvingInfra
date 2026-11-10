"""Convert local trace corpora into strict-replay (tier-A) trie workloads.

Thin entry point: puts this scripts directory on sys.path and dispatches to
the strict_convert package. See strict_convert/README.md for the corpus
adapters, conversion rules, and validation subcommand.

Usage examples:
    python convert_traces_to_strict.py trie_text \
        --input browsecomp-glm52.workload.jsonl \
        --target-tokenizer /path/to/deepseek-v4-flash \
        --output workloads-strict/browsecomp-glm52.strict.jsonl

    python convert_traces_to_strict.py validate \
        --workload workloads-strict/browsecomp-glm52.strict.jsonl \
        --tokenizer /path/to/deepseek-v4-flash
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from strict_convert.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

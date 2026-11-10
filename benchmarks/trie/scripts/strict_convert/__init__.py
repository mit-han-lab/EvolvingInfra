"""One-time conversion of local trace corpora into strict-replay workloads.

Every adapter normalizes one corpus into a canonical intermediate form, a
single core builder re-renders each conversation with the target chat
template and emits tier-A ("fully token-authoritative") generate events:
``recorded_prompt_token_ids`` plus ``recorded_output_token_ids`` with
``max_tokens == len(recorded_output_token_ids)``, per trie replay-trace
schema v2. See README.md in this package for the conversion rules and the
design decisions taken where the source corpora are ambiguous.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Trie and SGLang integration provenance

Trie is embedded at `benchmarks/trie/`; the matching inference runtime is
DecoupledEngine in `engine/` (derived from SGLang). This copy deliberately excludes the source repository's
nested SGLang checkout and private runtime capture adapters.

## Imported Trie revision

- Upstream: `https://github.com/Applied-Compute/trie`
- Source: `https://github.com/crmsndu/trie.git` (public fork)
- Branch: `strict-replay`
- Commit: `1db432d7a2dcdea6155d5676b112cc5a83a28a6a`
- License: Apache-2.0

Local generic extensions add authoritative prompt-token replay, server response
verification, sticky routing, and versioned schema fixtures.

## Matching SGLang runtime

The repository-level runtime owns the other half of the contract:

- official completion/chat token-ID responses, originally landed upstream as
  `410ab4fde`;
- forced output IDs through normal scheduler/decode state;
- forced output IDs through generic EAGLE/MTP verification;
- fail-closed validation for context limits, incompatible constraints, and
  unsupported speculative algorithms.

Model-specific launch settings and validation results are intentionally not part
of this provenance document.

## Updating Trie

Import only generic harness files and preserve the source commit above. Never
copy the source repository's nested `sglang/` directory. Port server changes to
the root SGLang tree and test both engines in the same revision.

The runtime client must not depend on private capture code. Generic offline
converters may live under `scripts/strict_convert/adapters/`; keep them outside
the runtime import graph and record their source formats in the converter README.

# Benchmarks

| Directory | Contents |
|---|---|
| [`trie/`](trie/README.md) | Trie, the strict-replay client: its own Python distribution (`trie`, Python >= 3.11, pins `transformers==4.57.6`); install it in a separate venv. Workload converters and the live conformance probe are in `trie/scripts/`. |
| [`replay/`](replay/README.md) | The adapter that runs Trie through the router's prepared path, and the checks that decide whether a run is valid. |
| [`reproduce/`](reproduce/README.md) | Benchmark setup and expected results. |

The agent traces used in the paper are on Hugging Face:
[crmsndu/evolving-infra-traces](https://huggingface.co/datasets/crmsndu/evolving-infra-traces).
[Workloads](workloads.md) describes each file, how to replay it, and how to run it on another model.

```bash
hf download crmsndu/evolving-infra-traces --repo-type dataset --local-dir traces
```

```bash
python3 -m venv .venv-trie && .venv-trie/bin/pip install -e 'benchmarks/trie[test]'
.venv-trie/bin/python -m pytest -q benchmarks/trie/tests
```

Trie's DSV4 converters and image replay read two standard-library files from the engine tree by
path (the DSV4 chat encoder and the image layout); set `DECOUPLED_ENGINE_DIR` to the
`decoupled_engine` package directory when Trie runs outside this checkout.

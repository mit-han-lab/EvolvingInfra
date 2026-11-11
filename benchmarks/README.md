# Benchmarks

One directory per workload of the paper, and the tools that replay and score them.

| Workload | What it stresses | Model of the paper's runs |
|---|---|---|
| [`claude-code/`](claude-code/README.md) | long shared prefixes, prefill-heavy | GLM-5.2, DeepSeek-V4-Flash |
| [`putnam/`](putnam/README.md) | long outputs, decode-heavy | GLM-5.2 |
| [`osworld/`](osworld/README.md) | a screenshot per step, prefill-heavy | Kimi-K2.6 |

Each has a README (what the workload is, the paper's setup and results, how to run it on
another model) and an `AGENTS.md` with the rules for an agent that runs it.

| Tools | Contents |
|---|---|
| [`trie/`](trie/README.md) | Trie, the strict-replay client: its own Python distribution (`trie`, Python >= 3.11, pins `transformers==4.57.6`); install it in a separate venv. Workload converters and the live conformance probe are in `trie/scripts/`. |
| [`replay/`](replay/README.md) | The adapter that runs Trie through the router's prepared path, the checks that decide whether a run is valid, and the scoring. |
| [`reproduce/`](reproduce/README.md) | The setup shared by the paper's experiments, how to compare a run with them, and the Modal benchmarks. |

The workloads are on Hugging Face:
[crmsndu/evolving-infra-traces](https://huggingface.co/datasets/crmsndu/evolving-infra-traces).
Each is one JSONL file that the replay client reads as it is (`workload_path=`), with its
sha256 beside it. One line is one task: an agent session with its sub-agents, as `generate`
events (a request with a fixed output length), `wait` events (tool and user time) and the
edges that say when a sub-agent starts and joins.

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

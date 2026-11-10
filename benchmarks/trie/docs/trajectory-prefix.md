# Isolating repeated Roadmap and EdgeBench trajectories

Add `trajectory_prefix_tokens=10` to the Trie replay command for **both** arms:

```bash
uv run trie workload_path=traces.jsonl \
  endpoint=http://127.0.0.1:30000/v1 model=served-model \
  tokenizer_model=/path/to/model strict_replay=true max_retries=0 \
  replay_output_mode=model trajectory_prefix_tokens=10 \
  cache_salt_mode=global concurrency=256 duration=5400
```

Use the same option with Roadmap's schema-v3 complete tasks and EdgeBench's
schema-v2 trajectories. Each admitted instance gets ten random ordinary token
IDs from the target tokenizer. Its first request begins with this prefix, and
every later request keeps it, including after context replacement/compaction.
All child agents in one v3 task share it. Repeated or concurrent instances of
the same task get fresh prefixes, including across separate replay processes.
Sampling uses OS randomness, independently of workload shuffle/warm-start seeds.
Full-prefix collision is probabilistic, negligible with ten ordinary tokens
from a normal model vocabulary; it is not a mathematical uniqueness guarantee.

This prevents a repeated trajectory from inheriting the previous instance's
cached context, while allowing prefix reuse inside its own instance. It also
isolates different tasks that originally shared a prompt: use this setting
when task isolation is the intended workload, not when evaluating cross-task
prefix sharing. Partial matches shorter than ten tokens remain possible.

The input sent for **each** request is the original input plus ten tokens;
output budgets, waits, dependencies, routing keys and cache-salt settings are
unchanged. Reserve ten additional tokens under the engine's context limit.
Adding real tokens can affect model output and speculative acceptance; this
is a workload change and must be identical in compared arms. Prior completed
measurements without it remain a different workload configuration.

Source token IDs/counts/hashes are validated before prepending. Returned prompt
IDs must match the entire sent prompt, including the prefix. Stationary warmup
uses the same prefix as the measured continuation. Source data is never edited.
Lifecycle logs record prefix length/hash and, on completion, original prompt
length (`source_prompt_tokens`); `prompt_tokens` and cache overlap metrics use
the actual prefixed input. Prefix contents are not added to lifecycle logs.

The default is `trajectory_prefix_tokens=0`, preserving existing replay inputs
(including CC/Codex). The option requires strict replay. Unlike the default
token-authoritative path, enabling it loads the target tokenizer and prepares
its ordinary-token pool once **before** the measured window; it does not
tokenize each request. Both real generation and forced-output modes work.
